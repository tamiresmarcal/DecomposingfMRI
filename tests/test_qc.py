"""Per-subject QC metrics and the ISC grouping they depend on."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from fmri_decomposition.io import write_table_atomic
from fmri_decomposition.qc import (QC_COLUMNS, activation_qc, add_stimulus_coverage,
                                   default_isc_parcels, pick_qc_atlas, qc_frame)
from fmri_decomposition.validate import isc_alignment, isc_gate


def toy_atlas(n_nodes=5, names=None, name="toy"):
    from fmri_decomposition.atlases.registry import AtlasSpec

    names = names or [f"P{i}" for i in range(n_nodes)]
    labels = pd.DataFrame({
        "index": np.arange(1, len(names) + 1), "name": names,
        "hemi": ["B"] * len(names), "x": 0.0, "y": 0.0, "z": 0.0, "network": "n",
    })
    return AtlasSpec(name=name, kind="labels", labels=labels)


def write_activation(tmp_path, sub, task, atlas, n_tr=100, good=None,
                     empty_cols=(), tr=1.0, signal=None):
    """A minimal stage-2 shard at a real hive path, so read_shard recovers keys."""
    rng = np.random.default_rng(abs(hash((sub, task))) % 2**32)
    cols = {}
    for j, c in enumerate(atlas.columns):
        if c in empty_cols:
            cols[c] = np.full(n_tr, np.nan, dtype=np.float32)
        elif signal is not None and j == 0:
            cols[c] = signal.astype(np.float32)
        else:
            cols[c] = rng.normal(size=n_tr).astype(np.float32)
    t = np.arange(n_tr)
    table = pa.table({
        "t": pa.array(t, pa.int32()),
        "time_s": pa.array(t * tr, pa.float32()),
        "stimulus_time_s": pa.array(t * tr, pa.float32()),
        "good_frame": pa.array(np.ones(n_tr, bool) if good is None else good, pa.bool_()),
        "run_idx": pa.array(np.zeros(n_tr, np.int8), pa.int8()),
        **{k: pa.array(v, pa.float32()) for k, v in cols.items()},
    })
    path = (tmp_path / f"atlas={atlas.name}" / "cohort=c" / f"task={task}"
            / f"sub={sub}" / "data.parquet")
    write_table_atomic(table, path)
    return path


class TestFracGoodFrames:
    def test_counts_the_censored_frames(self, tmp_path):
        atlas = toy_atlas()
        good = np.ones(100, bool)
        good[:20] = False
        p = write_activation(tmp_path, "01", "movie", atlas, good=good)
        qc = activation_qc([p], atlas, tr=1.0)
        assert qc[("01", "movie")]["frac_good_frames"] == pytest.approx(0.8)
        assert qc[("01", "movie")]["n_tr_total"] == 100

    def test_is_one_when_nothing_is_censored(self, tmp_path):
        """ds002837's case: the censor is disabled, so this carries no signal."""
        atlas = toy_atlas()
        p = write_activation(tmp_path, "01", "movie", atlas)
        assert activation_qc([p], atlas, tr=1.0)[("01", "movie")]["frac_good_frames"] == 1.0


class TestFracParcelsEmpty:
    def test_an_all_nan_parcel_counts_as_empty(self, tmp_path):
        """That is exactly how extract_parcels marks `counts == 0`."""
        atlas = toy_atlas(5)
        p = write_activation(tmp_path, "01", "movie", atlas, empty_cols=("P1", "P3"))
        row = activation_qc([p], atlas, tr=1.0)[("01", "movie")]
        assert row["n_parcels_empty"] == 2
        assert row["frac_parcels_empty"] == pytest.approx(0.4)

    def test_a_healthy_subject_has_none(self, tmp_path):
        atlas = toy_atlas(5)
        p = write_activation(tmp_path, "01", "movie", atlas)
        assert activation_qc([p], atlas, tr=1.0)[("01", "movie")]["frac_parcels_empty"] == 0.0

    def test_several_shards_report_the_worst_one(self, tmp_path):
        """One badly-registered session is a finding, not something to average."""
        atlas = toy_atlas(5)
        a = write_activation(tmp_path / "a", "01", "movie", atlas)
        b = write_activation(tmp_path / "b", "01", "movie", atlas,
                             empty_cols=("P0", "P1", "P2"))
        row = activation_qc([a, b], atlas, tr=1.0)[("01", "movie")]
        assert row["n_parcels_empty"] == 3


class TestFracStimulusCovered:
    def test_a_short_scan_is_flagged_against_the_longest_of_the_same_film(self, tmp_path):
        atlas = toy_atlas()
        files = [write_activation(tmp_path, "01", "movie", atlas, n_tr=1000),
                 write_activation(tmp_path, "02", "movie", atlas, n_tr=1000),
                 write_activation(tmp_path, "03", "movie", atlas, n_tr=600)]
        qc = activation_qc(files, atlas, tr=1.0)
        add_stimulus_coverage(qc)
        assert qc[("01", "movie")]["frac_stimulus_covered"] == pytest.approx(1.0)
        assert qc[("03", "movie")]["frac_stimulus_covered"] == pytest.approx(0.6)

    def test_the_reference_is_per_film_not_per_cohort(self, tmp_path):
        """ds002837's films run 95 to 154 minutes; a cohort-wide reference would
        mark every viewer of the short film as incomplete."""
        atlas = toy_atlas()
        files = [write_activation(tmp_path, "01", "long", atlas, n_tr=2000),
                 write_activation(tmp_path, "02", "short", atlas, n_tr=800),
                 write_activation(tmp_path, "03", "short", atlas, n_tr=800)]
        qc = activation_qc(files, atlas, tr=1.0)
        add_stimulus_coverage(qc)
        assert qc[("02", "short")]["frac_stimulus_covered"] == pytest.approx(1.0)
        assert qc[("03", "short")]["frac_stimulus_covered"] == pytest.approx(1.0)


class TestQcAtlasChoice:
    def test_the_finest_atlas_with_shards_wins(self, tmp_path):
        """Empty parcels are the point: Yeo's 7 networks are too coarse to
        register a normalisation failure that Harvard-Oxford's 111 would."""
        from fmri_decomposition.config import config_from_dict

        coarse, fine = toy_atlas(3, name="coarse"), toy_atlas(40, name="fine")
        cfg = config_from_dict({"cohort": "c", "tr": 1.0, "derivatives_root": str(tmp_path),
                                "output_root": str(tmp_path / "out"),
                                "atlases": ["coarse", "fine"]})
        root = tmp_path / "out" / "activation"
        write_activation(root, "01", "movie", coarse)
        write_activation(root, "01", "movie", fine)
        assert pick_qc_atlas(cfg, [coarse, fine]).name == "fine"


class TestIscSeed:
    def test_an_auditory_parcel_is_preferred(self):
        atlas = toy_atlas(names=["Frontal_Pole", "Heschls_Gyrus_includes_H1_and_H2",
                                 "Occipital_Pole"])
        cols, how = default_isc_parcels(atlas)
        assert cols == ["Heschls_Gyrus_includes_H1_and_H2"]
        assert "heschl" in how

    def test_the_arbitrary_fallback_says_it_is_arbitrary(self):
        atlas = toy_atlas(names=["A", "B", "C"])
        cols, how = default_isc_parcels(atlas)
        assert cols == ["A"] and "arbitrary" in how


class TestIscIsGroupedByStimulus:
    """The reference is only meaningful among people who saw the same thing."""

    def _cohort(self, tmp_path, atlas, n_per_task, n_tr=300, lag=None):
        rng = np.random.default_rng(0)
        files = []
        for task, n_subs in n_per_task.items():
            shared = rng.normal(size=n_tr + 100)
            for i in range(n_subs):
                sub = f"{task}{i:02d}"
                shift = (lag or {}).get(sub, 0)
                x = shared[50 + shift: 50 + shift + n_tr] + rng.normal(scale=0.3, size=n_tr)
                files.append(write_activation(tmp_path, sub, task, atlas,
                                              n_tr=n_tr, signal=x))
        return files

    def test_two_films_do_not_share_a_group_mean(self, tmp_path):
        """Pooled, each subject would be correlated against a mean of two
        unrelated soundtracks and report noise."""
        atlas = toy_atlas(names=["Heschls_Gyrus_includes_H1_and_H2", "B"])
        files = self._cohort(tmp_path, atlas, {"filmA": 4, "filmB": 4})
        isc = isc_alignment(files, parcels=("Heschls_Gyrus_includes_H1_and_H2",),
                            max_lag_tr=5)
        assert set(isc["movie"]) == {"filmA", "filmB"}
        assert (isc["n_subjects"] == 4).all()
        assert (isc["peak_isc"] > 0.5).all(), "within-film ISC should be high"
        assert (isc["best_lag_tr"] == 0).all()

    def test_a_shifted_subject_is_caught_within_its_own_film(self, tmp_path):
        atlas = toy_atlas(names=["Heschls_Gyrus_includes_H1_and_H2", "B"])
        files = self._cohort(tmp_path, atlas, {"filmA": 5}, lag={"filmA03": 8})
        isc = isc_alignment(files, parcels=("Heschls_Gyrus_includes_H1_and_H2",),
                            max_lag_tr=15)
        shifted = isc.loc[isc["sub"] == "filmA03", "best_lag_tr"].iloc[0]
        others = isc.loc[isc["sub"] != "filmA03", "best_lag_tr"]
        assert abs(shifted) == 8
        assert (others == 0).all()

    def test_a_film_with_too_few_subjects_gets_nan_and_a_reason(self, tmp_path):
        atlas = toy_atlas(names=["Heschls_Gyrus_includes_H1_and_H2", "B"])
        files = self._cohort(tmp_path, atlas, {"filmA": 4, "tiny": 2})
        isc = isc_alignment(files, parcels=("Heschls_Gyrus_includes_H1_and_H2",),
                            max_lag_tr=5)
        tiny = isc[isc["movie"] == "tiny"]
        assert len(tiny) == 2
        assert tiny["peak_isc"].isna().all()
        assert tiny["isc_note"].str.contains("2 subject").all()

    def test_the_gate_ignores_rows_with_no_computable_lag(self, tmp_path):
        isc = pd.DataFrame({"best_lag_tr": [0, 0, 1, np.nan, np.nan]})
        ok, msg = isc_gate(isc, 1.0)
        assert ok and "3 subject(s)" in msg and "2 row(s)" in msg

    def test_the_gate_fails_loudly_when_nothing_is_computable(self):
        ok, msg = isc_gate(pd.DataFrame({"best_lag_tr": [np.nan, np.nan]}), 1.0)
        assert not ok and "no subject" in msg


class TestQcFrame:
    """participants_qc.csv: machine-owned, keyed the way participants.csv is."""

    def test_it_is_keyed_for_the_join_the_models_have_to_do(self, tmp_path):
        from fmri_decomposition.config import config_from_dict

        cfg = config_from_dict({"cohort": "ds002837", "tr": 1.0,
                                "derivatives_root": str(tmp_path),
                                "output_root": str(tmp_path / "out")})
        df = qc_frame(cfg, {("12", "film"): {"mean_fd": 0.7},
                            ("03", "film"): {"mean_fd": 0.1}})
        assert list(df.columns) == QC_COLUMNS
        assert list(df["sub"]) == ["03", "12"]
        assert (df["cohort"] == "ds002837").all()
        assert list(df["participant_id"]) == ["sub-03", "sub-12"]

    def test_it_carries_no_exclusion_column(self):
        """Measurement here; the decision is at the models."""
        for forbidden in ("excluded", "exclusion_reason"):
            assert forbidden not in QC_COLUMNS

    def test_every_failure_mode_has_a_column(self):
        for col in ("mean_fd", "best_lag_tr", "frac_stimulus_covered",
                    "frac_good_frames", "frac_parcels_empty"):
            assert col in QC_COLUMNS
