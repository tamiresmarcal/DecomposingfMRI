"""Stage 5a's pooling invariants.

Everything here is a property that, when wrong, produces tables that LOOK fine
and cannot be pooled: a matrix whose width depends on which states a cohort
happened to visit, a dwell time computed with the wrong stride, or a state set
that silently never gets a table because nothing named it.
"""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fmri_decomposition import transitions as T


def latents(tmp_path, cols, metadata=None, n=12):
    d = pd.DataFrame({"cohort": "c", "task": "movie", "sub": "01",
                      "window_id": np.arange(n),
                      "pca0/3": 0.0, "pca1/3": 0.0, "pca2/3": 0.0,
                      "crosses_run_boundary": False})
    for name, labels in cols.items():
        d[name] = labels
    md = {k.encode(): json.dumps(v).encode() for k, v in (metadata or {}).items()}
    p = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pandas(d, preserve_index=False)
                     .replace_schema_metadata(md), p)
    return p


class TestNStatesFor:
    def test_recorded_k_wins_over_the_labels(self):
        # The cohort visited 3 of 8 states. K is 8 -- a property of the FIT --
        # and taking it from the labels would give this cohort a 3x3 matrix
        # while another cohort got 8x8 for the same state set.
        p = None
        assert T.n_states_for("MeanShift_pca3_8", path=p,
                              labels=np.array([0, 1, 2])) == 8

    def test_provenance_beats_a_name_that_disagrees(self, tmp_path):
        p = latents(tmp_path, {"MeanShift_pca3_8": 0},
                    metadata={"clusterers": {"MeanShift_pca3_8": {"k": 5}}})
        assert T.n_states_for("MeanShift_pca3_8", path=p) == 5

    def test_falls_back_to_the_labels_only_with_a_warning(self, capsys):
        got = T.n_states_for("Odd_pca3_x", path=None,
                             labels=np.array([0, 1, 2, 3]))
        assert got == 4
        assert "WARNING" in capsys.readouterr().out

    def test_refuses_to_guess_with_nothing_to_go_on(self):
        with pytest.raises(SystemExit, match="cannot determine"):
            T.n_states_for("Odd_pca3_x", path=None, labels=None)


class TestTimeAxis:
    def test_prefers_the_recorded_stride(self, tmp_path):
        p = latents(tmp_path, {"HMM_pca3_8": 0},
                    metadata={"stride_s": 2.47, "indep_factor": 1})
        assert T.time_axis(p, "-1", n_overlaps=5) == (2.47, 1)

    def test_derives_it_for_a_sliding_window(self, tmp_path):
        p = latents(tmp_path, {"HMM_pca3_8": 0})
        assert T.time_axis(p, "60", n_overlaps=5) == (12.0, 5)

    def test_refuses_a_negative_window_with_no_recorded_stride(self, tmp_path):
        # window_s / n_overlaps would be -0.2 s, and every dwell time and switch
        # rate derived from it would come out negative without complaint.
        p = latents(tmp_path, {"HMM_pca3_8": 0})
        with pytest.raises(SystemExit, match="no `stride_s`"):
            T.time_axis(p, "-1", n_overlaps=5)


class TestDiscoverStateColumns:
    def test_finds_every_family_without_being_told(self, tmp_path):
        p = latents(tmp_path, {"ThresholdCluster_pca3_8": 0,
                               "HMM_umap3_27": 0, "MeanShift_pca3_7": 0})
        assert T.discover_state_columns(p) == [
            "HMM_umap3_27", "MeanShift_pca3_7", "ThresholdCluster_pca3_8"]

    def test_does_not_mistake_the_embeddings_for_state_columns(self, tmp_path):
        p = latents(tmp_path, {"HMM_pca3_8": 0})
        found = T.discover_state_columns(p)
        assert found == ["HMM_pca3_8"]
        assert not any("pca0" in c for c in found)


class TestShorten:
    def test_never_merges_two_names(self):
        s = pd.Series(["ThresholdCluster_pca3_8", "MeanShift_umap3_8"])
        out = T._shorten(s)
        assert out.nunique() == 2

    def test_drops_a_prefix_shared_by_all(self):
        s = pd.Series(["HMM_pca3_8", "HMM_pca3_27"])
        assert T._shorten(s).to_list() == ["8", "27"]


class TestSubjectTransitions:
    def frame(self, labels, wid=None, boundary=None):
        n = len(labels)
        return pd.DataFrame({
            "window_id": np.arange(n) if wid is None else wid,
            "s": labels,
            "crosses_run_boundary": (np.zeros(n, bool) if boundary is None
                                     else boundary)})

    def test_cells_are_p_joint_and_sum_to_one(self):
        row = T.subject_transitions(self.frame([0, 1, 0, 1, 0]), "s", 2, 1.0, 1)
        cells = [row[c] for c in T.cell_names(2)]
        assert np.isclose(sum(cells), 1.0)
        assert row["n_transitions"] == 4

    def test_every_cell_is_present_even_when_never_visited(self):
        row = T.subject_transitions(self.frame([0] * 6), "s", 3, 1.0, 1)
        assert all(c in row for c in T.cell_names(3))
        assert row["0->0"] == 1.0 and row["2->1"] == 0.0

    def test_no_transition_is_counted_across_a_gap(self):
        # window_ids 0,1,2 then 10,11: four pairs, not five.
        row = T.subject_transitions(
            self.frame([0, 0, 0, 1, 1], wid=np.array([0, 1, 2, 10, 11])),
            "s", 2, 1.0, 1)
        assert row["n_transitions"] == 3

    def test_dwell_is_measured_within_segments_not_across_them(self):
        # One run of 3 and one of 2, never a run of 5.
        row = T.subject_transitions(
            self.frame([0, 0, 0, 0, 0], wid=np.array([0, 1, 2, 10, 11])),
            "s", 2, 1.0, 1)
        assert np.isclose(row["mean_dwell_s"], 2.5)

    def test_a_subject_whose_every_frame_is_a_boundary_still_appears(self):
        row = T.subject_transitions(
            self.frame([0, 1], boundary=np.ones(2, bool)), "s", 2, 1.0, 1)
        assert set(T.cell_names(2)) <= set(row)
        assert np.isnan(row["n_transitions"]) or row["n_transitions"] == 0

    def test_dwell_scales_with_the_stride(self):
        a = T.subject_transitions(self.frame([0, 0, 1, 1]), "s", 2, 1.0, 1)
        b = T.subject_transitions(self.frame([0, 0, 1, 1]), "s", 2, 2.47, 1)
        assert np.isclose(b["mean_dwell_s"], a["mean_dwell_s"] * 2.47)


class TestMaxK:
    """Discovery is how a NEW state set is picked up automatically. It must not
    also be how a RETIRED one comes back: latents written before K was cut to 8
    and 27 still carry ThresholdCluster_pca3_125 and _512, which are 15,625 and
    262,144 cells per subject."""

    def test_an_oversized_set_is_skipped(self):
        k_of = {"ThresholdCluster_pca3_8": 8, "ThresholdCluster_pca3_512": 512}
        keep, skip = T._within_k_band(sorted(k_of), k_of, 2, 64)
        assert keep == ["ThresholdCluster_pca3_8"]
        assert skip == ["ThresholdCluster_pca3_512"]

    def test_the_sets_in_use_survive_the_default(self):
        k_of = {"ThresholdCluster_pca3_8": 8, "HMM_umap3_27": 27,
                "MeanShift_pca3_7": 7}
        keep, skip = T._within_k_band(sorted(k_of), k_of, 2, 64)
        assert len(keep) == 3 and skip == []

    def test_an_unresolvable_k_is_kept_not_dropped(self):
        # `process` resolves it per cell with the labels in hand and says so.
        # Dropping it here would silently lose a state set over a missing field.
        keep, skip = T._within_k_band(["Odd_pca3_x"], {"Odd_pca3_x": None}, 2, 64)
        assert keep == ["Odd_pca3_x"] and skip == []

    def test_the_boundary_is_inclusive(self):
        k_of = {"MeanShift_pca3_64": 64, "MeanShift_pca3_65": 65}
        keep, skip = T._within_k_band(sorted(k_of), k_of, 2, 64)
        assert keep == ["MeanShift_pca3_64"]
        assert skip == ["MeanShift_pca3_65"]

    def test_k_quietly_returns_none_rather_than_guessing(self, tmp_path):
        p = latents(tmp_path, {"Odd_pca3_x": 0})
        assert T._k_quietly("Odd_pca3_x", p) is None

    def test_k_quietly_reads_the_recorded_k(self, tmp_path):
        p = latents(tmp_path, {"MeanShift_pca3_7": 0},
                    metadata={"clusterers": {"MeanShift_pca3_7": {"k": 7}}})
        assert T._k_quietly("MeanShift_pca3_7", p) == 7

    def test_a_single_state_set_is_skipped(self):
        # MeanShift at too wide a bandwidth wrote these before stage 4b refused
        # them. One state means one cell, the same value for every subject.
        k_of = {"MeanShift_pca3_1": 1, "MeanShift_pca3_7": 7}
        keep, skip = T._within_k_band(sorted(k_of), k_of, 2, 64)
        assert keep == ["MeanShift_pca3_7"]
        assert skip == ["MeanShift_pca3_1"]

    def test_both_ends_are_skipped_together(self):
        k_of = {"MeanShift_pca3_1": 1, "ThresholdCluster_pca3_8": 8,
                "ThresholdCluster_pca3_512": 512}
        keep, skip = T._within_k_band(sorted(k_of), k_of, 2, 64)
        assert keep == ["ThresholdCluster_pca3_8"]
        assert len(skip) == 2


class TestPerCellStateSets:
    """A method that DISCOVERS its K has a different column name in every cell.
    `MeanShift_pca3_16` exists at harvardoxford/30s and nowhere else, because
    that is where the bandwidth search landed on 16. Demanded across the grid it
    is missing from 14 cells, and stage 5a refused to start on an "incomplete"
    grid that was never incomplete. That is the failure these pin."""

    def cell(self, root, atlas, w, ms, cohorts=("a", "b"), mh=None, k_of=None):
        for c in cohorts:
            d = pd.DataFrame({"cohort": c, "task": "m", "sub": "s1",
                              "window_id": range(6),
                              "crosses_run_boundary": False,
                              "ThresholdCluster_pca3_8": 0, ms: 0})
            md = {b"model_hash": json.dumps(mh or f"h-{atlas}-{w}").encode(),
                  b"stride_s": json.dumps(6.0).encode(),
                  b"clusterers": json.dumps(
                      k_of or {ms: {"k": int(ms.rsplit("_", 1)[1])},
                               "ThresholdCluster_pca3_8": {"k": 8}}).encode()}
            p = (root / "latents" / f"atlas={atlas}" / f"window_s={w}"
                 / f"cohort={c}" / "data.parquet")
            p.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(d, preserve_index=False)
                             .replace_schema_metadata(md), p)

    def test_each_cell_uses_its_own_discovered_k(self, tmp_path):
        self.cell(tmp_path, "yeo7", "30", "MeanShift_pca3_16")
        self.cell(tmp_path, "yeo7", "60", "MeanShift_pca3_3")
        d = T.check_grid(tmp_path, ["yeo7"], ["30", "60"], ["a", "b"], None, 3, 64)
        ready = d[d["ok"]]
        at30 = set(ready[ready["window_s"] == "30"]["states"])
        at60 = set(ready[ready["window_s"] == "60"]["states"])
        assert "MeanShift_pca3_16" in at30 and "MeanShift_pca3_16" not in at60
        assert "MeanShift_pca3_3" in at60 and "MeanShift_pca3_3" not in at30
        # and the fixed-name set is usable in both
        assert "ThresholdCluster_pca3_8" in at30 & at60

    def test_a_cell_only_missing_a_set_is_not_fatal(self, tmp_path):
        self.cell(tmp_path, "yeo7", "30", "MeanShift_pca3_16")
        self.cell(tmp_path, "yeo7", "60", "MeanShift_pca3_3")
        d = T.check_grid(tmp_path, ["yeo7"], ["30", "60"], ["a", "b"], None, 3, 64)
        assert not d["fatal"].any()

    def test_an_absent_aperture_is_reported_not_fatal(self, tmp_path):
        self.cell(tmp_path, "yeo7", "30", "MeanShift_pca3_16")
        d = T.check_grid(tmp_path, ["yeo7"], ["30", "120"], ["a", "b"], None, 3, 64)
        gap = d[d["window_s"] == "120"]
        assert len(gap) and not gap["fatal"].any()
        assert (gap["reason"] == "no latents file").all()

    def test_differing_model_hash_IS_fatal(self, tmp_path):
        # The one that must stop everything: two cohorts from different fits.
        self.cell(tmp_path, "yeo7", "30", "MeanShift_pca3_4", cohorts=("a",),
                  mh="h1")
        self.cell(tmp_path, "yeo7", "30", "MeanShift_pca3_4", cohorts=("b",),
                  mh="h2")
        d = T.check_grid(tmp_path, ["yeo7"], ["30"], ["a", "b"], None, 3, 64)
        assert d["fatal"].any()
        assert any("model_hash differs" in r for r in d["reason"])

    def test_a_named_state_set_that_is_absent_IS_fatal(self, tmp_path):
        # Asked for by hand, so its absence is a mistake worth stopping on.
        self.cell(tmp_path, "yeo7", "30", "MeanShift_pca3_4")
        d = T.check_grid(tmp_path, ["yeo7"], ["30"], ["a", "b"],
                         ["HMM_umap3_8"], 3, 64)
        assert d["fatal"].all()

    def test_a_set_in_one_cohort_only_is_not_usable_by_either(self, tmp_path):
        # A state label is comparable across cohorts only if one fit defined it.
        self.cell(tmp_path, "yeo7", "30", "MeanShift_pca3_4", cohorts=("a",))
        self.cell(tmp_path, "yeo7", "30", "MeanShift_pca3_9", cohorts=("b",))
        usable, partial, _ = T.cell_states(tmp_path, "yeo7", "30", ["a", "b"],
                                           3, 64)
        assert "MeanShift_pca3_4" not in usable
        assert "MeanShift_pca3_9" not in usable
        assert usable == ["ThresholdCluster_pca3_8"]
        assert partial == {"a": ["MeanShift_pca3_4"], "b": ["MeanShift_pca3_9"]}

    def test_a_degenerate_k_is_excluded_per_cell(self, tmp_path):
        self.cell(tmp_path, "yeo7", "30", "MeanShift_pca3_1",
                  k_of={"MeanShift_pca3_1": {"k": 1},
                        "ThresholdCluster_pca3_8": {"k": 8}})
        usable, _, skipped = T.cell_states(tmp_path, "yeo7", "30", ["a", "b"],
                                           3, 64)
        assert usable == ["ThresholdCluster_pca3_8"]
        assert skipped == ["MeanShift_pca3_1"]
