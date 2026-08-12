import numpy as np
import pandas as pd
import pytest

from fmri_decomposition.activation import (RunRef, build_activation_table,
                                           extract_parcels, process_run)
from fmri_decomposition.config import ConfigError, config_from_dict
from fmri_decomposition.timing import (RunSegment, build_time_axis, censor_mask,
                                       segments_from_scans)

nib = pytest.importorskip("nibabel")


def cfg_for(tmp_path, **over):
    base = {
        "cohort": "t", "tr": 1.0,
        "derivatives_root": str(tmp_path), "output_root": str(tmp_path / "out"),
        "atlases": ["toy"],
    }
    base.update(over)
    return config_from_dict(base)


def toy_atlas_and_img(n_tr=60, n_parcels=4, shape=(8, 8, 6), seed=0):
    from fmri_decomposition.atlases.registry import AtlasSpec

    labels = np.zeros(shape, dtype=np.int16)
    bounds = np.linspace(0, shape[0], n_parcels + 1).astype(int)
    for p, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
        labels[lo:hi] = p
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    lab_img = nib.Nifti1Image(labels, affine)
    spec = AtlasSpec.from_label_image("toy", lab_img,
                                      [f"Left_P{i}" for i in range(n_parcels)])

    rng = np.random.default_rng(seed)
    data = rng.normal(size=shape + (n_tr,)).astype(np.float32)
    known = np.arange(1, n_parcels + 1, dtype=np.float32) * 10.0
    for p in range(1, n_parcels + 1):
        data[labels == p] += known[p - 1]
    return spec, nib.Nifti1Image(data, affine), labels, known


class TestExtractParcels:
    def test_parcel_mean_is_the_mean_of_its_voxels(self):
        spec, img, labels, known = toy_atlas_and_img()
        ts, counts = extract_parcels(img, spec)
        data = np.asarray(img.dataobj)
        for p in range(1, 5):
            expected = data[labels == p].mean(axis=0)
            assert np.allclose(ts[:, p - 1], expected, atol=1e-5)
        assert counts.tolist() == [int((labels == p).sum()) for p in range(1, 5)]

    def test_shape_and_dtype(self):
        spec, img, _, _ = toy_atlas_and_img(n_tr=37)
        ts, _ = extract_parcels(img, spec)
        assert ts.shape == (37, 4) and ts.dtype == np.float32

    def test_chunking_does_not_change_the_result(self):
        spec, img, _, _ = toy_atlas_and_img(n_tr=50)
        a, _ = extract_parcels(img, spec, time_chunk=7)
        b, _ = extract_parcels(img, spec, time_chunk=1000)
        assert np.array_equal(a, b)

    def test_empty_parcel_becomes_nan_not_a_missing_column(self):
        spec, img, labels, _ = toy_atlas_and_img()
        mask = np.ones(labels.shape, dtype=bool)
        mask[labels == 2] = False
        ts, counts = extract_parcels(img, spec, mask)
        assert ts.shape[1] == 4, "column count must not depend on the subject's mask"
        assert np.isnan(ts[:, 1]).all()
        assert np.isfinite(ts[:, 0]).all()

    def test_rejects_3d_input(self):
        spec, img, labels, _ = toy_atlas_and_img()
        vol = nib.Nifti1Image(np.asarray(img.dataobj)[..., 0], img.affine)
        with pytest.raises(ValueError, match="4D"):
            extract_parcels(vol, spec)


class TestTimeAxis:
    def test_identity_case_has_equal_time_columns(self):
        ax = build_time_axis(100, 2.0)
        assert np.allclose(ax.time_s, np.arange(100) * 2.0)
        assert np.array_equal(ax.time_s, ax.stimulus_time_s)
        assert ax.source == "identity"

    def test_segments_produce_a_continuous_stimulus_clock(self):
        segs = [RunSegment(50, 0.0), RunSegment(30, 50.0)]
        ax = build_time_axis(80, 1.0, segs)
        assert ax.stimulus_time_s[50] == pytest.approx(50.0)
        assert ax.run_idx[49] == 0 and ax.run_idx[50] == 1

    def test_segment_mismatch_raises_rather_than_shifting_silently(self):
        with pytest.raises(ValueError, match="segments sum"):
            build_time_axis(80, 1.0, [RunSegment(50), RunSegment(40)])

    def test_scans_tsv_gaps_do_not_advance_stimulus_time(self):
        """NNDb sub-1: runs at 13:14:18, 13:20:33, 14:17:53."""
        acq = [0.0, 375.0, 3815.0]
        segs = segments_from_scans(acq, [300, 300, 200], tr=1.0)
        assert [s.stimulus_offset_s for s in segs] == [0.0, 300.0, 600.0]

    def test_negative_gap_is_an_error(self):
        with pytest.raises(ValueError, match="negative inter-run gap"):
            segments_from_scans([0.0, 100.0], [300, 300], tr=1.0)

    def test_unknown_timing_source_rejected(self):
        with pytest.raises(ValueError):
            build_time_axis(10, 1.0, source="vibes")


class TestCensorMask:
    def test_dilation_removes_neighbours(self):
        good = censor_mask(10, censored=[5], dilate_tr=1)
        assert good.tolist() == [True] * 4 + [False, False, False] + [True] * 3

    def test_no_dilation_when_disabled(self):
        good = censor_mask(10, censored=[5], dilate_tr=0)
        assert good.sum() == 9

    def test_dilation_at_the_edges_does_not_wrap(self):
        good = censor_mask(5, censored=[0], dilate_tr=1)
        assert good.tolist() == [False, False, True, True, True]

    def test_accepts_a_good_array(self):
        g = np.array([1, 1, 0, 1, 1])
        assert censor_mask(5, good=g, dilate_tr=0).tolist() == [True, True, False, True, True]


class TestTableContract:
    def test_columns_and_dtypes(self, tmp_path):
        cfg = cfg_for(tmp_path)
        spec, img, _, _ = toy_atlas_and_img(n_tr=20)
        ts, _ = extract_parcels(img, spec)
        ax = build_time_axis(20, cfg.tr)
        ref = RunRef(cohort="t", sub="01", task="movie", bold=tmp_path / "x.nii.gz")
        table = build_activation_table(ref, spec, cfg, ts, ax, np.ones(20, bool))
        assert table.column_names[:5] == ["t", "time_s", "stimulus_time_s",
                                          "good_frame", "run_idx"]
        assert str(table.schema.field("t").type) == "int32"
        assert str(table.schema.field("time_s").type) == "float"
        assert str(table.schema.field("good_frame").type) == "bool"
        assert all(str(table.schema.field(c).type) == "float" for c in spec.columns)
        assert table.schema.metadata[b"atlas"] == b"toy"
        # Partition keys are the path's job, not the file's.
        for key in ("cohort", "atlas", "task", "sub"):
            assert key not in table.column_names
            assert key.encode() in table.schema.metadata
        for key in ("ses", "run", "acq", "run_key"):
            assert key in table.column_names

    def test_run_key_is_stable_and_includes_present_entities(self, tmp_path):
        ref = RunRef(cohort="c", sub="01", task="m", bold=tmp_path / "x", ses="003", run="01")
        assert ref.run_key == "cohort-c_sub-01_task-m_ses-003_run-01"


class TestProcessRun:
    def test_end_to_end_single_run(self, tmp_path):
        cfg = cfg_for(tmp_path)
        spec, img, _, _ = toy_atlas_and_img(n_tr=40)
        bold = tmp_path / "sub-01_task-movie_bold.nii.gz"
        nib.save(img, bold)
        ref = RunRef(cohort="t", sub="01", task="movie", bold=bold)
        entry = process_run(ref, spec, cfg)
        assert entry.status == "ok", entry.detail
        df = pd.read_parquet(entry.path)
        assert len(df) == 40 and df["good_frame"].all()

    def test_second_call_skips(self, tmp_path):
        cfg = cfg_for(tmp_path)
        spec, img, _, _ = toy_atlas_and_img(n_tr=20)
        bold = tmp_path / "sub-01_task-movie_bold.nii.gz"
        nib.save(img, bold)
        ref = RunRef(cohort="t", sub="01", task="movie", bold=bold)
        assert process_run(ref, spec, cfg).status == "ok"
        assert process_run(ref, spec, cfg).status == "skipped"

    def test_trim_longer_than_the_file_fails_loudly(self, tmp_path):
        cfg = cfg_for(tmp_path, trim={"column": "end_movie", "unit": "seconds"})
        spec, img, _, _ = toy_atlas_and_img(n_tr=20)
        bold = tmp_path / "sub-01_task-movie_bold.nii.gz"
        nib.save(img, bold)
        ref = RunRef(cohort="t", sub="01", task="movie", bold=bold, trim_end_s=999.0)
        entry = process_run(ref, spec, cfg)
        assert entry.status == "error" and "trim asks for" in entry.detail

    def test_trim_unit_tr_is_honoured(self, tmp_path):
        cfg = cfg_for(tmp_path, tr=2.0, trim={"column": "end_movie", "unit": "tr"})
        spec, img, _, _ = toy_atlas_and_img(n_tr=40)
        bold = tmp_path / "sub-01_task-movie_bold.nii.gz"
        nib.save(img, bold)
        ref = RunRef(cohort="t", sub="01", task="movie", bold=bold, trim_end_s=10.0)
        entry = process_run(ref, spec, cfg)
        assert entry.status == "ok", entry.detail
        assert len(pd.read_parquet(entry.path)) == 10


class TestConfig:
    def test_tr_is_mandatory(self, tmp_path):
        with pytest.raises(ConfigError, match="tr must be"):
            cfg_for(tmp_path, tr=None)

    def test_trim_unit_must_be_declared_correctly(self, tmp_path):
        with pytest.raises(ConfigError, match="trim.unit"):
            cfg_for(tmp_path, trim={"column": "end_movie", "unit": "frames"})

    def test_unknown_keys_are_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="unknown"):
            cfg_for(tmp_path, windows={"sizes_s": [30], "overlap_factor": 3})

    def test_missing_stimulus_duration_is_an_error(self, tmp_path):
        cfg = cfg_for(tmp_path)
        with pytest.raises(ConfigError, match="stimulus duration"):
            cfg.stimulus_duration_s("movie")
        assert cfg.stimulus_duration_s("movie", fallback=120.0) == 120.0

    def test_hash_is_stable_and_sensitive(self, tmp_path):
        a = cfg_for(tmp_path)
        b = cfg_for(tmp_path)
        c = cfg_for(tmp_path, tr=2.0)
        assert a.hash() == b.hash() != c.hash()

    def test_window_tr_uses_the_cohort_tr(self, tmp_path):
        assert cfg_for(tmp_path, tr=2.47).window_tr(300) == 121
