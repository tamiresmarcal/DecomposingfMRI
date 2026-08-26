"""Confound regression and FD-based censoring.

These paths exist for fMRIPrep-style cohorts (CNeuroMod), which -- unlike an
AFNI-preprocessed cohort such as ds002837 -- ship neither a censor .1D nor
already-regressed timeseries. Before this was wired up nothing ever set
`RunRef.confounds`, so `load_confounds` was unreachable and a config asking
for motion regression silently got none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmri_decomposition.activation import HMP_6, HMP_24, RunRef, _good_frames, load_confounds
from fmri_decomposition.cli import _attach_confounds
from fmri_decomposition.config import ConfigError, config_from_dict

BASE = {
    "cohort": "c", "tr": 1.49,
    "derivatives_root": "/d", "output_root": "/o",
}


def _cfg(**confounds):
    return config_from_dict({**BASE, "confounds": confounds})


def _write_confounds(path, n_tr, fd=None, extra_cols=True):
    df = pd.DataFrame({c: np.linspace(0, 1, n_tr) for c in HMP_24})
    if extra_cols:
        df["csf"] = np.arange(n_tr, dtype=float)
    if fd is not None:
        df["framewise_displacement"] = fd
    df.to_csv(path, sep="\t", index=False)
    return path


class TestHmpColumns:
    def test_24hmp_is_six_params_times_four_terms(self):
        assert len(HMP_6) == 6
        assert len(HMP_24) == 24
        assert len(set(HMP_24)) == 24

    def test_every_base_param_appears_with_all_four_terms(self):
        for b in HMP_6:
            for suffix in ("", "_derivative1", "_power2", "_derivative1_power2"):
                assert f"{b}{suffix}" in HMP_24


class TestLoadConfounds:
    def test_strategy_24hmp_selects_exactly_those_columns(self, tmp_path):
        p = _write_confounds(tmp_path / "c.tsv", 40)
        arr = load_confounds(p, _cfg(format="fmriprep_tsv", strategy="24hmp"), 40)
        assert arr.shape == (40, 24)          # not 25 -- 'csf' is excluded

    def test_explicit_columns_win_over_the_strategy(self, tmp_path):
        p = _write_confounds(tmp_path / "c.tsv", 40)
        cfg = _cfg(format="fmriprep_tsv", strategy="24hmp", columns=["trans_x", "csf"])
        assert load_confounds(p, cfg, 40).shape == (40, 2)

    def test_a_missing_requested_column_names_itself(self, tmp_path):
        df = pd.DataFrame({"trans_x": [0.0, 1.0]})
        p = tmp_path / "c.tsv"
        df.to_csv(p, sep="\t", index=False)
        cfg = _cfg(format="fmriprep_tsv", strategy="24hmp")
        with pytest.raises(ValueError, match="missing"):
            load_confounds(p, cfg, 2)

    def test_nans_do_not_propagate(self, tmp_path):
        """fMRIPrep leaves derivative columns undefined on the first sample."""
        df = pd.DataFrame({c: np.ones(10) for c in HMP_24})
        df.loc[0, "trans_x_derivative1"] = np.nan
        p = tmp_path / "c.tsv"
        df.to_csv(p, sep="\t", index=False)
        arr = load_confounds(p, _cfg(format="fmriprep_tsv", strategy="24hmp"), 10)
        assert np.isfinite(arr).all()


class TestFdCensoring:
    def _ref(self, tmp_path, fd):
        p = _write_confounds(tmp_path / "c.tsv", len(fd), fd=fd)
        return RunRef(cohort="c", sub="01", task="t", bold=tmp_path / "b.nii.gz",
                      confounds=p)

    def test_frames_over_the_threshold_are_censored(self, tmp_path):
        fd = np.full(20, 0.1); fd[10] = 0.9
        cfg = _cfg(format="fmriprep_tsv", fd_threshold=0.5, dilate_tr=0)
        good = _good_frames(self._ref(tmp_path, fd), cfg, 20)
        assert not good[10]
        assert good.sum() == 19

    def test_dilation_removes_the_neighbours_too(self, tmp_path):
        fd = np.full(20, 0.1); fd[10] = 0.9
        cfg = _cfg(format="fmriprep_tsv", fd_threshold=0.5, dilate_tr=1)
        good = _good_frames(self._ref(tmp_path, fd), cfg, 20)
        assert not good[9] and not good[10] and not good[11]
        assert good[8] and good[12]

    def test_a_nan_first_frame_is_kept_not_censored(self, tmp_path):
        """NaN means 'undefined', not 'bad'. Censoring frame 0 of every run on
        that basis would be a systematic bias across the whole cohort."""
        fd = np.full(20, 0.1); fd[0] = np.nan
        cfg = _cfg(format="fmriprep_tsv", fd_threshold=0.5, dilate_tr=0)
        assert _good_frames(self._ref(tmp_path, fd), cfg, 20).all()

    def test_no_threshold_means_no_censoring(self, tmp_path):
        fd = np.full(20, 9.9)                      # every frame would fail
        cfg = _cfg(format="fmriprep_tsv", fd_threshold=None)
        assert _good_frames(self._ref(tmp_path, fd), cfg, 20).all()

    def test_length_mismatch_raises_rather_than_misaligning(self, tmp_path):
        cfg = _cfg(format="fmriprep_tsv", fd_threshold=0.5)
        ref = self._ref(tmp_path, np.full(15, 0.1))
        with pytest.raises(ValueError, match="15 rows"):
            _good_frames(ref, cfg, 20)

    def test_an_explicit_censor_file_takes_precedence(self, tmp_path):
        """A cohort shipping its own censor is authoritative over a threshold
        we chose ourselves."""
        keep = np.ones(20); keep[3] = 0
        censor = tmp_path / "censor.1D"
        np.savetxt(censor, keep, fmt="%d")
        ref = self._ref(tmp_path, np.full(20, 9.9))    # FD would censor everything
        ref.censor = censor
        cfg = _cfg(format="fmriprep_tsv", fd_threshold=0.5, dilate_tr=0)
        good = _good_frames(ref, cfg, 20)
        assert not good[3] and good.sum() == 19


class TestSidecarAttachment:
    def test_a_sibling_session_is_not_picked_up(self, tmp_path):
        """sub+task alone matches every session of that subject; the first hit
        would then be used for all of them."""
        func = tmp_path / "func"
        func.mkdir()
        for ses in ("003", "004"):
            (func / f"sub-01_ses-{ses}_task-s01e01a_desc-confounds_timeseries.tsv").touch()
        refs = [RunRef(cohort="c", sub="01", task="s01e01a", ses=ses,
                       bold=func / f"sub-01_ses-{ses}_task-s01e01a_bold.nii.gz")
                for ses in ("003", "004")]
        cfg = _cfg(format="fmriprep_tsv",
                   confounds_glob="sub-*_task-*_desc-confounds_timeseries.tsv")
        for ref in _attach_confounds(cfg, refs):
            assert f"ses-{ref.ses}" in ref.confounds.name

    def test_no_glob_leaves_confounds_unset(self, tmp_path):
        ref = RunRef(cohort="c", sub="01", task="t", bold=tmp_path / "b.nii.gz")
        assert _attach_confounds(_cfg(), [ref])[0].confounds is None


class TestConfoundsConfigValidation:
    def test_unknown_strategy_is_rejected(self):
        with pytest.raises(ConfigError, match="strategy"):
            _cfg(strategy="aggressive")

    def test_custom_strategy_requires_columns(self):
        with pytest.raises(ConfigError, match="columns"):
            _cfg(strategy="custom")

    def test_negative_fd_threshold_is_rejected(self):
        with pytest.raises(ConfigError, match="fd_threshold"):
            _cfg(fd_threshold=-1.0)
