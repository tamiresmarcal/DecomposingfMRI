"""The ASR truncates T at 50 for its narrow scales, and that must be visible.

Measured on the real HCP table: DSM_Anxi spreads over 11 raw values with only
6% of subjects at zero, yet its T column piles up on a single spike at 50. The
raw histogram and the T histogram look like they disagree, and a reader
reasonably concludes one of them is wrong. Neither is: the lower part of the
raw range maps to one T.

It matters beyond the figure. A target column that is mostly one value has
almost no usable variance, so the truncation is a second reason -- on top of
the age/sex norming -- that the model is fitted on raw.
"""
import csv

import pytest

from fmri_decomposition import __name__ as _pkg  # noqa: F401  (import guard)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import symptom_distribution as sd  # noqa: E402


def write(tmp_path, t_values, stem="DSM_Anxi", family_raw=None):
    """A minimal phenotype pair plus the participants table that selects it."""
    n = len(t_values)
    subs = [f"{100000 + i:06d}" for i in range(n)]
    ph = tmp_path / "ph.csv"
    with ph.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Subject", "Family_ID", f"{stem}_Raw", f"{stem}_T"])
        for i, (s, t) in enumerate(zip(subs, t_values)):
            raw = family_raw[i] if family_raw else i % 11
            w.writerow([s, f"F{i // 2}", raw, t])
    pa = tmp_path / "participants.csv"
    with pa.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant_id", "sub", "cohort", "task"])
        for s in subs:
            w.writerow([f"sub-{s}", s, "c", "MOVIE1"])
    return ph, pa


class TestTruncationIsDetected:
    def test_a_pile_at_50_is_flagged(self, tmp_path):
        # 60 of 100 at the floor: what the real DSM_Anxi_T looks like
        ph, pa = write(tmp_path, [50] * 60 + list(range(51, 91)))
        r = sd.hcp_t_scales(ph, pa)[0]
        assert r["truncated_at_50"]
        assert r["t_floor"] == 50
        assert r["t_floor_frac"] == pytest.approx(60.0, abs=.1)

    def test_a_spread_column_is_not_flagged(self, tmp_path):
        # the broad-band shape: runs well below 50, no pile
        ph, pa = write(tmp_path, list(range(30, 90)), stem="ASR_Intn")
        r = sd.hcp_t_scales(ph, pa)[0]
        assert not r["truncated_at_50"]
        assert r["t_floor"] == 30

    def test_a_floor_at_50_without_a_pile_is_not_flagged(self, tmp_path):
        """min(T) == 50 alone is not truncation -- a healthy cohort can simply
        have nobody below it. The flag needs the PILE."""
        ph, pa = write(tmp_path, [50] * 5 + list(range(51, 96)))
        r = sd.hcp_t_scales(ph, pa)[0]
        assert r["t_floor"] == 50
        assert not r["truncated_at_50"]

    def test_the_raw_column_keeps_the_variance_the_t_column_lost(self, tmp_path):
        """The point of the whole thing: the same subjects, resolved on raw
        and collapsed on T."""
        raw = list(range(11)) * 9 + [0]
        ph, pa = write(tmp_path, [50] * 60 + list(range(51, 91)), family_raw=raw)
        t_row = sd.hcp_t_scales(ph, pa)[0]
        raw_row = sd.hcp_scales(ph, pa)[0]
        assert t_row["t_floor_frac"] > 50
        assert raw_row["pct_at_floor"] < 15
        assert raw_row["n_distinct"] > 10


class TestThresholdsTravelWithTheScale:
    def test_narrow_scales_use_65_and_70(self, tmp_path):
        ph, pa = write(tmp_path, list(range(50, 90)), stem="DSM_Anxi")
        r = sd.hcp_t_scales(ph, pa)[0]
        assert (r["borderline_t"], r["clinical_t"]) == (65, 70)

    def test_broad_band_scales_use_60_and_64(self, tmp_path):
        ph, pa = write(tmp_path, list(range(30, 90)), stem="ASR_Intn")
        r = sd.hcp_t_scales(ph, pa)[0]
        assert (r["borderline_t"], r["clinical_t"]) == (60, 64)

    def test_the_counts_above_each_line_are_right(self, tmp_path):
        ph, pa = write(tmp_path, [50] * 10 + [65, 66, 70, 71], stem="DSM_Anxi")
        r = sd.hcp_t_scales(ph, pa)[0]
        assert r["n_borderline_plus"] == 4
        assert r["n_clinical"] == 2

    def test_ssaga_has_no_t_row(self, tmp_path):
        ph, pa = write(tmp_path, list(range(50, 90)), stem="SSAGA_Depressive_Ep")
        assert sd.hcp_t_scales(ph, pa) == []
