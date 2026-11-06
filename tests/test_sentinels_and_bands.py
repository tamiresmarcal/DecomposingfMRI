"""Missing-data sentinels, and a band table whose rows compare with each other.

Both found on the real data.

1. Cam-CAN's additional_HADS_depression carries 9999 for "no answer". HADS
   runs 0-21, so it cannot be a score. Left in, it set the max to 9999,
   inflated the sd by two orders of magnitude, squashed the histogram into
   the first pixel column, and would have handed a raw-target regression a
   few enormous outliers to chase.

2. The zero split was decided per scale. Cam-CAN's band counts come from its
   own `_category` column and its zero count from the raw column; where the
   two have different missingness the split is impossible for that scale. One
   cohort then got "Normal" and the other "Normal (>0)", the pivot made them
   two columns, and the table stopped comparing its own rows.
"""
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import symptom_distribution as sd      # noqa: E402
import symptom_plots as sp             # noqa: E402


class TestSentinels:
    def rows(self, vals):
        return [{"h": str(v)} for v in vals]

    def test_a_value_above_the_scale_max_is_dropped(self):
        assert sd.nums(self.rows([3, 9999, 21]), "h", 21) == [3.0, 21.0]

    def test_a_negative_value_is_dropped(self):
        assert sd.nums(self.rows([-1, 0, 5]), "h", 21) == [0.0, 5.0]

    def test_the_boundary_values_are_kept(self):
        assert sd.nums(self.rows([0, 21]), "h", 21) == [0.0, 21.0]

    def test_nothing_is_dropped_without_a_known_max(self):
        """An ASR raw score has no published ceiling, so there is no
        principled line to drop on and a guess would be worse than showing
        the value."""
        assert sd.nums(self.rows([3, 9999]), "h") == [3.0, 9999.0]

    def test_every_drop_is_reported(self):
        r = sd.out_of_range(self.rows([3, 9999, -1, 21]), "h", 21)
        assert r["n_dropped"] == 2
        assert r["dropped_values"] == [-1.0, 9999.0]

    def test_without_a_max_a_sentinel_is_suspect_not_dropped(self):
        r = sd.out_of_range(self.rows([3, 9999, 7]), "h")
        assert r["n_dropped"] == 0
        assert r["n_suspect"] == 1 and r["suspect_values"] == [9999.0]

    def test_a_plausible_large_score_is_not_called_suspect(self):
        """999 is a run of 9s; 120 is a believable ASR total. Only the first
        is flagged."""
        r = sd.out_of_range(self.rows([120, 98, 101]), "h")
        assert r["n_suspect"] == 0

    def test_the_sentinel_does_not_reach_the_summary(self):
        v = sd.nums(self.rows([0, 5, 9999]), "h", 21)
        s = sd._summary(v, 21)
        assert s["max"] == 5.0
        assert s["sd"] < 3


def _block(band, zero_n, raw_n):
    return {"raw": [], "raw_col": "c", "t": None, "band": Counter(band),
            "band_order": ["Normal", "Mild", "Moderate", "Severe"],
            "cutoffs": [8, 11, 15], "scale_max": 21, "banded_on": "category",
            "summary": {"n": raw_n}, "zero_n": zero_n, "what": ""}


class TestBandTableStaysComparable:
    def two_cohorts(self, raw_n_b=100):
        return {
            "a": {"n": 100, "instrument": "x", "missing_pheno": [],
                  "scales": {"anxiety": _block({"Normal": 80, "Mild": 20}, 10, 100)}},
            "b": {"n": 100, "instrument": "y", "missing_pheno": [],
                  "scales": {"anxiety": _block({"Normal": 70, "Mild": 30}, 5, raw_n_b)}},
        }

    def test_when_both_can_split_both_do(self):
        rows = sp.band_table(self.two_cohorts(), constructs=("anxiety",))
        names = {r["band"] for r in rows}
        assert "Zero" in names
        assert "Normal (>0)" in names
        assert "Normal" not in names
        assert rows[0]["split_note"] == ""

    def test_when_one_cannot_split_neither_does(self):
        rows = sp.band_table(self.two_cohorts(raw_n_b=99), constructs=("anxiety",))
        names = {r["band"] for r in rows}
        assert "Zero" not in names
        assert "Normal (>0)" not in names
        assert "Normal" in names

    def test_the_reason_is_carried_on_every_row(self):
        rows = sp.band_table(self.two_cohorts(raw_n_b=99), constructs=("anxiety",))
        assert all("99 raw score" in r["split_note"] for r in rows)
        assert all("b/anxiety" in r["split_note"] for r in rows)

    def test_one_set_of_band_names_whatever_happens(self):
        """The property that matters: a pivot on `band` must never produce two
        columns meaning the same thing."""
        for raw_n in (100, 99):
            rows = sp.band_table(self.two_cohorts(raw_n), constructs=("anxiety",))
            per_cohort = {}
            for r in rows:
                per_cohort.setdefault(r["cohort"], set()).add(r["band"])
            assert len(set(map(frozenset, per_cohort.values()))) == 1

    def test_the_split_preserves_the_total(self):
        rows = sp.band_table(self.two_cohorts(), constructs=("anxiety",))
        for coh in ("a", "b"):
            got = [r for r in rows if r["cohort"] == coh]
            assert sum(r["n"] for r in got) == got[0]["total"]
