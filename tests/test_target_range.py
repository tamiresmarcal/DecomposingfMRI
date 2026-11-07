"""--target-range: a raw questionnaire score outside its own scale is a sentinel.

Cam-CAN's additional_HADS_depression carries 9999 for "no answer" and HADS
runs 0-21. This never mattered while the target was the *_category column --
text, so 9999 would not have coded -- and starts mattering the moment the raw
column is the target. Left in, a handful of 9999s are enormous outliers for a
regression to chase, and they set the scale of every plot of that column.
"""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fmri_decomposition.bstm_selection import read_phenotype  # noqa: E402


def spec(tmp_path, rows, name="ph.csv"):
    p = tmp_path / name
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return f"{p}:,"


def rows(values):
    return [{"ID": f"S{i:02d}", "hads": v, "Age": 30 + i, "Sex": "F"}
            for i, v in enumerate(values)]


class TestRange:
    def test_a_sentinel_is_dropped(self, tmp_path):
        s = spec(tmp_path, rows([0, 5, 9999, 12]))
        ph = read_phenotype([s], "ID", "hads", ["Age"], [], target_range=(0, 21))
        assert sorted(ph["y"]) == [0.0, 5.0, 12.0]

    def test_the_bounds_themselves_are_kept(self, tmp_path):
        s = spec(tmp_path, rows([0, 21]))
        ph = read_phenotype([s], "ID", "hads", ["Age"], [], target_range=(0, 21))
        assert sorted(ph["y"]) == [0.0, 21.0]

    def test_a_negative_value_is_dropped(self, tmp_path):
        s = spec(tmp_path, rows([-1, 3]))
        ph = read_phenotype([s], "ID", "hads", ["Age"], [], target_range=(0, 21))
        assert sorted(ph["y"]) == [3.0]

    def test_without_the_flag_nothing_changes(self, tmp_path):
        """The default must be byte-identical to the old behaviour, because
        every existing result was produced without this flag."""
        s = spec(tmp_path, rows([0, 5, 9999]))
        ph = read_phenotype([s], "ID", "hads", ["Age"], [])
        assert sorted(ph["y"]) == [0.0, 5.0, 9999.0]

    def test_the_drop_is_reported(self, tmp_path, capsys):
        s = spec(tmp_path, rows([0, 9999]))
        read_phenotype([s], "ID", "hads", ["Age"], [], target_range=(0, 21))
        out = capsys.readouterr().out
        assert "9999" in out and "dropped" in out

    def test_dropping_every_subject_fails_loudly(self, tmp_path):
        """If the range is wrong, the frame empties and the next error two
        steps later reads as an id mismatch. It must fail here instead."""
        s = spec(tmp_path, rows([9998, 9999]))
        with pytest.raises(SystemExit) as e:
            read_phenotype([s], "ID", "hads", ["Age"], [], target_range=(0, 21))
        assert "hads" in str(e.value)

    def test_an_ordinal_target_is_unaffected(self, tmp_path):
        """The *_category column is text; it codes by name and the range sees
        only the coded 0..3, so a sane range must not touch it."""
        r = [{"ID": f"S{i}", "cat": c, "Age": 30, "Sex": "F"}
             for i, c in enumerate(["Normal", "Mild", "Moderate", "Severe"])]
        s = spec(tmp_path, r)
        ph = read_phenotype([s], "ID", "cat", ["Age"], [], target_range=(0, 21))
        assert sorted(ph["y"]) == [0.0, 1.0, 2.0, 3.0]
