"""The participants table: adding motion without losing curation.

The tool's contract is that a hand-written exclusion survives every future
run of it. `validate_cohort` can only distinguish "excluded" from "someone
forgot" while the row and its reason are still there, so an automatic
threshold must never be able to overwrite or clear a human decision.
"""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "make_participants", REPO / "tools" / "make_participants.py")
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)


def row(sub, mean_fd=None, excluded=False, reason=""):
    return {"participant_id": f"sub-{sub}", "sub": sub, "cohort": "c", "task": "t",
            "excluded": excluded, "exclusion_reason": reason, "mean_fd": mean_fd}


class TestFdExclusions:
    def test_rows_over_the_threshold_are_excluded_with_a_recorded_reason(self):
        rows = [row("1", 0.1), row("2", 0.8)]
        added, cleared = mp.apply_fd_exclusions(rows, 0.5)
        assert (added, cleared) == (1, 0)
        assert rows[0]["excluded"] is False
        assert rows[1]["excluded"] is True
        assert rows[1]["exclusion_reason"] == "auto:mean_fd>0.5"

    def test_a_hand_written_exclusion_is_never_touched(self):
        rows = [row("1", 0.1, excluded=True, reason="corrupted run")]
        mp.apply_fd_exclusions(rows, 0.5)
        assert rows[0]["excluded"] is True
        assert rows[0]["exclusion_reason"] == "corrupted run"

    def test_a_hand_written_exclusion_is_not_re_reasoned_by_the_threshold(self):
        """Even when the row would also fail on motion, the reason stays theirs."""
        rows = [row("1", 9.9, excluded=True, reason="fell asleep, per scan log")]
        mp.apply_fd_exclusions(rows, 0.5)
        assert rows[0]["exclusion_reason"] == "fell asleep, per scan log"

    def test_raising_the_threshold_clears_the_tools_own_exclusions(self):
        rows = [row("1", 0.8, excluded=True, reason="auto:mean_fd>0.5")]
        added, cleared = mp.apply_fd_exclusions(rows, 1.0)
        assert (added, cleared) == (0, 1)
        assert rows[0]["excluded"] is False
        assert rows[0]["exclusion_reason"] == ""

    def test_moving_the_threshold_rewrites_the_marker_without_double_counting(self):
        rows = [row("1", 2.0, excluded=True, reason="auto:mean_fd>0.5")]
        added, cleared = mp.apply_fd_exclusions(rows, 1.0)
        assert (added, cleared) == (0, 0), "already excluded; only the marker moved"
        assert rows[0]["exclusion_reason"] == "auto:mean_fd>1"

    def test_a_row_with_no_mean_fd_is_left_alone(self):
        """No motion file is not evidence of good motion."""
        rows = [row("1", None), row("2", "")]
        added, cleared = mp.apply_fd_exclusions(rows, 0.5)
        assert (added, cleared) == (0, 0)
        assert not any(r["excluded"] for r in rows)


class TestCsvRoundTrip:
    def test_nan_becomes_an_empty_cell_not_the_string_nan(self):
        assert mp._fmt(float("nan")) == ""
        assert mp._fmt(None) == ""
        assert mp._fmt(0.089209123) == "0.0892091"   # 6 significant figures

    def test_update_preserves_unknown_columns(self, tmp_path):
        path = tmp_path / "p.csv"
        path.write_text("participant_id,sub,cohort,task,excluded,exclusion_reason,group\n"
                        "sub-1,1,c,t,False,,control\n")
        rows, columns = mp.read_existing(path)
        assert "group" in columns and rows[0]["group"] == "control"
        mp.write_table(path, rows, columns + ["mean_fd"])
        rows2, columns2 = mp.read_existing(path)
        assert rows2[0]["group"] == "control"
        assert columns2[-1] == "mean_fd"


class TestCli:
    def test_exclude_without_fd_is_refused(self, tmp_path, capsys):
        out = tmp_path / "p.csv"
        rc = mp.main(["cfg.yaml", "-o", str(out), "--exclude-mean-fd", "0.5"])
        assert rc == 1
        assert "needs --fd" in capsys.readouterr().err

    def test_update_on_a_missing_file_is_refused(self, tmp_path, capsys):
        rc = mp.main(["cfg.yaml", "-o", str(tmp_path / "nope.csv"), "--update"])
        assert rc == 1
        assert "needs an existing" in capsys.readouterr().err

    def test_an_existing_file_is_not_overwritten_by_default(self, tmp_path, capsys):
        out = tmp_path / "p.csv"
        out.write_text("participant_id\nsub-1\n")
        assert mp.main(["cfg.yaml", "-o", str(out)]) == 1
        err = capsys.readouterr().err
        assert "--update" in err and "--force" in err
        assert out.read_text() == "participant_id\nsub-1\n"
