"""The participants table: adding motion without losing curation.

The tool's contract is that a hand-written exclusion survives every future
run of it. `validate_cohort` can only distinguish "excluded" from "someone
forgot" while the row and its reason are still there, so an automatic
threshold must never be able to overwrite or clear a human decision.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "make_participants", REPO / "tools" / "make_participants.py")
mp = importlib.util.module_from_spec(_spec)
# Registered before exec: @dataclass resolves annotations through
# sys.modules[cls.__module__], which is None for an unregistered module.
sys.modules[_spec.name] = mp
_spec.loader.exec_module(mp)


def row(sub, mean_fd=None, excluded=False, reason="", **extra):
    r = {"participant_id": f"sub-{sub}", "sub": sub, "cohort": "c", "task": "t",
         "excluded": excluded, "exclusion_reason": reason, "mean_fd": mean_fd}
    r.update(extra)
    return r


FD = lambda t=0.5: mp.Rule("mean_fd", "gt", t)                          # noqa: E731
LAG = lambda t=1: mp.Rule("best_lag_tr", "gt", t, abs_=True)            # noqa: E731
COV = lambda t=0.95: mp.Rule("frac_stimulus_covered", "lt", t)          # noqa: E731
EMPTY = lambda t=0.05: mp.Rule("frac_parcels_empty", "gt", t)           # noqa: E731


class TestRuleMarkers:
    """The marker is the contract: it is what makes an auto exclusion
    recognisable, reversible, and distinguishable from a human's."""

    def test_markers_are_readable_and_carry_the_threshold(self):
        assert FD(0.5).marker == "auto:mean_fd>0.5"
        assert COV(0.95).marker == "auto:frac_stimulus_covered<0.95"
        assert EMPTY(0.05).marker == "auto:frac_parcels_empty>0.05"

    def test_a_lag_rule_thresholds_the_absolute_value(self):
        """Lag -12 is exactly as wrong as +12."""
        assert LAG(1).marker == "auto:abs(best_lag_tr)>1"
        assert LAG(1).fires(row("1", best_lag_tr=-12))
        assert LAG(1).fires(row("1", best_lag_tr=12))
        assert not LAG(1).fires(row("1", best_lag_tr=0))

    def test_lt_and_gt_point_the_right_way(self):
        assert COV(0.95).fires(row("1", frac_stimulus_covered=0.60))
        assert not COV(0.95).fires(row("1", frac_stimulus_covered=1.0))
        assert EMPTY(0.05).fires(row("1", frac_parcels_empty=0.30))
        assert not EMPTY(0.05).fires(row("1", frac_parcels_empty=0.0))

    @pytest.mark.parametrize("value", [None, "", "n/a", float("nan")])
    def test_a_missing_metric_never_fires(self, value):
        """Absence of evidence is not evidence: no motion file is not proof of
        good motion, and an uncomputable ISC is not proof of a bad subject."""
        assert not FD(0.5).fires(row("1", mean_fd=value))


class TestAutoExclusions:
    def test_rows_over_the_threshold_are_excluded_with_a_recorded_reason(self):
        rows = [row("1", 0.1), row("2", 0.8)]
        counts = mp.apply_auto_exclusions(rows, [FD(0.5)])
        assert counts["auto:mean_fd>0.5"] == 1
        assert rows[0]["excluded"] is False
        assert rows[1]["excluded"] is True
        assert rows[1]["exclusion_reason"] == "auto:mean_fd>0.5"

    def test_several_rules_all_appear_in_the_reason(self):
        rows = [row("1", 0.9, frac_stimulus_covered=0.5, best_lag_tr=0)]
        mp.apply_auto_exclusions(rows, [FD(0.5), LAG(1), COV(0.95)])
        assert rows[0]["exclusion_reason"] == (
            "auto:mean_fd>0.5; auto:frac_stimulus_covered<0.95")

    def test_a_hand_written_exclusion_is_never_touched(self):
        rows = [row("1", 0.1, excluded=True, reason="corrupted run")]
        counts = mp.apply_auto_exclusions(rows, [FD(0.5)])
        assert counts["_protected"] == 1
        assert rows[0]["excluded"] is True
        assert rows[0]["exclusion_reason"] == "corrupted run"

    def test_a_hand_written_exclusion_is_not_re_reasoned_by_a_rule(self):
        rows = [row("1", 9.9, excluded=True, reason="fell asleep, per scan log")]
        mp.apply_auto_exclusions(rows, [FD(0.5)])
        assert rows[0]["exclusion_reason"] == "fell asleep, per scan log"

    def test_a_reason_mixing_human_and_auto_counts_as_the_humans(self):
        """The safe direction is to leave it alone."""
        rows = [row("1", 0.1, excluded=True,
                    reason="auto:mean_fd>0.5; and the scan log agrees")]
        mp.apply_auto_exclusions(rows, [FD(5.0)])
        assert rows[0]["excluded"] is True

    def test_raising_the_threshold_releases_the_rows_it_used_to_catch(self):
        rows = [row("1", 0.8, excluded=True, reason="auto:mean_fd>0.5")]
        counts = mp.apply_auto_exclusions(rows, [FD(1.0)])
        assert counts["_cleared"] == 1
        assert rows[0]["excluded"] is False
        assert rows[0]["exclusion_reason"] == ""

    def test_dropping_a_rule_releases_only_that_rules_rows(self):
        """Exclusions are recomputed from the rules in force, not accumulated."""
        rows = [row("1", 0.1, frac_stimulus_covered=0.5, excluded=True,
                    reason="auto:mean_fd>0.5; auto:frac_stimulus_covered<0.95")]
        mp.apply_auto_exclusions(rows, [COV(0.95)])
        assert rows[0]["excluded"] is True
        assert rows[0]["exclusion_reason"] == "auto:frac_stimulus_covered<0.95"

    def test_moving_a_threshold_rewrites_the_marker_in_place(self):
        rows = [row("1", 2.0, excluded=True, reason="auto:mean_fd>0.5")]
        counts = mp.apply_auto_exclusions(rows, [FD(1.0)])
        assert counts["_cleared"] == 0
        assert rows[0]["exclusion_reason"] == "auto:mean_fd>1"

    def test_a_row_with_no_metric_is_left_alone(self):
        rows = [row("1", None), row("2", "")]
        counts = mp.apply_auto_exclusions(rows, [FD(0.5)])
        assert counts["auto:mean_fd>0.5"] == 0
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
