"""The participants table: human-owned curation, and nothing else.

This file records who is in the cohort and who was deliberately removed, with
a reason. Nothing writes an exclusion automatically -- QC metrics live in
participants_qc.csv and the thresholds that use them live with the models --
so the property to pin down is that an update never loses what a person wrote.
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
    def test_it_offers_no_way_to_write_an_exclusion(self):
        """Exclusions are a human's or the models'. Not this tool's."""
        flags = mp.main.__doc__ or ""
        parser_src = Path(mp.__file__).read_text()
        for gone in ("--exclude-", "--fd", "--qc", "auto:", "apply_auto_exclusions"):
            assert gone not in parser_src, f"{gone} should have been removed"
        assert not flags

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


class TestFromList:
    """The cohort a study MEANT to have is not derivable from what is on disk.

    camcan_ccfrail: 55 subjects submitted to fMRIPrep, 54 completed. The 55th
    has no preprocessed bold, so discovery cannot see it -- and a table built
    from discovery reports 54 with nothing saying one is missing. CC620139
    would have been absent rather than excluded.
    """

    def test_ids_are_read_with_or_without_the_sub_prefix(self, tmp_path):
        p = tmp_path / "subjects.txt"
        p.write_text("CC620139\nsub-CC321480\n\n# a comment\n  CC410026  \n"
                     "CC620139\n")
        assert mp.read_subject_list(p) == ["CC620139", "CC321480", "CC410026"]

    def test_an_empty_list_is_not_silently_accepted(self, tmp_path):
        p = tmp_path / "subjects.txt"
        p.write_text("# nothing but comments\n\n")
        assert mp.read_subject_list(p) == []


class TestFromListCli:
    """End to end, against a fake cohort where one listed subject has no run."""

    def _cohort(self, tmp_path, on_disk=("01", "02")):
        import yaml

        deriv = tmp_path / "deriv"
        for sub in on_disk:
            d = deriv / f"sub-{sub}" / "func"
            d.mkdir(parents=True)
            (d / f"sub-{sub}_task-Movie_bold.nii.gz").write_text("")
        cfg = {"cohort": "c", "tr": 1.0, "derivatives_root": str(deriv),
               "output_root": str(tmp_path / "out"),
               "discovery": {"bold_glob": "sub-*/func/*_bold.nii.gz"}}
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return str(path)

    def test_a_subject_with_no_run_still_gets_a_row(self, tmp_path, capsys):
        import csv as _csv

        cfg = self._cohort(tmp_path)
        listing = tmp_path / "subjects.txt"
        listing.write_text("01\n02\n03\n")          # 03 never preprocessed
        out = tmp_path / "p.csv"

        assert mp.main([cfg, "-o", str(out), "--from-list", str(listing)]) == 0
        rows = list(_csv.DictReader(open(out)))
        assert [r["sub"] for r in rows] == ["01", "02", "03"]
        assert all(r["excluded"] == "False" for r in rows)

        printed = capsys.readouterr().out
        assert "1 listed subject(s) with NO run on disk" in printed
        assert "sub-03" in printed
        assert "excluded=True WITH a reason" in printed

    def test_more_rows_than_runs_is_not_reported_as_negative_extras(
            self, tmp_path, capsys):
        """55 listed against 54 on disk is not "-1 extra from multiple ses/run"."""
        cfg = self._cohort(tmp_path)
        listing = tmp_path / "subjects.txt"
        listing.write_text("01\n02\n03\n")
        mp.main([cfg, "-o", str(tmp_path / "p.csv"), "--from-list", str(listing)])
        printed = capsys.readouterr().out
        assert "runs on disk: 2" in printed
        assert "extra from multiple" not in printed

    def test_without_the_flag_the_missing_subject_vanishes(self, tmp_path):
        """The behaviour --from-list exists to prevent."""
        import csv as _csv

        cfg = self._cohort(tmp_path)
        out = tmp_path / "p.csv"
        assert mp.main([cfg, "-o", str(out)]) == 0
        rows = list(_csv.DictReader(open(out)))
        assert [r["sub"] for r in rows] == ["01", "02"], "03 is invisible, as before"

    def test_ambiguous_task_is_refused_rather_than_guessed(self, tmp_path, capsys):
        deriv = tmp_path / "deriv"
        for task in ("Movie", "Rest"):
            d = deriv / "sub-01" / "func"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"sub-01_task-{task}_bold.nii.gz").write_text("")
        import yaml

        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(yaml.safe_dump({
            "cohort": "c", "tr": 1.0, "derivatives_root": str(deriv),
            "output_root": str(tmp_path / "out"),
            "discovery": {"bold_glob": "sub-*/func/*_bold.nii.gz",
                          "exclude_tasks": []}}))
        listing = tmp_path / "subjects.txt"
        listing.write_text("01\n02\n")
        rc = mp.main([str(cfg), "-o", str(tmp_path / "p.csv"),
                      "--from-list", str(listing)])
        assert rc == 1
        assert "--task" in capsys.readouterr().err
