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
