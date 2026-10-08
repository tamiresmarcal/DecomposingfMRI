"""preprocessing/camcan/01_build_bids.py -- the pilot-then-expand workflow.

Not part of the pipeline package, so it is loaded by path. The behaviour under
test is the one that actually bit: the script refused to expand a root that
already held the pilot's subjects, which is the workflow its own `--limit` help
recommends. Refusing an EXTRA subject is the thing that matters -- fMRIPrep
globs sub-* and never reads subjects.txt -- and that is a different predicate
from "any subject at all".
"""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parent.parent / "preprocessing" / "camcan"
          / "01_build_bids.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("build_bids", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def bidssep(tmp_path):
    """A BIDSsep tree: anat/ and func_rest/, single-echo Rest, 5 subjects."""
    root = tmp_path / "BIDSsep"
    for i in range(5):
        sub = f"CC{i:06d}"
        a = root / "anat" / f"sub-{sub}" / "anat"
        a.mkdir(parents=True)
        for ext in (".nii.gz", ".json"):
            (a / f"sub-{sub}_T1w{ext}").write_text("x")
        f = root / "func_rest" / f"sub-{sub}" / "func"
        f.mkdir(parents=True)
        for ext in (".nii.gz", ".json"):
            (f / f"sub-{sub}_task-Rest_bold{ext}").write_text("x")
    return root


def run(mod, bidssep, out, *extra):
    return mod.main(["--camcan-bidssep", str(bidssep), "-o", str(out),
                     "--task", "Rest", "--n-echoes", "0",
                     "--func-subdir", "func_rest", *map(str, extra)])


class TestSurvey:
    def test_single_echo_rest_is_found(self, mod, bidssep):
        usable, skipped = mod.survey(bidssep, "func_rest", 0, "Rest")
        assert len(usable) == 5 and not skipped

    def test_a_wrong_echo_count_reports_the_task_not_the_data(self, mod,
                                                              bidssep):
        """--n-echoes 5 against single-echo data must not look like a finding
        about the subjects."""
        usable, skipped = mod.survey(bidssep, "func_rest", 5, "Rest")
        assert not usable
        assert all("no Rest data" in r for r in skipped.values())

    def test_the_skip_reason_names_the_task(self, mod, bidssep):
        (bidssep / "func_rest" / "sub-CC000003").rename(
            bidssep / "func_rest" / "x-CC000003")
        _, skipped = mod.survey(bidssep, "func_rest", 0, "Rest")
        assert "no Rest data" in skipped["CC000003"]


class TestPilotThenExpand:
    def test_a_pilot_links_a_sorted_prefix(self, mod, bidssep, tmp_path):
        out = tmp_path / "bids"
        assert run(mod, bidssep, out, "--limit", 2) == 0
        got = sorted(p.name for p in out.glob("sub-*"))
        assert got == ["sub-CC000000", "sub-CC000001"]
        assert (out / "subjects.txt").read_text().split() == \
            ["CC000000", "CC000001"]

    def test_expanding_the_pilot_is_allowed_without_force(self, mod, bidssep,
                                                          tmp_path):
        """THE REGRESSION. This used to exit 1 with "already has sub-*
        directories", which blocked the workflow --limit exists for."""
        out = tmp_path / "bids"
        assert run(mod, bidssep, out, "--limit", 2) == 0
        assert run(mod, bidssep, out) == 0
        assert len(list(out.glob("sub-*"))) == 5
        assert len((out / "subjects.txt").read_text().split()) == 5

    def test_expansion_keeps_the_prefix_so_array_indices_stay_valid(
            self, mod, bidssep, tmp_path):
        """A running array reads subjects.txt per task, so line N must not
        move -- that is what makes expanding mid-flight safe."""
        out = tmp_path / "bids"
        run(mod, bidssep, out, "--limit", 2)
        before = (out / "subjects.txt").read_text().split()
        run(mod, bidssep, out)
        after = (out / "subjects.txt").read_text().split()
        assert after[:len(before)] == before

    def test_expansion_does_not_touch_the_existing_links(self, mod, bidssep,
                                                         tmp_path):
        """Without --force, `link()` returns early. That is the difference
        between safe-mid-flight and a window where a live job cannot resolve
        a path."""
        out = tmp_path / "bids"
        run(mod, bidssep, out, "--limit", 2)
        p = next((out / "sub-CC000000" / "func").iterdir())
        before = p.lstat().st_mtime_ns
        run(mod, bidssep, out)
        assert p.lstat().st_mtime_ns == before

    def test_force_does_relink_them(self, mod, bidssep, tmp_path):
        out = tmp_path / "bids"
        run(mod, bidssep, out, "--limit", 2)
        p = next((out / "sub-CC000000" / "func").iterdir())
        before = p.lstat().st_mtime_ns
        run(mod, bidssep, out, "--force")
        assert p.lstat().st_mtime_ns != before


class TestStaleIsStillRefused:
    def test_a_subject_outside_the_selection_is_refused(self, mod, bidssep,
                                                        tmp_path, capsys):
        """The hazard the guard exists for: fMRIPrep globs sub-* and never
        reads subjects.txt, so a leftover from an earlier run WOULD be
        processed."""
        out = tmp_path / "bids"
        run(mod, bidssep, out)
        (out / "sub-LEFTOVER" / "anat").mkdir(parents=True)
        assert run(mod, bidssep, out, "--limit", 2) == 1
        err = capsys.readouterr().err
        assert "LEFTOVER" in err
        assert "does not include" in err

    def test_the_refusal_names_all_three_ways_out(self, mod, bidssep,
                                                  tmp_path, capsys):
        out = tmp_path / "bids"
        run(mod, bidssep, out)
        (out / "sub-LEFTOVER" / "anat").mkdir(parents=True)
        run(mod, bidssep, out, "--limit", 2)
        err = capsys.readouterr().err
        assert "Remove them" in err and "--force" in err and "--out" in err

    def test_force_overrides_it(self, mod, bidssep, tmp_path):
        """--force relinks the whole root. The stale dir then shows up in
        verify()'s own stale report instead, which is the loud failure."""
        out = tmp_path / "bids"
        run(mod, bidssep, out)
        (out / "sub-LEFTOVER" / "anat").mkdir(parents=True)
        assert run(mod, bidssep, out, "--limit", 2, "--force") == 1


class TestWhatItWrites:
    def test_dataset_description_and_bidsignore(self, mod, bidssep, tmp_path):
        out = tmp_path / "bids"
        run(mod, bidssep, out, "--dataset-name", "Rest view")
        dd = json.loads((out / "dataset_description.json").read_text())
        assert dd["Name"] == "Rest view"
        assert "subjects.txt" in (out / ".bidsignore").read_text()

    def test_links_are_absolute_so_they_resolve_from_anywhere(self, mod,
                                                              bidssep,
                                                              tmp_path):
        out = tmp_path / "bids"
        run(mod, bidssep, out, "--limit", 1)
        for d in ("anat", "func"):
            for p in (out / "sub-CC000000" / d).iterdir():
                assert Path(p).readlink().is_absolute()
                assert p.exists()           # follows the link
