"""Moving a flat `outputs/` into the three halves, without losing anything.

WHY THIS FILE EXISTS. The migration runs ONCE, against a tree holding months
of compute that cannot be regenerated in an afternoon. Every other tool in
this repo can be re-run; this one cannot be un-run. So what is pinned here is
not "does it work" but the four ways it could quietly destroy something:
moving a tree it does not understand, merging two trees of one stage, copying
instead of renaming (and running out of disk halfway), and claiming a version
composition that is not what the data says.
"""
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import migrate_outputs as M  # noqa: E402

from fmri_decomposition import io  # noqa: E402


def old_tree(root, extras=(), latents=(("cneuromod", "train"),
                                       ("camcan", "projected"))):
    """A flat `outputs/` as every run before the split wrote it."""
    root.mkdir(parents=True, exist_ok=True)
    for rel in ("activation/atlas=yeo7/cohort=camcan/task=Movie/sub=S1",
                "dfc/atlas=yeo7/window_s=30/cohort=camcan/task=Movie/sub=S1",
                "censor/policy=motion/cohort=camcan/task=Movie",
                "static_fc/atlas=yeo7/cohort=camcan/task=Movie",
                "transitions/atlas=yeo7/window_s=30/states=S/cohort=camcan/task=Movie",
                "bstm_selection/task=movie/target=hads",
                "fcm_selection/task=movie/target=hads",
                "transferred_models/train=camcan/test=hcp/targets=raw",
                "meta/cohorts/cohort=camcan", "meta/models", "meta/clusterers",
                "meta/cluster", "meta/transitions", "meta/censor", *extras):
        d = root / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / ".keep").write_text("")
    (root / "meta" / "static_fc.json").write_text("{}")
    (root / "meta" / "atlas-yeo7_labels.csv").write_text("i,name\n0,a\n")
    for cohort, role in latents:
        d = root / f"latents/atlas=yeo7/window_s=30/cohort={cohort}/task=T"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"cohort": [cohort] * 2, "role": [role] * 2,
                                 "window_id": [0, 1]}), d / "data.parquet")
    return root


class TestItMovesEachTreeToItsOwnHalf:
    def test_the_extraction_trees_land_outside_every_version(self, tmp_path):
        root = old_tree(tmp_path / "outputs")
        M.main(["--output-root", str(root), "--apply"])
        for tree in io.EXTRACTION_TREES:
            assert (root / "extractions" / tree).is_dir(), tree
            assert not (root / tree).exists(), tree

    def test_the_fitted_trees_land_inside_one(self, tmp_path):
        root = old_tree(tmp_path / "outputs")
        M.main(["--output-root", str(root), "--apply"])
        for tree in io.UNSUPERVISED_TREES:
            assert io.tree_root(root, tree, "v0").is_dir(), tree
        for tree in io.SUPERVISED_TREES:
            assert io.tree_root(root, tree, "v0").is_dir(), tree

    def test_the_version_is_chosen_not_assumed(self, tmp_path):
        root = old_tree(tmp_path / "outputs")
        M.main(["--output-root", str(root), "--modelling-version", "v3",
                "--apply"])
        assert io.tree_root(root, "latents", "v3").is_dir()
        assert not (root / "modelling" / "version=v0").exists()

    def test_meta_splits_with_the_data_it_describes(self, tmp_path):
        """A fitted clusterer is true of ONE version; a cohort's stage-2
        manifest is true of all of them. They shared one directory."""
        root = old_tree(tmp_path / "outputs")
        M.main(["--output-root", str(root), "--apply"])
        for name in ("cohorts", "censor", "static_fc.json",
                     "atlas-yeo7_labels.csv"):
            assert (root / "extractions" / "meta" / name).exists(), name
        for name in ("models", "clusterers", "cluster", "transitions"):
            assert (io.meta_dir(root, "unsupervised", "v0") / name).is_dir(), name
        assert not (root / "meta").exists()      # emptied, then removed

    def test_nothing_is_left_behind_at_the_top(self, tmp_path):
        root = old_tree(tmp_path / "outputs")
        M.main(["--output-root", str(root), "--apply"])
        assert sorted(p.name for p in root.iterdir()) == ["extractions",
                                                          "modelling"]


class TestTheFourWaysItCouldDestroySomething:
    def test_a_tree_it_does_not_know_is_reported_and_left(self, tmp_path, capsys):
        """More likely yours than ours, so moving it would be a guess."""
        root = old_tree(tmp_path / "outputs", extras=("my_scratch/deep",))
        M.main(["--output-root", str(root), "--apply"])
        assert (root / "my_scratch" / "deep").is_dir()
        out = capsys.readouterr().out
        assert "my_scratch" in out and "NOT MOVED" in out

    def test_it_refuses_to_merge_two_trees_of_one_stage(self, tmp_path, capsys):
        root = old_tree(tmp_path / "outputs")
        (root / "extractions" / "dfc" / "atlas=already").mkdir(parents=True)
        M.main(["--output-root", str(root), "--apply"])
        out = capsys.readouterr().out
        assert "destination already exists" in out
        # BOTH survive: the old tree is not merged in and not deleted.
        assert (root / "dfc").is_dir()
        assert (root / "extractions" / "dfc" / "atlas=already").is_dir()

    def test_a_cross_filesystem_move_stops_instead_of_copying(self, tmp_path,
                                                              monkeypatch):
        """Renaming a 300 GB tree is milliseconds and needs no free space;
        copying it is hours and needs a second copy. Falling back silently is
        how a migration runs out of disk halfway through."""
        root = old_tree(tmp_path / "outputs")

        def exdev(*_a, **_k):
            raise OSError(18, "Invalid cross-device link")

        monkeypatch.setattr(M.os, "rename", exdev)
        with pytest.raises(SystemExit) as e:
            M.main(["--output-root", str(root), "--apply"])
        msg = str(e.value)
        assert "different filesystems" in msg and "rename" in msg
        assert (root / "dfc").is_dir()           # untouched

    def test_the_dry_run_touches_nothing(self, tmp_path):
        root = old_tree(tmp_path / "outputs")
        before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
        M.main(["--output-root", str(root)])
        assert sorted(p.relative_to(root).as_posix()
                      for p in root.rglob("*")) == before


class TestTheVersionCompositionComesFromTheData:
    """Not from a prompt. The latents carry `cohort` and `role` as COLUMNS, so
    what a version holds is already recorded in the files being moved -- and a
    manifest written from an argument would be whatever someone typed."""

    def test_train_and_project_are_read_off_the_role_column(self, tmp_path):
        root = old_tree(tmp_path / "outputs",
                        latents=(("cneuromod", "train"), ("ds002837", "train"),
                                 ("camcan", "projected")))
        M.main(["--output-root", str(root), "--apply"])
        man = io.read_version_manifest(root, "v0")
        assert man["unsupervised_train"] == ["cneuromod", "ds002837"]
        assert man["unsupervised_project"] == ["camcan"]

    def test_train_subset_still_counts_as_training(self, tmp_path):
        """`--restrict-subjects` writes role=train_subset. It is still the
        training half, and reading it as 'projected' would invert the claim."""
        root = old_tree(tmp_path / "outputs",
                        latents=(("cneuromod", "train_subset"),
                                 ("camcan", "projected")))
        M.main(["--output-root", str(root), "--apply"])
        man = io.read_version_manifest(root, "v0")
        assert man["unsupervised_train"] == ["cneuromod"]
        assert man["unsupervised_project"] == ["camcan"]

    def test_the_supervised_cohort_comes_from_the_summary(self, tmp_path):
        root = old_tree(tmp_path / "outputs")
        (root / "bstm_selection/task=movie/target=hads/summary.csv").write_text(
            "model,arm,atlas,cohort,mean\nridge,all [cond],yeo7,camcan,0.1\n")
        M.main(["--output-root", str(root), "--apply"])
        assert io.read_version_manifest(root, "v0")["supervised_train"] == ["camcan"]

    def test_a_key_it_cannot_read_is_left_unset_not_guessed(self, tmp_path):
        root = old_tree(tmp_path / "outputs")        # no summary.csv anywhere
        M.main(["--output-root", str(root), "--apply"])
        man = io.read_version_manifest(root, "v0")
        assert "supervised_train" not in man
        # and the next supervised run can still claim it
        io.claim_version(root, "v0", supervised_train="camcan")

    def test_no_claim_leaves_the_manifest_alone(self, tmp_path):
        root = old_tree(tmp_path / "outputs")
        M.main(["--output-root", str(root), "--apply", "--no-claim"])
        assert io.read_version_manifest(root, "v0") is None

    def test_a_latents_file_without_the_columns_is_skipped_not_fatal(self, tmp_path):
        root = old_tree(tmp_path / "outputs")
        d = root / "latents/atlas=yeo7/window_s=30/cohort=old/task=T"
        d.mkdir(parents=True)
        pq.write_table(pa.table({"window_id": [0, 1]}), d / "data.parquet")
        M.main(["--output-root", str(root), "--apply"])
        man = io.read_version_manifest(root, "v0")
        assert "old" not in man["unsupervised_train"]
        assert "old" not in man.get("unsupervised_project", [])


class TestTheManifestsStopLying:
    """Stage 2 and 3 record an absolute `path` per entry. NOTHING locates data
    by it -- both readers ignore the field and the trees are found by walking --
    so a stale value breaks nothing. It is provenance, and provenance that is
    wrong is worse than provenance that is missing."""

    def _manifest(self, root):
        return (root / "meta" / "cohorts" / "cohort=camcan"
                / "manifest_activation.json")

    def test_the_stored_path_is_brought_in_step(self, tmp_path):
        root = old_tree(tmp_path / "outputs")
        p = self._manifest(root)
        p.write_text(json.dumps({"entries": [
            {"stage": "activation",
             "path": f"{root}/activation/atlas=yeo7/cohort=camcan/"
                     f"task=Movie/sub=S1/data.parquet"}]}))
        M.main(["--output-root", str(root), "--apply"])
        moved = (root / "extractions" / "meta" / "cohorts" / "cohort=camcan"
                 / "manifest_activation.json")
        got = json.loads(moved.read_text())["entries"][0]["path"]
        assert "/extractions/activation/" in got
        assert f"{root}/activation/" not in got

    def test_a_path_outside_this_tree_is_left_alone(self, tmp_path):
        """Only paths under THIS output_root are rewritten. A manifest that
        records something else is recording something else."""
        root = old_tree(tmp_path / "outputs")
        other = "/somewhere/else/activation/atlas=yeo7/data.parquet"
        self._manifest(root).write_text(
            json.dumps({"entries": [{"stage": "activation", "path": other}]}))
        M.main(["--output-root", str(root), "--apply"])
        moved = (root / "extractions" / "meta" / "cohorts" / "cohort=camcan"
                 / "manifest_activation.json")
        assert json.loads(moved.read_text())["entries"][0]["path"] == other

    def test_the_dry_run_counts_them_without_writing(self, tmp_path, capsys):
        root = old_tree(tmp_path / "outputs")
        p = self._manifest(root)
        p.write_text(json.dumps({"entries": [
            {"path": f"{root}/dfc/atlas=yeo7/x.parquet"}]}))
        before = p.read_text()
        M.main(["--output-root", str(root)])
        assert p.read_text() == before
        assert "1 manifest(s)" in capsys.readouterr().out


class TestRunningItTwice:
    def test_the_second_run_has_nothing_to_do(self, tmp_path, capsys):
        root = old_tree(tmp_path / "outputs")
        M.main(["--output-root", str(root), "--apply"])
        M.main(["--output-root", str(root)])
        out = capsys.readouterr().out
        assert "nothing to move" in out
        assert "already in the new shape" in out

    def test_it_reports_what_the_version_holds(self, tmp_path, capsys):
        root = old_tree(tmp_path / "outputs")
        M.main(["--output-root", str(root), "--apply"])
        capsys.readouterr()
        M.main(["--output-root", str(root)])
        assert "unsupervised_train=cneuromod" in capsys.readouterr().out
