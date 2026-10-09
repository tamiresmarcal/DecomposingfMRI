"""Moving one cohort's output under another cohort's key.

`cohort` is a partition key and not a column, so for the immutable per-subject
shards a rename of the directory IS the migration -- nothing inside the files
changes and stage 1 is not re-run. The derived per-cohort tables are a
different matter and this is where the care goes: each cohort holds a complete
copy of its own, keyed per (sub, task), so moving one over the other would
silently keep whichever moved last.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import migrate_cohort as M        # noqa: E402


def tree(root: Path, cohort: str, task: str, subs=("01", "02"),
         atlases=("yeo7",), windows=("30",)):
    """An output tree shaped like the real one."""
    for atlas in atlases:
        for sub in subs:
            p = (root / "activation" / f"atlas={atlas}" / f"cohort={cohort}"
                 / f"task={task}" / f"sub={sub}" / "data.parquet")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"a")
        for w in windows:
            for sub in subs:
                p = (root / "dfc" / f"atlas={atlas}" / f"window_s={w}"
                     / f"cohort={cohort}" / f"task={task}" / f"sub={sub}"
                     / "data.parquet")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"d")
    meta = root / "meta" / "cohorts" / f"cohort={cohort}"
    (meta / "shards").mkdir(parents=True, exist_ok=True)
    (meta / "shards" / f"manifest_activation_{task}.json").write_text("{}")
    (meta / "manifest_activation.json").write_text('{"cohort": "%s"}' % cohort)
    (meta / "participants_qc.csv").write_text(f"sub,task\n01,{task}\n")
    cen = root / "censor" / "policy=motion" / f"cohort={cohort}"
    cen.mkdir(parents=True, exist_ok=True)
    (cen / "subjects.parquet").write_bytes(b"c")


class TestTheMove:
    def test_shards_move_and_the_existing_task_is_untouched(self, tmp_path):
        tree(tmp_path, "camcan", "Movie")
        tree(tmp_path, "camcan_rest", "Rest")
        assert M.main(["--from", "camcan_rest", "--to", "camcan",
                       "--output-root", str(tmp_path), "--apply"]) == 0

        act = tmp_path / "activation" / "atlas=yeo7"
        assert (act / "cohort=camcan" / "task=Rest" / "sub=01"
                / "data.parquet").exists()
        assert (act / "cohort=camcan" / "task=Movie" / "sub=01"
                / "data.parquet").exists()
        assert not (act / "cohort=camcan_rest").exists()

    def test_dfc_moves_under_every_window(self, tmp_path):
        tree(tmp_path, "camcan", "Movie", windows=("30", "60"))
        tree(tmp_path, "camcan_rest", "Rest", windows=("30", "60"))
        M.main(["--from", "camcan_rest", "--to", "camcan",
                "--output-root", str(tmp_path), "--apply"])
        for w in ("30", "60"):
            assert (tmp_path / "dfc" / "atlas=yeo7" / f"window_s={w}"
                    / "cohort=camcan" / "task=Rest" / "sub=02"
                    / "data.parquet").exists()

    def test_it_works_when_the_destination_does_not_exist_yet(self, tmp_path):
        """A plain rename, not a merge."""
        tree(tmp_path, "camcan_rest", "Rest")
        M.main(["--from", "camcan_rest", "--to", "camcan",
                "--output-root", str(tmp_path), "--apply"])
        assert (tmp_path / "activation" / "atlas=yeo7" / "cohort=camcan"
                / "task=Rest" / "sub=01" / "data.parquet").exists()


class TestItLooksBeforeItMoves:
    def test_the_default_is_a_dry_run(self, tmp_path):
        tree(tmp_path, "camcan_rest", "Rest")
        assert M.main(["--from", "camcan_rest", "--to", "camcan",
                       "--output-root", str(tmp_path)]) == 0
        assert (tmp_path / "activation" / "atlas=yeo7"
                / "cohort=camcan_rest").exists()
        assert not (tmp_path / "activation" / "atlas=yeo7"
                    / "cohort=camcan").exists()

    def test_a_colliding_task_is_refused_not_overwritten(self, tmp_path):
        """The case that must never be silent: both cohorts holding the same
        task means two files claim one (cohort, task, sub) leaf, and whichever
        moved last would win."""
        tree(tmp_path, "camcan", "Rest")
        tree(tmp_path, "camcan_rest", "Rest")
        assert M.main(["--from", "camcan_rest", "--to", "camcan",
                       "--output-root", str(tmp_path), "--apply"]) == 1
        # nothing moved, both still intact
        for c in ("camcan", "camcan_rest"):
            assert (tmp_path / "activation" / "atlas=yeo7" / f"cohort={c}"
                    / "task=Rest" / "sub=01" / "data.parquet").exists()

    def test_the_same_cohort_twice_is_refused(self, tmp_path):
        with pytest.raises(SystemExit, match="same cohort"):
            M.main(["--from", "camcan", "--to", "camcan",
                    "--output-root", str(tmp_path)])


class TestWhatItDeliberatelyLeaves:
    def test_derived_per_cohort_tables_are_not_moved(self, tmp_path):
        """Each cohort has a COMPLETE copy of these covering its own tasks, so
        a move would overwrite one task's table with the other's. They are
        rebuilt by finalize, which rewrites them from scratch."""
        tree(tmp_path, "camcan", "Movie")
        tree(tmp_path, "camcan_rest", "Rest")
        M.main(["--from", "camcan_rest", "--to", "camcan",
                "--output-root", str(tmp_path), "--apply"])
        movie_meta = tmp_path / "meta" / "cohorts" / "cohort=camcan"
        # the movie's own tables survive untouched
        assert '"camcan"' in (movie_meta / "manifest_activation.json").read_text()
        assert "Movie" in (movie_meta / "participants_qc.csv").read_text()

    def test_censor_is_not_moved(self, tmp_path):
        tree(tmp_path, "camcan", "Movie")
        tree(tmp_path, "camcan_rest", "Rest")
        M.main(["--from", "camcan_rest", "--to", "camcan",
                "--output-root", str(tmp_path), "--apply"])
        assert (tmp_path / "censor" / "policy=motion" / "cohort=camcan_rest"
                / "subjects.parquet").exists()

    def test_the_per_run_shard_manifests_DO_move(self, tmp_path):
        """These are per array task, not per cohort, so finalize needs them
        under the destination to rebuild the merged manifest."""
        tree(tmp_path, "camcan", "Movie")
        tree(tmp_path, "camcan_rest", "Rest")
        M.main(["--from", "camcan_rest", "--to", "camcan",
                "--output-root", str(tmp_path), "--apply"])
        shards = tmp_path / "meta" / "cohorts" / "cohort=camcan" / "shards"
        assert {p.name for p in shards.iterdir()} == {
            "manifest_activation_Movie.json", "manifest_activation_Rest.json"}

    def test_it_says_what_to_re_run(self, tmp_path, capsys):
        tree(tmp_path, "camcan_rest", "Rest")
        M.main(["--from", "camcan_rest", "--to", "camcan",
                "--output-root", str(tmp_path)])
        out = capsys.readouterr().out
        assert "finalize.sbatch config/camcan.yaml activation" in out
        assert "censor.sbatch" in out
        assert "participants_qc.csv" in out
