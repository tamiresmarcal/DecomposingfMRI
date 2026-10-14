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

import pandas as pd
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
    # Named by ARRAY TASK INDEX, exactly as cli.py:69 writes them -- which is
    # why two runs of one stage collide on the name while describing different
    # runs.
    for i in range(2):
        (meta / "shards"
         / f"manifest_dfc_shard-{i:04d}-of-0002.json").write_text(
            '{"entries": [{"stage": "dfc", "cohort": "%s", "atlas": "yeo7", '
            '"task": "%s", "sub": "0%d", "path": "p", "status": "ok"}]}'
            % (cohort, task, i))
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

    def test_a_clashing_shard_manifest_is_renamed_not_refused(self, tmp_path):
        """The case this hit in practice. Shard manifests are named by array
        task INDEX, so two runs of one stage both write shard-0000-of-0002.
        That is a name clash, not two files claiming one data leaf -- refusing
        would block the migration on provenance while the data was disjoint."""
        tree(tmp_path, "camcan", "Movie")
        tree(tmp_path, "camcan_rest", "Rest")
        assert M.main(["--from", "camcan_rest", "--to", "camcan",
                       "--output-root", str(tmp_path), "--apply"]) == 0
        shards = tmp_path / "meta" / "cohorts" / "cohort=camcan" / "shards"
        assert {p.name for p in shards.iterdir()} == {
            "manifest_dfc_shard-0000-of-0002.json",
            "manifest_dfc_shard-0001-of-0002.json",
            "manifest_dfc_shard-0000-of-0002-from-camcan_rest.json",
            "manifest_dfc_shard-0001-of-0002-from-camcan_rest.json"}

    def test_the_renamed_manifest_is_still_inside_the_merge_glob(self,
                                                                 tmp_path):
        """The suffix goes before `.json` for this reason: `merge-manifests`
        globs `manifest_<stage>_shard-*.json` and concatenates the entries of
        every match, so a renamed file is still merged and the consolidated
        manifest covers BOTH runs. A suffix after `.json` would silently drop
        one task's provenance."""
        import json

        tree(tmp_path, "camcan", "Movie")
        tree(tmp_path, "camcan_rest", "Rest")
        M.main(["--from", "camcan_rest", "--to", "camcan",
                "--output-root", str(tmp_path), "--apply"])
        shards = tmp_path / "meta" / "cohorts" / "cohort=camcan" / "shards"
        files = sorted(shards.glob("manifest_dfc_shard-*.json"))
        assert len(files) == 4
        tasks = {e["task"] for f in files
                 for e in json.loads(f.read_text())["entries"]}
        assert tasks == {"Movie", "Rest"}

    def test_it_says_what_to_re_run(self, tmp_path, capsys):
        tree(tmp_path, "camcan_rest", "Rest")
        M.main(["--from", "camcan_rest", "--to", "camcan",
                "--output-root", str(tmp_path)])
        out = capsys.readouterr().out
        assert "finalize.sbatch config/camcan.yaml activation" in out
        assert "censor.sbatch" in out
        assert "participants_qc.csv" in out


class TestSplitByTask:
    """The other half: give the four trees the `task=` key they lacked.

    A rewrite, not a move -- `task` was a COLUMN here, so each file is
    partitioned by its own rows. The part that needs testing is the schema
    metadata: all of it describes the FIT rather than the rows, so every part
    must carry it verbatim, and the per-column `clusterers` block is what
    gives the HMM2 label columns their provenance.
    """

    @staticmethod
    def _latents(root, cohort, tasks, states=("HMM2_pca3_8",)):
        import json

        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [(t, f"S{i}") for t in tasks for i in range(2)]
        d = pd.DataFrame({"cohort": cohort,
                          "task": [r[0] for r in rows],
                          "sub": [r[1] for r in rows],
                          "window_id": range(len(rows)),
                          "pca0/3": 0.5})
        for st in states:
            d[st] = 1
        t = pa.Table.from_pandas(d, preserve_index=False)
        md = {
            b"model_hash": json.dumps("abc123").encode(),
            b"censor_policy": json.dumps("motion").encode(),
            b"stride_s": json.dumps(2.47).encode(),
            b"clusterers": json.dumps(
                {st: {"fit_hash": f"fit-{st}", "k": 8,
                      "train_cohorts": ["ds002837", "cneuromod"],
                      "params": {"n_iter": 100}} for st in states}).encode(),
        }
        p = (root / "latents" / "atlas=yeo7" / "window_s=-1"
             / f"cohort={cohort}" / "data.parquet")
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t.replace_schema_metadata({**(t.schema.metadata or {}),
                                                  **md}), p)
        return p

    def test_the_default_is_a_dry_run(self, tmp_path):
        src = self._latents(tmp_path, "camcan", ["Movie", "Rest"])
        assert M.main(["--split-tasks", "--output-root", str(tmp_path)]) == 0
        assert src.exists()
        assert not (src.parent / "task=Movie").exists()

    def test_it_writes_one_file_per_task_and_removes_the_original(self,
                                                                  tmp_path):
        src = self._latents(tmp_path, "camcan", ["Movie", "Rest"])
        M.main(["--split-tasks", "--output-root", str(tmp_path), "--apply"])
        assert not src.exists()
        for t in ("Movie", "Rest"):
            assert (src.parent / f"task={t}" / "data.parquet").exists()

    def test_each_part_holds_only_its_own_rows(self, tmp_path):
        src = self._latents(tmp_path, "camcan", ["Movie", "Rest"])
        M.main(["--split-tasks", "--output-root", str(tmp_path), "--apply"])
        total = 0
        for t in ("Movie", "Rest"):
            d = pd.read_parquet(src.parent / f"task={t}" / "data.parquet")
            assert set(d["task"]) == {t}
            assert sorted(d["sub"]) == ["S0", "S1"]
            total += len(d)
        assert total == 4            # nothing lost, nothing duplicated

    def test_the_clusterers_provenance_survives_in_every_part(self, tmp_path):
        """THE POINT. Dropping this would leave the HMM2 columns in place while
        stripping the fit that produced them -- worse than losing them, because
        the labels would look usable."""
        import json

        import pyarrow.parquet as pq

        src = self._latents(tmp_path, "camcan", ["Movie", "Rest"])
        M.main(["--split-tasks", "--output-root", str(tmp_path), "--apply"])
        for t in ("Movie", "Rest"):
            md = pq.ParquetFile(src.parent / f"task={t}"
                                / "data.parquet").schema_arrow.metadata
            blk = json.loads(md[b"clusterers"].decode())
            assert blk["HMM2_pca3_8"]["fit_hash"] == "fit-HMM2_pca3_8"
            assert blk["HMM2_pca3_8"]["train_cohorts"] == ["ds002837",
                                                           "cneuromod"]
            assert json.loads(md[b"model_hash"].decode()) == "abc123"
            assert json.loads(md[b"censor_policy"].decode()) == "motion"
            assert json.loads(md[b"stride_s"].decode()) == 2.47

    def test_a_multi_task_latents_file_says_stride_s_is_inherited(self,
                                                                  tmp_path,
                                                                  capsys):
        """The one thing a split cannot fix: the file records ONE stride_s and
        every part inherits it, which is right only if those tasks share a TR.
        Said out loud rather than assumed, since it is the reason the level
        exists."""
        self._latents(tmp_path, "camcan", ["Movie", "Rest"])
        M.main(["--split-tasks", "--output-root", str(tmp_path)])
        out = capsys.readouterr().out
        assert "inherits the one stride_s" in out
        assert "Movie" in out and "Rest" in out

    def test_a_single_task_file_gets_no_such_note(self, tmp_path, capsys):
        self._latents(tmp_path, "camcan", ["Movie"])
        M.main(["--split-tasks", "--output-root", str(tmp_path)])
        assert "inherits the one stride_s" not in capsys.readouterr().out

    def test_an_already_split_tree_is_a_no_op(self, tmp_path, capsys):
        self._latents(tmp_path, "camcan", ["Movie", "Rest"])
        M.main(["--split-tasks", "--output-root", str(tmp_path), "--apply"])
        assert M.main(["--split-tasks", "--output-root", str(tmp_path),
                       "--apply"]) == 0
        assert "nothing to split" in capsys.readouterr().out

    def test_it_finds_every_tree_by_one_rule(self, tmp_path):
        """latents, transitions, static_fc and BOTH censor tables -- a leaf
        whose parent is a cohort= key has not been split. One rule, so there is
        no per-tree pattern list to keep in step with io.py."""
        for rel, leaf in (
                ("latents/atlas=yeo7/window_s=-1/cohort=c", "data.parquet"),
                ("transitions/atlas=yeo7/window_s=-1/states=S/cohort=c",
                 "subjects.parquet"),
                ("static_fc/atlas=yeo7/cohort=c", "subjects.parquet"),
                ("censor/policy=motion/cohort=c", "subjects.parquet"),
                ("censor/policy=motion/atlas=yeo7/window_s=30/cohort=c",
                 "windows.parquet")):
            d = tmp_path / rel
            d.mkdir(parents=True)
            pd.DataFrame({"task": ["m"], "sub": ["S0"]}).to_parquet(
                d / leaf, index=False)
        assert len(M.unsplit_leaves(tmp_path)) == 5
        M.main(["--split-tasks", "--output-root", str(tmp_path), "--apply"])
        assert M.unsplit_leaves(tmp_path) == []

    def test_a_leaf_already_under_a_task_key_is_not_a_candidate(self, tmp_path):
        d = tmp_path / "latents/atlas=yeo7/window_s=-1/cohort=c/task=Movie"
        d.mkdir(parents=True)
        pd.DataFrame({"task": ["Movie"], "sub": ["S0"]}).to_parquet(
            d / "data.parquet", index=False)
        assert M.unsplit_leaves(tmp_path) == []

    def test_either_a_move_or_a_split_is_required(self, tmp_path):
        with pytest.raises(SystemExit, match="--split-tasks"):
            M.main(["--output-root", str(tmp_path)])
