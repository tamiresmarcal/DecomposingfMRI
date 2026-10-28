"""Removing a state set has to remove the SOURCE, not just the derived tables.

`transitions` builds its plan from the state columns inside the latents, so
deleting `outputs/transitions/.../states=X` alone is undone by the next run.
These tests pin that the columns go, that the derived tables go with them, and
above all that nothing else in the file moves: the rows, the other state
columns, and the schema metadata are what make a latents file usable, and a
cleanup that quietly costs any of them is worse than the mess it tidies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import drop_state_columns as D  # noqa: E402

LEGACY = ["HMM_pca3_8", "HMM_pca3_27"]
CURRENT = ["HMM1_pca3_8", "HMM2_pca3_8"]

#: Everything a latents file says about itself besides its rows. Carried
#: verbatim, because a file that loses this looks usable and is not.
META = {
    b"model_hash": b'"abc123"',
    b"stride_s": b"2.47",
    b"censor_policy": b'"motion"',
    b"train_cohorts": b'["ds002837"]',
}


def write_cell(root, cohort, task, cols):
    d = (root / "latents" / "atlas=mini" / "window_s=-1"
         / f"cohort={cohort}" / f"task={task}")
    d.mkdir(parents=True, exist_ok=True)
    data = {"sub": ["S1", "S2"], "cohort": [cohort] * 2, "task": [task] * 2,
            "pca0/3": [0.1, 0.2]}
    for c in cols:
        data[c] = [0, 1]
    md = dict(META)
    md[b"clusterers"] = json.dumps(
        {c: {"method": "hmm", "fit_hash": f"h-{c}"} for c in cols}).encode()
    pq.write_table(
        pa.table(data).replace_schema_metadata(md), d / "data.parquet")
    return d / "data.parquet"


def write_transitions(root, states, task):
    d = (root / "transitions" / "atlas=mini" / "window_s=-1"
         / f"states={states}" / "cohort=camcan" / f"task={task}")
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"sub": ["S1"]}), d / "subjects.parquet")
    return d


@pytest.fixture
def tree(tmp_path):
    """Movie carries the legacy sets beside the current ones; rest carries
    only the current ones -- the asymmetry this tool exists to remove."""
    write_cell(tmp_path, "camcan", "Movie", CURRENT + LEGACY)
    write_cell(tmp_path, "camcan", "Rest", CURRENT)
    for s in CURRENT + LEGACY:
        write_transitions(tmp_path, s, "Movie")
    for s in CURRENT:
        write_transitions(tmp_path, s, "Rest")
    return tmp_path


def cols_of(path):
    return D.columns_in(path)


class TestItFindsOnlyWhatWasNamed:
    def test_the_plan_names_the_files_that_carry_them(self, tree):
        hits, missing = D.plan(tree, LEGACY)
        assert missing == []
        assert len(hits) == 1                       # only Movie has them
        assert "task=Movie" in str(next(iter(hits)))

    def test_a_name_that_matches_nothing_is_reported(self, tree):
        hits, missing = D.plan(tree, ["HMM_pca3_99"])
        assert hits == {} and missing == ["HMM_pca3_99"]

    def test_all_names_missing_refuses_rather_than_claiming_success(self, tree):
        with pytest.raises(SystemExit, match="none of the named state sets"):
            D.main(["--states", "HMM_pca3_99", "--output-root", str(tree),
                    "--apply"])

    def test_it_does_not_match_the_current_sets_by_prefix(self, tree):
        """`HMM_*` as a glob would also catch HMM1_/HMM2_. The tool takes
        exact names for exactly this reason."""
        hits, _ = D.plan(tree, LEGACY)
        carried = sorted(next(iter(hits.values())))
        assert carried == sorted(LEGACY)


class TestItRemovesTheColumnsAndKeepsEverythingElse:
    def test_the_named_columns_are_gone(self, tree):
        assert D.main(["--states", *LEGACY, "--output-root", str(tree),
                       "--apply"]) == 0
        movie = tree / ("latents/atlas=mini/window_s=-1/cohort=camcan/"
                        "task=Movie/data.parquet")
        cols, block = cols_of(movie)
        for s in LEGACY:
            assert s not in cols and s not in block

    def test_the_current_columns_survive(self, tree):
        D.main(["--states", *LEGACY, "--output-root", str(tree), "--apply"])
        movie = tree / ("latents/atlas=mini/window_s=-1/cohort=camcan/"
                        "task=Movie/data.parquet")
        cols, block = cols_of(movie)
        for s in CURRENT:
            assert s in cols and s in block
            assert block[s]["fit_hash"] == f"h-{s}"

    def test_the_rows_and_embedding_survive(self, tree):
        movie = tree / ("latents/atlas=mini/window_s=-1/cohort=camcan/"
                        "task=Movie/data.parquet")
        before = pq.ParquetFile(movie).read()
        D.main(["--states", *LEGACY, "--output-root", str(tree), "--apply"])
        after = pq.ParquetFile(movie).read()
        assert after.num_rows == before.num_rows
        assert after.column("pca0/3").to_pylist() == \
            before.column("pca0/3").to_pylist()
        assert after.column("sub").to_pylist() == ["S1", "S2"]

    def test_the_file_metadata_is_carried_verbatim(self, tree):
        """model_hash, stride_s and censor_policy describe the file, not the
        columns. stride_s especially: stage 5a reads it for every dwell time."""
        D.main(["--states", *LEGACY, "--output-root", str(tree), "--apply"])
        movie = tree / ("latents/atlas=mini/window_s=-1/cohort=camcan/"
                        "task=Movie/data.parquet")
        md = pq.ParquetFile(movie).schema_arrow.metadata
        for k, v in META.items():
            assert md[k] == v, k

    def test_a_cell_that_never_had_them_is_untouched(self, tree):
        rest = tree / ("latents/atlas=mini/window_s=-1/cohort=camcan/"
                       "task=Rest/data.parquet")
        before = rest.stat().st_mtime_ns
        D.main(["--states", *LEGACY, "--output-root", str(tree), "--apply"])
        assert rest.stat().st_mtime_ns == before


class TestItRemovesTheDerivedTablesToo:
    def test_the_transitions_for_those_sets_are_gone(self, tree):
        D.main(["--states", *LEGACY, "--output-root", str(tree), "--apply"])
        base = tree / "transitions" / "atlas=mini" / "window_s=-1"
        for s in LEGACY:
            assert not (base / f"states={s}").exists()

    def test_the_other_transitions_are_kept(self, tree):
        D.main(["--states", *LEGACY, "--output-root", str(tree), "--apply"])
        base = tree / "transitions" / "atlas=mini" / "window_s=-1"
        for s in CURRENT:
            assert (base / f"states={s}" / "cohort=camcan"
                    / "task=Movie" / "subjects.parquet").exists()
            assert (base / f"states={s}" / "cohort=camcan"
                    / "task=Rest" / "subjects.parquet").exists()

    def test_keep_transitions_leaves_them(self, tree):
        D.main(["--states", *LEGACY, "--output-root", str(tree),
                "--keep-transitions", "--apply"])
        base = tree / "transitions" / "atlas=mini" / "window_s=-1"
        assert (base / f"states={LEGACY[0]}").exists()
        # ...but the source is still gone, which is the point of the warning.
        movie = tree / ("latents/atlas=mini/window_s=-1/cohort=camcan/"
                        "task=Movie/data.parquet")
        assert LEGACY[0] not in cols_of(movie)[0]


class TestTheDryRunChangesNothing:
    def test_nothing_moves_without_apply(self, tree):
        movie = tree / ("latents/atlas=mini/window_s=-1/cohort=camcan/"
                        "task=Movie/data.parquet")
        before = movie.stat().st_mtime_ns
        assert D.main(["--states", *LEGACY,
                       "--output-root", str(tree)]) == 0
        assert movie.stat().st_mtime_ns == before
        assert LEGACY[0] in cols_of(movie)[0]
        base = tree / "transitions" / "atlas=mini" / "window_s=-1"
        assert (base / f"states={LEGACY[0]}").exists()
