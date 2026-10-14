"""A partition key that is also a column may only be read one way.

The derived tables -- censor's two, static_fc's, latents, transitions -- carry
provenance columns that repeat a key in their own path: a `policy` column under
`policy=<name>/`, an `atlas` and a `cohort` column under `atlas=<a>/cohort=<c>/`,
and, since the task split, a `task` column under `task=<t>/`. The per-subject
shards of stages 1 and 2 deliberately do not (dfc.py: "cohort / atlas / task /
sub are partition keys, carried by the path"), which is why the main pipeline
never met this.

`pq.read_table(path)` builds a ParquetDataset even when handed a single file,
infers those keys from the directory names, and then tries to merge an inferred
dictionary<values=string> field with the file's own string column:

    ArrowTypeError: Unable to merge: Field policy has incompatible types:
    string vs dictionary<values=string, indices=int32, ordered=0>

That is what killed `migrate_cohort --split-tasks` partway through an --apply on
nibi. `io.read_file` opens the one file through the parquet reader, so there is
nothing to infer and nothing to merge.

`pd.read_parquet(path)` on a single local file turns out to be safe, for a
reason worth writing down because it is not a guarantee: pandas opens the file
itself and hands pyarrow an open HANDLE rather than a path, so there are no
directory names to infer from. It does that only when it is "very certain that
it is not a directory" (pandas.io.parquet._get_path_or_handle), and older
pandas passed the path straight through. The library therefore uses read_file
for these files regardless, instead of resting on a pandas internal that
differs across the versions this project runs on.

WHY THESE TESTS EMULATE THE READER INSTEAD OF CALLING IT

The failure is version-dependent: pyarrow 18 infers the keys and raises,
pyarrow 25 infers nothing for a file path and returns the table. So a test that
merely reads a file proves which pyarrow is installed, and on a developer's 25
it passes while the cluster's 18 fails -- which is exactly how the migration
tool shipped broken. `as_pyarrow_18` reproduces the RULE rather than the
version: it raises if, and only if, a path is handed to the dataset layer for a
file that has a column its own path also names as a key. That has teeth on
every version, and it is faithful enough that a new offending read site fails
here instead of waiting to fail on nibi.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from fmri_decomposition import censor as C
from fmri_decomposition import static_fc as S
from fmri_decomposition.decompose import load_censor
from fmri_decomposition.io import parse_hive_keys, read_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import migrate_cohort as M  # noqa: E402


def offending_keys(source) -> list[str]:
    """Keys this file's path names that are also columns inside it.

    Empty for anything that is not a filesystem path -- an open handle has no
    directory names, which is the whole reason pandas' route is safe.
    """
    if not isinstance(source, (str, Path)):
        return []
    p = Path(source)
    if not p.is_file():
        return []
    try:
        names = set(pq.ParquetFile(p).schema_arrow.names)
    except Exception:                                            # noqa: BLE001
        return []
    return sorted(set(parse_hive_keys(p)) & names)


@pytest.fixture
def as_pyarrow_18(monkeypatch):
    """Make the dataset layer behave the way the cluster's pyarrow does.

    Patches the one door into it -- `pq.read_table`, which is also what
    `pd.read_parquet` calls underneath -- so handing it a PATH for a
    duplicated-key file raises instead of quietly working. Everything else
    still goes through, so a whole flow can run under this and only an
    offending call fails.
    """
    import pyarrow.parquet as pq_mod

    real = pq_mod.read_table
    allowed: list[str] = []

    def fake_read_table(source, *a, **k):
        bad = offending_keys(source)
        if bad:
            raise pa.lib.ArrowTypeError(
                f"Unable to merge: Field {bad[0]} has incompatible types: "
                f"string vs dictionary<values=string, indices=int32, "
                f"ordered=0>   [read_table on {Path(source).name}]")
        allowed.append(type(source).__name__)
        return real(source, *a, **k)

    monkeypatch.setattr(pq_mod, "read_table", fake_read_table)
    return allowed


# -------------------------------------------------------------- the emulator
# itself, because a guard that never fires would make every test below vacuous.

class TestTheEmulatorHasTeeth:
    def test_a_path_to_a_file_that_repeats_a_column_raises(
            self, tmp_path, as_pyarrow_18):
        d = tmp_path / "censor" / "policy=motion" / "cohort=camcan" / "task=Movie"
        d.mkdir(parents=True)
        f = d / "subjects.parquet"
        pq.write_table(pa.table({"sub": ["S1"], "task": ["Movie"],
                                 "keep": [True], "policy": ["motion"]}), f)
        with pytest.raises(pa.lib.ArrowTypeError, match="Field policy"):
            pq.read_table(f)

    def test_a_shard_carrying_no_key_as_a_column_reads_fine(
            self, tmp_path, as_pyarrow_18):
        # The stage 1/2 convention: keys live in the path and nowhere else.
        d = tmp_path / "dfc" / "atlas=mini" / "cohort=c" / "task=T" / "sub=S1"
        d.mkdir(parents=True)
        f = d / "data.parquet"
        pq.write_table(pa.table({"window_id": [0, 1], "run_key": ["a", "a"]}), f)
        assert pq.read_table(f).num_rows == 2
        assert len(pd.read_parquet(f)) == 2

    def test_read_file_is_never_stopped_by_it(self, tmp_path, as_pyarrow_18):
        d = tmp_path / "censor" / "policy=motion" / "cohort=c" / "task=T"
        d.mkdir(parents=True)
        f = d / "subjects.parquet"
        pq.write_table(pa.table({"sub": ["S1"], "task": ["T"],
                                 "policy": ["motion"]}), f)
        assert read_file(f).num_rows == 1
        assert read_file(f, ["sub"]).column_names == ["sub"]


class TestWhyPandasGetsAwayWithIt:
    """Pinned, because the library no longer relies on it but it explains why
    nothing but the migration tool ever broke."""

    def test_pandas_hands_pyarrow_a_handle_and_not_a_path(self, tmp_path):
        d = tmp_path / "censor" / "policy=motion" / "cohort=camcan"
        d.mkdir(parents=True)
        f = d / "subjects.parquet"
        pq.write_table(pa.table({"sub": ["S1"], "policy": ["motion"]}), f)

        seen = []
        real = pq.read_table

        def spy(source, *a, **k):
            seen.append(type(source).__name__)
            return real(source, *a, **k)

        pq.read_table = spy
        try:
            pd.read_parquet(f)
        finally:
            pq.read_table = real
        assert seen and "Path" not in seen[0] and seen[0] != "str", (
            f"pandas passed a {seen[0]} -- if it ever passes the path again, "
            f"every pd.read_parquet of a duplicated-key file is back in play")


# ------------------------------------------------------------------- censor ->
# decompose, the pair whose files carry BOTH a `policy` and a `task` duplicate.

POLICY = {"name": "motion",
          "subject": {"max_mean_fd": 0.5, "min_frac_good_frames": 0.5,
                      "max_frac_parcels_empty": None,
                      "max_abs_best_lag_tr": None,
                      "min_frac_stimulus_covered": None},
          "window": {"min_frac_good_frames": None, "min_n_tr_effective": None,
                     "drop_rank_deficient": False,
                     "drop_crosses_run_boundary": False,
                     "drop_crosses_clip_boundary": False}}

QC = pd.DataFrame([
    {"sub": "S1", "task": "Movie", "mean_fd": 0.10, "frac_good_frames": 0.99},
    {"sub": "S2", "task": "Movie", "mean_fd": 0.90, "frac_good_frames": 0.99},
    {"sub": "S1", "task": "Rest", "mean_fd": 0.11, "frac_good_frames": 0.97},
    {"sub": "S2", "task": "Rest", "mean_fd": 0.13, "frac_good_frames": 0.96},
])


@pytest.fixture
def censored(tmp_path):
    qc_dir = tmp_path / "meta" / "cohorts" / "cohort=camcan"
    qc_dir.mkdir(parents=True)
    QC.to_csv(qc_dir / "participants_qc.csv", index=False)
    pol = tmp_path / "motion.yaml"
    pol.write_text(yaml.safe_dump(POLICY))
    assert C.run(argparse.Namespace(
        policy=str(pol), cohorts=["camcan"], output_root=str(tmp_path),
        stage="subjects", atlas=None, window_s=None)) == 0
    return tmp_path


class TestTheCensorTableIsStillReadable:
    def test_the_written_file_does_duplicate_its_keys(self, censored):
        # Not hypothetical: this is what makes the rest of the class matter.
        f = next((censored / "censor").rglob("subjects.parquet"))
        assert offending_keys(f) == ["policy", "task"]

    def test_load_censor_survives_the_cluster_reader(self, censored,
                                                     as_pyarrow_18):
        got = load_censor(censored, "motion", "mini", 30.0, "camcan")
        assert got["subjects"] == {("Movie", "S1"), ("Rest", "S1"),
                                   ("Rest", "S2")}

    def test_it_still_reads_every_task_not_just_the_first(self, censored,
                                                          as_pyarrow_18):
        got = load_censor(censored, "motion", "mini", 30.0, "camcan")
        assert got["n_subject_rows"] == 4


# ------------------------------------------------------------------ static_fc,
# whose `prov` writes `atlas` and `cohort` as columns beside each row's `task`.

class TestTheStaticFcTableIsStillReadable:
    @pytest.fixture
    def table(self, tmp_path):
        d = pd.DataFrame({"sub": ["S1", "S2"], "task": ["Movie", "Movie"],
                          "n_tr_used": [100, 110], "A__B": [0.1, 0.2]})
        S.write_cohort(tmp_path, "mini", "camcan", d,
                       {"atlas": "mini", "cohort": "camcan", "pool": "task"},
                       "Movie")
        return tmp_path

    def test_the_written_file_does_duplicate_its_keys(self, table):
        f = next((table / "static_fc").rglob("subjects.parquet"))
        assert offending_keys(f) == ["atlas", "cohort", "task"]

    def test_the_whole_table_survives_the_cluster_reader(self, table,
                                                         as_pyarrow_18):
        got = S.read_cohort_table(table, "mini", "camcan")
        assert list(got["sub"]) == ["S1", "S2"]
        assert "A__B" in got.columns

    def test_a_projection_survives_it_too(self, table, as_pyarrow_18):
        got = S.read_cohort_table(table, "mini", "camcan",
                                  columns=["sub", "task"])
        assert list(got.columns) == ["sub", "task"]

    def test_a_projected_column_the_cell_lacks_is_still_dropped(
            self, table, as_pyarrow_18):
        # read_file applies the projection itself now; that behaviour is load
        # bearing for cells of different widths and must not have been lost.
        got = S.read_cohort_table(table, "mini", "camcan",
                                  columns=["sub", "not_a_column"])
        assert list(got.columns) == ["sub"]


# -------------------------------------------------------- the migration tool,
# which is where this actually bit: it died on `policy` partway through --apply.

class TestTheSplitSurvivesTheClusterReader:
    @pytest.fixture
    def tree(self, tmp_path):
        """A censor table at the pre-split depth, with its `policy` column."""
        d = tmp_path / "censor" / "policy=motion" / "cohort=camcan"
        d.mkdir(parents=True)
        pq.write_table(
            pa.table({"sub": ["S1", "S2", "S1"],
                      "task": ["Movie", "Movie", "Rest"],
                      "keep": [True, False, True],
                      "policy": ["motion"] * 3,
                      "policy_hash": ["abc"] * 3}),
            d / "subjects.parquet")
        return tmp_path

    def test_it_splits_a_file_that_repeats_its_own_key(self, tree,
                                                       as_pyarrow_18):
        assert M.do_split(tree, apply=True) == 0
        base = tree / "censor" / "policy=motion" / "cohort=camcan"
        assert (base / "task=Movie" / "subjects.parquet").exists()
        assert (base / "task=Rest" / "subjects.parquet").exists()
        assert not (base / "subjects.parquet").exists()

    def test_every_row_lands_in_its_own_task(self, tree, as_pyarrow_18):
        M.do_split(tree, apply=True)
        base = tree / "censor" / "policy=motion" / "cohort=camcan"
        movie = pq.ParquetFile(base / "task=Movie" / "subjects.parquet").read()
        rest = pq.ParquetFile(base / "task=Rest" / "subjects.parquet").read()
        assert movie.num_rows == 2 and rest.num_rows == 1
        assert set(movie.column("task").to_pylist()) == {"Movie"}
        assert set(rest.column("task").to_pylist()) == {"Rest"}
        assert movie.column("policy").to_pylist() == ["motion", "motion"]

    def test_the_decision_is_unchanged_by_the_split(self, tree, as_pyarrow_18):
        M.do_split(tree, apply=True)
        got = load_censor(tree, "motion", "mini", 30.0, "camcan")
        assert got["subjects"] == {("Movie", "S1"), ("Rest", "S1")}

    def test_a_crash_partway_leaves_the_rest_resumable(self, tree):
        """What her interrupted --apply left behind.

        The source is unlinked only after its parts are written, so a file the
        tool never reached is untouched and a re-run finds it. Nothing is
        half-written and nothing is lost.
        """
        second = tree / "censor" / "policy=strict" / "cohort=camcan"
        second.mkdir(parents=True)
        pq.write_table(pa.table({"sub": ["S1"], "task": ["Movie"],
                                 "policy": ["strict"]}),
                       second / "subjects.parquet")

        real = M.split_file
        done = []

        def die_on_the_second(f, dest):
            if done:
                raise pa.lib.ArrowTypeError("Unable to merge: Field policy")
            done.append(f)
            return real(f, dest)

        M.split_file = die_on_the_second
        try:
            with pytest.raises(pa.lib.ArrowTypeError):
                M.do_split(tree, apply=True)
        finally:
            M.split_file = real

        left = M.unsplit_leaves(tree)
        assert len(left) == 1                      # the one it never reached
        assert M.do_split(tree, apply=True) == 0   # and a re-run finishes it
        assert M.unsplit_leaves(tree) == []
