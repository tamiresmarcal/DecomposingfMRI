"""A partition key that is also a column, which is what the latents files are.

This broke on the cluster and passed every local test, because the failure is
VERSION-DEPENDENT: `pq.read_table` on a single file routes through
`ds.dataset`, which infers hive partitioning from the directory names and then
tries to merge the inferred `cohort` field with the file's own `cohort` column.
pyarrow 18 (the analysis container) raises; pyarrow 25 does not.

So these tests do not just call the helper and hope the local pyarrow agrees --
they assert the conflict is real by provoking it through `ds.dataset` directly,
and then assert the helper is immune. The first assertion is what makes the
second meaningful on a version where `read_table` happens to work.
"""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
import pytest

from fmri_decomposition.io import read_file


@pytest.fixture
def latents(tmp_path):
    """A latents file where `cohort` is both a path key and a real column."""
    p = (tmp_path / "latents" / "atlas=yeo7" / "window_s=30"
         / "cohort=camcan" / "data.parquet")
    p.parent.mkdir(parents=True)
    d = pd.DataFrame({"cohort": ["camcan"] * 6, "task": "m", "sub": "01",
                      "window_id": range(6), "pca0/3": 0.5,
                      "ThresholdCluster_pca3_8": 3})
    pq.write_table(
        pa.Table.from_pandas(d, preserve_index=False)
          .replace_schema_metadata({b"model_hash": b"abc"}), p)
    return p


class TestTheConflictIsReal:
    def test_hive_inference_cannot_merge_the_duplicated_key(self, latents):
        # The mechanism, provoked directly so the test does not depend on which
        # pyarrow version routes `read_table` through it.
        with pytest.raises(pa.ArrowTypeError, match="Unable to merge.*cohort"):
            pds.dataset(str(latents), format="parquet",
                        partitioning="hive").to_table()


class TestReadFile:
    def test_reads_the_whole_file(self, latents):
        t = read_file(latents)
        assert t.num_rows == 6
        assert "cohort" in t.column_names

    def test_keeps_the_schema_metadata(self, latents):
        # append_columns merges `clusterers` into this; losing it would silently
        # strip every file's provenance.
        assert (read_file(latents).schema.metadata or {}).get(b"model_hash") == b"abc"

    def test_projects_columns(self, latents):
        t = read_file(latents, ["cohort", "window_id"])
        assert t.column_names == ["cohort", "window_id"]

    def test_a_projection_including_the_partition_key_still_works(self, latents):
        # The case that raised on the cluster.
        assert read_file(latents, ["cohort"]).num_rows == 6

    def test_silently_drops_columns_the_file_does_not_have(self, latents):
        # Callers pass a superset -- `crosses_run_boundary` is absent from a
        # dfc-sourced file written before it existed.
        t = read_file(latents, ["window_id", "not_a_column"])
        assert t.column_names == ["window_id"]

    def test_the_cohort_column_is_not_a_dictionary(self, latents):
        # A dictionary type is the symptom of partition inference having happened,
        # which is the thing being avoided. Asserting the exact string type would
        # be wrong: pandas writes `string` under pyarrow 18 and `large_string`
        # under 25, and this test must mean the same thing on both.
        t = read_file(latents).schema.field("cohort").type
        assert not pa.types.is_dictionary(t)
        assert pa.types.is_string(t) or pa.types.is_large_string(t)


class TestCallersUseIt:
    def test_append_columns_round_trips_on_a_partitioned_path(self, latents):
        import numpy as np

        from fmri_decomposition.cluster import append_columns

        append_columns(latents, {"HMM_pca3_8": np.arange(6) % 8},
                       {"HMM_pca3_8": {"method": "hmm", "k": 8}})
        t = read_file(latents)
        assert "HMM_pca3_8" in t.column_names
        assert t.num_rows == 6
        md = t.schema.metadata or {}
        assert b"clusterers" in md and md[b"model_hash"] == b"abc"

    def test_append_columns_replaces_a_column_of_the_same_name(self, latents):
        import json

        import numpy as np

        from fmri_decomposition.cluster import append_columns

        append_columns(latents, {"ThresholdCluster_pca3_8": np.zeros(6, int)},
                       {"ThresholdCluster_pca3_8": {"k": 8}})
        t = read_file(latents)
        assert t.column_names.count("ThresholdCluster_pca3_8") == 1
        assert t.column("ThresholdCluster_pca3_8").to_pylist() == [0] * 6
        assert "ThresholdCluster_pca3_8" in json.loads(
            t.schema.metadata[b"clusterers"].decode())

    def test_read_embedding_works_on_a_partitioned_path(self, latents):
        from fmri_decomposition.cluster import read_embedding

        d = read_embedding(latents, ["pca0/3"])
        assert d is not None and len(d) == 6


class TestKeysAreReadByNameNotByPosition:
    """The failure mode this convention exists to prevent.

    `rglob` is depth-agnostic, so inserting a key below an existing one does
    not stop a glob finding its files -- it shifts every positional index by
    one. And `"cohort=camcan".split("=")[1]` succeeds exactly as well as
    `"atlas=yeo7".split("=")[1]`, so nothing raises: the atlas silently becomes
    the window size. `bstm_selection.discover` and `fcm_selection.discover`
    both did this, and both were converted when `task=` was added.
    """

    def test_hive_key_finds_each_key_wherever_it_sits(self):
        from fmri_decomposition.io import hive_key

        p = Path("/o/transitions/atlas=yeo7/window_s=-1/states=HMM2_pca3_8"
                 "/cohort=camcan/task=Rest/subjects.parquet")
        assert hive_key(p, "atlas") == "yeo7"
        assert hive_key(p, "window_s") == "-1"
        assert hive_key(p, "states") == "HMM2_pca3_8"
        assert hive_key(p, "cohort") == "camcan"
        assert hive_key(p, "task") == "Rest"

    def test_a_missing_key_is_none_not_a_wrong_answer(self):
        from fmri_decomposition.io import hive_key

        p = Path("/o/static_fc/atlas=yeo7/cohort=camcan/subjects.parquet")
        assert hive_key(p, "task") is None
        assert hive_key(p, "window_s") is None

    def test_inserting_a_key_does_not_move_the_others(self):
        """What positional parsing could not survive: the same lookups give the
        same answers before and after a new level appears."""
        from fmri_decomposition.io import hive_key

        before = Path("/o/transitions/atlas=yeo7/window_s=30/states=S"
                      "/cohort=camcan/subjects.parquet")
        after = Path("/o/transitions/atlas=yeo7/window_s=30/states=S"
                     "/cohort=camcan/task=Movie/subjects.parquet")
        for key in ("atlas", "window_s", "states", "cohort"):
            assert hive_key(before, key) == hive_key(after, key), key

    def test_positional_reads_would_have_been_wrong(self):
        """Stated as a fact rather than left implicit: the old code read
        parts[-5:-2], and after the insertion those four positions hold the
        WRONG keys while still splitting cleanly on '='."""
        after = Path("/o/transitions/atlas=yeo7/window_s=30/states=S"
                     "/cohort=camcan/task=Movie/subjects.parquet")
        positional = [after.parts[i].split("=") for i in (-5, -4, -3, -2)]
        assert [k for k, _ in positional] == [
            "window_s", "states", "cohort", "task"]        # not atlas/window/states/cohort
        assert all(len(kv) == 2 for kv in positional)      # so nothing raises


class TestEveryTreeHasOnePathBuilder:
    """static_fc, transitions and censor built their paths inline while latents
    went through io. That spread is why adding one key touched five files."""

    def test_all_four_accept_the_task_level(self):
        from fmri_decomposition.io import (censor_root, latents_root,
                                           static_fc_root, transitions_root)

        assert latents_root("/o", "yeo7", -1, "c", "Movie").as_posix() == (
            "/o/latents/atlas=yeo7/window_s=-1/cohort=c/task=Movie")
        assert transitions_root("/o", "yeo7", -1, "S", "c", "Rest").as_posix() == (
            "/o/transitions/atlas=yeo7/window_s=-1/states=S/cohort=c/task=Rest")
        assert static_fc_root("/o", "yeo7", "c", "Movie").as_posix() == (
            "/o/static_fc/atlas=yeo7/cohort=c/task=Movie")
        assert censor_root("/o", "motion", "c", "Rest").as_posix() == (
            "/o/censor/policy=motion/cohort=c/task=Rest")

    def test_each_stops_at_the_first_key_not_given(self):
        """So one builder serves both 'narrow to one leaf' and 'glob a level'."""
        from fmri_decomposition.io import (censor_root, latents_root,
                                           static_fc_root, transitions_root)

        assert latents_root("/o", "yeo7", -1).name == "window_s=-1"
        assert latents_root("/o", "yeo7", -1, "c").name == "cohort=c"
        assert transitions_root("/o").name == "transitions"
        assert transitions_root("/o", "yeo7").name == "atlas=yeo7"
        assert static_fc_root("/o").name == "static_fc"
        assert censor_root("/o", "motion").name == "policy=motion"

    def test_the_window_is_formatted_the_same_way_everywhere(self):
        """-1 and 30 as integers, not -1.0 and 30.0, in both trees that key on
        it -- otherwise one stage's path cannot find the other's."""
        from fmri_decomposition.io import latents_root, transitions_root

        assert latents_root("/o", "a", -1.0).name == "window_s=-1"
        assert latents_root("/o", "a", 30.0).name == "window_s=30"
        assert transitions_root("/o", "a", -1.0).name == "window_s=-1"
        assert transitions_root("/o", "a", 30.0).name == "window_s=30"
