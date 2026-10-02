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
