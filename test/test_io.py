from pathlib import Path

import pyarrow as pa
import pytest

from fmri_decomposition.io import (activation_path, activation_root, cleanup_stale_tmp,
                                   cohort_meta_dir, dfc_path, dfc_root, leaf_filename,
                                   should_skip, write_manifest, write_table_atomic,
                                   ManifestEntry)


class TestPaths:
    def test_activation_layout(self):
        p = activation_path("/out", "ds002837", "harvardoxford", "500daysofsummer", "1")
        assert p == Path("/out/activation/atlas=harvardoxford/cohort=ds002837/"
                         "task=500daysofsummer/sub=1/data.parquet")

    def test_dfc_layout_inserts_window_between_atlas_and_cohort(self):
        p = dfc_path("/out", "ds002837", "harvardoxford", 30, "500daysofsummer", "1")
        assert "atlas=harvardoxford/window_s=30/cohort=ds002837/task=" in str(p)

    def test_atlas_is_the_outermost_key(self):
        """Atlas is the only key that changes column count, so it must be a root.

        Everything below an atlas= directory shares a schema; if atlas sat
        deeper there would be no directory meaning "one atlas, many subjects",
        and no path could be handed to pyarrow.dataset without hitting 111- and
        14-column files in the same scan.
        """
        for p in (activation_path("/out", "c", "a", "t", "1"),
                  dfc_path("/out", "c", "a", 30, "t", "1")):
            keys = [x.split("=")[0] for x in p.parts if "=" in x]
            assert keys[0] == "atlas"
            assert keys.index("atlas") < keys.index("cohort")

    def test_roots_are_schema_homogeneous_prefixes_of_the_leaf(self):
        leaf = dfc_path("/out", "ds002837", "harvardoxford", 30, "movie", "1")
        assert str(leaf).startswith(str(dfc_root("/out", "harvardoxford")))
        assert str(leaf).startswith(str(dfc_root("/out", "harvardoxford", 30)))
        assert str(leaf).startswith(str(dfc_root("/out", "harvardoxford", 30, "ds002837")))

    def test_pooling_across_cohorts_is_one_directory(self):
        """The point of atlas-first: one path covers every cohort at one window."""
        root = dfc_root("/out", "harvardoxford", 30)
        for cohort in ("ds002837", "cneuromod", "hcp7t"):
            leaf = dfc_path("/out", cohort, "harvardoxford", 30, "m", "1")
            assert str(leaf).startswith(str(root))

    def test_activation_root_without_cohort_pools_cohorts(self):
        root = activation_root("/out", "yeo7")
        assert str(activation_path("/out", "camcan", "yeo7", "m", "CC110033")).startswith(
            str(root))

    def test_cohort_meta_is_keyed_the_same_hive_way(self):
        d = cohort_meta_dir("/out", "ds002837")
        assert d == Path("/out/meta/cohorts/cohort=ds002837")

    def test_task_sits_above_sub(self):
        """A fixed convention, chosen for the majority case where task count is 1.

        It costs CNeuroMod ~300 thin directories per atlas per window size; the
        alternative would make the path shape cohort-dependent, which is the
        same class of problem as a per-cohort run= level.
        """
        p = activation_path("/out", "c", "a", "movie", "1")
        parts = p.parts
        assert parts.index("task=movie") < parts.index("sub=1")

    def test_directory_depth_is_constant_across_cohorts(self):
        """Adding a run= directory for one cohort would break discovery for all."""
        no_ses = activation_path("/out", "c", "a", "t", "1")
        with_ses = activation_path("/out", "c", "a", "t", "01", ses="003", run="01")
        assert len(no_ses.parts) == len(with_ses.parts)
        assert with_ses.name == "ses-003_run-01.parquet"

    def test_leaf_filename_is_deterministic_not_part_0(self):
        assert leaf_filename() == "data.parquet"
        assert leaf_filename(ses="1") == "ses-1.parquet"
        assert leaf_filename(ses="1", run="2", acq="x") == "ses-1_run-2_acq-x.parquet"

    def test_key_values_are_filesystem_safe(self):
        p = activation_path("/out", "c", "a", "task/with slash", "CC110033")
        assert "task=task-with-slash" in str(p)
        assert "sub=CC110033" in str(p)

    def test_window_key_drops_a_pointless_decimal(self):
        assert "window_s=30/" in str(dfc_path("/o", "c", "a", 30.0, "t", "1"))
        assert "window_s=2.5/" in str(dfc_path("/o", "c", "a", 2.5, "t", "1"))


class TestAtomicWrite:
    def test_writes_and_leaves_no_tmp(self, tmp_path):
        table = pa.table({"a": pa.array([1, 2, 3])})
        out = tmp_path / "deep" / "nested" / "data.parquet"
        write_table_atomic(table, out)
        assert out.exists()
        assert list(tmp_path.rglob("*.tmp.*")) == []

    def test_a_failed_write_leaves_no_file_at_the_final_path(self, tmp_path, monkeypatch):
        """A file at its final path is complete by construction."""
        import pyarrow.parquet as pq

        def boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(pq, "write_table", boom)
        out = tmp_path / "data.parquet"
        with pytest.raises(OSError):
            write_table_atomic(pa.table({"a": [1]}), out)
        assert not out.exists()

    def test_round_trip(self, tmp_path):
        import pyarrow.parquet as pq

        table = pa.table({"a": pa.array([1.5, 2.5], type=pa.float32())})
        out = write_table_atomic(table, tmp_path / "x.parquet")
        assert pq.read_table(out).column("a").to_pylist() == [1.5, 2.5]

    def test_skip_if_exists(self, tmp_path):
        p = tmp_path / "x.parquet"
        assert not should_skip(p)
        p.write_bytes(b"")
        assert should_skip(p)
        assert not should_skip(p, overwrite=True)

    def test_stale_tmp_cleanup(self, tmp_path):
        stale = tmp_path / "data.parquet.tmp.123"
        stale.write_bytes(b"partial")
        import os
        import time

        os.utime(stale, (time.time() - 200_000,) * 2)
        assert cleanup_stale_tmp(tmp_path) == [stale]
        assert not stale.exists()


class TestManifest:
    def test_records_versions_and_config(self, tmp_path):
        import json

        entries = [ManifestEntry("dfc", "c", "a", "t", "1", "/p", "ok", n_rows=10,
                                 window_s=30)]
        path = write_manifest(tmp_path, entries, {"cohort": "c", "tr": 1.0})
        assert path.parent == tmp_path / "meta"
        payload = json.loads(path.read_text())
        assert payload["config"]["tr"] == 1.0
        assert "pyarrow" in payload["versions"]
        assert payload["entries"][0]["status"] == "ok"

    def test_cohort_scoped_manifest_lands_under_its_cohort(self, tmp_path):
        entries = [ManifestEntry("dfc", "ds002837", "a", "t", "1", "/p", "ok")]
        path = write_manifest(tmp_path, entries, {}, name="manifest_dfc.json",
                              cohort="ds002837")
        assert path == tmp_path / "meta" / "cohorts" / "cohort=ds002837" / "manifest_dfc.json"
