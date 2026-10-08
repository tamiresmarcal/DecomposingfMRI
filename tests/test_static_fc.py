"""Stage 3b -- the benchmark's control arm.

The claim the module is built on is that its edges come from the SAME frames
the `window_s = -1` state arm is fitted on, so that a difference between the
two arms is a difference between models. `test_edges_equal_pearson_on_the_very_
frames_the_state_arm_reads` is that claim; if it ever fails, the benchmark is
comparing preprocessing and the headline result means nothing.
"""
import json

import numpy as np
import pandas as pd
import pytest

from fmri_decomposition import static_fc as S

PARCELS = [f"P{i:02d}" for i in range(6)]


def write_activation(root, cohort, task, sub, n_tr=120, tr=2.0, seed=0,
                     bad=(), run_keys=("r0",), nan_parcel=None):
    """A stage-2 shard, as `extract` writes one: meta columns, parcels, `tr`."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    for k, run_key in enumerate(run_keys):
        t = np.arange(n_tr)
        X = rng.normal(size=(n_tr, len(PARCELS))).astype(np.float32)
        X += np.sin(2 * np.pi * 0.03 * t * tr)[:, None].astype(np.float32)
        if nan_parcel is not None:
            X[:, nan_parcel] = np.nan
        d = pd.DataFrame(X, columns=PARCELS)
        d["t"] = t.astype(np.int32)
        d["time_s"] = (t * tr).astype(np.float32)
        d["stimulus_time_s"] = (t * tr).astype(np.float32)
        good = np.ones(n_tr, bool)
        good[list(bad)] = False
        d["good_frame"] = good
        d["run_idx"] = np.int16(0)
        d["run_key"] = run_key
        p = (root / "activation" / "atlas=mini" / f"cohort={cohort}"
             / f"task={task}" / f"sub={sub}" / f"{run_key}.parquet")
        p.parent.mkdir(parents=True, exist_ok=True)
        tbl = pa.Table.from_pandas(d, preserve_index=False)
        pq.write_table(tbl.replace_schema_metadata({b"tr": str(tr).encode()}), p)


@pytest.fixture
def cohort(tmp_path):
    for i, sub in enumerate(["S1", "S2", "S3"]):
        write_activation(tmp_path, "c", "m", sub, seed=i, bad=(2, 3))
    return tmp_path


class TestEdgeNames:
    def test_upper_triangle_row_major_matching_stage_three(self):
        assert S.edge_names(["a", "b", "c"]) == ["a__b", "a__c", "b__c"]

    def test_one_name_per_edge(self):
        n = len(PARCELS)
        assert len(S.edge_names(PARCELS)) == n * (n - 1) // 2

    def test_the_order_comes_from_the_file_not_the_registry(self, cohort):
        """Reordering the label table must not silently relabel every edge, so
        the names are built from the activation file's own column order."""
        from fmri_decomposition import frames

        paths = frames.shard_paths(cohort, "mini", "c")
        feats = frames.feature_columns(paths[0])
        d, _ = S.fc_for_cohort(cohort, "mini", "c", band=None)
        assert S.fc_columns(d) == S.edge_names(feats)


class TestFrameCoverage:
    def test_counts_acquired_and_surviving_separately(self, cohort):
        from fmri_decomposition import frames

        cover = S.frame_coverage(frames.shard_paths(cohort, "mini", "c"))
        assert cover[("m", "S1")] == (120, 118)

    def test_sums_over_several_acquisitions_of_one_subject(self, tmp_path):
        write_activation(tmp_path, "c", "m", "S1", bad=(1,),
                         run_keys=("r0", "r1"))
        from fmri_decomposition import frames

        cover = S.frame_coverage(frames.shard_paths(tmp_path, "mini", "c"))
        assert cover[("m", "S1")] == (240, 238)


class TestTheFairnessProperty:
    def test_edges_equal_pearson_on_the_very_frames_the_state_arm_reads(
            self, cohort):
        """THE test. `frames.read_cohort` is what stage 4 feeds the HMM at
        window_s=-1; these edges have to be that matrix's correlation and
        nothing else -- not a close reimplementation."""
        from fmri_decomposition import frames
        from fmri_decomposition.dfc import pearson_upper

        feats = frames.feature_columns(frames.shard_paths(cohort, "mini", "c")[0])
        ident, X, _ = frames.read_cohort(cohort, "mini", "c", feats, band=None,
                                         log=lambda *a: None)
        d, _ = S.fc_for_cohort(cohort, "mini", "c", band=None)
        row = d.set_index("sub").loc["S2"]
        sel = ident.index[(ident["sub"] == "S2")].to_numpy()
        want = pearson_upper(X[sel])
        got = row[S.fc_columns(d)].to_numpy(float)
        assert np.allclose(got, want, atol=1e-6)

    def test_bad_frames_are_excluded_from_the_correlation(self, tmp_path):
        """A censored TR is zero in every parcel at once, so leaving it in
        inflates every edge in the same direction. Stage 3 removes it by
        pairwise deletion; this path removes it by the same gate."""
        write_activation(tmp_path, "c", "m", "S1", n_tr=80, seed=1,
                         bad=range(40, 80))
        d, stats = S.fc_for_cohort(tmp_path, "mini", "c", band=None, min_tr=10)
        assert int(d["n_tr_used"].iloc[0]) == 40
        assert int(d["n_tr_total"].iloc[0]) == 80
        assert d["frac_good_frames"].iloc[0] == pytest.approx(0.5)


class TestPooling:
    def test_task_pooling_gives_one_row_per_task_and_subject(self, tmp_path):
        write_activation(tmp_path, "c", "m1", "S1", seed=1)
        write_activation(tmp_path, "c", "m2", "S1", seed=2)
        d, _ = S.fc_for_cohort(tmp_path, "mini", "c", pool="task", band=None)
        assert sorted(d["task"]) == ["m1", "m2"]
        assert (d["n_tasks"] == 1).all()

    def test_subject_pooling_concatenates_them_into_one_row(self, tmp_path):
        write_activation(tmp_path, "c", "m1", "S1", seed=1)
        write_activation(tmp_path, "c", "m2", "S1", seed=2)
        d, _ = S.fc_for_cohort(tmp_path, "mini", "c", pool="subject", band=None)
        assert len(d) == 1
        assert d["n_tasks"].iloc[0] == 2
        assert d["n_tr_used"].iloc[0] == 240

    def test_a_pooled_row_is_not_labelled_with_a_real_task_name(self, tmp_path):
        """It must not be mistakable for a row that measured one task."""
        write_activation(tmp_path, "c", "m1", "S1", seed=1)
        write_activation(tmp_path, "c", "rest", "S1", seed=2)
        d, _ = S.fc_for_cohort(tmp_path, "mini", "c", pool="subject", band=None)
        assert d["task"].iloc[0] == "(pooled)"

    def test_the_two_agree_on_a_one_task_cohort(self, cohort):
        """Cam-CAN's movie is one task, so the choice must not move a number."""
        a, _ = S.fc_for_cohort(cohort, "mini", "c", pool="task", band=None)
        b, _ = S.fc_for_cohort(cohort, "mini", "c", pool="subject", band=None)
        cols = S.fc_columns(a)
        assert np.allclose(a.sort_values("sub")[cols].to_numpy(float),
                           b.sort_values("sub")[cols].to_numpy(float))


class TestRefusalsAndFlags:
    def test_a_thin_subject_is_skipped_and_reported_not_written(self, tmp_path):
        write_activation(tmp_path, "c", "m", "S1", n_tr=120, seed=1)
        write_activation(tmp_path, "c", "m", "S2", n_tr=120, seed=2,
                         bad=range(10, 120))
        d, stats = S.fc_for_cohort(tmp_path, "mini", "c", band=None, min_tr=30)
        assert d["sub"].to_list() == ["S1"]
        assert stats["skipped_thin"] == [("m", "S2", 10)]

    def test_everyone_thin_names_min_tr_rather_than_writing_nothing(self,
                                                                   tmp_path):
        write_activation(tmp_path, "c", "m", "S1", n_tr=40, seed=1)
        with pytest.raises(SystemExit) as e:
            S.fc_for_cohort(tmp_path, "mini", "c", band=None, min_tr=200)
        assert "--min-tr 200" in str(e.value)

    def test_an_empty_parcel_nans_its_edges_and_keeps_the_row(self, tmp_path):
        """A parcel outside one subject's brain mask is a coverage fact. The
        row survives it; how many missing edges a MODEL tolerates is the
        model step's decision, recorded there."""
        write_activation(tmp_path, "c", "m", "S1", seed=1, nan_parcel=0)
        d, _ = S.fc_for_cohort(tmp_path, "mini", "c", band=None)
        assert len(d) == 1
        assert d["n_edges_nan"].iloc[0] == len(PARCELS) - 1
        assert np.isnan(d["P00__P01"].iloc[0])
        assert np.isfinite(d["P01__P02"].iloc[0])

    def test_no_shards_points_at_extract_not_at_this_stage(self, tmp_path):
        with pytest.raises(SystemExit) as e:
            S.fc_for_cohort(tmp_path, "mini", "nope", band=None)
        assert "fmri-decomp extract" in str(e.value)

    def test_a_missing_table_names_the_command_that_makes_it(self, tmp_path):
        with pytest.raises(SystemExit) as e:
            S.read_cohort_table(tmp_path, "mini", "c")
        assert "fmri-decomp static-fc" in str(e.value)


class TestStorageModes:
    def test_a_narrow_atlas_keeps_one_column_per_edge(self, cohort):
        _, stats = S.fc_for_cohort(cohort, "mini", "c", band=None)
        assert stats["edge_storage"] == "columns"

    def test_a_packed_table_is_unpacked_back_into_named_columns(self, tmp_path):
        """Above stage 3's threshold the edges become one list column. A reader
        asking for edges should not have to branch on the atlas's width."""
        names = S.edge_names([f"N{i:03d}" for i in range(4)])
        d = pd.DataFrame({"task": ["m"], "sub": ["S1"], "n_tr_used": [100],
                          "edges": [np.arange(len(names), dtype=np.float32)],
                          "edge_names": [json.dumps(names)]})
        p = (tmp_path / "static_fc" / "atlas=big" / "cohort=c")
        p.mkdir(parents=True)
        d.to_parquet(p / "subjects.parquet", index=False)
        out = S.read_cohort_table(tmp_path, "big", "c")
        assert S.fc_columns(out) == names
        assert out["N000__N001"].iloc[0] == 0.0
        assert "edges" not in out.columns


class TestWrittenTable:
    def test_the_file_carries_what_produced_it(self, cohort):
        d, stats = S.fc_for_cohort(cohort, "mini", "c", band=None)
        path = S.write_cohort(cohort, "mini", "c", d,
                              {"atlas": "mini", "cohort": "c", "pool": "task",
                               "estimator": "pearson_over_good_frames"})
        back = pd.read_parquet(path)
        assert back["estimator"].iloc[0] == "pearson_over_good_frames"
        assert (back["pool"] == "task").all()


class TestProjection:
    def test_asking_for_keys_does_not_read_the_edges(self, cohort):
        d, _ = S.fc_for_cohort(cohort, "mini", "c", band=None)
        S.write_cohort(cohort, "mini", "c", d, {"pool": "task"})
        keys = S.read_cohort_table(cohort, "mini", "c",
                                   columns=["task", "sub", "n_tr_used"])
        assert keys.columns.to_list() == ["task", "sub", "n_tr_used"]
        assert S.fc_columns(keys) == []

    def test_a_projected_column_the_cell_lacks_is_dropped_not_raised(self,
                                                                    cohort):
        """The table's width depends on the storage mode and on which QC
        columns the writing version carried, so a projection names what it
        wants and takes what is there."""
        d, _ = S.fc_for_cohort(cohort, "mini", "c", band=None)
        S.write_cohort(cohort, "mini", "c", d, {"pool": "task"})
        out = S.read_cohort_table(cohort, "mini", "c",
                                  columns=["sub", "not_a_column"])
        assert out.columns.to_list() == ["sub"]

    def test_without_a_projection_the_edges_still_come_back(self, cohort):
        d, _ = S.fc_for_cohort(cohort, "mini", "c", band=None)
        S.write_cohort(cohort, "mini", "c", d, {"pool": "task"})
        assert len(S.fc_columns(S.read_cohort_table(cohort, "mini", "c"))) == 15
