"""Stage 4's activation source: the per-TR aperture written as window_s = -1.

The invariants under test are the ones no later stage can recover if they are
wrong. `window_id` is the only record of where a frame sat in the stimulus;
`crosses_run_boundary` is the only record of where the scanner stopped; and a
NaN parcel silently becoming 0 would hand the fit a value that was never
measured. None of the three is visible in the output once it is wrong.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fmri_decomposition import frames

PARCELS = ["P0", "P1", "P2"]


def leaf(n_per_run, tr=1.0, run_key="run-01", stim_offset=0.0, bad=(),
         values=None, stim_gap_tr=0):
    """One activation leaf as a DataFrame, runs concatenated as on disk.

    `stim_gap_tr` leaves that many TRs of stimulus unaccounted for between runs,
    which is the case where the window_ids already jump.
    """
    n = sum(n_per_run)
    t = np.arange(n, dtype=np.int32)
    run_idx = np.concatenate([np.full(L, i, dtype=np.int8)
                              for i, L in enumerate(n_per_run)])
    stim = np.empty(n, dtype=np.float32)
    pos, off = 0, stim_offset
    for L in n_per_run:
        stim[pos:pos + L] = off + np.arange(L) * tr
        off += (L + stim_gap_tr) * tr
        pos += L
    good = np.ones(n, dtype=bool)
    good[list(bad)] = False
    X = (np.tile(np.arange(n, dtype=np.float32)[:, None], (1, len(PARCELS)))
         if values is None else values)
    d = pd.DataFrame({"t": t, "time_s": (t * tr).astype(np.float32),
                      "stimulus_time_s": stim, "good_frame": good,
                      "run_idx": run_idx, "ses": None, "run": "01", "acq": None,
                      "run_key": run_key})
    for j, c in enumerate(PARCELS):
        d[c] = X[:, j]
    return d


def finalise(df, tr=1.0, band=None, zscore=False):
    stats = {"runs": 0, "runs_filtered": 0, "runs_too_short": 0,
             "boundary_frames": 0, "bad_frames": 0, "censored_windows": 0}
    prepped = frames._prepare_leaf(df, PARCELS, tr, band, zscore, stats)
    return frames._finalise_subject(prepped, tr, stats), stats


class TestWindowId:
    def test_is_the_frame_position_in_the_stimulus(self):
        out, _ = finalise(leaf([10], tr=2.0), tr=2.0)
        assert out["window_id"].to_list() == list(range(10))

    def test_offset_leaf_continues_the_index(self):
        out, _ = finalise(leaf([5], tr=1.49, stim_offset=5 * 1.49,
                               run_key="run-02"), tr=1.49)
        assert out["window_id"].to_list() == [5, 6, 7, 8, 9]

    def test_two_leaves_of_one_subject_share_one_index(self):
        a = leaf([6], run_key="run-01")
        b = leaf([6], run_key="run-02", stim_offset=6.0)
        stats = {"runs": 0, "runs_filtered": 0, "runs_too_short": 0,
                 "boundary_frames": 0, "bad_frames": 0, "censored_windows": 0}
        parts = [frames._prepare_leaf(x, PARCELS, 1.0, None, False, stats)
                 for x in (a, b)]
        out = frames._finalise_subject(pd.concat(parts, ignore_index=True),
                                       1.0, stats)
        assert out["window_id"].to_list() == list(range(12))


class TestRunBoundary:
    def test_marks_the_first_frame_of_a_later_run(self):
        out, stats = finalise(leaf([5, 5]))
        assert stats["boundary_frames"] == 1
        marked = out.loc[out["crosses_run_boundary"], "window_id"].to_list()
        assert marked == [5]

    def test_never_marks_the_earliest_run(self):
        out, _ = finalise(leaf([5]))
        assert not out["crosses_run_boundary"].any()

    def test_does_not_mark_when_the_index_already_jumps(self):
        # The stimulus itself skips 3 TRs between runs, so the gap splits the
        # sequence without spending a frame on it.
        out, stats = finalise(leaf([5, 5], stim_gap_tr=3))
        assert stats["boundary_frames"] == 0
        assert not out["crosses_run_boundary"].any()

    def test_marked_frame_leaves_a_gap_that_stage_5a_splits_on(self):
        from fmri_decomposition.transitions import subject_transitions

        out, _ = finalise(leaf([5, 5]))
        out["ThresholdCluster_pca3_2"] = 0          # one state: all self-pairs
        row = subject_transitions(out.sort_values("window_id"),
                                 "ThresholdCluster_pca3_2", 2, 1.0, 1)
        # 10 frames, 1 dropped as a boundary -> two segments of 5 and 4, so
        # 4 + 3 = 7 pairs, NOT the 8 a single 9-frame sequence would give.
        assert row["n_transitions"] == 7


class TestGoodFrames:
    def test_bad_frames_are_dropped_and_counted(self):
        out, stats = finalise(leaf([10], bad=(3, 7)))
        assert stats["bad_frames"] == 2
        assert out["window_id"].to_list() == [0, 1, 2, 4, 5, 6, 8, 9]

    def test_duplicate_window_id_is_fatal(self):
        # Two acquisitions covering the same stimulus span: not one time series.
        a = leaf([4], run_key="run-01")
        b = leaf([4], run_key="run-02", stim_offset=0.0)
        stats = {"runs": 0, "runs_filtered": 0, "runs_too_short": 0,
                 "boundary_frames": 0, "bad_frames": 0, "censored_windows": 0}
        parts = [frames._prepare_leaf(x, PARCELS, 1.0, None, False, stats)
                 for x in (a, b)]
        with pytest.raises(SystemExit, match="duplicate window_id"):
            frames._finalise_subject(pd.concat(parts, ignore_index=True),
                                     1.0, stats)


class TestZscore:
    def test_removes_scale_and_offset(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        a = frames._zscore(X.astype(np.float32))
        b = frames._zscore((X * 40 + 900).astype(np.float32))
        assert np.allclose(a, b, atol=1e-4)

    def test_a_constant_parcel_becomes_zero_not_nan(self):
        X = np.ones((50, 3), dtype=np.float32)
        out = frames._zscore(X)
        assert np.all(out == 0) and not np.isnan(out).any()

    def test_an_unmeasured_parcel_stays_nan(self):
        # activation.extract_parcels writes an all-NaN column for a parcel with
        # no voxels in the subject's mask. Letting it fall through to the
        # constant-parcel 0 would feed the fit a value nobody measured.
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, 3)).astype(np.float32)
        X[:, 1] = np.nan
        out = frames._zscore(X)
        assert np.isnan(out[:, 1]).all()
        assert not np.isnan(out[:, [0, 2]]).any()


class TestBandpass:
    def test_short_run_passes_through_unfiltered(self):
        X = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
        out, filtered = frames._bandpass(X, 1.0, (0.01, 0.1))
        assert filtered is False and np.shares_memory(out, X)

    def test_attenuates_a_tone_above_the_band(self):
        n, tr = 400, 1.0
        t = np.arange(n) * tr
        slow = np.sin(2 * np.pi * 0.05 * t)
        fast = np.sin(2 * np.pi * 0.35 * t)
        X = np.stack([slow + fast] * 3, axis=1).astype(np.float32)
        out, filtered = frames._bandpass(X, tr, (0.01, 0.1))
        assert filtered is True
        # the in-band component survives, the out-of-band one does not
        keep = slice(50, -50)                      # ignore filtfilt edge effects
        assert np.corrcoef(out[keep, 0], slow[keep])[0, 1] > 0.99

    def test_band_edge_at_or_above_nyquist_is_dropped_not_clipped(self):
        # TR 6 s -> Nyquist 0.083 Hz, so a 0.1 Hz low-pass does not exist. It
        # must degrade to the high-pass alone rather than to something whose
        # frequency content differs from every other cohort's.
        rng = np.random.default_rng(2)
        X = rng.normal(size=(200, 3)).astype(np.float32)
        out, filtered = frames._bandpass(X, 6.0, (0.01, 0.1))
        assert filtered is True and np.isfinite(out).all()


class TestSchemaReads:
    def write(self, tmp_path, tr=1.0, with_tr=True):
        d = leaf([8])
        tbl = pa.Table.from_pandas(d, preserve_index=False)
        md = {b"stage": b"activation"}
        if with_tr:
            md[b"tr"] = str(tr).encode()
        p = (tmp_path / "activation" / "atlas=toy" / "cohort=c"
             / "task=movie" / "sub=01" / "data.parquet")
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl.replace_schema_metadata(md), p)
        return p

    def test_feature_columns_are_the_parcels(self, tmp_path):
        p = self.write(tmp_path)
        assert frames.feature_columns(p) == PARCELS

    def test_tr_comes_from_the_shard(self, tmp_path):
        assert frames.shard_tr(self.write(tmp_path, tr=2.47)) == 2.47

    def test_a_shard_without_tr_is_fatal(self, tmp_path):
        with pytest.raises(SystemExit, match="no `tr` in its schema"):
            frames.shard_tr(self.write(tmp_path, with_tr=False))

    def test_shard_paths_finds_every_leaf_of_a_subject(self, tmp_path):
        p = self.write(tmp_path)
        for name in ("data_run-02.parquet", "data_run-03.parquet"):
            (p.parent / name).write_bytes(p.read_bytes())
        found = frames.shard_paths(tmp_path, "toy", "c")
        assert len(found) == 3


class TestOneCohortTwoTRs:
    """The bug the `task=` level exists to make impossible.

    `read_cohort` did `tr = shard_tr(paths[0])` -- one TR for the whole cohort,
    applied to every shard's band-pass and time axis, and written as the one
    `stride_s` a latents file can hold. Cam-CAN's movie (2.47s) and rest
    (1.97s) are the same people in one cohort, so paths[0] sorted to a Movie
    shard: every Rest frame would have been filtered at 2.47s -- a declared
    0.01-0.1 Hz band that is really 0.0126-0.126 Hz -- and every rest dwell
    time `transitions` reports would have been 25% wrong. Nothing errored.
    """

    FEATS = [f"P{i:02d}" for i in range(4)]

    def two_tr_cohort(self, root):
        from tests.test_static_fc import write_activation

        for sub in ("S0", "S1"):
            write_activation(root, "camcan", "Movie", sub, n_tr=120, tr=2.47)
            write_activation(root, "camcan", "Rest", sub, n_tr=120, tr=1.97)
        return root

    def test_the_tasks_are_discovered_from_the_directories(self, tmp_path):
        self.two_tr_cohort(tmp_path)
        assert frames.shard_tasks(tmp_path, "mini", "camcan") == ["Movie",
                                                                 "Rest"]

    def test_each_task_resolves_to_its_own_tr(self, tmp_path):
        self.two_tr_cohort(tmp_path)
        assert frames.cohort_tr(tmp_path, "mini", "camcan", "Movie") == 2.47
        assert frames.cohort_tr(tmp_path, "mini", "camcan", "Rest") == 1.97

    def test_asking_for_the_whole_cohort_is_REFUSED_not_guessed(self, tmp_path):
        """THE POINT. The old code returned 2.47 here -- the first shard's --
        and applied it to the 1.97s frames."""
        self.two_tr_cohort(tmp_path)
        with pytest.raises(SystemExit) as e:
            frames.cohort_tr(tmp_path, "mini", "camcan")
        msg = str(e.value)
        assert "more than one TR" in msg
        assert "2.47" in msg and "1.97" in msg
        assert "Movie" in msg and "Rest" in msg
        assert "one task at a time" in msg

    def test_read_cohort_refuses_the_same_way(self, tmp_path):
        self.two_tr_cohort(tmp_path)
        with pytest.raises(SystemExit, match="more than one TR"):
            frames.read_cohort(tmp_path, "mini", "camcan", self.FEATS,
                               log=lambda *a: None)

    def test_read_cohort_per_task_returns_that_tasks_tr(self, tmp_path):
        self.two_tr_cohort(tmp_path)
        for task, want in (("Movie", 2.47), ("Rest", 1.97)):
            ident, X, tr = frames.read_cohort(
                tmp_path, "mini", "camcan", self.FEATS, task=task,
                log=lambda *a: None)
            assert tr == want, task
            assert set(ident["task"]) == {task}

    def test_a_single_tr_cohort_still_needs_no_task(self, tmp_path):
        """The change must not make every existing config pass a task."""
        from tests.test_static_fc import write_activation

        for sub in ("S0", "S1"):
            write_activation(tmp_path, "ds002837", "film", sub, tr=1.0)
        assert frames.cohort_tr(tmp_path, "mini", "ds002837") == 1.0
        _, _, tr = frames.read_cohort(tmp_path, "mini", "ds002837", self.FEATS,
                                      log=lambda *a: None)
        assert tr == 1.0

    def test_two_tasks_at_the_SAME_tr_are_not_refused(self, tmp_path):
        """ds002837 has ten films at one TR. Only a TR disagreement is a
        problem; several tasks are not."""
        from tests.test_static_fc import write_activation

        for task in ("film_a", "film_b"):
            write_activation(tmp_path, "ds002837", task, "S0", tr=1.0)
        assert frames.cohort_tr(tmp_path, "mini", "ds002837") == 1.0

    def test_the_band_pass_actually_differs_between_the_two_trs(self, tmp_path):
        """Not just that the TR is reported right -- that it CHANGES the data.
        Same frames, two TRs, so the 0.01-0.1 Hz band keeps different content;
        if the filter ignored the TR these would be identical."""
        from tests.test_static_fc import write_activation

        write_activation(tmp_path, "a", "t", "S0", n_tr=200, tr=2.47, seed=3)
        write_activation(tmp_path, "b", "t", "S0", n_tr=200, tr=1.97, seed=3)
        _, slow, _ = frames.read_cohort(tmp_path, "mini", "a", self.FEATS,
                                        band=(0.01, 0.1), log=lambda *a: None)
        _, fast, _ = frames.read_cohort(tmp_path, "mini", "b", self.FEATS,
                                        band=(0.01, 0.1), log=lambda *a: None)
        assert slow.shape == fast.shape
        assert not np.allclose(slow, fast)
