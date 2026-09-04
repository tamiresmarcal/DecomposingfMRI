"""Framewise displacement from motion regressor files.

The subject-level mean is the number ds002837 can actually have (its censor
file cannot be aligned to the images); these tests pin down that it is
computed from the right columns, in the right units, without differencing
across a run boundary.
"""

import numpy as np
import pytest

from fmri_decomposition.motion import (FD_RADIUS_MM, MotionError,
                                       framewise_displacement, read_1d,
                                       run_starts_from_matrix,
                                       select_motion_columns,
                                       summarize_motion_file)


def write_1d(path, values, header_lines=()):
    with open(path, "w") as fh:
        for line in header_lines:
            fh.write(f"# {line}\n")
        for row in np.atleast_2d(values):
            fh.write(" ".join(f"{v:.6f}" for v in np.atleast_1d(row)) + "\n")
    return str(path)


class TestFramewiseDisplacement:
    def test_pure_translation_is_the_sum_of_absolute_steps(self):
        params = np.zeros((4, 6))
        params[:, 3] = [0.0, 1.0, 1.0, -0.5]      # dS, mm
        fd = framewise_displacement(params)
        assert np.isnan(fd[0])
        assert np.allclose(fd[1:], [1.0, 0.0, 1.5])

    def test_rotation_is_arc_length_on_a_50mm_sphere(self):
        params = np.zeros((2, 6))
        params[1, 0] = 1.0                         # roll, 1 degree
        fd = framewise_displacement(params)
        assert fd[1] == pytest.approx(np.deg2rad(1.0) * FD_RADIUS_MM)

    def test_afni_and_spm_orders_disagree_and_that_is_the_point(self):
        """Rotations first vs translations first is a 0.87x/1.15x error."""
        params = np.zeros((2, 6))
        params[1, 0] = 1.0
        afni = framewise_displacement(params, (0, 1, 2), (3, 4, 5), "deg")
        spm = framewise_displacement(params, (3, 4, 5), (0, 1, 2), "deg")
        assert afni[1] == pytest.approx(np.deg2rad(1.0) * FD_RADIUS_MM)
        assert spm[1] == pytest.approx(1.0)

    def test_radians_are_not_silently_treated_as_degrees(self):
        params = np.zeros((2, 6))
        params[1, 0] = 1.0
        deg = framewise_displacement(params, rot_unit="deg")
        rad = framewise_displacement(params, rot_unit="rad")
        assert rad[1] == pytest.approx(FD_RADIUS_MM)
        assert deg[1] < rad[1]

    def test_run_boundaries_are_not_differenced_across(self):
        """The between-run jump is a scanner restart, not head movement."""
        params = np.zeros((6, 6))
        params[3:, 3] = 20.0                       # a 20 mm step at the boundary
        naive = framewise_displacement(params)
        aware = framewise_displacement(params, run_starts=[0, 3])
        assert naive[3] == pytest.approx(20.0)
        assert np.isnan(aware[3]), "frame 0 of run 2 has nothing to difference against"
        assert np.nanmax(aware) == 0.0

    def test_first_frame_of_each_run_is_nan_not_zero(self):
        """Zero is a measurement; there is no measurement to make."""
        fd = framewise_displacement(np.zeros((10, 6)), run_starts=[0, 5])
        assert np.isnan(fd[0]) and np.isnan(fd[5])
        assert np.isfinite(fd[[1, 2, 3, 4, 6, 7, 8, 9]]).all()

    def test_demeaning_does_not_change_fd(self):
        """AFNI's X-matrix carries per-run demeaned motion. FD is a difference."""
        rng = np.random.default_rng(0)
        params = rng.normal(size=(50, 6))
        assert np.allclose(framewise_displacement(params),
                           framewise_displacement(params - params.mean(0)),
                           equal_nan=True)


class TestColumnSelection:
    def test_a_partial_label_set_is_not_completed_by_guessing(self, tmp_path):
        """Only roll and pitch are labelled. Four axes short is not four axes."""
        values = np.zeros((8, 8))
        values[:, 6] = np.arange(8)
        path = write_1d(tmp_path / "x.1D", values, [
            'ColumnLabels = "Run#1Pol#0 ; Run#1Pol#1 ; ventricles ; WMe ; '
            'bandpass#1 ; bandpass#2 ; roll_01 ; pitch_01"',
        ])
        arr, header = read_1d(path)
        with pytest.raises(MotionError):
            select_motion_columns(arr, header)

    def test_six_labelled_motion_columns_are_found_anywhere(self, tmp_path):
        n = 10
        values = np.zeros((n, 10))
        values[:, 4] = 1.0                          # roll
        values[:, 9] = 2.0                          # dP
        labels = ("Run#1Pol#0 ; Run#1Pol#1 ; ventricles ; WMe ; roll_01 ; "
                  "pitch_01 ; yaw_01 ; dS_01 ; dL_01 ; dP_01")
        path = write_1d(tmp_path / "x.1D", values, [f'ColumnLabels = "{labels}"'])
        arr, header = read_1d(path)
        params, (rot, trans, unit), how = select_motion_columns(arr, header)
        assert params.shape == (n, 6)
        assert (rot, trans, unit) == ((0, 1, 2), (3, 4, 5), "deg")
        assert np.allclose(params[:, 0], 1.0) and np.allclose(params[:, 5], 2.0)
        assert "ColumnLabels" in how

    def test_per_run_motion_blocks_are_summed_back_together(self, tmp_path):
        """`-regress_motion_per_run` gives one zero-padded column per run."""
        n = 10
        bases = ["roll", "pitch", "yaw", "dS", "dL", "dP"]
        arr = np.zeros((n, 12))
        for j in range(6):
            arr[:5, 2 * j] = j + 1                  # run 1 block, zero after
            arr[5:, 2 * j + 1] = -(j + 1)           # run 2 block, zero before
        labels = " ; ".join(f"{b}_{r:02d}" for b in bases for r in (1, 2))
        path = write_1d(tmp_path / "x.1D", arr, [f'ColumnLabels = "{labels}"'])
        values, header = read_1d(path)
        params, _, how = select_motion_columns(values, header)
        assert params.shape == (n, 6)
        assert np.allclose(params[:5, 0], 1.0) and np.allclose(params[5:, 0], -1.0)
        assert "per-run" in how

    def test_plain_six_column_file_assumes_the_declared_order(self, tmp_path):
        path = write_1d(tmp_path / "m.1D", np.zeros((5, 6)))
        values, header = read_1d(path)
        _, order, how = select_motion_columns(values, header, order="spm")
        assert order == ((3, 4, 5), (0, 1, 2), "deg")
        assert "no ColumnLabels" in how

    def test_unlabelled_wide_matrix_refuses_to_guess(self, tmp_path):
        path = write_1d(tmp_path / "big.1D", np.zeros((5, 40)))
        values, header = read_1d(path)
        with pytest.raises(MotionError, match="motion-columns"):
            select_motion_columns(values, header)

    def test_explicit_columns_accept_negative_indices(self, tmp_path):
        values = np.zeros((5, 40))
        values[:, -6] = 3.0
        path = write_1d(tmp_path / "big.1D", values)
        arr, header = read_1d(path)
        params, _, how = select_motion_columns(arr, header,
                                               columns=[-6, -5, -4, -3, -2, -1])
        assert np.allclose(params[:, 0], 3.0)
        assert "explicit" in how


class TestRunStarts:
    def test_runstart_header_is_used(self, tmp_path):
        path = write_1d(tmp_path / "x.1D", np.ones((10, 6)),
                        ['RunStart = "0 4 7"'])
        values, header = read_1d(path)
        assert run_starts_from_matrix(values, header) == [0, 4, 7]

    def test_block_structure_of_a_design_matrix_gives_the_boundaries(self):
        """Per-run polort columns are zero outside their own run."""
        values = np.zeros((10, 4))
        values[:4, 0] = 1.0
        values[:4, 1] = np.linspace(-1, 1, 4)
        values[4:, 2] = 1.0
        values[4:, 3] = np.linspace(-1, 1, 6)
        assert run_starts_from_matrix(values) == [0, 4]

    def test_a_plain_motion_file_is_one_run(self):
        rng = np.random.default_rng(1)
        assert run_starts_from_matrix(rng.normal(size=(20, 6))) == [0]


class TestSummarizeMotionFile:
    def test_end_to_end_on_an_afni_style_design_matrix(self, tmp_path):
        n = 100
        rng = np.random.default_rng(2)
        values = np.zeros((n, 10))
        values[:50, 0] = 1.0                        # run 1 polort
        values[50:, 1] = 1.0                        # run 2 polort
        motion = rng.normal(scale=0.05, size=(n, 6))
        motion[70] += 5.0                           # one big movement
        values[:, 4:10] = motion
        labels = ("Run#1Pol#0 ; Run#2Pol#0 ; ventricles ; WMe ; roll_01 ; pitch_01 ; "
                  "yaw_01 ; dS_01 ; dL_01 ; dP_01")
        path = write_1d(tmp_path / "sub-1_task-film_motion.1D", values,
                        [f'ColumnLabels = "{labels}"'])

        s = summarize_motion_file(path)
        assert s.n_motion_runs == 2
        assert s.n_fd_frames == n - 2               # one NaN per run start
        assert s.max_fd > 5.0
        assert 0.0 < s.mean_fd < s.max_fd
        assert s.fd_source.endswith(".1D")
        assert 0.0 <= s.frac_fd_gt_0p5 <= 1.0

    def test_a_wide_unlabelled_file_raises_rather_than_inventing_a_number(self, tmp_path):
        path = write_1d(tmp_path / "x.1D", np.zeros((20, 30)))
        with pytest.raises(MotionError):
            summarize_motion_file(path)

    def test_mean_fd_is_insensitive_to_a_20_frame_offset(self, tmp_path):
        """The ds002837 argument, as a test.

        The regressors are on the acquisition clock and the images on the
        stimulus clock, ~15-28 frames apart. Frame-level censoring cannot
        survive that; a mean over ~5,000 frames barely notices.
        """
        rng = np.random.default_rng(3)
        motion = np.cumsum(rng.normal(scale=0.02, size=(5470, 6)), axis=0)
        full = write_1d(tmp_path / "full.1D", motion)
        trimmed = write_1d(tmp_path / "trimmed.1D", motion[20:])
        a = summarize_motion_file(full).mean_fd
        b = summarize_motion_file(trimmed).mean_fd
        assert abs(a - b) / a < 0.01


class TestAfniOrtvecLayout:
    """ds002837's real layout, as read from sub-1's header (2026-09).

        0-8     Run#kPol#j          9   polort, 3 runs
        9-112   bandpass[0..103]  104
        113-121 ROIPC.fsvent.rNN    9   ventricle PCs
        122-139 mot_demean_rNN[0-5] 18  <- the parameters
        140-157 mot_deriv_rNN[0-5]  18  <- derivatives, must NOT be used

    The motion reached 3dDeconvolve through `-ortvec mot_demean.r01.1D
    mot_demean_r01`, so AFNI named the columns after the ortvec rather than
    after the axis. Matching only roll/pitch/yaw/dS/dL/dP found nothing here
    and the reader refused on all 86 subjects.
    """

    N, RUNS = 300, (0, 100, 200)

    def _file(self, tmp_path, n_bandpass=4, deriv_scale=99.0):
        rng = np.random.default_rng(0)
        labels, cols = [], []

        for r in range(1, 4):
            for pol in range(3):
                labels.append(f"Run#{r}Pol#{pol}")
                cols.append(rng.normal(size=self.N))
        for b in range(n_bandpass):
            labels.append(f"bandpass[{b}]#0")
            cols.append(rng.normal(size=self.N))
        for r in range(1, 4):
            for pc in range(3):
                labels.append(f"ROIPC.fsvent.r{r:02d}[{pc}]#0")
                cols.append(rng.normal(size=self.N))

        # the parameters: one block per run, each meaningful only in its run
        self.truth = np.zeros((self.N, 6))
        bounds = list(zip(self.RUNS, list(self.RUNS[1:]) + [self.N]))
        for r, (lo, hi) in enumerate(bounds, start=1):
            for axis in range(6):
                col = rng.normal(scale=0.05, size=self.N)
                labels.append(f"mot_demean_r{r:02d}[{axis}]#0")
                cols.append(col)
                self.truth[lo:hi, axis] = col[lo:hi]
        # derivatives, deliberately huge so using them would be obvious
        for r in range(1, 4):
            for axis in range(6):
                labels.append(f"mot_deriv_r{r:02d}[{axis}]#0")
                cols.append(rng.normal(scale=deriv_scale, size=self.N))

        values = np.column_stack(cols)
        path = tmp_path / "sub-1_task-film_polort_bandpass_vent_wm_motion.1D"
        with open(path, "w") as fh:
            fh.write("# <matrix\n")
            fh.write(f'#  ni_type = "{values.shape[1]}*double"\n')
            fh.write(f'#  ColumnLabels = "{" ; ".join(labels)}"\n')
            fh.write(f'#  RunStart = "{",".join(str(s) for s in self.RUNS)}"\n')
            fh.write("# >\n")
            for row in values:
                fh.write(" ".join(f"{v:.6f}" for v in row) + "\n")
        return str(path)

    def test_the_parameters_are_found_and_the_derivatives_are_not(self, tmp_path):
        path = self._file(tmp_path)
        values, header = read_1d(path)
        starts = run_starts_from_matrix(values, header)
        params, order, how = select_motion_columns(values, header, run_starts=starts)
        assert starts == list(self.RUNS)
        assert params.shape == (self.N, 6)
        assert np.allclose(params, self.truth, atol=1e-6), "wrong columns selected"
        assert order == ((0, 1, 2), (3, 4, 5), "deg")
        assert "mot_demean" in how and "3 run block(s)" in how

    def test_each_run_is_read_over_its_own_rows(self, tmp_path):
        """Run 2's block must supply rows 100-199, not run 1's."""
        path = self._file(tmp_path)
        values, header = read_1d(path)
        params, _, _ = select_motion_columns(
            values, header, run_starts=run_starts_from_matrix(values, header))
        assert np.allclose(params[100:200], self.truth[100:200], atol=1e-6)

    def test_fd_is_finite_and_not_computed_from_the_derivatives(self, tmp_path):
        """The synthetic derivatives are ~2000x larger; using them would show."""
        s = summarize_motion_file(self._file(tmp_path))
        assert s.n_motion_runs == 3
        assert s.n_fd_frames == self.N - 3      # one NaN per run start
        assert 0 < s.mean_fd < 1.0, f"mean_fd={s.mean_fd} suggests the wrong columns"

    def test_run_boundaries_come_from_the_header(self, tmp_path):
        """Not differenced across the two boundaries RunStart declares."""
        path = self._file(tmp_path)
        values, header = read_1d(path)
        starts = run_starts_from_matrix(values, header)
        params, _, _ = select_motion_columns(values, header, run_starts=starts)
        fd = framewise_displacement(params, run_starts=starts)
        assert np.isnan(fd[[0, 100, 200]]).all()
        assert np.isfinite(np.delete(fd, [0, 100, 200])).all()

    def test_a_header_disagreeing_with_its_own_columns_raises(self, tmp_path):
        path = self._file(tmp_path)
        values, header = read_1d(path)
        with pytest.raises(MotionError, match="disagrees"):
            select_motion_columns(values, header, run_starts=[0])   # says 1 run
