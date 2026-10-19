"""HCP motion -> fMRIPrep dialect, so one cohort is not gated differently.

Two of the three existing cohorts regress 24HMP and censor at FD 0.5. HCP can
join that path only by speaking fMRIPrep's column names: `strategy` is honoured
for `fmriprep_tsv` ONLY, and `fd_threshold` needs a `framewise_displacement`
COLUMN, which HCP does not ship. Without the conversion this cohort would get a
different motion model and no frame censoring at all.

So these tests are mostly about units and conventions, because that is where a
silent wrong answer lives: degrees read as radians makes FD 57x too small, and a
forward difference shifts every regressor one TR against the data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import hcp_confounds as H  # noqa: E402


def movement(trans=None, rot_deg=None, n=4):
    """An HCP Movement_Regressors.txt as an array: 6 params + 6 derivatives."""
    mov = np.zeros((n, 12))
    if trans is not None:
        mov[:, 0] = trans
    if rot_deg is not None:
        mov[:, 3] = rot_deg
    mov[1:, 6:] = np.diff(mov[:, :6], axis=0)       # their backward difference
    return mov


class TestTheColumnsAreWhatThePipelineAsksFor:
    def test_it_writes_24hmp_plus_fd(self):
        assert len(H.COLUMNS) == 25
        assert H.COLUMNS[-1] == "framewise_displacement"

    def test_the_names_match_activation_HMP_24_exactly(self):
        """Not a copy to keep in step by hand: a mismatch here is the
        'missing N requested confound column(s)' error at stage 2."""
        from fmri_decomposition.activation import HMP_24

        assert H.HMP_24 == HMP_24

    def test_every_column_is_produced(self):
        cols, _ = H.confounds_from(movement(trans=[0, 1, 2, 3]))
        assert set(cols) == set(H.COLUMNS)
        assert all(len(v) == 4 for v in cols.values())


class TestUnits:
    def test_rotations_are_converted_to_radians(self):
        """fMRIPrep's rot_* are radians and HCP's are degrees. Left in degrees
        the regression would be unharmed -- least squares absorbs a per-column
        scale -- but FD would be 57x too small and the censor would pass
        everything."""
        cols, _ = H.confounds_from(movement(rot_deg=[0, 0, 180, 180]))
        assert cols["rot_x"] == pytest.approx([0, 0, np.pi, np.pi])

    def test_translations_are_left_in_mm(self):
        cols, _ = H.confounds_from(movement(trans=[0, 1, 2, 3]))
        assert cols["trans_x"] == pytest.approx([0, 1, 2, 3])


class TestFramewiseDisplacement:
    def test_a_pure_translation_is_the_absolute_difference(self):
        cols, _ = H.confounds_from(movement(trans=[0.0, 1.0, 1.0, 3.0]))
        assert cols["framewise_displacement"] == pytest.approx([0, 1, 0, 2])

    def test_a_rotation_is_scaled_by_the_50mm_radius(self):
        """Power's convention, and fMRIPrep's, so 0.5 means the same thing
        here as it does for camcan."""
        cols, _ = H.confounds_from(movement(rot_deg=[0.0, 0.0, 180.0, 180.0]))
        fd = cols["framewise_displacement"]
        assert fd[2] == pytest.approx(50.0 * np.pi)
        assert fd[3] == pytest.approx(0.0)

    def test_the_first_frame_is_zero_not_undefined(self):
        cols, _ = H.confounds_from(movement(trans=[5.0, 6.0, 7.0, 8.0]))
        assert cols["framewise_displacement"][0] == 0.0

    def test_translation_and_rotation_add(self):
        mov = movement(trans=[0.0, 1.0], rot_deg=[0.0, 180.0], n=2)
        fd = H.confounds_from(mov)[0]["framewise_displacement"]
        assert fd[1] == pytest.approx(1.0 + 50.0 * np.pi)


class TestDerivatives:
    def test_backward_difference_with_a_zero_first_sample(self):
        cols, _ = H.confounds_from(movement(trans=[0.0, 1.0, 3.0, 6.0]))
        assert cols["trans_x_derivative1"] == pytest.approx([0, 1, 2, 3])

    def test_power2_is_the_square(self):
        cols, _ = H.confounds_from(movement(trans=[0.0, 1.0, 2.0, 3.0]))
        assert cols["trans_x_power2"] == pytest.approx([0, 1, 4, 9])
        assert cols["trans_x_derivative1_power2"] == pytest.approx([0, 1, 1, 1])

    def test_it_reports_agreement_with_hcps_own_derivative_columns(self):
        """Theirs are not used -- the convention is undocumented -- but the
        disagreement is measured so the assumption is visible."""
        _, drift = H.confounds_from(movement(trans=[0.0, 1.0, 3.0, 6.0]))
        assert drift == pytest.approx(0.0, abs=1e-12)

    def test_a_forward_difference_in_their_columns_is_reported_not_hidden(self):
        mov = movement(trans=[0.0, 1.0, 3.0, 6.0])
        mov[:-1, 6:] = np.diff(mov[:, :6], axis=0)      # forward instead
        mov[-1, 6:] = 0
        cols, drift = H.confounds_from(mov)
        assert drift > 0.5
        # ...and ours is still the backward one, computed from the parameters.
        assert cols["trans_x_derivative1"] == pytest.approx([0, 1, 2, 3])


class TestTheFileIsNamedSoItIsActuallyFound:
    def test_it_carries_the_entities_the_sidecar_matcher_demands(self):
        """`cli._attach_sidecar` globs the BOLD's directory and then keeps only
        hits whose NAME contains every entity -- sub- and task-. A file missing
        either is silently not attached, and the run gets no confounds at all
        with nothing said."""
        name = H.sidecar_name("100610", "MOVIE1")
        assert "sub-100610" in name
        assert "task-MOVIE1" in name
        assert name.endswith("_desc-confounds_timeseries.tsv")

    def test_a_default_glob_matches_it(self):
        from fnmatch import fnmatch

        assert fnmatch(H.sidecar_name("100610", "MOVIE1"),
                       "*_desc-confounds_timeseries.tsv")


class TestReadingAndWriting:
    def test_a_file_that_is_not_hcp_motion_is_refused(self, tmp_path):
        f = tmp_path / "Movement_Regressors.txt"
        np.savetxt(f, np.zeros((10, 6)))
        with pytest.raises(SystemExit, match="expected 12 columns"):
            H.read_movement(f)

    def test_the_round_trip_preserves_the_values(self, tmp_path):
        cols, _ = H.confounds_from(movement(trans=[0.0, 1.5, 2.25, 3.125]))
        out = tmp_path / H.sidecar_name("100610", "MOVIE1")
        H.write_tsv(cols, out)

        import pandas as pd

        back = pd.read_csv(out, sep="\t")
        assert list(back.columns) == H.COLUMNS
        assert back["trans_x"].to_numpy() == pytest.approx(
            [0.0, 1.5, 2.25, 3.125])
        assert back["framewise_displacement"].to_numpy() == pytest.approx(
            cols["framewise_displacement"])

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        cols, _ = H.confounds_from(movement(trans=[0.0, 1.0], n=2))
        out = tmp_path / "x.tsv"
        H.write_tsv(cols, out)
        assert out.exists()
        assert list(tmp_path.glob("*.tmp")) == []
