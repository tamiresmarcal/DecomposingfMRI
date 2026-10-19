#!/usr/bin/env python3
"""HCP Movement_Regressors.txt -> an fMRIPrep-style confounds TSV.

    python tools/hcp_confounds.py config/hcp7t.yaml
    python tools/hcp_confounds.py config/hcp7t.yaml --apply

WHY THIS EXISTS
---------------
Two of the three existing cohorts clean motion inside the pipeline:

    filtering: {already_applied: false, bandpass: [0.01, 0.1], detrend: true}
    confounds: {format: fmriprep_tsv, strategy: 24hmp, fd_threshold: 0.5}

HCP can join that path exactly, but only by speaking fMRIPrep's dialect.
`format: afni_1D` would read Movement_Regressors.txt with np.loadtxt, but
`strategy` is honoured for `fmriprep_tsv` ONLY (activation.py), so afni_1D
means all 12 columns and no 24HMP; and `fd_threshold` needs an
`framewise_displacement` COLUMN, which HCP does not ship at all. Without this
converter the cohort gets a different motion model and NO frame censoring,
while the other three are gated at FD 0.5 -- the one asymmetry that would
undermine the comparison, for the reason static_fc.py states about censor
tables.

WHAT IT WRITES, AND WHY EACH PIECE
----------------------------------
Columns 1-6 of Movement_Regressors.txt are translations in mm and rotations in
DEGREES; 7-12 are their derivatives. fMRIPrep's `rot_*` are RADIANS, so the
rotations are converted -- not because the regression cares (least squares
absorbs a per-column scale) but because framewise displacement does, and
because a column called `rot_x` that holds degrees is a trap for the next
reader.

  * `*_derivative1`  backward difference, first sample 0, as fMRIPrep defines
    it. Computed from columns 1-6 rather than taken from HCP's 7-12, because
    their differencing convention is undocumented and a forward difference
    would shift every regressor one TR. The two are compared and the largest
    disagreement is reported, so the assumption is visible rather than silent.
  * `*_power2`       the square, which is what the 24 in 24HMP buys.
  * `framewise_displacement`  Power's FD: the summed absolute translation
    difference plus 50mm x the summed absolute rotation difference in radians.
    Derived from the six parameters rather than read from
    Movement_RelativeRMS.txt, which is a root-mean-square displacement and NOT
    Power FD -- so a 0.5 threshold on it would not mean what 0.5 means for the
    other three cohorts.

The file lands beside its BOLD run, named `sub-<id>_task-<name>_desc-
confounds_timeseries.tsv`, because `cli._attach_sidecar` globs the BOLD's own
directory and then requires every entity the run carries to appear in the
sidecar's NAME. A file missing `sub-` or `task-` is silently not attached.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HMP_6 = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
HMP_24 = [f"{b}{s}" for b in HMP_6
          for s in ("", "_derivative1", "_power2", "_derivative1_power2")]
COLUMNS = HMP_24 + ["framewise_displacement"]

#: Power et al.'s rotation-to-displacement radius, in mm. The same constant
#: fMRIPrep uses, so a threshold means the same thing in both.
FD_RADIUS_MM = 50.0


def read_movement(path: Path) -> np.ndarray:
    """The 12 columns HCP writes, as float. Raises if the shape is not 12."""
    arr = np.atleast_2d(np.loadtxt(path))
    if arr.ndim != 2 or arr.shape[1] != 12:
        raise SystemExit(
            f"{path}: expected 12 columns (6 parameters + 6 derivatives), "
            f"got shape {arr.shape}. This is not an HCP "
            f"Movement_Regressors.txt.")
    return arr


def confounds_from(mov: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
    """({column: values}, the largest disagreement with HCP's own derivatives).

    The disagreement is returned rather than acted on: it is a statement about
    their differencing convention, and the caller decides whether to care.
    """
    native = mov[:, :6]                       # mm, mm, mm, deg, deg, deg
    base = native.copy()
    base[:, 3:] = np.deg2rad(base[:, 3:])     # fMRIPrep's rot_* are radians

    d = np.zeros_like(base)
    d[1:] = np.diff(base, axis=0)             # backward difference, first = 0

    out: dict[str, np.ndarray] = {}
    for j, name in enumerate(HMP_6):
        out[name] = base[:, j]
        out[f"{name}_derivative1"] = d[:, j]
        out[f"{name}_power2"] = base[:, j] ** 2
        out[f"{name}_derivative1_power2"] = d[:, j] ** 2

    fd = np.zeros(len(base))
    fd[1:] = (np.abs(np.diff(native[:, :3], axis=0)).sum(axis=1)
              + FD_RADIUS_MM * np.abs(
                  np.diff(np.deg2rad(native[:, 3:]), axis=0)).sum(axis=1))
    out["framewise_displacement"] = fd

    # Their columns 7-12 are in NATIVE units, so compare in native units.
    theirs = mov[:, 6:]
    ours = np.zeros_like(native)
    ours[1:] = np.diff(native, axis=0)
    drift = float(np.abs(theirs - ours).max()) if theirs.size else 0.0
    return out, drift


def write_tsv(cols: dict[str, np.ndarray], path: Path) -> None:
    body = np.column_stack([cols[c] for c in COLUMNS])
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for row in body:
            fh.write("\t".join(f"{v:.10g}" for v in row) + "\n")
    tmp.replace(path)


def sidecar_name(sub: str, task: str) -> str:
    """What `cli._attach_sidecar` will accept for this run."""
    return f"sub-{sub}_task-{task}_desc-confounds_timeseries.tsv"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="the cohort YAML, for its discovery rules")
    p.add_argument("--apply", action="store_true",
                   help="actually write. Without it, only reports.")
    a = p.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fmri_decomposition.cohort import discover_runs
    from fmri_decomposition.config import load_config

    cfg = load_config(a.config)
    refs = discover_runs(cfg)
    if not refs:
        raise SystemExit(
            f"{a.config} discovers no runs -- fix discovery before this. "
            f"`check_cohort.py` says why.")
    print(f"{len(refs)} run(s) discovered")

    worst = 0.0
    n_done = n_missing = 0
    for ref in refs:
        src = Path(ref.bold).parent / "Movement_Regressors.txt"
        dest = Path(ref.bold).parent / sidecar_name(ref.sub, ref.task)
        if not src.exists():
            print(f"  MISSING {src}")
            n_missing += 1
            continue
        mov = read_movement(src)
        nt = _n_volumes(ref.bold)
        if nt is not None and len(mov) != nt:
            raise SystemExit(
                f"{src}: {len(mov)} motion row(s) but {nt} volume(s) in "
                f"{Path(ref.bold).name}.\n"
                f"  A confounds file one row out is not a small error: every "
                f"regressor, and every censored frame, would be shifted "
                f"against the data it describes.")
        cols, drift = confounds_from(mov)
        worst = max(worst, drift)
        if a.apply:
            write_tsv(cols, dest)
        n_done += 1
        if n_done <= 3 or not a.apply:
            fd = cols["framewise_displacement"]
            print(f"  {dest.name}  n={len(mov)}  "
                  f"FD mean={fd.mean():.3f} max={fd.max():.3f}  "
                  f"frames over 0.5mm: {(fd > 0.5).sum()}")

    print(f"\n{n_done} file(s)" + ("" if a.apply else "  (--apply to write)")
          + (f", {n_missing} run(s) with no Movement_Regressors.txt"
             if n_missing else ""))
    print(f"largest disagreement with HCP's own derivative columns: {worst:.3e}"
          "\n  Near zero means their columns ARE the backward difference and "
          "the convention assumed here is theirs. A large value means it is "
          "not, and the derivatives written here are still correct -- they are "
          "computed from the parameters -- but say so before trusting theirs.")
    return 0


def _n_volumes(bold) -> int | None:
    try:
        import nibabel as nib

        return int(nib.load(str(bold)).shape[3])
    except Exception:                                            # noqa: BLE001
        return None


if __name__ == "__main__":
    sys.exit(main())
