#!/usr/bin/env python3
"""Power framewise displacement for HCP, from Movement_Regressors.txt.

    apptainer exec --bind /project,/scratch,/home --pwd $PWD $FMRIDECOMP_SIF45 \
        python3 tools/hcp_fd.py --root /project/6008063/tamires/cohorts/hcp_fix

DESCRIPTIVE ONLY. HCP's ICA-FIX data are not censored or excluded on motion
(see config/hcp7t_fix.yaml and the Methods); this reports the same quantity
the other cohorts are summarised by, so the Supplementary Table can show it.
Read-only.

Each run directory of the HCP tree,
    <sub>/MNINonLinear/Results/{t,r}fMRI_<TASK>_7T_<PE>/Movement_Regressors.txt
holds 12 columns: trans x/y/z (mm), rot x/y/z (DEGREES), then their
derivatives. FD is computed by fmri_decomposition's own
`framewise_displacement` (Power et al. 2012, 50-mm sphere, first frame NaN),
NOT from Movement_RelativeRMS.txt, which is an RMS displacement and not FD.

Per (sub, task) the mean FD over the run's frames; per condition (MOVIE, REST)
the frame-weighted mean over that participant's four runs. Writes
hcp_fd.csv and prints mean (s.d.) across the participants in the latents.

The files are small. If they were not downloaded with the BOLD, fetch them
from the HCP S3 bucket with your HCP AWS credentials, e.g.
    aws s3 cp --recursive --exclude "*" --include "*/Movement_Regressors.txt" \
        s3://hcp-openaccess/HCP_1200/<sub>/MNINonLinear/Results/ \
        <root>/<sub>/MNINonLinear/Results/
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

from fmri_decomposition.extractions.motion import (                  # noqa: E402
    framewise_displacement)
from fmri_decomposition.io import default_output_root, latents_root  # noqa: E402

RUN = re.compile(r"[tr]fMRI_((?:MOVIE|REST)\d)_7T")


def run_fd(path: Path) -> np.ndarray:
    params = np.loadtxt(path)[:, :6]          # trans (mm) then rot (deg)
    return framewise_displacement(params, rot_idx=(3, 4, 5),
                                  trans_idx=(0, 1, 2), rot_unit="deg")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="the HCP tree (derivatives_root)")
    p.add_argument("--cohort", default="hcp7t_fix")
    p.add_argument("--atlas", default="harvardoxford")
    p.add_argument("--window-s", default="30")
    p.add_argument("--output-root", default=None)
    p.add_argument("-o", "--out", default="hcp_fd.csv")
    a = p.parse_args(argv)

    files = sorted(Path(a.root).glob("*/MNINonLinear/Results/*_7T_*/Movement_Regressors.txt"))
    print(f"{len(files)} Movement_Regressors.txt under {a.root}")
    if not files:
        raise SystemExit("none found: fetch them first (see the docstring)")
    rows = []
    for f in files:
        m = RUN.search(f.parent.name)
        if not m:
            continue
        fd = run_fd(f)
        ok = np.isfinite(fd)
        rows.append(dict(sub=f.parts[-5], task=m.group(1),
                         fd_sum=float(fd[ok].sum()), fd_n=int(ok.sum())))
    runs = pd.DataFrame(rows)
    runs["mean_fd"] = runs["fd_sum"] / runs["fd_n"]
    runs["condition"] = runs["task"].str.extract(r"^(MOVIE|REST)")[0]
    runs[["sub", "task", "condition", "mean_fd", "fd_n"]].to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(runs)} runs, {runs['sub'].nunique()} participants)")

    per = (runs.groupby(["sub", "condition"])[["fd_sum", "fd_n"]].sum()
           .assign(mean_fd=lambda d: d["fd_sum"] / d["fd_n"]).reset_index())

    root = Path(a.output_root) if a.output_root else default_output_root()
    keep = []
    for f in sorted(latents_root(root, a.atlas, a.window_s, a.cohort).glob("task=*/data.parquet")):
        cond = re.match(r"(MOVIE|REST)", f.parent.name.split("=", 1)[1]).group(1)
        s = pd.read_parquet(f, columns=["sub"])["sub"].astype(str).unique()
        keep.append(pd.DataFrame({"sub": s, "condition": cond}))
    k = pd.concat(keep).drop_duplicates() if keep else per[["sub", "condition"]]
    per["sub"] = per["sub"].astype(str)
    k = k.merge(per, on=["sub", "condition"], how="left")
    print(f"\nanalysed participants ({a.cohort}, {a.atlas}, {a.window_s}s):")
    for cond, g in k.groupby("condition"):
        fd = g["mean_fd"]
        print(f"  {cond:6s} n={len(g):4d}  with FD={int(fd.notna().sum()):4d}  "
              f"mean FD={fd.mean():.3f} ({fd.std():.3f}) mm  > 0.5 mm: {int((fd > 0.5).sum())}")


if __name__ == "__main__":
    main()
