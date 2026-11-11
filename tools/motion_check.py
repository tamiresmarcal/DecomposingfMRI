#!/usr/bin/env python3
"""Subject-level mean FD for one cohort, and who among the ANALYSED would fail.

    apptainer exec --bind /project,/scratch,/home --pwd $PWD $FMRIDECOMP_SIF45 \
        python3 tools/motion_check.py config/camcan.yaml

Why it exists: the censor policy reads `mean_fd` from participants_qc.csv, and
a rule whose column is entirely empty is reported and NOT applied. A cohort
whose participants_qc.csv lacks mean_fd was therefore never gated on it. This
measures the FD that rule would have used, so its effect can be judged before
deciding to re-run anything. Read-only on outputs/.

Two routes, in order:
  1. the pipeline's own `qc.motion_qc` (what `fmri-decomp diagnose` runs);
  2. if that finds nothing, every fMRIPrep `*desc-confounds_timeseries.tsv`
     under the cohort's derivatives_root, its `framewise_displacement` column
     averaged per (sub, task) over runs. The first frame (NaN) is skipped.

Writes <cohort>_motion.csv beside the call and prints, per task, how many of
the participants in the latents exceed the threshold.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd                                                  # noqa: E402

from fmri_decomposition.config import load_config                    # noqa: E402
from fmri_decomposition.io import default_output_root, latents_root  # noqa: E402


def via_pipeline(cfg) -> pd.DataFrame:
    from fmri_decomposition.extractions.qc import motion_qc
    rows, msg = motion_qc(cfg)
    print(f"[route 1] {msg}")
    if not rows:
        return pd.DataFrame()
    m = pd.DataFrame([dict(sub=str(k[0]), task=k[1], **v)
                      for k, v in rows.items()])
    if "fd_note" in m:
        notes = m["fd_note"].fillna("").value_counts().head(5)
        print(f"[route 1] notes: {notes.to_dict()}")
    return m[["sub", "task", "mean_fd"]] if "mean_fd" in m else pd.DataFrame()


def via_files(cfg) -> pd.DataFrame:
    root = Path(cfg.derivatives_root)
    pat = cfg.confounds.confounds_glob or "*desc-confounds_timeseries.tsv"
    files = sorted(root.rglob(pat))
    print(f"[route 2] {len(files)} confounds file(s) matching {pat!r} under {root}")
    col = cfg.confounds.fd_column or "framewise_displacement"
    rows = []
    for f in files:
        sub = re.search(r"sub-([A-Za-z0-9]+)", f.name)
        task = re.search(r"task-([A-Za-z0-9]+)", f.name)
        if not (sub and task):
            continue
        try:
            fd = pd.read_csv(f, sep="\t", usecols=[col])[col]
        except (ValueError, OSError) as e:
            print(f"[route 2] skip {f.name}: {e}")
            continue
        rows.append(dict(sub=sub.group(1), task=task.group(1),
                         fd_sum=fd.sum(skipna=True), fd_n=fd.notna().sum()))
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows).groupby(["sub", "task"], as_index=False)[["fd_sum", "fd_n"]].sum()
    d["mean_fd"] = d["fd_sum"] / d["fd_n"]
    return d[["sub", "task", "mean_fd"]]


def analysed(root: Path, cohort: str, atlas: str, window_s: str) -> pd.DataFrame:
    base = latents_root(root, atlas, window_s, cohort)
    out = []
    for f in sorted(base.glob("task=*/data.parquet")):
        task = f.parent.name.split("=", 1)[1]
        s = pd.read_parquet(f, columns=["sub"])["sub"].astype(str).unique()
        out.append(pd.DataFrame({"sub": s, "task": task}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["sub", "task"])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--atlas", default="harvardoxford")
    p.add_argument("--window-s", default="30")
    p.add_argument("--output-root", default=None)
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    m = via_pipeline(cfg)
    if m.empty or m["mean_fd"].notna().sum() == 0:
        m = via_files(cfg)
    if m.empty:
        raise SystemExit("no FD found by either route; see the messages above")
    m["sub"] = m["sub"].astype(str).str.replace(r"^sub-", "", regex=True)
    out = Path(f"{cfg.cohort}_motion.csv")
    m.to_csv(out, index=False)
    print(f"wrote {out} ({len(m)} rows, {m['mean_fd'].notna().sum()} with FD)")

    root = Path(a.output_root) if a.output_root else default_output_root()
    k = analysed(root, cfg.cohort, a.atlas, a.window_s)
    k["sub"] = k["sub"].str.replace(r"^sub-", "", regex=True)
    k = k.merge(m, on=["sub", "task"], how="left")
    print(f"\nanalysed participants (latents, {a.atlas}, {a.window_s}s):")
    for task, g in k.groupby("task"):
        fd = g["mean_fd"]
        print(f"  {task:8s} n={len(g):4d}  with FD={int(fd.notna().sum()):4d}  "
              f"mean FD={fd.mean():.3f} ({fd.std():.3f}) mm  "
              f"> {a.threshold} mm: {int((fd > a.threshold).sum())}")


if __name__ == "__main__":
    main()
