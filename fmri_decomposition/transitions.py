#!/usr/bin/env python3
"""Stage 5a -- per-subject brain-state transition matrices, one table per state set.

    fmri-decomp transitions --check
    fmri-decomp transitions --window-s 15 30 60 120 300

writes, per (atlas, window_s, state definition, cohort),

    outputs/transitions/atlas=<a>/window_s=<w>/states=<def>/cohort=<c>/pairs.parquet
    outputs/transitions/atlas=<a>/window_s=<w>/states=<def>/cohort=<c>/summary.parquet
    outputs/meta/transitions/manifest.json

WHAT A TRANSITION IS HERE
-------------------------
A latents file holds one row per window, in time order within (task, sub), each
carrying a discrete state label (`ThresholdCluster_pca3_8` and `_27`). A
transition is one neighbouring pair in that sequence, so N windows give N-1
transitions. `pairs.parquet` has one row per (subject, from_state, to_state)
that was actually observed -- long, not wide, because a wide table is K*K
columns and most of them are zero.

THE DIAGONAL IS KEPT
--------------------
`i -> i` is a transition like any other and stays in the table. Windows overlap
by `1 - 1/n_overlaps` (80% at every window size, since stride = window_s /
n_overlaps), so the diagonal is inflated by the sliding window itself. That
inflation is the same factor at every aperture and near-constant across
subjects, so it does not bias the between-subject contrast a phenotype model
reads -- it is a near-constant offset, not a differential confound.

WHAT THE OVERLAP DOES AND DOES NOT COST
---------------------------------------
Windows are NOT thinned to a non-overlapping subset: keeping all of them gives
74 transitions per camcan subject at 30 s instead of 14, and the correlation
between neighbouring windows costs power, not validity. The unit of observation
in any across-subject test is the SUBJECT, and subjects are independent --
within-subject dependence adds measurement error to each subject's p, which
attenuates an association toward zero rather than inflating type I error.

`n_transitions_independent` travels with every row anyway: it is the count a
non-overlapping grid would have given, so the ~5x inflation is a number in the
table rather than something to remember.

TWO NORMALISATIONS
------------------
`p_cond` is the Markov transition probability, n(i->j) / n(i->.), which is what
"transition probability" usually means. `p_joint` is n(i->j) / n_transitions.

Prefer `p_joint` for across-subject modelling. Its denominator is the same for
every subject (one fixed-length clip), so subjects are comparable; `p_cond`
divides by a per-subject, per-row count that is often 1 or 2, which both makes
the value jump to 1.0 on a single observation and makes two subjects' values
mean different things.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .io import latents_root, meta_dir

# The state-label columns this stage reads. K = 8 and 27 only: at 125 and 512
# cells a subject's 74 transitions leave >99% of the matrix at exactly zero, so
# there is no probability to correlate with anything.
STATE_COLUMNS = ["ThresholdCluster_pca3_8", "ThresholdCluster_pca3_27"]

# Identity and ordering. `window_id` is the time order within (task, sub) and
# is globally meaningful within a movie -- see windows.make_stimulus_grid.
IDENT = ["cohort", "task", "sub", "window_id"]
COORDS = ["pca0/3", "pca1/3", "pca2/3"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def latents_path(root: Path, atlas: str, window_s, cohort: str) -> Path:
    return latents_root(root, atlas, window_s, cohort) / "data.parquet"


# ------------------------------------------------------------- pre-flight ---
def _schema(path: Path):
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).schema_arrow


def _meta_value(path: Path, key: str):
    md = _schema(path).metadata or {}
    raw = md.get(key.encode())
    if raw is None:
        return None
    try:
        return json.loads(raw.decode())
    except Exception:                                            # noqa: BLE001
        return raw.decode()


def check_grid(root: Path, atlases, windows, cohorts, states) -> pd.DataFrame:
    """One row per grid cell, with `ok` and the reason it is not.

    Checked before anything is written, because the alternative is discovering a
    gap after a cohort has been processed -- which is how stage 4 left window
    sizes half-done.

    The `model_hash` check is the one that would otherwise fail silently. State
    `5` is only the same state in two cohorts if both came from the same fit;
    if camcan's latents were written by a different `decompose` run than
    ds002837's, every transition compared across them is meaningless and
    nothing downstream would notice.
    """
    rows = []
    for atlas in atlases:
        for w in windows:
            present = {}
            for cohort in cohorts:
                p = latents_path(root, atlas, w, cohort)
                rec = {"atlas": atlas, "window_s": w, "cohort": cohort,
                       "path": str(p)}
                if not p.exists():
                    # One row per STATE even when the file is missing, so the
                    # denominator in the report equals the grid size. Emitting a
                    # single `states="*"` row here made "48/69" print under a
                    # header that said 90 cells.
                    for st in states:
                        rows.append({**rec, "states": st, "ok": False,
                                     "reason": "no latents file"})
                    continue
                names = set(_schema(p).names)
                present[cohort] = {
                    "model_hash": _meta_value(p, "model_hash"),
                    "censor_policy": _meta_value(p, "censor_policy"),
                    "umap_fitted": _meta_value(p, "umap_fitted"),
                    "names": names,
                }
                import pyarrow.parquet as pq
                n_rows = pq.ParquetFile(p).metadata.num_rows
                for st in states:
                    reason = ""
                    if st not in names:
                        reason = f"column {st} absent (was --bins run for it?)"
                    rows.append({**rec, "states": st, "n_rows": n_rows,
                                 "model_hash": present[cohort]["model_hash"],
                                 "censor_policy": present[cohort]["censor_policy"],
                                 "ok": not reason, "reason": reason})

            # Cross-cohort consistency, within this (atlas, window_s).
            hashes = {c: v["model_hash"] for c, v in present.items()}
            if len(set(hashes.values())) > 1:
                for r in rows:
                    if r["atlas"] == atlas and r["window_s"] == w and r["ok"]:
                        r["ok"] = False
                        r["reason"] = (f"model_hash differs across cohorts "
                                       f"{hashes} -- state labels are not "
                                       f"comparable; re-run decompose for all "
                                       f"cohorts together")
    return pd.DataFrame(rows)


def report_check(df: pd.DataFrame) -> int:
    """Readable at 90 rows: state sets that are ready, then gaps by reason.

    A row-per-cell table is unreadable once the grid is a few atlases wide, and
    the thing you need from it is two lists -- what can run, and what to fix.
    """
    if df.empty:
        print("nothing in the grid at all -- check --atlas / --window-s")
        return 1
    ok, total = int(df["ok"].sum()), len(df)
    short = df.assign(K=df["states"].str.replace("ThresholdCluster_pca3_", "",
                                                 regex=False))
    print(f"\ngrid: {ok}/{total} cell(s) ready\n")

    ready = short[short["ok"]]
    if len(ready):
        tab = (ready.groupby(["atlas", "window_s", "K"])
                    .agg(cohorts=("cohort", "nunique"),
                         rows=("n_rows", "sum"),
                         policy=("censor_policy", "first"))
                    .reset_index())
        tab["window_s"] = tab["window_s"].astype(float)
        print("READY")
        print(tab.sort_values(["atlas", "window_s", "K"])
                 .to_string(index=False))

    gaps = short[~short["ok"]]
    if len(gaps):
        print("\nNOT READY")
        for reason, g in gaps.groupby("reason"):
            where = (g.assign(w=g["window_s"].astype(float))
                      .groupby("atlas")["w"]
                      .apply(lambda s: ", ".join(f"{x:g}s" for x in sorted(set(s)))))
            print(f"  {len(g)} cell(s): {reason}")
            for atlas, windows in where.items():
                print(f"      {atlas:<16} {windows}")

    policies = {p for p in df.get("censor_policy", pd.Series(dtype=object)).dropna()}
    if len(policies) > 1:
        print(f"\nWARNING: more than one censor policy in the grid: {policies}.")
        print("         One state set censored and another not is not a fair "
              "comparison.")
    if ok < total:
        print(f"\n{total - ok} cell(s) are not ready. Fix them, or pass "
              f"--allow-partial to run the rest.")
    return 0 if ok == total else 1


# --------------------------------------------------------------- compute ---
def _runs(labels: np.ndarray) -> np.ndarray:
    """Run lengths of consecutive identical labels -- dwell, in windows."""
    if len(labels) == 0:
        return np.array([], dtype=int)
    change = np.flatnonzero(np.diff(labels) != 0) + 1
    return np.diff(np.concatenate([[0], change, [len(labels)]]))


def _entropy_rate(counts: np.ndarray) -> float:
    """Plug-in entropy rate of the subject's own matrix, in bits.

    Biased downward at these counts -- 74 transitions over 64 cells cannot
    estimate 64 probabilities -- and reported as a comparative feature only,
    never as an absolute quantity. A shrinkage estimator is the fix if it ever
    carries a claim.
    """
    n = counts.sum()
    if n == 0:
        return np.nan
    row = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(row > 0, counts / row, 0.0)
        term = np.where(p > 0, p * np.log2(p), 0.0)
    pi = (row / n).ravel()
    return float(-(pi * term.sum(axis=1)).sum())


def subject_transitions(g: pd.DataFrame, state_col: str, n_states: int,
                        stride_s: float, indep_factor: int):
    """One subject x task -> (long pair rows, one summary row).

    `g` must already be sorted by window_id. A run boundary is not a
    transition, so pairs that straddle one are dropped -- `crosses_run_boundary`
    marks the window, and consecutive windows from different runs are not
    neighbours in time.
    """
    # A window that straddles a run boundary was computed across a
    # discontinuity, so its own label is suspect -- it is dropped outright
    # rather than merely excluded from transitions. Dropping leaves GAPS in
    # window_id, and two windows either side of a gap are not neighbours in
    # time, so the sequence is split into contiguous segments and every
    # sequential quantity -- transitions and dwell alike -- is computed within a
    # segment. Doing this for transitions but not for dwell was the bug this
    # replaces: the two would have been measured over different data.
    if "crosses_run_boundary" in g.columns:
        g = g.loc[~g["crosses_run_boundary"].to_numpy(dtype=bool)]
    if g.empty:
        return (pd.DataFrame(columns=["from_state", "to_state", "n", "transition",
                                      "p_cond", "p_joint", "n_transitions"]),
                _empty_summary(n_states))

    lab = g[state_col].to_numpy(dtype=np.int64, copy=True)
    wid = g["window_id"].to_numpy(dtype=np.int64, copy=True)
    # segment boundaries: a jump of more than one window_id
    seg_start = np.concatenate([[0], np.flatnonzero(np.diff(wid) != 1) + 1,
                                [len(wid)]])
    segments = [lab[seg_start[k]:seg_start[k + 1]]
                for k in range(len(seg_start) - 1)]

    counts = np.zeros((n_states, n_states), dtype=np.int64)
    runs_all = []
    for seg in segments:
        if len(seg) > 1:
            np.add.at(counts, (seg[:-1], seg[1:]), 1)
        runs_all.append(_runs(seg))
    n_tr = int(counts.sum())

    nz = np.argwhere(counts > 0)
    row_tot = counts.sum(axis=1)
    pairs = pd.DataFrame({
        "from_state": nz[:, 0],
        "to_state": nz[:, 1],
        "n": counts[nz[:, 0], nz[:, 1]],
    })
    pairs["transition"] = [f"{i}->{j}" for i, j in nz]
    pairs["p_cond"] = pairs["n"] / row_tot[pairs["from_state"]]
    pairs["p_joint"] = pairs["n"] / n_tr if n_tr else np.nan
    pairs["n_transitions"] = n_tr

    runs = np.concatenate(runs_all) if runs_all else np.array([], dtype=int)
    n_change = n_tr - int(np.trace(counts))
    occupancy = np.bincount(lab, minlength=n_states) / max(len(lab), 1)
    xyz = g[COORDS].to_numpy(dtype=float, copy=True) if all(
        c in g.columns for c in COORDS) else None

    summary = {
        "n_windows": len(lab),
        "n_transitions": n_tr,
        # What a non-overlapping grid would have given: one window per stride
        # step instead of `n_overlaps` of them. The ratio against
        # `n_transitions` is the effective-information penalty for keeping the
        # overlaps, so the ~5x inflation is a number in the table.
        "n_transitions_independent": max(len(lab) // indep_factor - 1, 0),
        "n_states_visited": int((occupancy > 0).sum()),
        "n_distinct_transitions": int(len(nz)),
        "switch_rate": (n_change / n_tr) if n_tr else np.nan,
        "self_transition_rate": (1 - n_change / n_tr) if n_tr else np.nan,
        "switches_per_min": (n_change / (len(lab) * stride_s / 60)
                             if len(lab) else np.nan),
        "mean_dwell_s": float(runs.mean() * stride_s) if len(runs) else np.nan,
        "entropy_rate_bits": _entropy_rate(counts),
        "dispersion": (float(np.linalg.norm(xyz - xyz.mean(0), axis=1).mean())
                       if xyz is not None and len(xyz) else np.nan),
    }
    summary.update({f"occ_{k}": float(occupancy[k]) for k in range(n_states)})
    return pairs, summary


def _empty_summary(n_states: int) -> dict:
    """Every column present, all NaN -- a subject whose every window crossed a
    run boundary must still appear, or a cohort's n would silently shrink."""
    d = {k: np.nan for k in
         ("n_windows", "n_transitions", "n_transitions_independent",
          "n_states_visited", "n_distinct_transitions", "switch_rate",
          "self_transition_rate", "switches_per_min", "mean_dwell_s",
          "entropy_rate_bits", "dispersion")}
    d.update({f"occ_{k}": np.nan for k in range(n_states)})
    return d


def n_states_for(state_col: str) -> int:
    """8 or 27, from the column name -- the grid is K**3 cells by construction."""
    return int(state_col.rsplit("_", 1)[1])


def process(root: Path, atlas: str, window_s, cohort: str, state_col: str,
            n_overlaps: int, overwrite: bool) -> dict:
    out_dir = (root / "transitions" / f"atlas={atlas}"
               / f"window_s={window_s}" / f"states={state_col}"
               / f"cohort={cohort}")
    pairs_path, summary_path = out_dir / "pairs.parquet", out_dir / "summary.parquet"
    if not overwrite and pairs_path.exists() and summary_path.exists():
        return {"status": "skipped", "cohort": cohort}

    src = latents_path(root, atlas, window_s, cohort)
    cols = IDENT + [state_col] + [c for c in COORDS] + ["crosses_run_boundary"]
    have = set(_schema(src).names)
    df = pd.read_parquet(src, columns=[c for c in cols if c in have])

    K = n_states_for(state_col)
    stride_s = float(window_s) / n_overlaps
    model_hash = _meta_value(src, "model_hash")
    policy = _meta_value(src, "censor_policy")

    all_pairs, all_summary = [], []
    for (task, sub), g in df.sort_values("window_id").groupby(["task", "sub"],
                                                             sort=True):
        pairs, summary = subject_transitions(g, state_col, K, stride_s, n_overlaps)
        if len(pairs):
            all_pairs.append(pairs.assign(task=task, sub=sub))
        all_summary.append({"task": task, "sub": sub, **summary})

    prov = {"cohort": cohort, "atlas": atlas, "window_s": str(window_s),
            "states": state_col, "n_states": K, "model_hash": model_hash,
            "censor_policy": policy, "n_overlaps": n_overlaps}

    pairs_df = (pd.concat(all_pairs, ignore_index=True) if all_pairs
                else pd.DataFrame(columns=["from_state", "to_state", "n",
                                           "transition", "p_cond", "p_joint",
                                           "n_transitions", "task", "sub"]))
    summary_df = pd.DataFrame(all_summary)
    for k, v in prov.items():
        pairs_df[k] = v
        summary_df[k] = v

    out_dir.mkdir(parents=True, exist_ok=True)
    _write(pairs_df, pairs_path)
    _write(summary_df, summary_path)
    return {"status": "ok", "cohort": cohort, "subjects": len(summary_df),
            "pair_rows": len(pairs_df),
            "median_transitions": float(summary_df["n_transitions"].median())
            if len(summary_df) else np.nan,
            **prov}


def _write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        df.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


# ------------------------------------------------------------------- run ---
def _default_root() -> Path:
    if os.environ.get("FMRIDECOMP_OUTPUTS"):
        return Path(os.environ["FMRIDECOMP_OUTPUTS"])
    import yaml

    repo = Path(__file__).resolve().parent.parent
    return Path(yaml.safe_load(
        (repo / "config" / "camcan_movie.yaml").read_text())["output_root"])


def _discover_cohorts(root: Path, atlases, windows) -> list[str]:
    found = set()
    for atlas in atlases:
        for w in windows:
            d = latents_root(root, atlas, w)
            if d.is_dir():
                found |= {p.name.split("=", 1)[1] for p in d.glob("cohort=*")}
    return sorted(found)


def run(args) -> int:
    root = Path(args.output_root) if args.output_root else _default_root()
    if not root.is_dir():
        raise SystemExit(f"output_root does not exist: {root}")
    log(f"output_root {root}")

    windows = [str(w) for w in args.window_s]
    atlases = list(args.atlas)
    states = list(args.states)
    cohorts = args.cohorts or _discover_cohorts(root, atlases, windows)
    if not cohorts:
        raise SystemExit("no cohort has latents under any grid cell -- run "
                         "`fmri-decomp decompose` first")
    log(f"grid: {len(atlases)} atlas x {len(windows)} window x {len(states)} "
        f"state def x {len(cohorts)} cohort(s) = "
        f"{len(atlases) * len(windows) * len(states) * len(cohorts)} cell(s)")

    check = check_grid(root, atlases, windows, cohorts, states)
    rc = report_check(check)
    if args.check:
        return rc
    if rc and not args.allow_partial:
        raise SystemExit("\nrefusing to start on an incomplete grid; see above, "
                         "or pass --allow-partial")

    ready = check[check["ok"]]
    entries = []
    for (atlas, w, st), grp in ready.groupby(["atlas", "window_s", "states"],
                                             sort=True):
        log(f"atlas={atlas} window_s={w} states={st}")
        for cohort in sorted(grp["cohort"].unique()):
            e = process(root, atlas, w, cohort, st, args.n_overlaps,
                        args.overwrite)
            entries.append(e)
            if e["status"] == "skipped":
                log(f"    {cohort:<14} already written, skipped")
            else:
                log(f"    {cohort:<14} {e['subjects']:>4} subject(s), "
                    f"{e['pair_rows']:>7,} pair row(s), "
                    f"median {e['median_transitions']:.0f} transitions each")

    out = meta_dir(root) / "transitions" / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"entries": entries, "grid": {"atlases": atlases, "windows": windows,
                                      "states": states, "cohorts": cohorts},
         "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        indent=2, default=str))
    log(f"manifest -> {out.relative_to(root)}")
    return 0


def add_arguments(p) -> None:
    p.add_argument("--atlas", nargs="+",
                   default=["harvardoxford", "yeo7", "networks"])
    p.add_argument("--window-s", nargs="+",
                   default=["15", "30", "60", "120", "300"])
    p.add_argument("--states", nargs="+", default=STATE_COLUMNS,
                   help="state-label columns; K=125 and 512 are excluded on "
                        "purpose (>99%% of their cells are zero per subject)")
    p.add_argument("--cohorts", nargs="*", default=None)
    p.add_argument("--n-overlaps", type=int, default=5,
                   help="windows.n_overlaps the shards were written with; sets "
                        "the stride used for dwell times in seconds")
    p.add_argument("--check", action="store_true",
                   help="report grid readiness and write nothing")
    p.add_argument("--allow-partial", action="store_true",
                   help="process the ready cells instead of refusing")
    p.add_argument("--output-root")
    p.add_argument("--overwrite", action="store_true")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
