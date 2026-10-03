#!/usr/bin/env python3
"""Stage 5a -- per-subject brain-state transition matrices, one table per state set.

    fmri-decomp transitions --check
    fmri-decomp transitions --window-s 15 30 60 120 300

writes, per (atlas, window_s, state definition, cohort),

    outputs/transitions/atlas=<a>/window_s=<w>/states=<def>/cohort=<c>/subjects.parquet
    outputs/meta/transitions/manifest.json

ONE ROW PER SUBJECT, ONE TABLE
------------------------------
    cohort, task, sub | 0->0, 0->1, ... (every K*K cell) | switch_rate, ... | provenance

Everything a model reads is at subject level, so it is one table rather than a
long pair table plus a summary to join. An earlier version stored the cells long
because K=512 would have been 262,144 columns; with K capped at 27 that reason
is gone -- 729 columns is nothing, and unobserved cells are exact zeros, which
parquet stores for almost free.

WHAT A TRANSITION IS HERE
-------------------------
A latents file holds one row per window, in time order within (task, sub), each
carrying a discrete state label, written by stage 4b (`cluster`). A
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

WHICH PROBABILITY IS IN THE CELLS
---------------------------------
`p_joint` -- n(i->j) divided by all of that subject's transitions. The whole
table sums to 1 per subject.

The alternative is `p_cond`, n(i->j) / n(i->.), which is what "transition
probability" usually means and makes each ROW sum to 1. It is the wrong view for
an across-subject model: its denominator is a per-subject, per-row count often
equal to 1, so a single observation gives 1.0, and `0.5` means "1 of 2" for one
subject and "37 of 74" for another.

Nothing is lost by choosing. `count = p_joint * n_transitions` exactly, and
`p_cond` is a row-wise renormalisation of those counts, so both are one line
away from the table as written.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .io import STATE_K_BAND, latents_root, meta_dir

# The state-label columns this stage reads. K = 8 and 27 only: at 125 and 512
# cells a subject's 74 transitions leave >99% of the matrix at exactly zero, so
# there is no probability to correlate with anything.
# No fixed list. `--states` defaults to every state column found in the
# schema, so a method added by stage 4b is picked up without being named here.
# The convention is `<Method>_<embedding>_<K>`; anything matching it qualifies.
STATE_SUFFIX_RE = r"^[A-Za-z]+_[a-z]+\d*_\d+$"

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


def cell_states(root: Path, atlas: str, w, cohorts, min_k: int, max_k: int):
    """The state sets usable in ONE (atlas, aperture) cell.

    PER CELL, not a union across the grid, and that distinction is the whole
    point. A union works while every state set has a fixed name -- threshold and
    HMM are `_8` and `_27` everywhere -- and breaks the moment a method DISCOVERS
    its K: `MeanShift_pca3_16` exists at harvardoxford/30s and nowhere else,
    because 30s harvardoxford is where the bandwidth search landed on 16. Demanded
    across the grid it is missing from 14 cells, and stage 5a refused to start on
    an "incomplete" grid that was never incomplete.

    Returns (usable, partial, skipped):
      usable   present in EVERY cohort of this cell, K inside the band. A state
               label is comparable across cohorts only if one fit defined it, so
               a set missing from one cohort is not usable in any of them.
      partial  per cohort, the sets it has that its siblings do not -- stage 4b
               ran unevenly, which is worth saying rather than silently dropping.
      skipped  shared but outside the K band.
    """
    per, first = {}, None
    for c in cohorts:
        p = latents_path(root, atlas, w, c)
        if p.exists():
            per[c] = set(discover_state_columns(p))
            first = first or p
    if not per:
        return [], {}, []

    shared = set.intersection(*per.values())
    partial = {c: sorted(v - shared) for c, v in per.items() if v - shared}
    k_of = {st: _k_quietly(st, first) for st in shared}
    usable, skipped = _within_k_band(sorted(shared), k_of, min_k, max_k)
    return usable, partial, skipped


def check_grid(root: Path, atlases, windows, cohorts, states, min_k: int,
               max_k: int) -> pd.DataFrame:
    """One row per (atlas, aperture, state set, cohort), with `ok` and why not.

    Checked before anything is written, because the alternative is discovering a
    gap after a cohort has been processed -- which is how stage 4 left window
    sizes half-done.

    The `model_hash` check is the one that would otherwise fail silently. State
    `5` is only the same state in two cohorts if both came from the same fit; if
    camcan's latents were written by a different `decompose` run than
    ds002837's, every transition compared across them is meaningless and nothing
    downstream would notice.

    `states` names the sets explicitly and applies to every cell; None discovers
    them per cell via `cell_states`.
    """
    import pyarrow.parquet as pq

    rows = []
    for atlas in atlases:
        for w in windows:
            if states:
                here, partial, skipped = list(states), {}, []
            else:
                here, partial, skipped = cell_states(root, atlas, w, cohorts,
                                                     min_k, max_k)
            base = {"atlas": atlas, "window_s": w}

            present = {}
            for cohort in cohorts:
                p = latents_path(root, atlas, w, cohort)
                rec = {**base, "cohort": cohort, "path": str(p)}
                if not p.exists():
                    # Reported, not fatal: an aperture stage 3 never ran is a
                    # gap in the plan, not a corruption of what exists. The
                    # default --window-s spans more apertures than any one tree
                    # has, and aborting on that made the command unusable.
                    rows.append({**rec, "states": "(no latents)", "ok": False,
                                 "reason": "no latents file", "n_rows": 0,
                                 "fatal": False})
                    continue
                names = set(_schema(p).names)
                present[cohort] = _meta_value(p, "model_hash")
                n_rows = pq.ParquetFile(p).metadata.num_rows
                for st in here:
                    reason = ("" if st in names else
                              f"column {st} absent (run `fmri-decomp cluster` "
                              f"for this atlas and aperture)")
                    rows.append({**rec, "states": st, "n_rows": n_rows,
                                 "model_hash": present[cohort],
                                 "censor_policy": _meta_value(p, "censor_policy"),
                                 "ok": not reason, "reason": reason,
                                 # Only when NAMED does an absent column stop the
                                 # run: the user asked for it by hand. A set that
                                 # discovery simply did not find in this cell is
                                 # not a fault of the cell.
                                 "fatal": bool(reason) and bool(states)})
                for st in partial.get(cohort, []):
                    rows.append({**rec, "states": st, "n_rows": n_rows,
                                 "model_hash": present[cohort],
                                 "ok": False,
                                 "fatal": False,
                                 "reason": f"{st} is in {cohort} but not in "
                                           f"every cohort of this cell -- stage "
                                           f"4b ran unevenly here"})
                for st in skipped:
                    rows.append({**rec, "states": st, "n_rows": n_rows,
                                 "ok": False, "fatal": False,
                                 "reason": f"K outside [{min_k}, {max_k}]"})

            if len(set(present.values())) > 1:
                for r in rows:
                    if r["atlas"] == atlas and r["window_s"] == w and r["ok"]:
                        r["ok"] = False
                        # THIS one stops everything. Two cohorts from different
                        # fits cannot be pooled and nothing downstream notices.
                        r["fatal"] = True
                        r["reason"] = (f"model_hash differs across cohorts "
                                       f"{present} -- state labels are not "
                                       f"comparable; re-run decompose for all "
                                       f"cohorts together")
    return pd.DataFrame(rows)


def _shorten(states: pd.Series) -> pd.Series:
    """Drop a prefix every state set shares, and only then.

    This used to strip the literal `ThresholdCluster_pca3_` from every name,
    which was fine while that was the only family. With several it was actively
    misleading: `ThresholdCluster_pca3_8` printed as `8` in a table that also
    listed `MeanShift_umap3_8` in full, so two different state sets were one
    column apart and indistinguishable. A prefix is only hidden when hiding it
    cannot merge two names.
    """
    uniq = sorted(set(states))
    if len(uniq) < 2:
        return states
    pre = os.path.commonprefix(uniq)
    pre = pre[:pre.rfind("_") + 1]          # whole underscore-separated tokens
    return states.str.slice(len(pre)) if pre else states


def report_check(df: pd.DataFrame) -> int:
    """Readable at 90 rows: state sets that are ready, then gaps by reason.

    A row-per-cell table is unreadable once the grid is a few atlases wide, and
    the thing you need from it is two lists -- what can run, and what to fix.
    """
    if df.empty:
        print("nothing in the grid at all -- check --atlas / --window-s")
        return 1
    ok, total = int(df["ok"].sum()), len(df)
    short = df.assign(states=_shorten(df["states"]))
    print(f"\ngrid: {ok}/{total} cell(s) ready\n")

    ready = short[short["ok"]]
    if len(ready):
        tab = (ready.groupby(["atlas", "window_s", "states"])
                    .agg(cohorts=("cohort", "nunique"),
                         rows=("n_rows", "sum"),
                         policy=("censor_policy", "first"))
                    .reset_index())
        tab["window_s"] = tab["window_s"].astype(float)
        print("READY")
        print(tab.sort_values(["atlas", "window_s", "states"])
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
    n_fatal = int(df.get("fatal", pd.Series(dtype=bool)).fillna(False).sum())
    if ok < total:
        print(f"\n{total - ok} cell(s) are not ready, {n_fatal} of them "
              f"INCONSISTENT rather than merely absent.")
        if not n_fatal:
            print("         None of these stops the run: a state set absent "
                  "from a cell, or an aperture stage 3 never ran, is a gap in "
                  "the plan and not a fault in what exists.")
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


def cell_names(n_states: int) -> list[str]:
    """`0->0`, `0->1`, ... every cell, in (from, to) order.

    All K*K are emitted even when a subject never made that transition, so the
    schema is identical across subjects and cohorts -- which is what a model
    reading the table needs.
    """
    return [f"{i}->{j}" for i in range(n_states) for j in range(n_states)]


def subject_transitions(g: pd.DataFrame, state_col: str, n_states: int,
                        stride_s: float, indep_factor: int) -> dict:
    """One subject x task -> ONE row: every cell, then the features.

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
        return _empty_row(n_states)

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

    runs = np.concatenate(runs_all) if runs_all else np.array([], dtype=int)
    n_change = n_tr - int(np.trace(counts))
    n_distinct = int((counts > 0).sum())
    occupancy = np.bincount(lab, minlength=n_states) / max(len(lab), 1)
    xyz = g[COORDS].to_numpy(dtype=float, copy=True) if all(
        c in g.columns for c in COORDS) else None

    # The cells, as p_joint: n(i->j) / all of this subject's transitions.
    #
    # One value per cell and still lossless: count = p_joint * n_transitions
    # exactly, and p_cond (the Markov probability, n(i->j) / n(i->.)) is a
    # row-wise renormalisation of those counts. Storing p_joint beside
    # n_transitions therefore throws nothing away, and it is the view a model
    # should read -- its denominator is the same for every subject, while
    # p_cond divides by a per-row count that is often 1, so `0.5` can mean
    # "1 of 2" for one subject and "37 of 74" for another.
    flat = (counts.ravel() / n_tr) if n_tr else np.full(n_states ** 2, np.nan)
    row = dict(zip(cell_names(n_states), flat.astype(float)))

    row.update({
        "n_windows": len(lab),
        "n_transitions": n_tr,
        # What a non-overlapping grid would have given: one window per stride
        # step instead of `n_overlaps` of them. The ratio against
        # `n_transitions` is the effective-information penalty for keeping the
        # overlaps, so the ~5x inflation is a number in the table.
        "n_transitions_independent": max(len(lab) // indep_factor - 1, 0),
        "n_states_visited": int((occupancy > 0).sum()),
        "n_distinct_transitions": n_distinct,
        "switch_rate": (n_change / n_tr) if n_tr else np.nan,
        "self_transition_rate": (1 - n_change / n_tr) if n_tr else np.nan,
        "switches_per_min": (n_change / (len(lab) * stride_s / 60)
                             if len(lab) else np.nan),
        "mean_dwell_s": float(runs.mean() * stride_s) if len(runs) else np.nan,
        "entropy_rate_bits": _entropy_rate(counts),
        "dispersion": (float(np.linalg.norm(xyz - xyz.mean(0), axis=1).mean())
                       if xyz is not None and len(xyz) else np.nan),
    })
    row.update({f"occ_{k}": float(occupancy[k]) for k in range(n_states)})
    return row


def _empty_row(n_states: int) -> dict:
    """Every column present, all NaN -- a subject whose every window crossed a
    run boundary must still appear, or a cohort's n would silently shrink."""
    d = {c: np.nan for c in cell_names(n_states)}
    d.update({k: np.nan for k in
              ("n_windows", "n_transitions", "n_transitions_independent",
               "n_states_visited", "n_distinct_transitions", "switch_rate",
               "self_transition_rate", "switches_per_min", "mean_dwell_s",
               "entropy_rate_bits", "dispersion")})
    d.update({f"occ_{k}": np.nan for k in range(n_states)})
    return d


def discover_state_columns(path: Path) -> list[str]:
    """State columns in one latents file, from the schema.

    Read rather than listed, so a clusterer added by stage 4b needs no edit
    here. `<Method>_<embedding>_<K>` is the convention stage 4b writes.
    """
    import re

    import pyarrow.parquet as pq

    names = pq.ParquetFile(path).schema_arrow.names
    skip = {"n_states", "n_transitions", "n_transitions_independent"}
    return sorted(n for n in names
                  if n not in skip and "->" not in n
                  and re.match(STATE_SUFFIX_RE, n))


def n_states_for(state_col: str, path: Path | None = None,
                 labels=None) -> int:
    """How many states this state SET has -- never how many a cohort used.

    Resolution order, and the order matters:

      1. the `clusterers` provenance stage 4b wrote into the latents schema,
      2. the `<Method>_<embedding>_<K>` column name,
      3. only then `max(label) + 1`.

    Deriving it from the labels alone is wrong and was a real bug: a cohort that
    happens to visit 6 of 8 states would get a 6x6 matrix while another gets
    8x8, so the same state set would have different columns per cohort and
    nothing downstream could pool them. K is a property of the fit, which is
    why the fit records it.
    """
    if path is not None:
        md = _schema(path).metadata or {}
        raw = md.get(b"clusterers")
        if raw:
            entry = json.loads(raw.decode()).get(state_col)
            if entry and entry.get("k"):
                return int(entry["k"])
    tail = state_col.rsplit("_", 1)[1]
    if tail.isdigit():
        return int(tail)
    if labels is not None and len(labels):
        print(f"  WARNING: {state_col} has no recorded K and none in its name; "
              f"falling back to max(label)+1, which can differ per cohort")
        return int(np.max(labels)) + 1
    raise SystemExit(f"cannot determine the number of states for {state_col!r}")


def time_axis(path: Path, window_s, n_overlaps: int) -> tuple[float, int]:
    """(seconds between consecutive rows, rows per independent sample).

    Preferred from the latents schema, where stage 4 records it per cohort, and
    only then derived as `window_s / n_overlaps`. Deriving it is right for a
    sliding window and wrong for anything else: at `window_s = -1` the rows are
    single TRs, so the derivation gives a NEGATIVE stride and every dwell time
    and switch rate computed from it comes out negative. The TR also differs per
    cohort, which is why it is a per-cohort metadata field and not something
    this stage could compute from the path.
    """
    stride = _meta_value(path, "stride_s")
    indep = _meta_value(path, "indep_factor")
    if stride is not None:
        return float(stride), int(indep) if indep is not None else 1
    w = float(window_s)
    if w <= 0:
        raise SystemExit(
            f"{path} has window_s={window_s} and no `stride_s` in its schema "
            f"metadata, so the time between consecutive rows is unknown. It was "
            f"written by a stage 4 that predates the activation source; re-run "
            f"`fmri-decomp decompose --source activation` for it.")
    return w / n_overlaps, n_overlaps


def _k_quietly(state_col: str, path: Path) -> int | None:
    """K for a state column, or None if it cannot be resolved without guessing.

    Deliberately does not read the labels: this runs over every file in the grid
    just to size the columns, and `n_states_for`'s last-resort path both reads
    data and prints a warning that would repeat once per cohort.
    """
    try:
        return n_states_for(state_col, path=path)
    except SystemExit:
        return None


def _within_k_band(states: list[str], k_of: dict, min_k: int, max_k: int):
    """(states to use, states skipped for a K outside the band).

    Both ends matter, for opposite reasons. Too wide (125, 512) is a table with
    more columns than a subject has transitions. Too narrow is worse because it
    looks fine: at K=1 every subject's table is the single cell `0->0` = 1.0, a
    column of a constant that costs fits and dilutes an FDR family. Stage 4b
    refuses to write those now, but latents written before it did still carry
    them -- `MeanShift_pca3_1` and friends -- and discovery would pick them up.

    A state set with no resolvable K is KEPT: `process` resolves it per cell with
    the labels in hand and will say so. Only a KNOWN, out-of-band K is dropped.
    """
    keep, skip = [], []
    for st in states:
        k = k_of.get(st)
        (skip if (k is not None and not min_k <= k <= max_k) else keep).append(st)
    return keep, skip


def process(root: Path, atlas: str, window_s, cohort: str, state_col: str,
            n_overlaps: int, overwrite: bool) -> dict:
    out_dir = (root / "transitions" / f"atlas={atlas}"
               / f"window_s={window_s}" / f"states={state_col}"
               / f"cohort={cohort}")
    out_path = out_dir / "subjects.parquet"
    if not overwrite and out_path.exists():
        return {"status": "skipped", "cohort": cohort}

    src = latents_path(root, atlas, window_s, cohort)
    cols = IDENT + [state_col] + [c for c in COORDS] + ["crosses_run_boundary"]
    # `cols` includes `cohort`, which is ALSO the name of a directory key on this
    # path, so pd.read_parquet's dataset layer would try to merge a string column
    # with an inferred dictionary partition field and raise on pyarrow 18.
    # io.read_file opens the one file and infers nothing. It also drops columns
    # the file does not have, which is what the old comprehension did.
    from .io import read_file

    df = read_file(src, cols).to_pandas()

    K = n_states_for(state_col, path=src, labels=df[state_col].to_numpy())
    stride_s, indep = time_axis(src, window_s, n_overlaps)
    model_hash = _meta_value(src, "model_hash")
    policy = _meta_value(src, "censor_policy")

    rows = [{"task": task, "sub": sub,
             **subject_transitions(g, state_col, K, stride_s, indep)}
            for (task, sub), g in df.sort_values("window_id")
                                    .groupby(["task", "sub"], sort=True)]

    prov = {"cohort": cohort, "atlas": atlas, "window_s": str(window_s),
            "states": state_col, "n_states": K, "model_hash": model_hash,
            "censor_policy": policy, "n_overlaps": n_overlaps,
            # What was actually used, which is not n_overlaps at every aperture.
            "stride_s": stride_s, "indep_factor": indep}

    # Column order is the contract: keys, every cell, the features, provenance.
    cells = cell_names(K)
    feats = [c for c in rows[0] if c not in set(cells) | {"task", "sub"}] if rows else []
    out = pd.DataFrame(rows, columns=["task", "sub"] + cells + feats)
    for k, v in prov.items():
        out[k] = v

    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out, out_path)
    return {"status": "ok", "cohort": cohort, "subjects": len(out),
            "n_cells": len(cells), "n_columns": out.shape[1],
            "median_transitions": float(out["n_transitions"].median())
            if len(out) else np.nan,
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
    states = list(args.states) if args.states else []
    cohorts = args.cohorts or _discover_cohorts(root, atlases, windows)
    if not cohorts:
        raise SystemExit("no cohort has latents under any grid cell -- run "
                         "`fmri-decomp decompose` first")
    # NO GLOBAL UNION. The state sets are resolved per (atlas, aperture) by
    # check_grid, because a method that discovers its K has a different column
    # name in every cell -- see cell_states. `--states` still overrides, for
    # forcing one set across the grid.
    check = check_grid(root, atlases, windows, cohorts, states or None,
                       args.min_k, args.max_k)
    if check.empty:
        raise SystemExit("nothing in the grid at all -- check --atlas / "
                         "--window-s against what stage 4 produced")
    if not states:
        found = sorted(set(check.loc[check["ok"], "states"]))
        if not found:
            raise SystemExit(
                "no usable state set in any cell. Run `fmri-decomp cluster` to "
                "add state definitions, or pass --states explicitly.\n"
                "See the report above for what was found and why it was "
                "rejected.")
        log(f"discovered {len(found)} distinct state set(s) across the grid; "
            f"each cell uses the ones its own cohorts all have")

    log(f"grid: {len(atlases)} atlas x {len(windows)} window x "
        f"{len(cohorts)} cohort(s) -> {int(check['ok'].sum())} ready cell(s) "
        f"of {len(check)}")

    rc = report_check(check)
    if args.check:
        return rc
    fatal = check[check.get("fatal", False) == True]           # noqa: E712
    if len(fatal) and not args.allow_partial:
        raise SystemExit(
            f"\nrefusing to start: {len(fatal)} cell(s) are not merely missing "
            f"but INCONSISTENT -- see the reasons above. Pass --allow-partial to "
            f"run the rest anyway.")

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
                log(f"    {cohort:<14} {e['subjects']:>4} subject(s) x "
                    f"{e['n_columns']:>4} column(s) "
                    f"({e['n_cells']} cell(s)), median "
                    f"{e['median_transitions']:.0f} transitions each")

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
                   default=["15", "30", "60", "120", "300", "-1"],
                   help="-1 is the activation aperture: one frame per row "
                        "instead of a sliding window. Cells with no latents are "
                        "reported by --check, not fatal.")
    p.add_argument("--states", nargs="*", default=None,
                   help="state-label columns. Default: every one found in the "
                        "latents schema, so a method added by `fmri-decomp "
                        "cluster` is picked up without being named here.")
    p.add_argument("--cohorts", nargs="*", default=None)
    p.add_argument("--min-k", type=int, default=STATE_K_BAND[0],
                   help="skip a DISCOVERED state set with fewer states than "
                        "this. K=1 is one cell, `0->0`=1.0 for every subject -- "
                        "a constant. Stage 4b refuses to write those now, but "
                        "latents written before it did still carry them.")
    p.add_argument("--max-k", type=int, default=STATE_K_BAND[1],
                   help="skip a DISCOVERED state set with more than this many "
                        "states; a set named in --states is always used. "
                        "Default 64, which keeps 8 and 27 and any K MeanShift "
                        "plausibly finds, and drops the retired 125 and 512: at "
                        "K=512 a subject's matrix is 262,144 cells of which "
                        ">99%% are exactly zero.")
    p.add_argument("--n-overlaps", type=int, default=5,
                   help="windows.n_overlaps the shards were written with; sets "
                        "the stride used for dwell times in seconds. Ignored "
                        "for any latents file that records its own `stride_s` "
                        "-- see time_axis.")
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
