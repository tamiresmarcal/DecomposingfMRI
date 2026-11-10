"""Turn stage 3 measurements into a stage 4 decision, once, in a named policy.

    fmri-decomp censor --policy config/censor/default.yaml
    fmri-decomp censor --policy config/censor/default.yaml \\
        --stage dfc --atlas harvardoxford --window-s 30

writes

    outputs/extractions/censor/policy=<name>/cohort=<c>/task=<t>/subjects.parquet
    outputs/extractions/censor/policy=<name>/atlas=<a>/window_s=<w>/cohort=<c>/task=<t>/windows.parquet
    outputs/extractions/meta/censor/policy=<name>.json

WHY THIS IS ITS OWN STAGE
-------------------------
The pipeline's rule is that no threshold appears anywhere in it: `diagnose`
writes `mean_fd = 0.52` and stops, because `0.52 > 0.5 -> exclude` is a claim
about the analysis rather than a fact about the scan. That rule is right, and
it leaves a gap -- the claim still has to be made somewhere, and if it is made
inline in whatever notebook happens to need it, it gets made differently every
time and travels with nothing.

So: one step, between measurement and modelling. It reads only QC columns,
never imaging data, runs in seconds, and writes a decision that stage 4
consumes instead of re-deriving. The policy is versioned by `name` and hashed
into every output, so two policies coexist and a sensitivity analysis is two
files rather than a re-run with different constants.

WHAT IT DOES NOT DO
-------------------
It does not censor frames. Per-TR censoring already happened at stage 2 and is
in `good_frame`; where it could not happen -- ds002837, whose regressor and
image timelines cannot be reconciled -- it cannot be recovered here either,
which is exactly why `max_mean_fd` at the subject level is doing the work for
that cohort.

It does not edit `participants.csv`. That file is human curation ("corrupted
run", "consent withdrawn") and is not the place for a threshold someone will
want to move.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..io import (censor_root, cohort_meta_dir, dfc_root, meta_dir,
                  parse_hive_keys, read_file)

# (column, comparison, policy key). `lo` means the column must be >= the
# threshold, `hi` that it must be <= it, `abs_hi` that its magnitude must be.
SUBJECT_RULES = [
    ("mean_fd", "hi", "max_mean_fd"),
    ("frac_good_frames", "lo", "min_frac_good_frames"),
    ("frac_parcels_empty", "hi", "max_frac_parcels_empty"),
    ("frac_stimulus_covered", "lo", "min_frac_stimulus_covered"),
    ("best_lag_tr", "abs_hi", "max_abs_best_lag_tr"),
]

_SENSE = {"lo": ("<", lambda v, t: v < t),
          "hi": (">", lambda v, t: v > t),
          "abs_hi": (">", lambda v, t: v.abs() > t)}


def policy_hash(policy: dict) -> str:
    return hashlib.sha256(
        json.dumps(policy, sort_keys=True, default=str).encode()).hexdigest()[:16]


def load_policy(path: str | Path) -> dict:
    import yaml

    policy = yaml.safe_load(Path(path).read_text()) or {}
    policy.setdefault("name", Path(path).stem)
    policy.setdefault("subject", {})
    policy.setdefault("window", {})
    return policy


def gate_subjects(qc: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """One row per (sub, task) with `keep` and the reason it was not kept.

    A threshold set against a column that is entirely NaN is reported and NOT
    applied: dropping every row for a missing input is how a cohort disappears
    without anyone noticing. `best_lag_tr` is the live case -- it is NaN for
    any task with fewer than three subjects.
    """
    out = qc.copy()
    reasons = [[] for _ in range(len(out))]
    applied, skipped = {}, {}

    for column, sense, key in SUBJECT_RULES:
        threshold = policy["subject"].get(key)
        if threshold is None:
            continue
        if column not in out.columns or out[column].isna().all():
            skipped[key] = f"{column} absent or all-NaN"
            continue
        op, test = _SENSE[sense]
        values = pd.to_numeric(out[column], errors="coerce")
        # NaN is not a failure, it is unknown -- and a rule whose column is
        # only PARTLY missing still applies to the rows that have it.
        fails = test(values, threshold).fillna(False)
        applied[key] = int(fails.sum())
        label = f"|{column}|" if sense == "abs_hi" else column
        for i in np.flatnonzero(fails.to_numpy()):
            reasons[i].append(f"{label}{op}{threshold}")

    out["reason"] = ["; ".join(r) for r in reasons]
    out["keep"] = out["reason"] == ""
    out.attrs["applied"] = applied
    out.attrs["skipped"] = skipped
    return out


def gate_windows(win: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """One row per window with `keep` and the reason, for the DFC path."""
    out = win.copy()
    reasons = [[] for _ in range(len(out))]
    applied = {}

    def fail(mask: pd.Series, label: str, key: str) -> None:
        mask = mask.fillna(False)
        applied[key] = int(mask.sum())
        for i in np.flatnonzero(mask.to_numpy()):
            reasons[i].append(label)

    w = policy["window"]
    if w.get("min_frac_good_frames") is not None and "frac_good_frames" in out:
        t = w["min_frac_good_frames"]
        fail(out["frac_good_frames"] < t, f"frac_good_frames<{t}",
             "min_frac_good_frames")
    if w.get("min_n_tr_effective") is not None and "n_tr_effective" in out:
        t = w["min_n_tr_effective"]
        fail(out["n_tr_effective"] < t, f"n_tr_effective<{t}", "min_n_tr_effective")
    for flag, key in (("rank_deficient", "drop_rank_deficient"),
                      ("crosses_run_boundary", "drop_crosses_run_boundary"),
                      ("crosses_clip_boundary", "drop_crosses_clip_boundary")):
        if w.get(key) and flag in out:
            fail(out[flag].astype(bool), flag, key)

    out["reason"] = ["; ".join(r) for r in reasons]
    out["keep"] = out["reason"] == ""
    out.attrs["applied"] = applied
    return out


def _write(df: pd.DataFrame, path: Path, policy: dict, phash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.assign(policy=policy["name"], policy_hash=phash)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def run(args) -> int:
    root = Path(args.output_root) if args.output_root else _default_root()
    policy = load_policy(args.policy)
    phash = policy_hash(policy)
    base = censor_root(root, policy["name"])
    print(f"policy {policy['name']}  hash {phash}\noutput_root {root}")

    cohorts = args.cohorts or sorted(
        p.name.split("=", 1)[1]
        for p in (meta_dir(root, "extractions") / "cohorts").glob("cohort=*")
        if (p / "participants_qc.csv").exists())
    if not cohorts:
        raise SystemExit("no cohort has a participants_qc.csv; run "
                         "`fmri-decomp diagnose` first")

    summary = {"policy": policy, "policy_hash": phash, "cohorts": {},
               "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    for cohort in cohorts:
        qc_path = cohort_meta_dir(root, cohort) / "participants_qc.csv"
        if not qc_path.exists():
            print(f"{cohort:<16} no participants_qc.csv, skipped")
            continue
        qc = pd.read_csv(qc_path, dtype={"sub": str})
        gated = gate_subjects(qc, policy)
        # ONE FILE PER TASK. The table is already one row per (sub, task), so
        # this is a partition of it rather than a change to what it says -- and
        # it means a cohort's two conditions can be re-gated independently, and
        # that every tree in outputs/ keys on task= at the same depth.
        for task, g in gated.groupby("task", sort=True):
            _write(g.reset_index(drop=True),
                   base / f"cohort={cohort}" / f"task={task}"
                   / "subjects.parquet", policy, phash)

        kept, total = int(gated["keep"].sum()), len(gated)
        dropped = {k: int(v) for k, v in
                   gated.loc[~gated["keep"], "reason"].value_counts().items()}
        print(f"{cohort:<16} subjects {kept}/{total} kept"
              + (f"   dropped: {dropped}" if dropped else ""))
        for key, reason in gated.attrs["skipped"].items():
            print(f"    NOT APPLIED  {key}: {reason}")
        summary["cohorts"][cohort] = {
            "subject_rows": total, "subject_kept": kept,
            "applied": gated.attrs["applied"], "skipped": gated.attrs["skipped"]}

        if args.stage != "dfc":
            continue

        shard_root = dfc_root(root, args.atlas, float(args.window_s), cohort)
        paths = sorted(shard_root.rglob("*.parquet"))
        if not paths:
            print(f"    no dfc shards under {shard_root}, window gate skipped")
            continue

        # Read only the QC columns, and only the ones this cohort actually has:
        # `crosses_clip_boundary` exists where a config defines clips and not
        # otherwise, and asking parquet for an absent column is a hard error.
        want = ["window_id", "start_s", "n_tr_effective", "frac_good_frames",
                "rank_deficient", "crosses_run_boundary", "crosses_clip_boundary"]
        have = set(pq.ParquetFile(paths[0]).schema_arrow.names)
        cols = [c for c in want if c in have]
        missing = [c for c in want if c not in have]
        if missing:
            print(f"    columns absent from the shards, not gated on: {missing}")

        frames = []
        for p in paths:
            keys = parse_hive_keys(p)
            d = pd.read_parquet(p, columns=cols)
            frames.append(d.assign(cohort=keys["cohort"], task=keys["task"],
                                   sub=keys["sub"]))
        win = gate_windows(pd.concat(frames, ignore_index=True), policy)

        # A window belonging to a dropped subject is dropped too, and says so.
        drop_subs = set(map(tuple, gated.loc[~gated["keep"], ["sub", "task"]].values))
        by_sub = pd.Series(list(zip(win["sub"], win["task"]))).isin(drop_subs)
        win.loc[by_sub.to_numpy(), "reason"] = (
            win.loc[by_sub.to_numpy(), "reason"]
               .radd("subject excluded; ").str.rstrip("; "))
        win.loc[by_sub.to_numpy(), "keep"] = False

        window_key = shard_root.parent.name.split("=", 1)[1]
        for task, g in win.groupby("task", sort=True):
            out = g.reset_index(drop=True)
            out.attrs = win.attrs
            _write(out, base / f"atlas={args.atlas}"
                        / f"window_s={window_key}" / f"cohort={cohort}"
                        / f"task={task}" / "windows.parquet", policy, phash)
        wk, wt = int(win["keep"].sum()), len(win)
        print(f"{'':<16} windows  {wk:,}/{wt:,} kept  "
              f"({100 * wk / wt:.1f}%)  {win.attrs['applied']}")
        summary["cohorts"][cohort].update(
            {"atlas": args.atlas, "window_s": window_key,
             "window_rows": wt, "window_kept": wk,
             "window_applied": win.attrs["applied"]})

    out = (meta_dir(root, "extractions") / "censor"
           / f"policy={policy['name']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nsummary -> {out.relative_to(root)}")
    return 0


def _default_root() -> Path:
    """Delegates to io.default_output_root -- see it for why this is not
    eight copies of a cohort filename any more."""
    from ..io import default_output_root

    return default_output_root()


def add_arguments(p) -> None:
    p.add_argument("--policy", default="config/censor/default.yaml")
    p.add_argument("--stage", choices=["subject", "dfc"], default="subject",
                   help="`subject` gates subjects only; `dfc` also gates "
                        "windows, and needs --atlas and --window-s")
    p.add_argument("--atlas")
    p.add_argument("--window-s")
    p.add_argument("--cohorts", nargs="*", default=None)
    p.add_argument("--output-root")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    args = p.parse_args(argv)
    if args.stage == "dfc" and not (args.atlas and args.window_s):
        raise SystemExit("--stage dfc needs --atlas and --window-s")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------- reading it back ---
# `load_censor` used to live in decompose.py, so the module that CONSUMES the
# decision owned the reader and the module that WRITES it did not -- and
# static_fc, an extraction, had to import it out of the unsupervised model to
# read a table this file produced. The reader belongs beside the writer: the
# path shape below and the one `write`s above build are the same claim, and
# keeping them in one file is what stops them drifting.

def load_censor(root: Path, policy: str | None, atlas: str, window_s, cohort: str):
    """The stage 3.5 decision for one cohort -> (kept subjects, kept windows).

    Returns None when no policy was named, which is how this stage stays
    runnable on a tree that predates `censor`. Everything else is a hard error:
    a policy that was asked for and not found must not silently become no
    censoring at all, because the run would look identical and be a different
    analysis.

    The window gate is optional even when the policy exists -- `censor` writes
    windows.parquet only for the atlas x aperture it was given `--stage dfc`
    for. A policy with subjects only is the normal first-pass state, and gates
    whole subjects rather than individual windows.
    """
    if not policy:
        return None

    base = censor_root(root, policy)
    # EVERY task of the cohort, unioned. The decision is per (sub, task) and
    # the set below is keyed by that pair, so reading the parts and unioning
    # them is the same answer as one file was -- and a cohort whose conditions
    # were gated in separate runs is handled without either overwriting the
    # other.
    subj_paths = sorted((base / f"cohort={cohort}")
                        .glob("task=*/subjects.parquet"))
    if not subj_paths:
        raise SystemExit(
            f"--censor-policy {policy!r} but no "
            f"{base / f'cohort={cohort}'}/task=*/subjects.parquet exists.\n"
            f"Run:  fmri-decomp censor --policy config/censor/{policy}.yaml "
            f"--cohorts {cohort}")

    # read_file, not pd.read_parquet: a censor table carries a `policy` column
    # under `policy=<name>/` and a `task` column under `task=<t>/`, so the
    # dataset layer infers both keys from the path and cannot merge them with
    # the file's own columns. Raises on pyarrow 18, passes on 25. See
    # io.read_file.
    subj = pd.concat([read_file(p, ["sub", "task", "keep"]).to_pandas()
                      for p in subj_paths], ignore_index=True)
    keep_subs = set(zip(subj.loc[subj["keep"], "task"].astype(str),
                        subj.loc[subj["keep"], "sub"].astype(str)))

    win_paths = sorted((base / f"atlas={atlas}" / f"window_s={window_s}"
                        / f"cohort={cohort}").glob("task=*/windows.parquet"))
    keep_windows = None
    if win_paths:
        win = pd.concat(
            [read_file(p, ["sub", "task", "window_id", "keep"]).to_pandas()
             for p in win_paths], ignore_index=True)
        win = win.loc[win["keep"]]
        keep_windows = set(zip(win["task"].astype(str), win["sub"].astype(str),
                               win["window_id"].astype("int64")))
    return {"policy": policy, "subjects": keep_subs, "windows": keep_windows,
            "n_subject_rows": len(subj), "n_subject_kept": int(subj["keep"].sum())}
