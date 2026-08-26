#!/usr/bin/env python3
"""Pre-flight conformance checks for a cohort config.

Cohort-agnostic: the same four checks run against ds002837, CNeuroMod and
Cam-CAN alike, because everything cohort-specific already lives in the YAML.
Run this BEFORE burning core-hours on a full extraction.

    python tools/check_cohort.py config/ds002837.yaml               # checks 1-2
    python tools/check_cohort.py config/ds002837.yaml --all --limit 3

The four checks map to the four properties that actually break when a new
cohort is added:

  1. paths    -- every (run, atlas) maps to a DISTINCT output file.
                 This is the assumption the lock-free parallelism rests on:
                 "each worker owns one leaf". Two runs colliding on one path
                 means two workers race and one silently wins. Needs no data.

  2. tr       -- the config TR equals the TR in each NIfTI header.
                 A config TR that disagrees with the header does not crash;
                 it silently rescales every window and every time column.
                 Needs the real files (headers only -- cheap, no voxels read).

  3. outputs  -- written shards honour the stage-2 contract: required columns
                 present, partition keys NOT duplicated as columns, dtypes as
                 documented. Needs `extract` to have run.

  4. parallel -- n-jobs=1 and n-jobs=N produce identical values.
                 Guards against worker-count-dependent results. Re-runs a
                 small extraction twice, so use --limit.

Exit code is 0 only if every selected check passes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OK, BAD, WARN = "  ok  ", " FAIL ", " warn "


def _hdr(title: str) -> None:
    print(f"\n\033[1m== {title}\033[0m")


def _load(cfg_path: str):
    from fmri_decomposition.config import load_config

    cfg = load_config(cfg_path)
    if cfg.participants is not None and not Path(cfg.participants).is_absolute():
        # Relative participants paths resolve against the CWD, not the config
        # file. That is a silent "no such file" the moment you run from a
        # scheduler's working directory, so say so plainly.
        if not Path(cfg.participants).exists():
            print(f"[{WARN}] participants path {cfg.participants} is relative and does "
                  f"not exist from cwd={Path.cwd()}. Use an absolute path in the YAML.")
    return cfg


def _refs(cfg):
    from fmri_decomposition.cli import _attach_censor, _attach_confounds
    from fmri_decomposition.cohort import (attach_participants, discover_runs,
                                           load_participants)

    refs = discover_runs(cfg)
    if cfg.participants is not None and Path(cfg.participants).exists():
        refs = attach_participants(refs, load_participants(cfg), cfg)
    return _attach_confounds(cfg, _attach_censor(cfg, refs))


def _atlases(cfg):
    from fmri_decomposition.atlases.registry import get_atlas

    return [get_atlas(n, **cfg.atlas_params.get(n, {})) for n in cfg.atlases]


# --------------------------------------------------------------- check 1 ---
def check_paths(cfg, refs, atlas_names) -> bool:
    """Every (run, atlas) must own a distinct output leaf."""
    from fmri_decomposition.io import activation_path

    _hdr("1. output-path uniqueness (the lock-free-parallelism precondition)")
    if not refs:
        print(f"[{BAD}] discovery found 0 runs -- check discovery.bold_glob "
              f"against {cfg.derivatives_root}")
        return False

    owners: dict[str, list[str]] = defaultdict(list)
    for atlas in atlas_names:
        for r in refs:
            p = activation_path(cfg.output_root, r.cohort, atlas, r.task,
                                r.sub, r.ses, r.run, r.acq)
            owners[str(p)].append(f"{Path(r.bold).name}")

    collisions = {p: srcs for p, srcs in owners.items() if len(srcs) > 1}
    print(f"  runs discovered : {len(refs)}")
    print(f"  atlases         : {len(atlas_names)}")
    print(f"  distinct outputs: {len(owners)}  (expected {len(refs) * len(atlas_names)})")
    if collisions:
        print(f"[{BAD}] {len(collisions)} output path(s) claimed by more than one run:")
        for p, srcs in list(collisions.items())[:5]:
            print(f"    {p}")
            for s in srcs:
                print(f"      <- {s}")
        print("  Fix: the colliding runs differ by an entity that discovery is not "
              "capturing (ses/run/acq). They belong in the FILENAME, via the "
              "discovery patterns -- never as a new directory level.")
        return False
    print(f"[{OK}] no collisions -- every worker owns exactly one leaf")

    depths = {len(Path(p).relative_to(cfg.output_root).parts) for p in owners}
    if len(depths) == 1:
        print(f"[{OK}] directory depth constant across all runs ({depths.pop()} levels)")
    else:
        print(f"[{BAD}] inconsistent directory depth {sorted(depths)} -- a hive dataset "
              "must have one shape")
        return False

    ex = sorted(owners)[0]
    print(f"  example leaf: {Path(ex).relative_to(cfg.output_root)}")
    return True


# --------------------------------------------------------------- check 2 ---
def check_tr(cfg, refs, limit: int | None) -> bool:
    """Config TR must equal the TR in the NIfTI header. Reads headers only."""
    import nibabel as nib

    _hdr("2. config TR vs NIfTI header TR")
    subset = refs if limit is None else refs[:limit]
    if not subset:
        print(f"[{WARN}] no runs to check")
        return True

    bad, unset, checked = [], [], 0
    for r in subset:
        try:
            hdr = nib.load(str(r.bold)).header
            hdr_tr = float(hdr.get_zooms()[3])
        except Exception as exc:                                  # noqa: BLE001
            print(f"[{BAD}] cannot read header for {Path(r.bold).name}: "
                  f"{type(exc).__name__}: {exc}")
            return False
        checked += 1
        if hdr_tr in (0.0, 1.0):
            # 0 means unset, and a bare 1.0 is nibabel's default for a header
            # that never had a TR written. Either way the header corroborates
            # nothing -- INCLUDING when the config also says 1.0, which is the
            # ds002837 case exactly. A coincidental match is not evidence.
            unset.append((Path(r.bold).name, hdr_tr))
        elif abs(hdr_tr - cfg.tr) > 1e-3:
            bad.append((Path(r.bold).name, hdr_tr))

    print(f"  config tr = {cfg.tr}s;  headers read = {checked}")
    if bad:
        print(f"[{BAD}] {len(bad)} file(s) disagree with the config TR:")
        for name, tr in bad[:5]:
            print(f"    {name}: header says {tr}s, config says {cfg.tr}s")
        print("  A wrong TR does not crash -- it rescales every window and every "
              "time column. Resolve before extracting.")
        return False
    if unset:
        coincidental = " (which equals your config TR -- a coincidence, not a check)" \
            if abs(cfg.tr - unset[0][1]) < 1e-9 else ""
        print(f"[{WARN}] {len(unset)}/{checked} file(s) carry an unset/default header "
              f"TR of {unset[0][1]}s{coincidental}:")
        for name, tr in unset[:3]:
            print(f"    {name}")
        print("  The header corroborates NOTHING here. Confirm the TR from the "
              "dataset's own documentation or a sidecar JSON, not from this check.")
        print("  (NNDb/ds002837 omits RepetitionTime from its task JSON -- a BIDS "
              "violation -- so its real TR of 1.0s can only come from the paper.)")
    if not unset and not bad:
        print(f"[{OK}] every header carries a real TR and it agrees with the config")
    return True


# --------------------------------------------------------------- check 3 ---
def check_outputs(cfg, atlas_names) -> bool:
    """Written shards must honour the stage-2 contract."""
    import pandas as pd
    import pyarrow.parquet as pq

    from fmri_decomposition.activation import META_COLUMNS
    from fmri_decomposition.io import PARTITION_KEYS, activation_root

    _hdr("3. written-output contract")
    ok = True
    for atlas in atlas_names:
        root = activation_root(cfg.output_root, atlas, cfg.cohort)
        files = sorted(Path(root).rglob("*.parquet")) if Path(root).exists() else []
        if not files:
            print(f"[{WARN}] atlas {atlas!r}: no shards under {root} -- run `extract` first")
            continue
        t = pq.read_table(files[0])
        df = t.to_pandas()
        print(f"  atlas {atlas!r}: {len(files)} shard(s); first has "
              f"{len(df)} rows x {len(df.columns)} cols")

        missing = [c for c in META_COLUMNS if c not in df.columns]
        if missing:
            print(f"[{BAD}]   missing contract column(s): {missing}")
            ok = False
        else:
            print(f"[{OK}]   all {len(META_COLUMNS)} contract columns present")

        leaked = [k for k in PARTITION_KEYS["activation"] if k in df.columns]
        if leaked:
            print(f"[{BAD}]   partition key(s) duplicated as columns: {leaked}")
            print("        pyarrow infers sub=01 as int32, which destroys "
                  "leading zeros and Cam-CAN ids like CC110033.")
            ok = False
        else:
            print(f"[{OK}]   partition keys carried by the path only")

        stray = list(Path(root).rglob("*.tmp.*"))
        if stray:
            print(f"[{WARN}]   {len(stray)} stray .tmp file(s) from killed workers")

        meta = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
        for key in ("atlas", "cohort", "task", "sub", "tr"):
            if key not in meta:
                print(f"[{WARN}]   shard metadata missing {key!r}")
        if meta.get("tr") and abs(float(meta["tr"]) - cfg.tr) > 1e-6:
            print(f"[{BAD}]   shard was written with tr={meta['tr']}, config now says "
                  f"{cfg.tr} -- outputs are stale")
            ok = False

        nan_cols = [c for c in df.columns if c not in META_COLUMNS
                    and df[c].dtype.kind == "f" and df[c].isna().all()]
        if nan_cols:
            print(f"[{WARN}]   {len(nan_cols)} all-NaN parcel column(s) "
                  f"(empty under the brain mask), e.g. {nan_cols[:3]}")
    return ok


# --------------------------------------------------------------- check 4 ---
def check_parallel(cfg, refs, atlas_names, n_jobs: int, limit: int) -> bool:
    """n-jobs=1 and n-jobs=N must produce identical values."""
    import pandas as pd
    from joblib import Parallel, delayed

    from fmri_decomposition.activation import process_run
    from fmri_decomposition.io import activation_path

    _hdr(f"4. serial vs parallel determinism (n-jobs=1 vs n-jobs={n_jobs})")
    subset = refs[:limit]
    if not subset:
        print(f"[{WARN}] no runs to check")
        return True
    atlases = _atlases(cfg)[:1]      # one atlas is enough to prove determinism
    print(f"  {len(subset)} run(s) x 1 atlas, extracted twice into temp dirs")

    import dataclasses

    results = {}
    tmpdirs = []
    try:
        for tag, jobs_n in (("serial", 1), ("parallel", n_jobs)):
            tmp = Path(tempfile.mkdtemp(prefix=f"conform_{tag}_"))
            tmpdirs.append(tmp)
            cfg_t = dataclasses.replace(cfg, output_root=tmp)
            entries = Parallel(n_jobs=jobs_n, backend="loky")(
                delayed(process_run)(r, a, cfg_t, True)
                for a in atlases for r in subset
            )
            errs = [e for e in entries if e.status == "error"]
            if errs:
                print(f"[{BAD}] {tag} run produced {len(errs)} error(s):")
                for e in errs[:3]:
                    print(f"    {e.sub}/{e.task}: {e.detail}")
                return False
            results[tag] = (tmp, cfg_t)
            print(f"[{OK}]   {tag} pass wrote {len(entries)} shard(s)")

        mismatched = []
        for a in atlases:
            for r in subset:
                pa_ = activation_path(results["serial"][0], r.cohort, a.name, r.task,
                                      r.sub, r.ses, r.run, r.acq)
                pb_ = activation_path(results["parallel"][0], r.cohort, a.name, r.task,
                                      r.sub, r.ses, r.run, r.acq)
                if not (pa_.exists() and pb_.exists()):
                    mismatched.append((r.sub, "a shard is missing on one side"))
                    continue
                da, db = pd.read_parquet(pa_), pd.read_parquet(pb_)
                if da.shape != db.shape:
                    mismatched.append((r.sub, f"shape {da.shape} vs {db.shape}"))
                elif not da.equals(db):
                    num = da.select_dtypes("number")
                    delta = (num - db.select_dtypes("number")).abs().to_numpy()
                    mismatched.append((r.sub, f"values differ, max|delta|={delta.max():.3e}"))
        if mismatched:
            print(f"[{BAD}] {len(mismatched)} run(s) differ between serial and parallel:")
            for sub, why in mismatched[:5]:
                print(f"    sub {sub}: {why}")
            return False
        print(f"[{OK}] identical output -- worker count does not change results")
        return True
    finally:
        for d in tmpdirs:
            shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ main ---
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--all", action="store_true",
                   help="also run the checks that need data on disk / a prior extract")
    p.add_argument("--checks", default=None,
                   help="comma-separated subset of: paths,tr,outputs,parallel")
    p.add_argument("--limit", type=int, default=3,
                   help="runs to use for the tr and parallel checks (default 3)")
    p.add_argument("--n-jobs", type=int, default=4)
    args = p.parse_args(argv)

    selected = (set(args.checks.split(",")) if args.checks
                else {"paths", "tr", "outputs", "parallel"} if args.all
                else {"paths", "tr"})

    cfg = _load(args.config)
    print(f"cohort={cfg.cohort}  tr={cfg.tr}  atlases={cfg.atlases}")
    print(f"derivatives_root={cfg.derivatives_root}")
    print(f"output_root={cfg.output_root}")
    if not Path(cfg.derivatives_root).exists():
        print(f"\n[{BAD}] derivatives_root does not exist. Edit the YAML first.")
        return 1

    refs = _refs(cfg)
    # Atlas names only for the path check: it needs no atlas image, so a cohort
    # can be checked on a login node with no atlas cache present.
    atlas_names = list(cfg.atlases)

    results = {}
    if "paths" in selected:
        results["paths"] = check_paths(cfg, refs, atlas_names)
    if "tr" in selected:
        results["tr"] = check_tr(cfg, refs, args.limit)
    if "outputs" in selected:
        results["outputs"] = check_outputs(cfg, atlas_names)
    if "parallel" in selected:
        results["parallel"] = check_parallel(cfg, refs, atlas_names,
                                             args.n_jobs, args.limit)

    _hdr("summary")
    for name, passed in results.items():
        print(f"  {name:<9} {'PASS' if passed else 'FAIL'}")
    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"\n\033[31m{len(failed)} check(s) failed: {failed}\033[0m")
        return 1
    print("\n\033[32mall selected checks passed\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
