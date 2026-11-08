#!/usr/bin/env python3
"""One shard of the Cam-CAN -> HCP transfer grid. Fits, scores, writes parquet.

    python tools/transfer_grid.py --target raw --shard 0 --n-shards 40
    python tools/transfer_grid.py --target raw --tree fcm        # one job, no shards

WHY THIS IS A SCRIPT AND NOT A NOTEBOOK CELL
--------------------------------------------
The grid is one fit per (state set x task x target x arm x model) on EVERY
training subject: at 282 sets that is ~9,000 fits plus ~2,250 baselines, and
boosting dominates. In a kernel that is hours of a session that cannot be
resumed if the connection drops. As an array it is minutes, and a shard that
dies is one shard to resubmit.

WHAT IS SHARDED, AND WHY THAT UNIT
----------------------------------
The STATE SETS, round-robin: shard i takes sets[i::n] of the grid sorted by
(atlas, window_s, states). Round-robin rather than contiguous blocks because
cost per set is uneven -- K=27 is 729 transition cells against K=8's 64 -- and
contiguous blocks would put every large K in one task.

Everything else stays together inside a shard ON PURPOSE. Both tasks, both
targets and every arm of one state set share the same loaded, pooled
transition table, and the covariates-only baseline is fitted once per
(set, task, target) and reused across arms. Splitting those out would re-read
and re-pool the same tables and refit the same baseline per task.

`--tree fcm` is a SEPARATE, unsharded job: static FC has no aperture and no
state set, so its whole grid is atlases x tasks x targets x arms x models --
small enough for one process, and nothing to shard it by.

WHAT IT REFUSES
---------------
A shard with no set in it (more shards than sets), a grid that does not match
the one the other shards saw (`grid_hash`), and anything the transfer itself
refuses: no common target label, no common task key, no shared state set.
Nothing here writes an empty frame -- a run that scored nothing must not leave
a file behind that reads as "scored nothing is the answer".
"""
from __future__ import annotations

import argparse
import faulthandler
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# A segfault in a C extension leaves nothing but "Segmentation fault" in the
# shell's log -- which is exactly what the first real array produced, forty
# times. faulthandler prints the PYTHON stack that was running when the signal
# arrived, to stderr, so the next one says which line. SIGUSR1 does the same on
# demand, for a task that is not dead but stuck:
#
#   scancel --signal=USR1 <jobid>_<task>     then read the .err
# Guarded: faulthandler needs a stderr with a real file descriptor, and a test
# runner that captures output does not supply one. A diagnostic that cannot
# install itself must not take the run down with it.
try:
    faulthandler.enable()
    for _sig in ("SIGUSR1", "SIGUSR2"):
        if hasattr(signal, _sig):
            faulthandler.register(getattr(signal, _sig), chain=True)
except Exception as _e:                                   # pragma: no cover
    print(f"NOTE: faulthandler not installed ({_e}); a segfault will leave no "
          f"python traceback", flush=True)


def say(*a):
    """print that reaches the log even if the next line segfaults."""
    print(*a, flush=True)

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "notebooks"), str(REPO)]


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="raw", choices=["raw", "category"],
                    help="which Cam-CAN target columns: the raw HADS score or "
                         "the banded category. Two different analyses; they are "
                         "written under separate `targets=` partitions.")
    ap.add_argument("--tree", default="bstm", choices=["bstm", "fcm"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out", default=None,
                    help="default: <outputs>/transferred_models")
    # The three paths, overridable so a run can be pointed at another tree
    # without editing transfer_config -- and so the tests can drive this
    # script against a fixture instead of somebody's scratch space.
    ap.add_argument("--outputs", default=None, help="the outputs tree to read")
    ap.add_argument("--camcan-pheno", default=None,
                    help="Cam-CAN's phenotype DIRECTORY")
    ap.add_argument("--hcp-pheno", default=None, help="HCP's phenotype CSV")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                    help="threads for ONE fit. There is no joblib loop here, so "
                         "this is the task's whole cpu allocation.")
    ap.add_argument("--atlas", nargs="*", default=None, help="restrict the grid")
    ap.add_argument("--window-s", nargs="*", type=float, default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--max-missing", type=float, default=.10,
                    help="fcm only: drop an edge NaN in more than this share of "
                         "TRAINING subjects")
    ap.add_argument("--overwrite", action="store_true",
                    help="refit a shard whose rows.parquet already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the grid, print what this shard would do, fit "
                         "nothing")
    a = ap.parse_args(argv)

    import pandas as pd
    import transfer_config as tc
    import transfer_tables as tt

    t0 = time.time()
    root_out = Path(a.out) if a.out else None
    train, test, train_cfg, test_cfg = tc.studies(
        a.target, models=a.models, root=a.outputs, camcan=a.camcan_pheno,
        hcp_pheno=a.hcp_pheno)
    root_out = root_out or (train_cfg.root / "transferred_models")
    import numpy, sklearn
    say(f"target={a.target} tree={a.tree} shard={a.shard}/{a.n_shards} "
        f"threads={a.threads}")
    # The thread caps, printed because an UNSET one does not mean "one thread":
    # inside the container the runtimes read the NODE's core count, and 192
    # OpenBLAS threads in a 4-cpu cgroup is how the first array segfaulted.
    say("  env " + "  ".join(
        f"{k}={os.environ.get(k, 'UNSET')}"
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")))
    say(f"  numpy {numpy.__version__}  pandas {pd.__version__}  "
        f"sklearn {sklearn.__version__}")
    say(f"train {len(train.pheno):,} subject(s) with phenotype, "
          f"test {len(test.pheno):,}")

    sets = tt.common_sets(train, test)
    if a.atlas:
        sets = sets[sets["atlas"].isin(a.atlas)]
    if a.window_s is not None and len(a.window_s):
        sets = sets[sets["window_s"].isin(a.window_s)]
    if not len(sets):
        raise SystemExit(
            "no state set is on disk for BOTH cohorts under the filters given "
            f"(--atlas {a.atlas} --window-s {a.window_s}). Stage 5a has to have "
            "run for the test cohort with the same fit -- see "
            "slurm/project_and_select.sh with STOP_AFTER=transitions.")
    gh = tt.grid_hash(sets)

    prov = dict(grid_hash=gh, n_sets_total=len(sets), n_boot=a.n_boot,
                seed=a.seed, git_sha=git_sha(),
                written_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    path = (tt.transfer_dir(root_out, "camcan", "hcp7t_fix", a.target, a.tree)
            / f"shard={a.shard:02d}" / "rows.parquet")
    if path.exists() and not a.overwrite:
        say(f"exists, skipping (pass --overwrite to refit): {path}")
        return 0

    if a.tree == "fcm":
        if a.n_shards != 1:
            raise SystemExit("--tree fcm is one job: it has no state set to "
                             "shard by. Submit it with --n-shards 1.")
        mine = sets
        say(f"fcm: atlases from the static_fc tree, {len(sets)} bstm set(s) "
              f"ignored (static FC has no aperture and no state set)")
    else:
        mine = tt.shard_of(sets, a.shard, a.n_shards)
        say(f"grid {len(sets)} set(s), hash {gh}; this shard has {len(mine)}: "
              + ", ".join(f"{r.states}/{r.atlas}/{r.window_s:g}"
                          for r in mine.head(4).itertuples())
              + (" ..." if len(mine) > 4 else ""))
        if not len(mine):
            raise SystemExit(
                f"shard {a.shard} of {a.n_shards} is EMPTY: there are only "
                f"{len(sets)} set(s) in the grid. Submit an array of at most "
                f"{len(sets)} task(s).")

    if a.dry_run:
        n_arm = len(a.arms) if a.arms else 4
        n_mod = len(a.models or train_cfg.models or tt.DEFAULT_MODELS)
        say(f"dry run: {len(mine)} set(s) x 2 task(s) x 2 target(s) x "
              f"{n_arm} arm(s) x {n_mod} model(s) = "
              f"{len(mine) * 4 * n_arm * n_mod:,} fit(s); would write {path}")
        return 0

    if a.tree == "fcm":
        rows = tt.fcm_transfer_rows(
            train, test, train_tasks=dict(train_cfg.tasks),
            test_tasks=dict(test_cfg.tasks),
            covariates=["Age", "Sex", "frac_good_frames", "n_tr_used"],
            models=a.models, max_missing=a.max_missing, seed=a.seed,
            n_boot=a.n_boot, threads=a.threads, log=say)
    else:
        rows = tt.transfer_rows(train, test, arms=a.arms, models=a.models,
                                sets=mine, seed=a.seed, n_boot=a.n_boot,
                                threads=a.threads, log=say)
    out = tt.write_transferred(rows, root_out, train="camcan", test="hcp7t_fix",
                               target_set=a.target, tree=a.tree, shard=a.shard,
                               n_shards=a.n_shards, **prov)
    say(f"{len(rows):,} row(s) -> {out}")
    say(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
