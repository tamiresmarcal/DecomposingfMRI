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
import json
import math
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


def build_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--threads", type=int, default=1,
                    help="threads for ONE fit. CLAMPED to what the process "
                         "already allows: raising a BLAS thread count mid-run "
                         "segfaults, so a real raise has to happen in the "
                         "ENVIRONMENT, which slurm/transfer.sbatch does.")
    ap.add_argument("--atlas", nargs="*", default=None, help="restrict the grid")
    ap.add_argument("--window-s", nargs="*", type=float, default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--allow-frame-mismatch", action="store_true",
                    help="fcm only: proceed when the two sides' static-FC "
                         "tables, or one side's two conditions, were built "
                         "from different frames. A DECISION, not a workaround: "
                         "every row then carries train_censor_policy, "
                         "test_censor_policy and frames_mismatch, and the "
                         "figure has to say so. Use it when the state trees "
                         "already carry the same asymmetry -- mirroring it "
                         "keeps fcm comparable to bstm, which is what E1 is "
                         "about.")
    ap.add_argument("--max-missing", type=float, default=.10,
                    help="fcm only: drop an edge NaN in more than this share of "
                         "TRAINING subjects")
    ap.add_argument("--overwrite", action="store_true",
                    help="refit a shard whose rows.parquet already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the grid, print what this shard would do, fit "
                         "nothing")
    ap.add_argument("--no-write", action="store_true",
                    help="fit and score, report, write NOTHING. This is how to "
                         "try the chain on real data: a trial run with --shard "
                         "and --n-shards writes a real shard file into the real "
                         "tree, where it reads as part of the array.")
    ap.add_argument("--status", action="store_true",
                    help="read-only: which shards are on disk and which are "
                         "missing, for both trees. The question an array leaves "
                         "behind is whether the tasks FINISHED or were killed, "
                         "and the queue cannot answer it.")
    return ap


def status(a, root_out) -> int:
    """What is on disk, per tree: shards, gaps, and whether they AGREE.

    An array that has left the queue has either finished or been killed, and
    squeue cannot tell you which afterwards. This says it in terms of the thing
    that matters -- the shards whose rows exist, and whether they were all
    computed the same way.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    import transfer_tables as tt

    # The two trees are located INDEPENDENTLY: `--out` says where the transfer
    # rows are, `--outputs` says where the pipeline's tree is. Deriving one
    # from the other by string surgery on the path worked in the one layout
    # where they are siblings and silently looked in the wrong place anywhere
    # else.
    outputs = Path(a.outputs) if a.outputs else Path(os.environ.get(
        "FMRIDECOMP_OUTPUTS", "/project/6008063/tamires/DecomposingfMRI/outputs"))
    root = root_out or outputs / "transferred_models"
    say(f"{root}")
    todo = []
    for tree in ("bstm", "fcm"):
        inv = tt.inventory(root, a.target, "camcan", "hcp7t_fix", tree)
        head = f"  targets={a.target} tree={tree}:"
        for f in inv["empty"]:
            say(f"{head} EMPTY FILE {f} -- nothing was scored; read that job's "
                f"log for the skip reasons, delete it and resubmit")
            todo.append(f"{tree}: an empty shard file to delete")
        if inv["rows"] is None:
            if not inv["empty"]:
                say(f"{head} NOTHING under {inv['base']}")
                todo.append(f"{tree}: nothing on disk")
            continue
        n, have, gaps = inv["n_shards"], inv["have"], inv["gaps"]
        # no state-set count here: it belongs on the coverage line below, where
        # a set is (atlas, window_s, states). Two numbers both called "state
        # sets" and counted differently is how a report stops being read.
        say(f"{head} {len(have)}/{n} shard(s), {len(inv['rows']):,} row(s)"
            + (f"  MISSING {gaps}" if gaps else "  COMPLETE"))
        if gaps:
            todo.append(f"{tree}: shard(s) {gaps}")
        # "40/40 shards" answers whether the TASKS finished. It does not answer
        # whether every model scored in every atlas and every window, which is
        # a different question with a different failure: a cell can be skipped
        # (a missing arm, a test set with one distinct score) or scored and
        # come out as NaN (the model predicted one value for everyone, so
        # Spearman has nothing to rank). Both leave the shard count at 40/40.
        d = inv["rows"]
        want = int(d["n_sets_total"].iloc[0]) if "n_sets_total" in d else None
        got = d.groupby(["atlas", "window_s", "states"]).ngroups
        # A shortfall has TWO causes and they need different actions. A missing
        # shard accounts for a known number of sets -- the grid is sets[i::n] of
        # a sorted list, so shard i holds ceil((N - i) / n) of them -- and
        # anything beyond that is cells that were skipped. Saying "grep for
        # skipped" when the cause is the shard you already know is missing
        # sends you looking for a second problem that is not there.
        owed = sum(math.ceil((want - i) / n) for i in gaps) if want and n else 0
        note = ""
        if want and got < want:
            note = (f"   <-- {owed} of them are the missing shard(s) {gaps}"
                    if got + owed == want else
                    f"   <-- SHORT by {want - got - owed} beyond the missing "
                    f"shard(s): cells were skipped, grep the logs for 'skipped'")
        say(f"      {got}" + (f"/{want}" if want else "")
            + " state set(s) with at least one row" + note)
        for m, h in d.groupby("model"):
            sets_m = h.groupby(["atlas", "window_s", "states"]).ngroups
            nan = int(h["score"].isna().sum()) if "score" in h else 0
            say(f"      {m:>6}: {sets_m} set(s), {len(h):,} row(s)"
                + (f", {nan} with NO score (predicted a constant)" if nan else "")
                + ("" if sets_m == got else "   <-- fewer sets than the other models"))
        for col in ("atlas", "window_s", "arm"):
            if col not in d:
                continue
            # a dict comprehension rather than groupby.apply: `include_groups`
            # moved between pandas 2 and 3 and this has to read the same on the
            # cluster's 2.2 and anywhere else
            per = {k: h.groupby(["atlas", "window_s", "states"]).ngroups
                   for k, h in d.groupby(col, dropna=False)}
            if per:
                say(f"      {col:>6}: " + ", ".join(
                    f"{k}={v}" for k, v in per.items()) + "  (state sets each)")
        # The check the grid hash cannot make: same grid, different analysis.
        hard, soft = tt.disagreements(inv["recipes"])
        if len(hard) > 1:
            say(f"{head} the shards DISAGREE on how they were computed -- "
                f"concatenating them would concatenate two analyses:")
            for k, v in sorted(hard.items(), key=lambda kv: -len(kv[1])):
                say(f"      shard(s) {v if len(v) < 12 else f'{len(v)} of them'}"
                    f": n_shards={k[0]}, n_boot={k[1]}, seed={k[2]}")
            odd = min(hard.items(), key=lambda kv: len(kv[1]))[1]
            say(f"    the odd one out is shard(s) {odd}. A shard tried by hand "
                f"is the usual cause, and `n_shards` above is read from it, "
                f"which is why the count can look absurd. Delete and resubmit:")
            for sh in odd:
                say(f"      rm -r {inv['base']}/shard={int(sh):02d}")
            todo.append(f"{tree}: shard(s) {odd} were computed differently")
        if len(soft) > 1:
            say(f"{head} NOTE -- not every shard carries the same arms and "
                f"models. Either one was run by hand with fewer, or those "
                f"cells were skipped; the per-model counts above say which:")
            for k, v in sorted(soft.items(), key=lambda kv: -len(kv[1])):
                say(f"      shard(s) {v if len(v) < 12 else f'{len(v)} of them'}"
                    f": arms={list(k[0])}, models={list(k[1])}")
    # THE fcm INPUTS. "Did the fcm job finish" is half the question; the other
    # half is whether its two inputs are comparable, and that is knowable from
    # disk before anything is submitted. Printed here because this is where
    # the question gets asked.
    import transfer_config as tc

    tr_co, te_co = tc.TRAIN_COHORTS, tc.TEST_COHORTS
    fc_root = outputs / "static_fc"
    recs, seen = {}, False
    for q in sorted(fc_root.rglob("subjects.parquet")):
        seen = True
        at, co = tt.hive(q, "atlas"), tt.hive(q, "cohort")
        # Which provenance columns the file HAS, from the schema, instead of
        # reading them and catching the failure. A bare `except Exception`
        # here labelled every table "predates the provenance columns" -- the
        # exception it was swallowing was a NameError of mine. A catch-all
        # around a read is a catch-all around the bug in the read.
        names = set(pq.read_schema(q).names)
        cols = [c for c in tt.FRAME_RECIPE if c in names]
        if not cols:
            recs[(at, co)] = None            # predates the provenance columns
            continue
        try:
            recs[(at, co)] = tt.frame_recipe(
                pd.read_parquet(q, columns=cols), f"{at}/{co}")
        except SystemExit as e:
            say(f"  static FC {at}/{co}: {e}")
    # THE STATE TREES' OWN POLICY, printed whether or not static FC exists.
    # The comparison between the two is what makes bstm and fcm comparable,
    # and this is also the only way to see that a cohort's state tables were
    # built with NO policy -- which decides whether running stage 3.5 for it
    # now would fix the fcm arm or break its agreement with the bstm arm.
    state = tt.censor_policies(outputs, list(tr_co) + list(te_co))
    say("  censor policy of the STATE trees (transitions/latents):")
    for c in list(tr_co) + list(te_co):
        v = state.get(c)
        say(f"      cohort={c:<12} "
            + ("none recorded -- its tables were built WITHOUT a policy, or "
               "predate the stamp" if not v else repr(v)))

    if not seen:
        say(f"  static FC: nothing under {fc_root} -- stage 3b has not run")
        todo.append("static FC: stage 3b has not run")
    else:
        gaps_fc = {}
        say(f"  static FC inputs under {fc_root}:")
        for (at, co), r in sorted(recs.items()):
            say(f"      atlas={at:<14} cohort={co:<12} "
                + ("no provenance columns (predates them)" if r is None else
                   ", ".join(f"{k}={v!r}" for k, v in r.items())))
        for at in sorted({x for x, _ in recs}):
            # EVERY cohort of a side. `_fc_cell` reads one table per cohort
            # and concatenates before filtering by task, so one absent cohort
            # skips every cell of that side -- which is exactly how a job
            # dropped all six (atlas, task) cells over a missing camcan_rest
            # while camcan's own table sat right there.
            missing = {s_: [c for c in co if (at, c) not in recs]
                       for s_, co in (("train", tr_co), ("test", te_co))}
            if any(missing.values()):
                for s_, co in missing.items():
                    if co:
                        say(f"      atlas={at}: the {s_} side is MISSING {co}"
                            f" -- every cell of that side skips, movie as well "
                            f"as rest")
                        todo.append(f"static FC atlas={at}: {s_} side missing {co}")
                        for c in co:
                            gaps_fc.setdefault((s_, c), set()).add(at)
                continue
            # lists, not sets: a recipe is a dict and a dict is unhashable
            tr = [recs[(at, c)] for c in tr_co if (at, c) in recs]
            te = [recs[(at, c)] for c in te_co if (at, c) in recs]
            # NOT `a, b`: `a` is this function's argparse namespace, and
            # rebinding it here made `a.target` twenty lines below raise
            # AttributeError on a dict. One-letter names do not get to shadow
            # a parameter.
            # each side's static FC against its own state trees
            for s_, co in (("train", tr_co), ("test", te_co)):
                for c in co:
                    r = recs.get((at, c))
                    if not r or "censor_policy" not in r:
                        continue
                    fc_p, st_p = (r["censor_policy"] or None), state.get(c)
                    if st_p is not None and fc_p != st_p:
                        say(f"      atlas={at} cohort={c}: static FC says "
                            f"censor_policy={fc_p!r} and its STATE tables say "
                            f"{st_p!r}. bstm and fcm would differ in frames as "
                            f"well as in model; the fcm transfer refuses this.")
                        todo.append(f"static FC {at}/{c}: disagrees with its "
                                    f"state trees")
            rec_tr, rec_te = tr[0], te[0]
            if rec_tr is not None and rec_te is not None and rec_tr != rec_te:
                diff = {k: (rec_tr.get(k), rec_te.get(k)) for k in rec_tr
                        if rec_tr.get(k) != rec_te.get(k)}
                say(f"      atlas={at}: the two sides DISAGREE -- "
                    + "; ".join(f"{k}: train {x!r} test {y!r}"
                                for k, (x, y) in diff.items())
                    + ". The fcm transfer refuses this: a difference between "
                      "the two would be a difference in frames, not in cohort.")
                todo.append(f"static FC atlas={at}: the two sides disagree")

        # The command, BUILT -- not a line with a placeholder in it. Every
        # flag comes from a cohort of the SAME side that already has a table,
        # so the new one reproduces its sibling's frames exactly, which is the
        # only version of this command worth printing.
        for (s_, c), ats in sorted(gaps_fc.items()):
            sib = next((r for (a_, c_), r in sorted(recs.items())
                        if r and c_ in (tr_co if s_ == "train" else te_co)), None)
            flags = []
            if sib:
                if sib.get("censor_policy"):
                    flags += ["--censor-policy", str(sib["censor_policy"])]
                if sib.get("match_bandpass"):
                    try:
                        lo, hi = json.loads(str(sib["match_bandpass"]))
                        flags += ["--match-bandpass", f"{lo:g}", f"{hi:g}"]
                    except Exception:
                        pass
                if sib.get("zscore_runs") is False:
                    flags += ["--no-zscore-runs"]
                if sib.get("pool"):
                    flags += ["--pool", str(sib["pool"])]
            say(f"    run, with the frames its own side already uses:")
            say(f"      sbatch slurm/static_fc.sbatch {c} -- --atlas "
                + " ".join(sorted(ats)) + (" " + " ".join(flags) if flags else "")
                + ("" if sib else
                   "   # NOTE no sibling table to copy the frame settings from"))

    if todo:
        say("\nNOT finished: " + "; ".join(todo))
        say(f"  Resubmitting is the same command -- a shard whose rows.parquet "
            f"exists is skipped, so only the gaps run:")
        say(f"  TARGETS={a.target} ./slurm/transfer_grid.sh")
    else:
        say(f"\ntargets={a.target}: finished. Read it in notebook 13 section 2.")
    return 0


def require_runtime() -> None:
    """Refuse on the login node's bare python, by name.

    `$RUN` is only set by `source slurm/env.sh`; unset, `$RUN python3 ...`
    silently runs the LOGIN NODE's python, which has no pandas -- so the
    failure is `ModuleNotFoundError: No module named 'pandas'` from a command
    that looks right and worked an hour ago in another shell. env.sh's own
    comments warn about this; a tool that needs the container can say it.
    """
    import importlib.util

    missing = [m for m in ("pandas", "pyarrow", "sklearn", "scipy")
               if importlib.util.find_spec(m) is None]
    if not missing:
        return
    raise SystemExit(
        f"cannot import {', '.join(missing)} with {sys.executable}.\n"
        f"  If that path is not inside a container, this is the login node's "
        f"python and `$RUN` was empty -- it is set by sourcing the env file, "
        f"not by being in the repo:\n"
        f"    source slurm/env.sh\n"
        f"    $RUN python3 tools/transfer_grid.py ...\n"
        f"  An unset $RUN does not fail; it runs `python3` alone.")


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    require_runtime()

    import pandas as pd
    import transfer_config as tc
    import transfer_tables as tt

    t0 = time.time()
    root_out = Path(a.out) if a.out else None
    if a.status:
        return status(a, root_out)
    train, test, train_cfg, test_cfg = tc.studies(
        a.target, models=a.models, root=a.outputs, camcan=a.camcan_pheno,
        hcp_pheno=a.hcp_pheno)
    root_out = root_out or (train_cfg.root / "transferred_models")
    import numpy, sklearn
    say(f"target={a.target} tree={a.tree} shard={a.shard}/{a.n_shards} "
        f"threads={a.threads}")
    # The caps AS THE PROCESS SEES THEM. Printed because the export in the job
    # script is not what python gets: `apptainer exec --cleanenv` drops a plain
    # export, so this line is the only place the two can be compared.
    say("  env " + "  ".join(
        f"{k}={os.environ.get(k, 'UNSET')}"
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")))
    say(f"  numpy {numpy.__version__}  pandas {pd.__version__}  "
        f"sklearn {sklearn.__version__}")
    use = tt.usable_threads(a.threads)
    say(f"  threads requested {a.threads}, usable {use}"
        + ("" if use >= a.threads else
           f"  -- CLAMPED. This process started with {use}, and raising a BLAS "
           f"thread count mid-run is what segfaulted the first two arrays. To "
           f"give the fits more, set APPTAINERENV_OPENBLAS_NUM_THREADS before "
           f"apptainer starts; a plain export does not survive --cleanenv."))
    # ROWS IN THE PHENOTYPE FILE, not the number fitted. HCP's table carries
    # all 1,206 S1200 subjects and only the 7T ones have transitions; the n
    # that matters is per cell and appears on the [i/N] lines below.
    say(f"phenotype rows: train {len(train.pheno):,}, test {len(test.pheno):,}"
        f"  (the FITTED n is per state set, on the lines below)")

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
    # NOT before --dry-run. A pre-flight on a tree whose shard 0 is already
    # written took this path and returned, so it printed "exists, skipping"
    # and said nothing at all about the grid it was asked to check -- the one
    # job of a pre-flight, skipped silently.
    if path.exists() and not a.overwrite and not (a.dry_run or a.no_write):
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
        if a.tree == "fcm":
            # the fcm grid is atlases x tasks x targets x 2 arms x models, and
            # the atlases are the ones BOTH cohorts have a static_fc table for
            pres = {}
            for q in (train_cfg.root / "static_fc").rglob("subjects.parquet"):
                pres.setdefault(tt.hive(q, "atlas"), set()).add(tt.hive(q, "cohort"))
            # ALL cohorts of each side, as `fcm_transfer_rows` requires:
            # `_fc_cell` reads one table per cohort and concatenates before
            # filtering by task, so ANY-of-them said three atlases were ready
            # while the real run refused all three.
            need = set(train_cfg.cohorts) | set(test_cfg.cohorts)
            both = sorted(a_ for a_, v in pres.items() if need <= v)
            say(f"dry run: static FC on {len(both)} atlas(es) both cohorts have"
                f" {both} x 2 task(s) x 2 target(s) x 2 arm(s) x {n_mod} "
                f"model(s) = {len(both) * 2 * 2 * 2 * n_mod:,} fit(s)")
            if not both:
                short = {a_: sorted(need - v) for a_, v in pres.items()}
                for a_, co in sorted(short.items()):
                    say(f"  atlas={a_}: missing {co}")
                say(f"  NONE: the static_fc tree has "
                    + ("; ".join(f"atlas={k}: {sorted(v)}"
                                 for k, v in sorted(pres.items())) or "nothing")
                    + f", and this needs an atlas held by BOTH "
                      f"{list(train_cfg.cohorts)} and {list(test_cfg.cohorts)}. "
                      f"Stage 3b -- slurm/static_fc.sbatch -- has not run for "
                      f"one of them.")
        else:
            say(f"dry run: {len(mine)} set(s) x 2 task(s) x 2 target(s) x "
                f"{n_arm} arm(s) x {n_mod} model(s) = "
                f"{len(mine) * 4 * n_arm * n_mod:,} fit(s)")
        say(f"  would write {path}"
            + ("   (already there; the real run would skip it)"
               if path.exists() and not a.overwrite else ""))
        return 0

    if a.tree == "fcm":
        rows = tt.fcm_transfer_rows(
            train, test, train_tasks=dict(train_cfg.tasks),
            test_tasks=dict(test_cfg.tasks),
            covariates=["Age", "Sex", "frac_good_frames", "n_tr_used"],
            models=a.models, max_missing=a.max_missing, seed=a.seed,
            n_boot=a.n_boot, threads=a.threads,
            allow_frame_mismatch=a.allow_frame_mismatch, log=say)
    else:
        rows = tt.transfer_rows(train, test, arms=a.arms, models=a.models,
                                sets=mine, seed=a.seed, n_boot=a.n_boot,
                                threads=a.threads, log=say)
    if a.no_write:
        say(f"{len(rows):,} row(s) scored, NOT written (--no-write). "
            f"{rows['over_base'].mean():.0%} above their transferred baseline; "
            f"best {rows['score'].max():+.3f}")
        say(f"done in {time.time() - t0:.0f}s")
        return 0
    out = tt.write_transferred(rows, root_out, train="camcan", test="hcp7t_fix",
                               target_set=a.target, tree=a.tree, shard=a.shard,
                               n_shards=a.n_shards, **prov)
    say(f"{len(rows):,} row(s) -> {out}")
    say(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
