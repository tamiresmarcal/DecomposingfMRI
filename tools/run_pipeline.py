#!/usr/bin/env python3
"""The whole grid, from Python, with no SLURM.

    python tools/run_pipeline.py --scenario train+project --dry-run
    python tools/run_pipeline.py --scenario project --cohorts camcan camcan_rest

The point is that there is nothing in slurm/*.sbatch except job directives and
this same CLI. Each sbatch forwards everything after a bare `--` verbatim, so a
laptop, a login node and a notebook all drive the same code with the same flags
-- `decompose` and `cluster` loop over --window-s and --atlas internally, which
is what SLURM's arrays only parallelise.

THE THREE SCENARIOS, which are the whole reason `--train` and `--project` mean
the same thing in both stages:

    train+project   fit on --train, apply to --project
    train           fit on --train, write/label only those  (omit --project)
    project         reuse the fit on disk, write/label only the new cohorts

`--train` is NOT "train now" -- it names the fit. In the `project` scenario it
is what finds the saved one; nothing is refitted, which the log says out loud.

--dry-run prints the commands instead of running them, which is how to read
this file without a cluster under it.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

ATLASES = ["harvardoxford", "yeo7", "networks"]
WINDOWED = ["30", "60", "120", "300"]

# The per-atlas stage-4 flags. Only the ACTIVATION aperture differs by atlas,
# and only because the named-parcel arm needs a lossless rotation: the extra
# --pca-latents is the atlas's own width. harvardoxford has none because a
# full-covariance HMM in 111 dimensions is not estimable -- 6,327 parameters
# per state, one per 4 observations at K=27.
PASSTHROUGH = {"yeo7": 7, "networks": 14}


def cmd(verb: str, *args: str) -> list[str]:
    return [sys.executable, "-m", "fmri_decomposition.cli", verb, *args]


def run(argv: list[str], dry: bool) -> None:
    print("$ " + " ".join(shlex.quote(a) for a in argv))
    if dry:
        return
    subprocess.run(argv, check=True)


def cohort_flags(scenario: str, train: list[str], project: list[str]
                 ) -> list[str]:
    """--train / --project for one scenario, identical for both stages.

    `train` keeps --train in every scenario on purpose. It names the fit, so
    the `project` scenario needs it to FIND the saved one -- and leaving it to
    the default would mean the hash moves the day that default changes, which
    silently refits instead of reusing.
    """
    flags = ["--train", *train]
    if scenario in ("train+project", "project"):
        flags += ["--project", *project]
    return flags


def decompose(scenario, train, project, policy, dry):
    co = cohort_flags(scenario, train, project)
    for atlas in ATLASES:
        run(cmd("decompose", "--atlas", atlas, "--window-s", *WINDOWED,
                *co, "--censor-policy", policy), dry)
    for atlas in ATLASES:
        extra = []
        if atlas in PASSTHROUGH:
            extra = ["--pca-latents", "3", str(PASSTHROUGH[atlas]),
                     "--umap-latents", "3", "--passthrough-features"]
        run(cmd("decompose", "--atlas", atlas, "--window-s", "-1",
                "--source", "activation", *co,
                "--censor-policy", policy, *extra), dry)


def cluster(scenario, train, project, dry, check=False):
    co = cohort_flags(scenario, train, project)
    # One call covers every atlas and every aperture: cluster loops over both.
    # SLURM splits it per cell only to buy wall-clock -- a cell is one parquet
    # file and parquet cannot append, so two writers would rename over each
    # other.
    run(cmd("cluster", "--atlas", *ATLASES,
            "--window-s", *WINDOWED, "-1", *co,
            *(["--check"] if check else [])), dry)


def analyse(cohort, output_name, targets, pheno, dry):
    run(cmd("transitions", "--cohorts", cohort, "--window-s", *WINDOWED, "-1"),
        dry)
    for target in targets:
        run(cmd("select-model", "--target", target, "--cohort", cohort,
                "--output-name", output_name, "--pheno", *pheno), dry)


def fc_arm(cohorts, atlases, targets, pheno, dry):
    run(cmd("static-fc", "--atlas", *atlases, "--cohorts", *cohorts), dry)
    for target in targets:
        run(cmd("select-fcm", "--target", target, "--cohorts", *cohorts,
                "--atlas", *atlases, "--pheno", *pheno), dry)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", required=True,
                   choices=["train+project", "train", "project"])
    p.add_argument("--train", nargs="+", default=["ds002837", "cneuromod"])
    p.add_argument("--project", nargs="+", default=["camcan", "camcan_rest"])
    p.add_argument("--censor-policy", default="motion")
    p.add_argument("--stages", nargs="+",
                   default=["decompose", "cluster"],
                   choices=["decompose", "cluster", "check", "analyse", "fc"])
    p.add_argument("--cohort", default="camcan",
                   help="whose transitions to rank, for --stages analyse")
    p.add_argument("--output-name", default="bstm_selection")
    p.add_argument("--targets", nargs="+",
                   default=["additional_HADS_anx_category",
                            "additional_HADS_dep_category"])
    p.add_argument("--pheno", nargs="+", default=None, metavar="PATH:SEP",
                   help="required for --stages analyse / fc; this script has "
                        "no default path either")
    p.add_argument("--atlas", nargs="+", default=ATLASES)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    if {"analyse", "fc"} & set(a.stages) and not a.pheno:
        raise SystemExit("--pheno is required for --stages analyse / fc")

    print(f"# scenario: {a.scenario}")
    print(f"#   train   {a.train}")
    print(f"#   project {a.project if a.scenario != 'train' else '(none)'}\n")

    if "decompose" in a.stages:
        decompose(a.scenario, a.train, a.project, a.censor_policy, a.dry_run)
    if "check" in a.stages:
        cluster(a.scenario, a.train, a.project, a.dry_run, check=True)
    if "cluster" in a.stages:
        cluster(a.scenario, a.train, a.project, a.dry_run)
    if "analyse" in a.stages:
        analyse(a.cohort, a.output_name, a.targets, a.pheno, a.dry_run)
    if "fc" in a.stages:
        fc_arm(a.project, a.atlas, a.targets, a.pheno, a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
