#!/usr/bin/env python3
"""Every number the Methods still has in red, read from what the pipeline wrote.

    source slurm/env.sh
    apptainer exec --bind /project,/scratch,/home --pwd $PWD $FMRIDECOMP_SIF45 \
        python3 tools/methods_numbers.py --states HMM2_umap3_8

Read-only: nothing under outputs/ is written or moved. Each block is
independent, so a missing tree prints why and the rest still runs.

What it answers (the Methods placeholder in brackets):

  1. the fit       train cohorts, censor policy, UMAP subsample, seed  [C1]
  2. the policy    the thresholds of that censor policy, and whether
                   the ISC lag was used to exclude anyone              [C1, C2]
  3. the samples   subjects and windows per cohort and condition,
                   after QC; training windows and participants          [B1, B2]
  4. the PCA       variance explained by the 3-component PCA            [B3]
  5. Cam-CAN       subjects with transitions AND a HADS score           [B1]
  6. maps          which spatial null the contextualisation used        [C3]
  7. models        which targets (raw / category) were run              [C4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd
import pyarrow.parquet as pq

from fmri_decomposition.io import (default_output_root, latents_root,
                                   meta_dir, transitions_root, tree_root)


def header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def guarded(fn):
    """Run one block; a missing tree is reported, not fatal."""
    def run(*a, **kw):
        try:
            fn(*a, **kw)
        except (SystemExit, FileNotFoundError, KeyError, ValueError,
                OSError, ImportError) as e:
            print(f"  !! not available: {type(e).__name__}: "
                  f"{str(e).splitlines()[0] if str(e) else ''}")
    return run


def latents_files(root, atlas, window_s, version):
    return sorted(latents_root(root, atlas, window_s, version=version)
                  .glob("cohort=*/task=*/data.parquet"))


def hive(path: Path, key: str) -> str:
    return next((p.split("=", 1)[1] for p in path.parts
                 if p.startswith(f"{key}=")), "")


@guarded
def the_fit(root, a):
    header("1. THE FIT (decompose)  -> Methods 3.3, censor policy [C1]")
    from fmri_decomposition.unsupervised_model.decompose import (
        latents_fit_description)
    files = latents_files(root, a.atlas, a.window_s, a.version)
    if not files:
        raise FileNotFoundError(f"no latents for atlas={a.atlas} "
                                f"window_s={a.window_s}")
    meta, states = latents_fit_description(files[0])
    for k in ("train_cohorts", "censor_policy", "umap_fit_rows", "seed",
              "n_latents"):
        print(f"  {k:16s} {meta.get(k)}")
    print(f"  state columns    {', '.join(states)}")
    if a.states not in states:
        print(f"  !! {a.states} is not among them")
    return meta


@guarded
def the_policy(root, a, policy):
    header(f"2. CENSOR POLICY '{policy}'  -> QC criteria [C1], ISC [C2]")
    if not policy:
        print("  the fit used NO censor policy (every frame and window kept)")
        return
    f = meta_dir(root, "extractions") / "censor" / f"policy={policy}.json"
    s = json.loads(f.read_text())
    pol = s.get("policy", s)
    print("  thresholds:")
    print("   ", json.dumps({k: pol[k] for k in ("subject", "window")
                             if k in pol}, indent=2).replace("\n", "\n    "))
    lag = (pol.get("subject") or {}).get("max_abs_best_lag_tr")
    print(f"  ISC lag used to exclude subjects: "
          f"{'NO (max_abs_best_lag_tr is null)' if lag is None else f'YES, |lag| > {lag} TR'}")
    for cohort, c in (s.get("cohorts") or {}).items():
        keys = {k: v for k, v in c.items()
                if any(x in k for x in ("kept", "rows", "subjects", "n_"))}
        print(f"  {cohort:12s} {keys}")


@guarded
def the_samples(root, a):
    header("3. SAMPLES after QC (from the latents)  -> n per cohort [B1], "
           "training windows [B2]")
    rows = []
    for f in latents_files(root, a.atlas, a.window_s, a.version):
        cols = [c for c in ("sub", "role", "window_id")
                if c in pq.read_schema(f).names]
        t = pd.read_parquet(f, columns=cols)
        rows.append(dict(cohort=hive(f, "cohort"), task=hive(f, "task"),
                         role=(t["role"].iloc[0] if "role" in t and len(t)
                               else "?"),
                         subjects=t["sub"].nunique(), windows=len(t)))
    if not rows:
        raise FileNotFoundError(f"no latents for atlas={a.atlas} "
                                f"window_s={a.window_s}")
    d = pd.DataFrame(rows)
    print(d.to_string(index=False))
    print("\n  per cohort (any condition):")
    subs = {}
    for f in latents_files(root, a.atlas, a.window_s, a.version):
        subs.setdefault(hive(f, "cohort"), set()).update(
            pd.read_parquet(f, columns=["sub"])["sub"].astype(str))
    for c, s in sorted(subs.items()):
        print(f"    {c:12s} {len(s)} participants")
    tr = d[d["role"] == "train"]
    train_cohorts = sorted(tr["cohort"].unique())
    n_train_subs = sum(len(subs[c]) for c in train_cohorts)
    print(f"\n  TRAINING: {int(tr['windows'].sum()):,} windows from "
          f"{n_train_subs} participants ({', '.join(train_cohorts)})")


@guarded
def the_pca(root, a):
    header("4. PCA  -> variance explained by 3 components [B3]")
    from fmri_decomposition.unsupervised_model.state_means import load_models
    m = load_models(root, a.atlas, a.window_s)
    pca = m["pca"][3]
    evr = pca.explained_variance_ratio_
    print(f"  per component: {', '.join(f'{v:.3%}' for v in evr)}")
    print(f"  total (3 PCs): {evr.sum():.2%}")


@guarded
def the_transitions(root, a):
    header(f"5a. TRANSITION TABLES for {a.states}  -> subjects per cohort/task")
    base = transitions_root(root, a.atlas, a.window_s, a.states,
                            version=a.version)
    rows = []
    for f in sorted(base.rglob("subjects.parquet")):
        t = pd.read_parquet(f, columns=["sub", "task"])
        for task, g in t.groupby("task"):
            rows.append(dict(cohort=hive(f, "cohort"), task=task,
                             subjects=g["sub"].nunique()))
    if not rows:
        raise FileNotFoundError(f"nothing under {base}")
    print(pd.DataFrame(rows).to_string(index=False))


@guarded
def the_camcan_hads(root, a):
    header("5b. CAM-CAN WITH HADS  -> n in the clinical analyses [B1]")
    from fmri_decomposition.supervised_model import hads as ht
    space = dict(atlas=a.atlas, window_s=a.window_s, states=a.states)
    for target_set, targets in (
            ("category", {"anxiety": "additional_HADS_anx_category",
                          "depression": "additional_HADS_dep_category"}),
            ("raw", {"anxiety": "additional_HADS_anxiety",
                     "depression": "additional_HADS_depression"})):
        try:
            st = ht.Study(ht.Settings(
                root=root, targets=targets, modelling_version=a.version,
                target_range=({k: (0, 21) for k in targets}
                              if target_set == "raw" else None)))
            for task in st.cfg.tasks:
                d = st.table(space, task)
                g = st.groups(d)
                print(f"  {target_set:8s} {task:6s} n={len(d):4d}  "
                      + "  ".join(f"{k}={int(v.sum())}" for k, v in g.items()))
        except (SystemExit, KeyError, ValueError, FileNotFoundError) as e:
            print(f"  {target_set:8s} !! {str(e).splitlines()[0]}")


@guarded
def the_maps(root, a):
    header("6. CONTEXTUALISATION  -> spatial null used [C3]")
    base = tree_root(root, "contextualization", a.version)
    found = sorted(base.rglob("provenance.json"))
    if not found:
        raise FileNotFoundError(f"no provenance.json under {base}")
    for f in found:
        p = json.loads(f.read_text())
        print(f"  {f.relative_to(base)}")
        print(f"     states={p.get('states')}  null={p.get('null')}")


@guarded
def the_models(root, a):
    header("7. SUPERVISED RUNS  -> which targets exist [C4]")
    for tree in ("bstm_selection", "fcm_selection", "transferred_models"):
        base = tree_root(root, tree, a.version)
        if not base.exists():
            print(f"  {tree:20s} (absent)")
            continue
        targets = sorted({hive(p, "target") for p in base.rglob("*")
                          if "target=" in str(p)} - {""})
        tasks = sorted({hive(p, "task") for p in base.rglob("*")
                        if "task=" in str(p)} - {""})
        print(f"  {tree:20s} tasks={tasks}")
        for t in targets:
            print(f"  {'':20s} target={t}")
        if tree == "transferred_models" and not targets:
            print(f"  {'':20s} {sorted(p.name for p in base.iterdir())[:10]}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", default=None)
    p.add_argument("--atlas", default="harvardoxford")
    p.add_argument("--window-s", default="30")
    p.add_argument("--states", default="HMM2_umap3_8")
    p.add_argument("--version", default=None,
                   help="modelling version (default: $FMRIDECOMP_MODELLING_VERSION, else v0)")
    a = p.parse_args(argv)
    root = Path(a.output_root) if a.output_root else default_output_root()
    print(f"outputs: {root}\nspace:   atlas={a.atlas} window_s={a.window_s} "
          f"states={a.states}")

    meta = {}
    try:
        from fmri_decomposition.unsupervised_model.decompose import (
            latents_fit_description)
        files = latents_files(root, a.atlas, a.window_s, a.version)
        meta = latents_fit_description(files[0])[0] if files else {}
    except Exception:                                            # noqa: BLE001
        pass
    the_fit(root, a)
    the_policy(root, a, meta.get("censor_policy"))
    the_samples(root, a)
    the_pca(root, a)
    the_transitions(root, a)
    the_camcan_hads(root, a)
    the_maps(root, a)
    the_models(root, a)


if __name__ == "__main__":
    main()
