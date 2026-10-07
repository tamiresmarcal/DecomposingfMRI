#!/usr/bin/env python3
"""What is actually on disk, read from the tree rather than from memory.

    fmri-decomp status
    fmri-decomp status --atlas yeo7 --window-s 30 -1

Every other `--check` answers "can the next stage run?" for one stage. This
answers "where is the whole pipeline?", by walking the output tree and reading
only parquet FOOTERS -- no row group is touched, so it is seconds on a tree with
thousands of shards and safe to run while jobs are writing.

It is READ-ONLY and takes no decisions. Its job is to make a disagreement
visible: the same state set with two different `model_hash`es across cohorts,
one aperture censored and another not, a state column at a K nothing downstream
will use. Those are the failures that look like success until much later.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .io import RAW_PREFIX, STATE_COLUMN_RE, STATE_K_BAND

STATE_RE = STATE_COLUMN_RE
EMBEDDINGS = {"pca3": "pca0/3", "umap3": "umap0/3"}


def _embeddings_present(cols) -> list[str]:
    """Which embeddings this file offers, raw<N> included.

    pca3/umap3 are a fixed column list, so membership is a lookup. The raw
    embedding is named by HOW MANY `raw/<name>` columns there are, which only
    the file knows -- so it is counted here rather than looked up.
    """
    have = [e for e, c in EMBEDDINGS.items() if c in cols]
    n_raw = sum(1 for c in cols if c.startswith(RAW_PREFIX))
    if n_raw:
        have.append(f"raw{n_raw}")
    return have


def _meta(path: Path, key: str):
    import pyarrow.parquet as pq

    md = pq.ParquetFile(path).schema_arrow.metadata or {}
    raw = md.get(key.encode())
    if raw is None:
        return None
    try:
        return json.loads(raw.decode())
    except Exception:                                            # noqa: BLE001
        return raw.decode()


def _footer(path: Path):
    """(n_rows, column names) from the footer alone."""
    import pyarrow.parquet as pq

    f = pq.ParquetFile(path)
    return f.metadata.num_rows, list(f.schema_arrow.names)


def _key(p: Path, name: str):
    for part in p.parts:
        if part.startswith(f"{name}="):
            return part.split("=", 1)[1]
    return None


# ------------------------------------------------------------------ stages ---
def activation(root: Path) -> pd.DataFrame:
    rows = []
    for d in sorted((root / "activation").glob("atlas=*/cohort=*")):
        rows.append({"atlas": _key(d, "atlas"), "cohort": _key(d, "cohort"),
                     "shards": sum(1 for _ in d.rglob("*.parquet"))})
    return pd.DataFrame(rows)


def dfc(root: Path) -> pd.DataFrame:
    rows = []
    for d in sorted((root / "dfc").glob("atlas=*/window_s=*/cohort=*")):
        rows.append({"atlas": _key(d, "atlas"), "window_s": _key(d, "window_s"),
                     "cohort": _key(d, "cohort"),
                     "shards": sum(1 for _ in d.rglob("*.parquet"))})
    return pd.DataFrame(rows)


def censor(root: Path) -> pd.DataFrame:
    rows = []
    for d in sorted((root / "censor").glob("policy=*")):
        # The window gate is written only for the (atlas, aperture) `censor` was
        # given --stage dfc for. Subjects-only is the normal first-pass state, so
        # this reports which it is rather than treating absence as a fault.
        gates = sorted({f"{_key(q, 'atlas')}/{_key(q, 'window_s')}s"
                        for q in d.glob(
                            "atlas=*/window_s=*/cohort=*/windows.parquet")})
        for sp in sorted(d.glob("cohort=*/subjects.parquet")):
            try:
                t = pd.read_parquet(sp, columns=["keep"])
                kept, total = int(t["keep"].sum()), len(t)
            except Exception as exc:                             # noqa: BLE001
                kept, total = -1, -1
                print(f"  (could not read {sp}: {type(exc).__name__})")
            rows.append({"policy": _key(d, "policy"),
                         "cohort": _key(sp, "cohort"),
                         "subjects_kept": kept, "of": total,
                         "window_gate": ",".join(gates) or "subjects only"})
    return pd.DataFrame(rows)


def latents(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((root / "latents").glob("atlas=*/window_s=*/cohort=*/data.parquet")):
        n, cols = _footer(p)
        states = sorted(c for c in cols if STATE_RE.match(c))
        rows.append({
            "atlas": _key(p, "atlas"), "window_s": _key(p, "window_s"),
            "cohort": _key(p, "cohort"), "rows": n,
            "emb": ",".join(_embeddings_present(cols)) or "-",
            "censor": str(_meta(p, "censor_policy")),
            "source": str(_meta(p, "source") or "dfc"),
            "role": str(_meta(p, "role")),
            "model_hash": str(_meta(p, "model_hash")),
            "n_states": len(states), "states": states})
    return pd.DataFrame(rows)


def transitions(root: Path, deep: bool = True) -> pd.DataFrame:
    """One row per table. `rows` is (task, sub) pairs; `subs` is PEOPLE.

    The two are not the same and the difference matters: cneuromod is 5 people
    across ~48 episodes, so a table with 240 rows has 48 correlated rows per
    person. Stage 6 treats a row as an independent observation, which is true
    for camcan (one task) and false for cneuromod. Reporting only the row count
    would hide that, so `subs` is read -- one small column, the only place this
    command touches a row group.
    """
    rows = []
    for p in sorted((root / "transitions").rglob("subjects.parquet")):
        n, cols = _footer(p)
        subs = None
        if deep:
            try:
                from .io import read_file
                subs = read_file(p, ["sub"]).to_pandas()["sub"].nunique()
            except Exception:                                    # noqa: BLE001
                subs = None
        rows.append({"atlas": _key(p, "atlas"), "window_s": _key(p, "window_s"),
                     "states": _key(p, "states"), "cohort": _key(p, "cohort"),
                     "rows": n, "subs": subs,
                     "cells": sum(1 for c in cols if "->" in c)})
    return pd.DataFrame(rows)


def selection(root: Path) -> pd.DataFrame:
    rows = []
    for d in sorted((root / "bstm_selection").glob("target=*")):
        sc = d / "scores.parquet"
        rows.append({"target": _key(d, "target"),
                     "scores": sc.exists(),
                     "n_rows": _footer(sc)[0] if sc.exists() else 0,
                     "design": (d / "DESIGN.md").exists(),
                     "figures": sum(1 for _ in (d / "figures").glob("*.png"))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- problems ---
def stale(lat: pd.DataFrame, tr: pd.DataFrame) -> list[str]:
    """Transition tables for a state set the latents no longer carry.

    This is the condition that made the whole command worth writing. Nothing
    else notices it: the tables are well-formed, they just describe states that
    no longer exist. Two ways in, both ordinary:

      * Stage 3 re-run after stage 4. It is the only stage that REWRITES a
        latents file wholesale, so it drops every state column stage 4 added.
      * A K-free method finding a different K. `MeanShift_pca3_7` becomes
        `MeanShift_pca3_4` after a bandwidth change, and the old table keeps its
        own directory under `states=MeanShift_pca3_7`.
    """
    if lat.empty or tr.empty:
        return []
    out = []
    have = {(a, w): set().union(*(set(x) for x in g["states"]))
            for (a, w), g in lat.groupby(["atlas", "window_s"])}
    for (a, w), g in tr.groupby(["atlas", "window_s"]):
        gone = sorted(set(g["states"]) - have.get((a, w), set()))
        if gone:
            out.append(f"{a} {w}s: STALE transition table(s) for state set(s) "
                       f"the latents no longer have: {gone}. Either stage 3 was "
                       f"re-run after stage 4, or a discovered K changed. Stage "
                       f"6 reads whatever tables it finds, so delete these or "
                       f"re-run stage 5 before trusting a selection.")
    return out


def behind(lat: pd.DataFrame, tr: pd.DataFrame, min_k: int, max_k: int
           ) -> list[str]:
    """State sets that exist in the latents with no transition table.

    The mirror of `stale`, and the more common state: stage 4b is cheap and gets
    re-run, stage 5a is a separate job and gets forgotten. Stage 6 then selects
    over whatever subset of the state sets happens to have tables, and reports a
    winner without ever mentioning the ones it never saw.

    A state set OUTSIDE the K band does NOT count as behind. Stage 5a skips those
    deliberately and says so, so flagging them again here reports a correctly
    working pipeline as broken -- 12 of the 37 problems on the first clean run
    were this, every one an orphan column that is supposed to have no table. A
    check that cries wolf on the normal state teaches people to skim past it,
    which is the opposite of what it is for.
    """
    if lat.empty:
        return []
    out = []
    done = {(a, w): set(g["states"]) for (a, w), g in
            (tr.groupby(["atlas", "window_s"]) if not tr.empty else [])}
    for (a, w), g in lat.groupby(["atlas", "window_s"]):
        have = set().union(*(set(x) for x in g["states"])) if len(g) else set()
        have = {st for st in have if min_k <= _k_from_name(st) <= max_k}
        todo = sorted(have - done.get((a, w), set()))
        if todo:
            out.append(f"{a} {w}s: {len(todo)} state set(s) in the latents have "
                       f"NO transition table -- stage 5 is behind stage 4 here. "
                       f"Stage 6 would pick a winner without ever seeing them: "
                       f"{todo}")
    return out


def _k_from_name(state: str) -> int:
    """K out of `<Method>_<embedding>_<K>`. An unparseable name counts as usable,
    so a naming convention that changes cannot silently hide state sets."""
    tail = state.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 1 << 30



def shard_gap(act: pd.DataFrame, dfcs: pd.DataFrame) -> list[str]:
    """A cohort with activation shards but fewer dfc shards.

    Stage 3 reads stage 2's output one shard at a time and tolerates a failure,
    so a run that produced no windows -- too short for the aperture, or a read
    error -- just leaves a gap. Nothing downstream can tell a missing subject
    from one that never existed.
    """
    if act.empty or dfcs.empty:
        return []
    out = []
    a = act.set_index(["atlas", "cohort"])["shards"]
    for (atlas, w, cohort), n in dfcs.set_index(
            ["atlas", "window_s", "cohort"])["shards"].items():
        have = a.get((atlas, cohort))
        if have is not None and 0 < n < have:
            out.append(f"{atlas} {w}s {cohort}: {n} dfc shard(s) from {have} "
                       f"activation shard(s) -- {have - n} subject-task(s) "
                       f"produced no windows at this aperture.")
    return out


def problems(lat: pd.DataFrame, args) -> list[str]:
    """Disagreements that no single stage's --check would surface.

    Each one is a thing that looks like success: the files exist, the columns are
    there, and the analysis is quietly not the one anybody intended.
    """
    out = []
    if lat.empty:
        return ["no latents at all -- stage 3 has not run"]

    for (atlas, w), g in lat.groupby(["atlas", "window_s"]):
        if g["model_hash"].nunique() > 1:
            out.append(f"{atlas} {w}s: cohorts came from DIFFERENT fits "
                       f"({sorted(g['model_hash'].unique())}). State 5 is not "
                       f"the same state in two of them; nothing downstream can "
                       f"pool them.")
        if g["censor"].nunique() > 1:
            out.append(f"{atlas} {w}s: cohorts disagree on the censor policy "
                       f"({sorted(g['censor'].unique())}).")
        miss = [e for e in args.embeddings if e not in set(
            ",".join(g["emb"]).split(","))]
        if miss:
            out.append(f"{atlas} {w}s: embedding(s) {miss} absent -- stage 4 "
                       f"will refuse the state sets built on them.")
        if (g["n_states"] == 0).any():
            out.append(f"{atlas} {w}s: no state columns -- stage 4 has not run "
                       f"for it.")

    pol = set(lat["censor"].unique())
    if len(pol) > 1:
        out.append(f"THE GRID SPANS {len(pol)} CENSOR POLICIES {sorted(pol)}. "
                   f"One aperture censored and another not is not a fair "
                   f"comparison of apertures.")

    # State sets at a K nothing downstream will use.
    bad_k = sorted({s for row in lat["states"] for s in row
                    if not args.min_k <= int(s.rsplit("_", 1)[1]) <= args.max_k})
    if bad_k:
        out.append(f"state column(s) at a K outside [{args.min_k}, {args.max_k}], "
                   f"skipped by stage 5: {bad_k}")
    return out


# --------------------------------------------------------------------- run ---
def _show(title: str, df: pd.DataFrame, cols=None) -> None:
    print(f"\n{title}")
    if df.empty:
        print("  (nothing)")
        return
    d = df if cols is None else df[cols]
    print("  " + d.to_string(index=False).replace("\n", "\n  "))


def run(args) -> int:
    root = Path(args.output_root) if args.output_root else _default_root()
    if not root.is_dir():
        raise SystemExit(f"output_root does not exist: {root}")
    print(f"output_root {root}")

    act, dfcs, cen = activation(root), dfc(root), censor(root)
    lat, tr, sel = (latents(root), transitions(root, deep=not args.shallow),
                    selection(root))

    def narrow(d):
        if d.empty:
            return d
        if args.atlas and "atlas" in d:
            d = d[d["atlas"].isin(args.atlas)]
        if args.window_s and "window_s" in d:
            d = d[d["window_s"].isin([str(w) for w in args.window_s])]
        return d

    dfcs, lat, tr = narrow(dfcs), narrow(lat), narrow(tr)

    _show("1  ACTIVATION   shards per cohort", act)
    _show("2  DFC          shards per aperture",
          dfcs.pivot_table(index=["atlas", "window_s"], columns="cohort",
                           values="shards", fill_value=0).reset_index()
          if not dfcs.empty else dfcs)
    _show("2.5 CENSOR      subjects kept per policy", cen)
    _show("3  LATENTS      embeddings per aperture", lat,
          ["atlas", "window_s", "cohort", "rows", "emb", "source", "censor",
           "role", "model_hash", "n_states"])

    print("\n4  STATE SETS   per aperture (K in the name)")
    if lat.empty:
        print("  (nothing)")
    else:
        for (atlas, w), g in lat.groupby(["atlas", "window_s"]):
            per = {c: sorted(set(x)) for c, x in
                   zip(g["cohort"], g["states"])}
            common = sorted(set.intersection(*(set(v) for v in per.values()))
                            if per else [])
            odd = {c: sorted(set(v) - set(common)) for c, v in per.items()
                   if set(v) - set(common)}
            print(f"  {atlas:<14} {w:>4}  {len(common)} shared: {common}")
            if odd:
                print(f"  {'':<14} {'':>4}  NOT in every cohort: {odd}")

    _show("5  TRANSITIONS  tables written  (rows = task x sub; subs = people)",
          tr)
    _show("6  SELECTION    targets", sel)

    probs = (problems(lat, args) + stale(lat, tr)
             + behind(lat, tr, args.min_k, args.max_k)
             + shard_gap(act, dfcs))
    print(f"\n{'=' * 70}")
    if probs:
        print(f"{len(probs)} PROBLEM(S)")
        for i, p_ in enumerate(probs, 1):
            print(f"  {i}. {p_}")
    else:
        print("no disagreement found between cohorts or apertures")
    return 1 if probs else 0


def _default_root() -> Path:
    if os.environ.get("FMRIDECOMP_OUTPUTS"):
        return Path(os.environ["FMRIDECOMP_OUTPUTS"])
    import yaml
    repo = Path(__file__).resolve().parent.parent
    return Path(yaml.safe_load(
        (repo / "config" / "camcan_movie.yaml").read_text())["output_root"])


def add_arguments(p) -> None:
    p.add_argument("--atlas", nargs="*", default=None)
    p.add_argument("--window-s", nargs="*", default=None)
    p.add_argument("--embeddings", nargs="+", default=["pca3", "umap3"],
                   help="embeddings the plan needs; a missing one is a problem")
    p.add_argument("--min-k", type=int, default=STATE_K_BAND[0])
    p.add_argument("--max-k", type=int, default=STATE_K_BAND[1])
    p.add_argument("--shallow", action="store_true",
                   help="skip the one column read that counts distinct PEOPLE "
                        "per transition table; footers only")
    p.add_argument("--output-root")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
