#!/usr/bin/env python3
"""Remove named state-label columns from the latents, and their transitions.

    python tools/drop_state_columns.py --states HMM_pca3_8 HMM_pca3_27
    python tools/drop_state_columns.py --states HMM_pca3_8 HMM_pca3_27 --apply

WHY THIS EXISTS, AND WHY DELETING THE TRANSITIONS IS NOT ENOUGH
---------------------------------------------------------------
A state set that only some cells carry is a permanent nuisance: `transitions
--check` reports every cell that lacks it ("stage 4b ran unevenly"), and
`bstm_selection` ranks it for the conditions that have it, so one condition's
summary.csv gets rows the other cannot.

The obvious move -- `rm -rf outputs/transitions/.../states=X` -- does not hold.
`transitions` builds its plan from the state COLUMNS inside the latents files,
so the next run regenerates exactly what was deleted. The columns are the
source; the transitions tables are derived. So this removes the columns first
and the derived tables second.

WHAT IT PRESERVES
-----------------
Everything else in the file: the rows, the embedding columns, every OTHER state
column, and the schema metadata verbatim -- `model_hash`, `stride_s`,
`censor_policy`, and the remaining per-column `clusterers` entries. Only the
named columns and their own `clusterers` keys go. A latents file that loses its
provenance is worse than one with an extra column in it.

Dry-run by default. It reports every file and directory it would touch, and
names any state set it could not find, because a typo that silently matches
nothing is the failure mode here.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

LEAF = "data.parquet"


def latents_files(root: Path) -> list[Path]:
    """Every latents leaf, at whatever depth the tree has."""
    base = root / "latents"
    return sorted(base.rglob(LEAF)) if base.is_dir() else []


def columns_in(path: Path) -> tuple[list[str], dict]:
    """(column names, the clusterers provenance block) of one latents file."""
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(path).schema_arrow
    md = schema.metadata or {}
    block = {}
    if b"clusterers" in md:
        try:
            block = json.loads(md[b"clusterers"].decode())
        except ValueError:
            block = {}
    return list(schema.names), block


def plan(root: Path, states: list[str]) -> tuple[dict, list[str]]:
    """{file: [state sets it carries]}, plus the sets found nowhere."""
    hits, seen = {}, set()
    for f in latents_files(root):
        cols, block = columns_in(f)
        here = [s for s in states if s in cols or s in block]
        if here:
            hits[f] = here
            seen.update(here)
    return hits, [s for s in states if s not in seen]


def drop_from(path: Path, states: list[str]) -> None:
    """Rewrite one file without `states`, carrying everything else over."""
    import pyarrow.parquet as pq

    # ParquetFile, not read_table: a latents path ends in cohort=/task=, and
    # the file has `cohort` and `task` COLUMNS, which the dataset layer tries
    # to merge with the keys it infers from the path. See io.read_file.
    table = pq.ParquetFile(path).read()
    md = dict(table.schema.metadata or {})

    block = {}
    if b"clusterers" in md:
        try:
            block = json.loads(md[b"clusterers"].decode())
        except ValueError:
            block = {}
    for s in states:
        block.pop(s, None)
    if block or b"clusterers" in md:
        md[b"clusterers"] = json.dumps(block).encode()

    keep = [c for c in table.column_names if c not in states]
    out = table.select(keep).replace_schema_metadata(md)
    tmp = path.with_name(f"{path.name}.tmp")
    pq.write_table(out, tmp, compression="zstd")
    tmp.replace(path)


def transitions_dirs(root: Path, states: list[str]) -> list[Path]:
    """The derived tables for these state sets, at whatever depth."""
    base = root / "transitions"
    if not base.is_dir():
        return []
    want = {f"states={s}" for s in states}
    return sorted(p for p in base.rglob("states=*")
                  if p.is_dir() and p.name in want)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", nargs="+", required=True, metavar="NAME",
                   help="state-set column names, exactly as they appear "
                        "(e.g. HMM_pca3_8). No globbing: a wildcard that "
                        "matched HMM1_* as well would delete the current "
                        "definitions along with the legacy ones.")
    p.add_argument("--output-root", default=None)
    p.add_argument("--keep-transitions", action="store_true",
                   help="drop the columns but leave the derived transitions "
                        "tables. Rarely what you want -- they then describe "
                        "state sets that no longer exist.")
    p.add_argument("--apply", action="store_true",
                   help="actually do it. Without this, only reports.")
    a = p.parse_args(argv)

    if a.output_root:
        root = Path(a.output_root)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from fmri_decomposition.io import default_output_root
        root = default_output_root()
    if not root.is_dir():
        raise SystemExit(f"output_root does not exist: {root}")
    print(f"output_root {root}")

    hits, missing = plan(root, a.states)
    for s in missing:
        print(f"  NOT FOUND in any latents file: {s}")
    if missing and not hits:
        raise SystemExit(
            "none of the named state sets exist -- check the spelling against\n"
            "  the `clusterers` block rather than guessing, since a name that\n"
            "  matches nothing would otherwise read as a successful cleanup.")

    for f, here in hits.items():
        print(f"  {f.relative_to(root)}: {', '.join(here)}")
    dirs = [] if a.keep_transitions else transitions_dirs(root, a.states)
    for d in dirs:
        print(f"  rm -r {d.relative_to(root)}")

    print(f"\n{len(hits)} latents file(s), "
          f"{sum(len(v) for v in hits.values())} column(s), "
          f"{len(dirs)} transitions directory(ies)"
          + ("" if a.apply else "  (--apply to do it)"))
    if not a.apply:
        return 0

    for f, here in hits.items():
        drop_from(f, here)
    for d in dirs:
        shutil.rmtree(d)
    print("dropped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
