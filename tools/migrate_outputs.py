"""Move a flat `outputs/` into the three-half shape. Look first; move on --apply.

    # what would move (the default -- nothing is touched)
    python tools/migrate_outputs.py

    # do it
    python tools/migrate_outputs.py --apply

THE SHAPE THIS PRODUCES, and the one question behind it -- does this tree's
content change when a different cohort plays a different role?

    outputs/extractions/                    NO   activation dfc censor
                                                 static_fc meta
    outputs/modelling/version=v0/
        unsupervised_model/                 YES  latents transitions meta
        supervised_model/                   YES  bstm_selection fcm_selection
                                                 transferred_models meta

WHY THIS IS A RENAME AND NOT A COPY. Every move here is `os.rename` of a
DIRECTORY within one filesystem: it rewrites a directory entry and touches no
file contents, so a 300 GB tree moves in milliseconds and needs no free space.
A copy would need a second copy of the tree and hours. The cost of that choice
is that source and destination must be on one filesystem, which they are --
both are under `outputs/`. If they ever are not, rename raises EXDEV and this
tool stops rather than silently falling back to a copy it cannot finish.

WHAT IT DOES NOT MOVE: anything it was not told about. A directory under
`outputs/` that is not one of the eleven known trees is REPORTED and left
alone. A tree this tool has never heard of is more likely to be something of
yours than something of ours, and moving it would be a guess.

THE VERSION'S COMPOSITION IS READ OFF THE DATA, not asked for. Every latents
row carries `cohort` and `role` (train / projected / train_subset) as COLUMNS,
so which cohorts were fitted and which were projected is already recorded in
the files being moved. The supervised half is read the same way, from the
selections' own `cohort` column. The version manifest this writes therefore
describes what is actually there -- not what someone typed at a prompt a year
ago -- and every later run is checked against it.

RUN IT WITH NOTHING IN THE QUEUE. The move is atomic per directory, but a job
holding an open file under the old path keeps reading the old inode while
every new process resolves the new one, so a run that spans the migration can
read two different trees. `squeue -u $USER` first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# The repo root, so `fmri_decomposition` imports from a checkout with no
# install. At module scope: the helpers below are used by functions this
# module exposes, not only by the command line.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fmri_decomposition.io import (EXTRACTION_TREES, SUPERVISED_TREES,
                                   UNSUPERVISED_TREES, claim_version,
                                   default_output_root, extractions_root,
                                   modelling_version, read_version_manifest,
                                   supervised_root, tree_root,
                                   unsupervised_root, version_manifest_path)

#: `meta/` splits with the data it describes. The KEY is the subdirectory (or
#: file) directly under the old flat `outputs/meta/`; the value is which half
#: it belongs to. A fitted clusterer is true of one modelling version and a
#: cohort's stage-2 manifest is true of all of them, and they shared a folder.
META_SPLIT = {
    "cohorts": "extractions",
    "censor": "extractions",
    "static_fc.json": "extractions",
    "models": "unsupervised",
    "clusterers": "unsupervised",
    "cluster": "unsupervised",
    "transitions": "unsupervised",
    "bstm_selection": "supervised",
    "fcm_selection": "supervised",
    "transferred_models": "supervised",
    # The pre-rename name of the bstm selection's output folder (commit
    # "bstm_selection: rename the output folder"). A tree that ran stage 5b
    # before that rename has `meta/selection/`, and it is supervised
    # provenance like the two above.
    "selection": "supervised",
}

#: `--output-name` on both selections means META_SPLIT can never be complete:
#: a run under any other name writes `meta/<that name>/`. So an unknown entry
#: directly under the old flat `meta/` is far more likely to be one of those
#: than something of the user's, and the report says so instead of leaving
#: them to guess. Anything matching this is still NOT moved -- naming a
#: destination for a folder whose stage this tool cannot establish would be
#: the guess the whole tool avoids.
META_LIKELY_SELECTION = ("select", "_selection")

#: Files directly under the old `outputs/meta/` that are not in META_SPLIT. The
#: atlas label tables are the only ones, and they are extraction-side: they
#: describe the parcellation, which is upstream of every fit.
META_FILE_PREFIX_EXTRACTIONS = ("atlas-",)


def _half(root: Path, tree: str, version: str) -> Path:
    return tree_root(root, tree, version)


def plan(root: Path, version: str):
    """-> (moves, meta_moves, unknown, already, notes).

    `moves` and `meta_moves` are (source, destination) pairs of DIRECTORIES
    (or, for one file, a file). Nothing is read or written here.
    """
    moves, meta_moves, unknown, already, notes = [], [], [], [], []
    known = set(EXTRACTION_TREES) | set(UNSUPERVISED_TREES) | set(SUPERVISED_TREES)

    for name in sorted(p.name for p in root.iterdir() if p.is_dir()):
        src = root / name
        if name in ("extractions", "modelling"):
            already.append(name)
            continue
        if name == "meta":
            continue                              # handled below, it splits
        if name not in known:
            unknown.append(name)
            continue
        dst = _half(root, name, version)
        if dst.exists():
            notes.append(
                f"  {name}: destination already exists ({_rel(dst, root)}). "
                f"Leaving BOTH in place -- merging two trees of the same "
                f"stage is not something this tool should guess at.")
            continue
        moves.append((src, dst))

    meta = root / "meta"
    if meta.is_dir():
        for p in sorted(meta.iterdir()):
            stage = META_SPLIT.get(p.name)
            if stage is None and p.is_file() and p.name.startswith(
                    META_FILE_PREFIX_EXTRACTIONS):
                stage = "extractions"
            if stage is None:
                hint = ("  <- looks like a selection run under its own "
                        "--output-name; it is supervised provenance, so "
                        "move it under modelling/version=<v>/"
                        "supervised_model/meta/ or delete it"
                        if any(k in p.name for k in META_LIKELY_SELECTION)
                        else "")
                unknown.append(f"meta/{p.name}{hint}")
                continue
            base = {"extractions": extractions_root(root),
                    "unsupervised": unsupervised_root(root, version),
                    "supervised": supervised_root(root, version)}[stage]
            dst = base / "meta" / p.name
            if dst.exists():
                notes.append(f"  meta/{p.name}: destination already exists "
                             f"({_rel(dst, root)}), left in place")
                continue
            meta_moves.append((p, dst))

    return moves, meta_moves, unknown, already, notes


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def composition(root: Path, version: str) -> dict:
    """What the version holds, READ OFF THE MOVED DATA.

    The latents carry `cohort` and `role` as columns, so this is recorded in
    the files themselves. Returns only the keys it could establish; a key it
    cannot read is left out rather than guessed, so `claim_version` leaves it
    unset instead of recording a fiction.
    """
    import pyarrow.parquet as pq

    out: dict[str, list] = {}
    train, project = set(), set()
    for f in unsupervised_root(root, version).joinpath("latents").rglob(
            "data.parquet"):
        try:
            t = pq.ParquetFile(f)
            have = set(t.schema_arrow.names)
            if not {"cohort", "role"} <= have:
                continue
            tbl = t.read(columns=["cohort", "role"])
        except Exception:                                        # noqa: BLE001
            continue
        for c, r in zip(tbl.column("cohort").to_pylist(),
                        tbl.column("role").to_pylist()):
            # `train_subset` is still training, with --restrict-subjects on.
            (train if str(r).startswith("train") else project).add(str(c))
    if train:
        out["unsupervised_train"] = sorted(train)
        out["unsupervised_project"] = sorted(project - train)

    sup = set()
    for tree in ("bstm_selection", "fcm_selection"):
        for f in supervised_root(root, version).joinpath(tree).rglob(
                "summary.csv"):
            try:
                head = f.read_text().splitlines()[:2]
            except OSError:
                continue
            if len(head) < 2:
                continue
            cols = head[0].split(",")
            if "cohort" not in cols:
                continue
            sup.add(head[1].split(",")[cols.index("cohort")].strip())
    if sup:
        out["supervised_train"] = sorted(c for c in sup if c)
    return out


def apply_moves(pairs, root: Path) -> None:
    for src, dst in pairs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(src, dst)
        except OSError as exc:
            if getattr(exc, "errno", None) == 18:                # EXDEV
                raise SystemExit(
                    f"{_rel(src, root)} and {_rel(dst, root)} are on "
                    f"different filesystems, so this cannot be a rename.\n"
                    f"  This tool only renames -- a 300 GB tree moves in "
                    f"milliseconds and needs no free space. Copying it is a "
                    f"different operation with a different cost, and doing it "
                    f"silently is how a migration runs out of disk halfway.\n"
                    f"  Move it yourself (rsync, then delete the source), or "
                    f"put both halves on one filesystem.") from exc
            raise


def fix_manifest_paths(root: Path, apply: bool) -> int:
    """Rewrite the absolute paths stored INSIDE the manifests.

    Stage 2 and 3 record `path` per entry as an absolute path. Nothing LOCATES
    data by it -- both manifest readers ignore the field, and the trees are
    found by walking -- so a stale value breaks nothing. It is provenance, and
    provenance that is wrong is worse than provenance that is missing, so the
    strings are brought in step with where the files now are.
    """
    import re

    pat = re.compile(
        r'(?P<root>' + re.escape(str(root)) + r')/(?P<tree>'
        + "|".join(sorted(EXTRACTION_TREES, key=len, reverse=True)) + r')/')
    n = 0
    for f in sorted(root.rglob("manifest*.json")):
        try:
            s = s0 = f.read_text()
        except OSError:
            continue
        s = pat.sub(lambda m: f"{m.group('root')}/extractions/{m.group('tree')}/", s)
        if s == s0:
            continue
        n += 1
        if apply:
            json.loads(s)                      # still valid before writing
            f.write_text(s)
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", default=None,
                   help="the outputs tree. Default: $FMRIDECOMP_OUTPUTS, else "
                        "the output_root: in config/*.yaml.")
    p.add_argument("--modelling-version", default=None, metavar="V",
                   help="which version the EXISTING fitted trees become "
                        "(default: $FMRIDECOMP_MODELLING_VERSION, else v0). "
                        "The current analysis is v0; use another only if this "
                        "tree is not the one v0 names.")
    p.add_argument("--apply", action="store_true",
                   help="actually move. Without it nothing is touched.")
    p.add_argument("--no-claim", action="store_true",
                   help="do not write version.json. The composition is read "
                        "off the latents' own `role` column, so there is "
                        "rarely a reason to skip it.")
    a = p.parse_args(argv)

    root = Path(a.output_root) if a.output_root else default_output_root()
    if not root.is_dir():
        raise SystemExit(f"output_root does not exist: {root}")
    version = modelling_version(a.modelling_version)
    print(f"output_root {root}")
    print(f"modelling version {version} "
          f"(the existing fitted trees become this)")

    moves, meta_moves, unknown, already, notes = plan(root, version)

    if already:
        print(f"\nalready in the new shape: {', '.join(already)}/")
    if unknown:
        print(f"\nNOT MOVED -- this tool does not know these, and a tree it "
              f"has never heard of is more likely yours than ours:")
        for u in unknown:
            print(f"  {u}")
    for n in notes:
        print(n)

    if not moves and not meta_moves:
        print(f"\nnothing to move under {root}")
        man = read_version_manifest(root, version)
        if man:
            held = ", ".join(f"{k}={'+'.join(v)}" for k, v in sorted(man.items())
                             if isinstance(v, list) and v)
            print(f"  version {version} holds: {held or '(nothing recorded)'}")
        return 0

    print(f"\n{len(moves)} tree(s) and {len(meta_moves)} meta entry/entries "
          f"to move:")
    for src, dst in moves + meta_moves:
        # The trailing slash only where it is true: two entries under meta/
        # are FILES (static_fc.json and the atlas label tables), and printing
        # them as directories misdescribes what is about to be moved.
        tail = "/" if src.is_dir() else ""
        print(f"  {_rel(src, root)}{tail}  ->  {_rel(dst, root)}{tail}")

    n_man = fix_manifest_paths(root, apply=False)
    if n_man:
        print(f"\n{n_man} manifest(s) record an absolute path under an old "
              f"stem; those strings will be rewritten (provenance only -- "
              f"nothing locates data by them).")

    if not a.apply:
        print("\n--apply to do it. Nothing has been touched.")
        print("Run it with NOTHING IN THE QUEUE: a job holding an open file "
              "under the old path keeps reading the old inode while new "
              "processes resolve the new one.")
        return 0

    apply_moves(moves, root)
    apply_moves(meta_moves, root)
    n_man = fix_manifest_paths(root, apply=True)
    print(f"\nmoved {len(moves)} tree(s), {len(meta_moves)} meta entry/entries,"
          f" rewrote {n_man} manifest(s)")

    meta = root / "meta"
    if meta.is_dir() and not any(meta.iterdir()):
        meta.rmdir()
        print("  removed the now-empty meta/")

    if not a.no_claim:
        comp = composition(root, version)
        if comp:
            claim_version(root, version, **comp)
            print(f"  {_rel(version_manifest_path(root, version), root)}: "
                  + ", ".join(f"{k}={'+'.join(v)}" for k, v in sorted(comp.items())))
        else:
            print("  NOT claiming the version: could not read a cohort/role "
                  "from the latents or a cohort from a selection summary. "
                  "The next run of a modelling stage will claim it from its "
                  "own arguments.")

    print("\nmigrated. `fmri-decomp status` should now find everything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
