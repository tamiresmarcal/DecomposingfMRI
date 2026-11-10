"""Storage layout, atomic writes, manifest.

Hive-partitioned parquet. A partitioned dataset is not a file that gets
appended to -- partition keys are directory names, and the dataset only
exists at read time when pyarrow walks the tree. Each worker owns a distinct
leaf and writes one file, so there are no locks and nothing to consolidate.

Directory depth is constant across cohorts by construction: leftover entities
(ses, run, acq) go in the filename, never in a directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

COMPRESSION = "zstd"


# ---------------------------------------------------------------- paths ---
def _key(value: Any) -> str:
    """Hive key values must be filesystem-safe and stable across platforms."""
    s = str(value)
    for bad in ("/", "\\", "=", " "):
        s = s.replace(bad, "-")
    return s


def leaf_filename(ses: str | None = None, run: str | None = None,
                  acq: str | None = None, suffix: str = ".parquet") -> str:
    """Deterministic leaf name so skip_if_exists can just stat() it.

    ALWAYS `data.parquet`, in every cohort. The arguments are kept because the
    call sites pass them and because they are what `validate` hashes to prove
    the name is safe -- see below -- but they no longer appear in the name.

    This used to embed whichever of ses/run/acq the source file carried, so
    ds002837 and Cam-CAN got `data.parquet` while CNeuroMod got
    `ses-002.parquet`. Two conventions in one tree, for no gain: ses, run, acq
    and run_key are already COLUMNS in the table (activation.py:139-142, and
    dfc.py carries them forward), so the filename was a second, partial copy
    of information the data already holds.

    Deliberately not the Spark `part-0` convention: that exists for when one
    logical partition spans several physical files, which never happens here.

    WHAT THIS GIVES UP, and where it is paid back. The entities in the name
    were also what made two runs of the same (cohort, atlas, task, sub) land
    on different paths. A cohort where one subject saw one task twice -- two
    sessions, two runs -- would now have both write to the same leaf, and the
    second would silently overwrite the first. So `validate` now fails on any
    such collision BEFORE anything is submitted (cli.py, cmd_validate). That
    check was previously only in tools/check_cohort.py, which activation_and_dfc.sh
    does not run; making the name constant is what forces it onto the path
    everyone actually takes.

    If a cohort does collide, the fix is NOT to put the entity back in the
    name: it is that (cohort, atlas, task, sub) is not the unit of analysis
    for that cohort, and the config needs to say so.
    """
    return "data" + suffix


# ------------------------------------------------- the three output trees ---
# `outputs/` has three halves, and which half a tree belongs to is decided by
# ONE question: does its content change when a different cohort plays a
# different role in the analysis?
#
#   outputs/extractions/                         no  -- shared by every version
#       activation/ dfc/ censor/ static_fc/ meta/
#   outputs/modelling/version=<v>/unsupervised_model/   yes -- WHICH cohorts fit
#       latents/ transitions/ contextualization/ meta/
#   outputs/modelling/version=<v>/supervised_model/     yes -- which cohort is
#       bstm_selection/ fcm_selection/ transferred_models/ meta/     the target
#
# A parcel timeseries, a windowed correlation matrix, a censor mask and a
# static-FC matrix are the same bytes whichever cohort trains -- so they sit
# OUTSIDE the version and are never recomputed or re-stored for one. Everything
# fitted sits inside it.
#
# `version=` IS A HIVE KEY, like every other level of this tree. That buys
# `outputs/modelling/version=*/...` as the glob that lists every version at
# once, and lets `hive_key` read it with no new parsing -- which is the whole
# reason it is not the flat name `modelling_v0`.
EXTRACTION_TREES = ("activation", "dfc", "censor", "static_fc")
UNSUPERVISED_TREES = ("latents", "transitions", "contextualization")
SUPERVISED_TREES = ("bstm_selection", "fcm_selection", "transferred_models")

#: The version used when nothing says otherwise. A string, not a number: it is
#: a hive value and it appears in paths.
DEFAULT_MODELLING_VERSION = "v0"


def modelling_version(version: str | None = None) -> str:
    """Which modelling version this call means.

    Explicit argument, else `$FMRIDECOMP_MODELLING_VERSION`, else `v0`.

    A silent default is how a v1 run writes into v0's tree, so every command
    that resolves one PRINTS it (see `cli`), and the version manifest refuses a
    run whose cohort composition does not match what the version already
    holds. The default exists because v0 is what every current path means; it
    is not a licence to leave it unsaid.
    """
    import os

    # `version is None` means "nobody said", which falls through to the
    # environment and then to v0. An EXPLICIT empty string is someone saying it
    # badly -- `--modelling-version ""` or a shell variable that expanded to
    # nothing -- and silently treating that as v0 is exactly the accident this
    # whole mechanism exists to prevent. So only None falls through.
    if version is None:
        version = os.environ.get("FMRIDECOMP_MODELLING_VERSION")
    v = str(version if version is not None
            else DEFAULT_MODELLING_VERSION).strip()
    if not v or "/" in v or "=" in v:
        raise SystemExit(
            f"modelling version {v!r} is not usable as a hive value: it "
            f"becomes the directory `version={v}`, so it cannot be empty or "
            f"contain '/' or '='.")
    return v


def announce_version(output_root: str | Path, version: str | None = None,
                     log=print) -> str:
    """Print which modelling version this run resolved, and what it holds.

    Called once per command that touches a versioned tree. The point is that
    nobody should ever have to GUESS which version a log belongs to: a default
    that is never said out loud is how a v1 run lands in v0 and is discovered
    weeks later from the figures.
    """
    v = modelling_version(version)
    man = read_version_manifest(output_root, v) or {}
    held = ", ".join(f"{k}={'+'.join(man[k])}" for k in VERSION_KEYS
                     if man.get(k))
    how = "explicit" if version else (
        "$FMRIDECOMP_MODELLING_VERSION" if os.environ.get(
            "FMRIDECOMP_MODELLING_VERSION") else "default")
    # PINNED INTO THE ENVIRONMENT, deliberately. The version is a fact about
    # the whole run, and the path builders resolve it per call -- so without
    # this, threading `version=` through every internal function would be the
    # only thing keeping them in agreement, and ONE missed call site reads the
    # wrong version with nothing saying so.
    #
    # It also reaches joblib's loky workers, which re-read the environment on
    # import: parent and children then agree by construction. That is the same
    # class of bug as a worker re-importing different code mid-fit, and the
    # same fix.
    os.environ["FMRIDECOMP_MODELLING_VERSION"] = v
    log(f"modelling version {v} ({how}) -> "
        f"{modelling_root(output_root, v)}")
    log(f"  version holds: {held}" if held
        else "  version holds: nothing recorded yet -- this run will claim it")
    return v


def reference_maps_dir() -> Path:
    """Where the contextualisation arm's reference maps live, ~285 MB.

    `$FMRIDECOMP_REFMAPS`, else `<repo>/reference_atlases` -- gitignored, the
    same arrangement as `.nilearn_data`. Not hidden, because unlike a cache
    this is a quarter of a gigabyte somebody fetched on purpose and will want
    to find. EXPORT IT if the checkout is on a filesystem that should not hold
    285 MB.

    NOT under `outputs/`: these are an INPUT, downloaded from elsewhere, in
    the same class as an atlas or the BIDS data -- nothing in the pipeline
    produces them and no modelling version owns them. And not beside the
    module either, which is where they ended up when contextualization moved
    into the package and its `Path(__file__).parent` default came along: a
    fetch script would then write 285 MB into site-packages.
    """
    v = os.environ.get("FMRIDECOMP_REFMAPS")
    return (Path(v) if v
            else Path(__file__).resolve().parent.parent / "reference_atlases")


def require_reference_maps() -> Path:
    """`reference_maps_dir()`, refused by name if it is not there.

    Without this the failure lands inside whichever loader asked for a file
    first -- a FileNotFoundError on `MNI152.nii.gz` deep in refmaps.py, which
    says nothing about a 285 MB download or an environment variable. The
    directory is the thing that is missing, so the directory is what the
    message is about.
    """
    d = reference_maps_dir()
    if d.is_dir() and any(d.iterdir()):
        return d
    where = ("$FMRIDECOMP_REFMAPS" if os.environ.get("FMRIDECOMP_REFMAPS")
             else "the default, since FMRIDECOMP_REFMAPS is not set")
    raise SystemExit(
        f"no reference maps at\n  {d}\n({where})\n\n"
        f"They are ~285 MB and are fetched once:\n"
        f"  fmri_decomposition/unsupervised_model/contextualization/"
        f"fetch_reference_maps.sh\n\n"
        f"If they are already somewhere else -- shared between projects, say "
        f"-- point this at it instead of fetching again:\n"
        f"  export FMRIDECOMP_REFMAPS=/path/to/reference_atlases\n"
        f"and put that line in slurm/env.sh so every run has it.")


def extractions_root(output_root: str | Path) -> Path:
    """Where the stage 1-3b trees live -- outside every modelling version."""
    return Path(output_root) / "extractions"


def modelling_root(output_root: str | Path, version: str | None = None) -> Path:
    """One modelling version's root: `outputs/modelling/version=<v>/`."""
    return (Path(output_root) / "modelling"
            / f"version={_key(modelling_version(version))}")


def unsupervised_root(output_root: str | Path, version: str | None = None) -> Path:
    return modelling_root(output_root, version) / "unsupervised_model"


def supervised_root(output_root: str | Path, version: str | None = None) -> Path:
    return modelling_root(output_root, version) / "supervised_model"


def tree_root(output_root: str | Path, tree: str,
              version: str | None = None) -> Path:
    """The root of one named tree, wherever its half puts it.

    The ONE place that maps a tree name to a location. Every stage used to
    spell `root / "transitions"` inline -- about forty sites across the package,
    the notebooks' modules and tools/ -- so the shape of the tree was a fact
    repeated forty times instead of stated once, and changing it meant finding
    all forty.
    """
    if tree in EXTRACTION_TREES:
        return extractions_root(output_root) / tree
    if tree in UNSUPERVISED_TREES:
        return unsupervised_root(output_root, version) / tree
    if tree in SUPERVISED_TREES:
        return supervised_root(output_root, version) / tree
    raise SystemExit(
        f"unknown output tree {tree!r}. Known: "
        f"{', '.join(EXTRACTION_TREES + UNSUPERVISED_TREES + SUPERVISED_TREES)}."
        f"\n  A selection written under its own --output-name is addressed by "
        f"`supervised_root(root, version) / name`, not through here.")


# ------------------------------------------- what a modelling version IS ---
# `version=v0` is a directory name, and a directory name is not a claim anyone
# can check. Six months on, "v0" says nothing about which cohorts were fitted
# or which was transferred to -- and worse, a job run with a different
# composition writes into v0 and the tree keeps the old name over new content.
#
# So a version records its own composition, once, and every later run under it
# is CHECKED against that record. The first run writes the manifest; a run
# whose composition differs is refused by name, with the two compositions
# printed. That is the same failure this project has already paid for twice --
# a path claiming something the data did not support.
VERSION_MANIFEST = "version.json"

#: The fields that DEFINE a version, in the order they are reported. Anything
#: not here (an atlas, an aperture, a model list, a seed) is a knob INSIDE a
#: version and may differ between runs without making a new one.
VERSION_KEYS = ("unsupervised_train", "unsupervised_project",
                "supervised_train", "supervised_test", "unsupervised_from")


def version_manifest_path(output_root: str | Path,
                          version: str | None = None) -> Path:
    return modelling_root(output_root, version) / VERSION_MANIFEST


def read_version_manifest(output_root: str | Path,
                          version: str | None = None) -> dict | None:
    """What this version says it is, or None if nothing has claimed it yet."""
    p = version_manifest_path(output_root, version)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


def _as_list(v):
    if v is None:
        return None
    return [str(x) for x in (v if isinstance(v, (list, tuple, set)) else [v])]


def claim_version(output_root: str | Path, version: str | None = None,
                  *, write: bool = True, **composition) -> dict:
    """Record or verify what a modelling version holds. Returns the manifest.

    Pass whichever of `VERSION_KEYS` the calling stage knows: an unsupervised
    run knows `unsupervised_train` / `unsupervised_project`, a supervised one
    knows `supervised_train` / `supervised_test`. A key the caller does not
    pass is left alone, so the two halves fill in their own half and neither
    overwrites the other's.

    `unsupervised_from` is how a version reuses another's state space: v2 that
    only swaps the supervised roles sets it to "v0" and holds no
    unsupervised_model/ of its own. Saying so is what makes the reuse readable
    instead of a mystery empty directory.

    `write=False` verifies without writing -- for a reader, or a --dry-run.
    """
    v = modelling_version(version)
    given = {k: _as_list(composition[k]) for k in composition
             if composition[k] is not None}
    bad = sorted(set(given) - set(VERSION_KEYS))
    if bad:
        raise SystemExit(
            f"claim_version got {bad}, which do not define a version. The "
            f"fields that do: {', '.join(VERSION_KEYS)}. An atlas, an "
            f"aperture, a model or a seed is a knob INSIDE a version.")

    have = read_version_manifest(output_root, v) or {}
    clash = {k: (have[k], given[k]) for k in given
             if k in have and have[k] is not None and have[k] != given[k]}
    if clash:
        detail = "\n".join(
            f"    {k}:\n      version holds: {was}\n      this run has:  {now}"
            for k, (was, now) in sorted(clash.items()))
        raise SystemExit(
            f"modelling version {v!r} already holds a DIFFERENT composition:\n"
            f"{detail}\n"
            f"  {version_manifest_path(output_root, v)}\n"
            f"  A version is which cohorts play which role. Writing this run "
            f"into {v!r} would leave the tree named for one composition and "
            f"filled with two.\n"
            f"  Run it under a new version instead -- pass "
            f"--modelling-version or set FMRIDECOMP_MODELLING_VERSION -- or, "
            f"if the composition recorded in {v!r} is simply wrong, edit that "
            f"file deliberately.")

    merged = {**have, **given, "version": v}
    if write and merged != {**have, "version": v} or (write and not have):
        import time as _time
        merged["updated_utc"] = _time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               _time.gmtime())
        p = version_manifest_path(output_root, v)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    return merged


def activation_root(output_root: str | Path, atlas: str,
                    cohort: str | None = None) -> Path:
    """The shallowest schema-homogeneous root: one atlas, any number of cohorts.

    Atlas is the outermost key because it is the only one that changes *column
    count* (111 vs 14 vs 7). Everything below a given `atlas=` directory shares
    a schema, so this path can be handed straight to `pyarrow.dataset`. Pass a
    cohort to narrow to one; omit it to pool across cohorts in a single read.
    """
    p = tree_root(output_root, "activation") / f"atlas={_key(atlas)}"
    return p if cohort is None else p / f"cohort={_key(cohort)}"


def dfc_root(output_root: str | Path, atlas: str, window_s: float | None = None,
             cohort: str | None = None) -> Path:
    """Same idea for stage 3, with `window_s` between atlas and cohort.

    window_s sits above cohort so that "one atlas, one window size, every
    cohort" is a single directory -- the cross-cohort pooling query stage 4
    actually runs. It changes row count, never column count, so it is safe
    anywhere below atlas.
    """
    p = tree_root(output_root, "dfc") / f"atlas={_key(atlas)}"
    if window_s is not None:
        p = p / f"window_s={_key(_fmt_window(window_s))}"
    return p if cohort is None else p / f"cohort={_key(cohort)}"


def activation_path(output_root: str | Path, cohort: str, atlas: str, task: str,
                    sub: str, ses: str | None = None, run: str | None = None,
                    acq: str | None = None) -> Path:
    return (
        activation_root(output_root, atlas, cohort)
        / f"task={_key(task)}"
        / f"sub={_key(sub)}"
        / leaf_filename(ses, run, acq)
    )


def dfc_path(output_root: str | Path, cohort: str, atlas: str, window_s: float,
             task: str, sub: str, ses: str | None = None, run: str | None = None,
             acq: str | None = None) -> Path:
    return (
        dfc_root(output_root, atlas, window_s, cohort)
        / f"task={_key(task)}"
        / f"sub={_key(sub)}"
        / leaf_filename(ses, run, acq)
    )


def latents_root(output_root: str | Path, atlas: str, window_s=None,
                 cohort: str | None = None, task: str | None = None,
                 version: str | None = None) -> Path:
    """Stage 3 output. Same key order as `dfc_root`, deliberately.

    Latents are written per (atlas, window_s, cohort, task) because each window
    size is an independent fit -- so the keys that select one fit are the ones
    above cohort, exactly as in stage 2. `window_s` is accepted as a number or
    as the string already in the path, since callers have it both ways.

    `task` SITS BELOW COHORT, AND IT IS NOT OPTIONAL IN PRACTICE. A latents file
    records ONE `stride_s` in its metadata, and stride_s is the TR of the rows
    in it -- so a file holding two tasks at two TRs (Cam-CAN's movie at 2.47s
    and rest at 1.97s, the same people) can only be right about one of them.
    `transitions` reads stride_s for every dwell time in seconds, so the wrong
    one is a 25% error in `mean_dwell_s` and `switches_per_min` with nothing
    saying so. One file per task is what makes a single stride_s a true claim.

    It also makes the per-file skip in `decompose` per (cohort, task): a cohort
    that GAINS a condition writes a new file instead of rewriting the one it
    has, so the state columns stage 4 wrote into the existing condition survive.
    """
    p = tree_root(output_root, "latents", version) / f"atlas={_key(atlas)}"
    if window_s is not None:
        w = window_s if isinstance(window_s, str) else _fmt_window(window_s)
        p = p / f"window_s={_key(w)}"
    if cohort is None:
        return p
    p = p / f"cohort={_key(cohort)}"
    return p if task is None else p / f"task={_key(task)}"


def latents_cells(output_root: str | Path, atlas: str, window_s,
                  version: str | None = None) -> dict[tuple[str, str], Path]:
    """{(cohort, task): latents file} for one (atlas, aperture) cell.

    The one enumerator for stage 3's output, because stage 4 and stage 5a both
    walk it and a second copy is how the two drift. Keyed by the PAIR since
    `decompose` writes one file per (cohort, task).
    """
    d = latents_root(output_root, atlas, window_s, version=version)
    if not d.is_dir():
        return {}
    out = {}
    for sub in sorted(d.glob("cohort=*/task=*")):
        f = sub / "data.parquet"
        if f.exists():
            out[(hive_key(sub, "cohort"), hive_key(sub, "task"))] = f
    return out


def transitions_root(output_root: str | Path, atlas: str | None = None,
                     window_s=None, states: str | None = None,
                     cohort: str | None = None, task: str | None = None,
                     version: str | None = None) -> Path:
    """Stage 5a output, narrowed by as many keys as are given.

    Built here rather than inline in transitions.py so that adding a key -- as
    `task` was -- is one edit instead of one per stage. The keys above `cohort`
    select a state SET; `cohort` and `task` select whose dynamics.
    """
    p = tree_root(output_root, "transitions", version)
    for name, v in (("atlas", atlas), ("window_s", window_s),
                    ("states", states), ("cohort", cohort), ("task", task)):
        if v is None:
            return p
        if name == "window_s" and not isinstance(v, str):
            v = _fmt_window(v)
        p = p / f"{name}={_key(v)}"
    return p


def static_fc_root(output_root: str | Path, atlas: str | None = None,
                   cohort: str | None = None,
                   task: str | None = None) -> Path:
    """Stage 3b output, narrowed by as many keys as are given.

    No `window_s`: one matrix per subject over every usable frame, which is the
    whole point of the arm -- it is the `window_s=-1` aperture's static twin.
    """
    p = tree_root(output_root, "static_fc")
    for name, v in (("atlas", atlas), ("cohort", cohort), ("task", task)):
        if v is None:
            return p
        p = p / f"{name}={_key(v)}"
    return p


def censor_root(output_root: str | Path, policy: str | None = None,
                cohort: str | None = None, task: str | None = None) -> Path:
    """Stage 2.5 output, narrowed by as many keys as are given.

    `policy` is outermost because it names the DECISION: two policies over one
    cohort are two answers to be compared, not two cohorts. The window-level
    gate hangs off the same policy root under its own atlas=/window_s= keys and
    is built by censor.py, which owns that shape.
    """
    p = tree_root(output_root, "censor")
    for name, v in (("policy", policy), ("cohort", cohort), ("task", task)):
        if v is None:
            return p
        p = p / f"{name}={_key(v)}"
    return p


def _fmt_window(window_s: float) -> str:
    return str(int(window_s)) if float(window_s).is_integer() else str(window_s)


# The usable range for K, the number of brain states in a state set. One
# definition, because three stages consult it and they had drifted apart: stage
# 4b refused to WRITE K<3 while stage 5a happily built tables for K=2, so a
# state set could be rejected and analysed in the same run.
#
#   floor 3  K=1 is a single cell, `0->0`=1.0 for every subject -- a constant.
#            K=2 is four cells of which one is free given the row sums, so it is
#            `switch_rate` under another name, and switch_rate is already a
#            summary feature. Neither earns a slot in the selection grid.
#   ceiling 64
#            at K=125 a subject's matrix is 15,625 cells and at 512 it is
#            262,144, against a few hundred transitions -- >99% exactly zero for
#            every subject, so there is no probability to correlate with.
STATE_K_BAND = (3, 64)


# Prefix for the input features `decompose --passthrough-features` writes into
# the latents table beside the PCA/UMAP coordinates. One definition, because
# `decompose` writes them and `cluster` resolves a `raw<N>` embedding from them,
# and a prefix that drifted between the two would read as "no raw columns".
#
# The `<prefix><feature name>` form is deliberate: `raw/AM` keeps the network's
# NAME, which is the whole reason this option exists -- a state mean over
# `raw/AM .. raw/WM` says "high in autobiographical memory, low in working
# memory", and the same mean over `pca0/14 .. pca13/14` says nothing until you
# invert the rotation. The `/` matches the existing `pca0/3` style and cannot
# collide with a transition cell (`i->j`) or an occupancy column (`occ_k`).
RAW_PREFIX = "raw/"

# Refuse `--passthrough-features` above this many features. The option is meant
# for the activation source, where a row is 7, 14 or 111 parcels. On the dfc
# source a row is 21, 91 or 6,105 EDGES, and the last would roughly double the
# whole latents tree for an embedding no clusterer here can fit -- a
# full-covariance HMM on 6,105 dimensions is 18.6M parameters per state.
PASSTHROUGH_MAX_FEATURES = 200


# `<Method>_<embedding>_<K>` -- the state-column convention stage 4b writes and
# stages 5a, 5b and `status` parse. ONE definition, because there were two and
# they are the kind of thing that silently disagrees: `status` had
# `^[A-Za-z]+_[a-z]+\d*_\d+$` and `cluster` had the same pattern inline, and
# BOTH rejected `HMM1_pca3_8` -- the method group was letters-only, so renaming
# `HMM` to `HMM1`/`HMM2` would have left stage 5 reporting zero state sets with
# the columns sitting in the files.
#
#   method      letters then letters-or-digits: HMM1, HMM2, MeanShift
#   embedding   lowercase then optional digits: pca3, umap3, raw14
#   K           digits
STATE_COLUMN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*_[a-z]+\d*_\d+$")


# Keys carried by the directory tree. They are deliberately NOT written as
# columns: a partition key duplicated as a column must match its inferred type
# exactly, and pyarrow reads `sub=01` as int32 -- which both collides with the
# string column and quietly destroys Cam-CAN's `CC110033` and NNDb's leading
# zeros. The path is the single source of truth; `read_shard` puts them back
# when a single file is opened on its own.
# The identity columns of a LATENTS file: what `decompose` carries through from
# the shards, plus what it adds once the fit exists. A schema contract, which
# is what this module is for -- it lived in decompose.py, and `frames.py`
# (shared by extraction, the unsupervised fit and the contextualisation arm)
# had to import the unsupervised model to learn the shape of its own output.
LATENTS_IDENT = ["cohort", "role", "model_hash", "task", "sub", "window_id",
                 "start_s", "stimulus_start_s", "n_tr_effective",
                 "frac_good_frames", "rank_deficient", "crosses_run_boundary"]

# Kept out of read_cohort's projection: added after the fit, not read from the
# dfc shard.
LATENTS_DERIVED = ("cohort", "task", "sub", "role", "model_hash")


PARTITION_KEYS = {
    "activation": ["atlas", "cohort", "task", "sub"],
    "dfc": ["atlas", "window_s", "cohort", "task", "sub"],
}


def parse_hive_keys(path: str | Path) -> dict[str, str]:
    """Recover partition keys from a leaf path. Everything comes back as a string."""
    return dict(
        part.split("=", 1) for part in Path(path).parts if "=" in part
    )


def hive_key(path: str | Path, name: str) -> str | None:
    """One partition key from a leaf path, BY NAME. None if it is not there.

    Always this rather than `path.parts[-N]`. A positional read is correct only
    for the tree depth it was written against, and it fails SILENTLY when a key
    is inserted: adding `task=` below `cohort=` shifted every index by one in
    `bstm_selection.discover`, and because `"cohort=camcan".split("=")[1]`
    succeeds just as well as `"atlas=yeo7".split("=")[1]`, the atlas quietly
    became the window size. Searching for the prefix cannot do that, and
    survives the next key as well.
    """
    want = f"{name}="
    for part in Path(path).parts:
        if part.startswith(want):
            return part.split("=", 1)[1]
    return None


def hive_partitioning(stage: str = "dfc"):
    """Explicit all-string partitioning, so no key is ever type-inferred."""
    import pyarrow.dataset as pads

    keys = PARTITION_KEYS[stage]
    return pads.partitioning(pa.schema([pa.field(k, pa.string()) for k in keys]),
                             flavor="hive")


def open_dataset(root: str | Path, stage: str = "dfc"):
    """Open a partitioned dataset with the correct key types.

    Always use this rather than `partitioning="hive"`: the latter infers types
    per key and will turn subject ids into integers.
    """
    import pyarrow.dataset as pads

    return pads.dataset(str(root), format="parquet", partitioning=hive_partitioning(stage))


def read_file(path: str | Path, columns: list[str] | None = None):
    """One parquet FILE -> pyarrow Table. Never a dataset, never partitioning.

    `pq.read_table` and `pd.read_parquet` both build a ParquetDataset, which
    infers hive partitioning from the directory names even for a single file. For
    a path like `.../cohort=camcan/data.parquet` that invents a `cohort`
    partition field and tries to merge it with the file's own `cohort` COLUMN --
    see PARTITION_KEYS below on why a duplicated key is a trap. The merge raises

        ArrowTypeError: Unable to merge: Field cohort has incompatible types:
        string vs dictionary<values=string, indices=int32, ordered=0>

    and it is VERSION-DEPENDENT: pyarrow 18 (the analysis container) raises,
    pyarrow 25 does not, so it passes every local test and fails on the cluster.
    `ParquetFile` opens the one file through the parquet reader with no dataset
    layer, so there is nothing to infer and nothing to merge.

    Use this anywhere a single latents or shard file is read whole, or read with
    a projection that includes a partition key.
    """
    import pyarrow.parquet as pq

    f = pq.ParquetFile(path)
    if columns is None:
        return f.read()
    have = set(f.schema_arrow.names)
    return f.read(columns=[c for c in columns if c in have])


def read_shard(path: str | Path):
    """Read one leaf file and restore its partition keys as columns."""
    import pandas as pd

    df = pd.read_parquet(path)
    for k, v in parse_hive_keys(path).items():
        if k not in df.columns:
            df[k] = v
    return df


def default_output_root() -> Path:
    """Where outputs live when no --output-root was given.

    `$FMRIDECOMP_OUTPUTS` wins. Otherwise it is read from the cohort configs in
    `config/`, which all name the SAME tree on purpose -- `cohort=` separates
    them inside it, which is what makes "one atlas, one window size, every
    cohort" a single readable path.

    This used to be three lines repeated in eight modules, each naming
    `config/camcan_movie.yaml`. That put one project's cohort filename into the
    library eight times: renaming the config broke every stage, and a new user
    with no Cam-CAN got a FileNotFoundError naming a dataset they do not have.
    Reading whatever configs exist is both shorter and less of a claim.

    Disagreement is REFUSED rather than resolved by picking one. A stage that
    silently wrote to a different root than the previous stage read from is the
    worst outcome available here: every file is present, and the tree is two
    half-built trees.
    """
    import os

    if os.environ.get("FMRIDECOMP_OUTPUTS"):
        return Path(os.environ["FMRIDECOMP_OUTPUTS"])

    import yaml

    cfg_dir = Path(__file__).resolve().parent.parent / "config"
    roots: dict[str, list[str]] = {}
    for f in sorted(cfg_dir.glob("*.yaml")):
        try:
            raw = yaml.safe_load(f.read_text()) or {}
        except Exception:                                        # noqa: BLE001
            continue
        got = raw.get("output_root") if isinstance(raw, dict) else None
        if got:
            roots.setdefault(str(got), []).append(f.name)
    if not roots:
        raise SystemExit(
            f"no output_root found. Either set FMRIDECOMP_OUTPUTS, pass "
            f"--output-root, or give a cohort config in {cfg_dir} an "
            f"`output_root:`.")
    if len(roots) > 1:
        detail = "\n".join(f"    {r}  <- {', '.join(f)}"
                            for r, f in sorted(roots.items()))
        raise SystemExit(
            f"cohort configs disagree about output_root:\n{detail}\n"
            f"  They are meant to name ONE tree, with `cohort=` separating "
            f"them inside it. Fix the configs, or pass --output-root / set "
            f"FMRIDECOMP_OUTPUTS to say which one this run means.")
    return Path(next(iter(roots)))


#: Which half of the tree each `meta/` subtree belongs to. Metadata follows the
#: data it describes, for the same reason the data splits: a cohort's stage-2
#: manifest and its atlas labels are true whatever is fitted on them, a fitted
#: PCA and a clusterer are true of ONE version, and a selection's design note
#: is true of one version's supervised run.
META_STAGE = {
    "cohorts": "extractions", "censor": "extractions",
    "static_fc.json": "extractions", "atlas_labels": "extractions",
    "models": "unsupervised", "clusterers": "unsupervised",
    "cluster": "unsupervised", "transitions": "unsupervised",
    "bstm_selection": "supervised", "fcm_selection": "supervised",
    "transferred_models": "supervised",
}


def meta_dir(output_root: str | Path, stage: str = "extractions",
             version: str | None = None) -> Path:
    """The `meta/` of one half of the tree.

    `stage` is "extractions", "unsupervised" or "supervised" -- NOT optional in
    meaning, only in syntax: it defaults to extractions because that is where
    the cohort manifests and atlas labels live, which is most of the callers.
    `META_STAGE` maps a subtree name to its stage, so a caller holding a name
    rather than a stage can look it up instead of deciding again.

    One `outputs/meta/` for everything was the old shape, and it put a fitted
    clusterer (true of one version) in the same directory as a cohort's
    stage-2 manifest (true of all of them).
    """
    if stage == "extractions":
        return extractions_root(output_root) / "meta"
    if stage == "unsupervised":
        return unsupervised_root(output_root, version) / "meta"
    if stage == "supervised":
        return supervised_root(output_root, version) / "meta"
    raise SystemExit(
        f"meta_dir stage must be 'extractions', 'unsupervised' or "
        f"'supervised', not {stage!r}. For a subtree name, look it up in "
        f"META_STAGE.")


def cohort_meta_dir(output_root: str | Path, cohort: str) -> Path:
    """Everything scoped to one cohort: manifests, coverage, ISC, participants.

    Now that atlas sits above cohort in the data tree, a cohort no longer has a
    subtree of its own. This is where its provenance lives instead, keyed the
    same hive way so it can be discovered by the same walk.
    """
    return (meta_dir(output_root, "extractions") / "cohorts"
            / f"cohort={_key(cohort)}")


def atlas_labels_path(output_root: str | Path, atlas: str) -> Path:
    """Atlas label tables are cohort-independent: one file, shared by all."""
    return (meta_dir(output_root, "extractions")
            / f"atlas-{_key(atlas)}_labels.csv")


# --------------------------------------------------------------- writes ---
def write_table_atomic(table: pa.Table, final_path: str | Path,
                       compression: str = COMPRESSION) -> Path:
    """Write then rename. A file at its final path is complete by construction.

    This is what makes a walltime kill mid-write harmless: the partial file is
    a .tmp that skip_if_exists ignores, not a truncated shard that reads as
    valid until it doesn't.
    """
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_name(f"{final_path.name}.tmp.{os.getpid()}")
    try:
        pq.write_table(table, tmp, compression=compression)
        os.replace(tmp, final_path)   # atomic within a filesystem
    finally:
        if tmp.exists():
            tmp.unlink()
    return final_path


def should_skip(path: str | Path, overwrite: bool = False) -> bool:
    return (not overwrite) and Path(path).exists()


def cleanup_stale_tmp(root: str | Path, older_than_s: float = 86_400) -> list[Path]:
    """Remove .tmp shards abandoned by killed workers."""
    now = time.time()
    removed = []
    for p in Path(root).rglob("*.tmp.*"):
        if now - p.stat().st_mtime > older_than_s:
            p.unlink()
            removed.append(p)
    return removed


# ------------------------------------------------------------- manifest ---
@dataclass
class ManifestEntry:
    stage: str
    cohort: str
    atlas: str
    task: str
    sub: str
    path: str
    status: str                 # ok | skipped | empty | error
    n_rows: int = 0
    window_s: float | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def write_manifest(output_root: str | Path, entries: Iterable[ManifestEntry],
                   config: dict[str, Any] | None = None,
                   name: str = "manifest.json", extra: dict | None = None,
                   cohort: str | None = None, stage: str = "extractions",
                   version: str | None = None) -> Path:
    from . import __version__

    # Every current caller is a stage-2/3 manifest, hence the default; a
    # modelling stage passes its own, so the record lands beside the data it
    # describes rather than in a tree that outlives it.
    d = (meta_dir(output_root, stage, version) if cohort is None
         else cohort_meta_dir(output_root, cohort))
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "package_version": __version__,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": _versions(),
        "config": config or {},
        "entries": [e.as_dict() for e in entries],
    }
    if extra:
        payload.update(extra)
    path = d / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)
    return path


def _versions() -> dict[str, str]:
    import numpy
    import pandas

    v = {"numpy": numpy.__version__, "pandas": pandas.__version__, "pyarrow": pa.__version__}
    for mod in ("nibabel", "nilearn"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = "absent"
    return v


# --------------------------------------------------------- edge storage ---
def edge_storage_mode(n_edges: int, threshold: int = 20_000) -> str:
    """'columns' below the threshold, 'list' above it.

    Parquet carries per-column metadata and per-column-per-row-group
    statistics, so pyarrow degrades in the low tens of thousands of columns.
    Harvard-Oxford-111 gives 6,105 edges (columns, interpretable loadings);
    Schaefer-1000 gives 499,500 (one fixed_size_list column).
    """
    return "columns" if n_edges <= threshold else "list"


# ------------------------------------------------------- subject lists ---
def read_subject_list(path: str, flag: str = "--restrict-subjects") -> set[str]:
    """One subject id per line, or a CSV with a `sub` column.

    Shared by every stage that is handed a list of ids -- the selection trees,
    which use it to pin who was SCORED, and `decompose`, which uses it to pin
    who the fit was fitted ON. Its whole purpose is that those stages agree
    about a set of subjects, and two implementations of "read a list of ids" is
    exactly the kind of thing that drifts on whitespace or case. `flag` only
    names the option in the error, so each caller's message points at its own.

    Ids are upper-cased and stripped, because every join downstream does the
    same -- a list normalised differently would match nothing and read as an id
    mismatch in the data rather than in this file.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{flag} {path}: no such file")
    lines = p.read_text().splitlines()
    if lines and "sub" in [c.strip().lower() for c in lines[0].split(",")]:
        with p.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        col = next((c for c in (rows[0] if rows else {})
                    if c and c.strip().lower() == "sub"), None)
        vals = [r.get(col) or "" for r in rows] if col else []
    else:
        vals = lines
    out = {v.strip().upper() for v in vals if v and v.strip()}
    if not out:
        raise SystemExit(f"{flag} {path}: no ids in it")
    return out


def subject_list_digest(subs: Iterable[str]) -> str:
    """A stable short digest of a SET of subject ids.

    What goes into `model_hash` when a fit was restricted to some subjects. The
    list itself must not go in: it is hundreds of ids, it would dwarf the rest
    of the payload in any diff, and -- the reason that matters -- a camcan
    split stratified on HADS encodes phenotype in its membership, which is not
    something to write into every output file's metadata. The digest says
    "these subjects and no others" without saying who they are.
    """
    joined = "\n".join(sorted(str(s).strip().upper() for s in subs))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]
