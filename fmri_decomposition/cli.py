"""Command line: `fmri-decomp validate | extract | dfc | diagnose | fixture`.

Parallelism is at or below the deepest partition key, so each worker owns a
distinct leaf and no locks are needed. Workers never write shared metadata --
the manifest is written once, serially, by the parent.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# BLAS threading inside workers would oversubscribe the CPUs joblib is already
# using. Must be set before numpy is imported anywhere.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def _load(cfg_path):
    from .config import load_config

    return load_config(cfg_path)


def _atlases(cfg):
    from .atlases.registry import get_atlas

    return [get_atlas(name, **cfg.atlas_params.get(name, {})) for name in cfg.atlases]


def _refs(cfg, strict=True):
    from .cohort import attach_participants, discover_runs, load_participants, validate_cohort

    refs = discover_runs(cfg)
    if cfg.participants is None:
        return refs, []
    participants = load_participants(cfg)
    problems = validate_cohort(cfg, refs, participants, strict=strict)
    return attach_participants(refs, participants, cfg), problems


def _shard(jobs, spec):
    """Select this array task's slice of the job list.

    Round-robin rather than contiguous blocks: work is unevenly sized (a
    90-minute movie next to an 8-minute one), and interleaving spreads the
    long runs across tasks instead of piling them into one.
    """
    if not spec:
        return jobs, None
    try:
        i, n = (int(x) for x in str(spec).split("/"))
    except ValueError:
        raise SystemExit(f"--shard must look like I/N, got {spec!r}")
    if not 0 <= i < n:
        raise SystemExit(f"--shard index {i} out of range for {n} shard(s)")
    return jobs[i::n], (i, n)


def _manifest_name(stage: str, shard) -> str:
    """Array tasks must never write the same file. Merge afterwards, serially."""
    if shard is None:
        return f"manifest_{stage}.json"
    i, n = shard
    return f"shards/manifest_{stage}_shard-{i:04d}-of-{n:04d}.json"


def _attach_sidecar(cfg, refs, glob_pat, attr):
    """Attach one per-run sidecar file (censor or confounds) to each RunRef."""
    if not glob_pat:
        return refs
    for ref in refs:
        hits = sorted(Path(ref.bold).parent.glob(glob_pat))
        entities = {f"sub-{ref.sub}", f"task-{ref.task}"}
        for k, v in (("ses", ref.ses), ("run", ref.run), ("acq", ref.acq)):
            if v:
                entities.add(f"{k}-{v}")
        # Every entity this run carries must appear in the sidecar's name, so a
        # sibling session or run cannot be picked up by accident.
        hits = [h for h in hits if all(e in h.name for e in entities)]
        if hits:
            setattr(ref, attr, hits[0])
    return refs


def _attach_censor(cfg, refs):
    return _attach_sidecar(cfg, refs, cfg.confounds.censor_glob, "censor")


def _attach_confounds(cfg, refs):
    return _attach_sidecar(cfg, refs, cfg.confounds.confounds_glob, "confounds")


# ------------------------------------------------------------- commands ---
def cmd_validate(args) -> int:
    cfg = _load(args.config)
    refs, problems = _refs(cfg, strict=False)
    print(f"cohort={cfg.cohort} tr={cfg.tr} runs_discovered={len(refs)}")
    print(f"atlases={cfg.atlases} config_hash={cfg.hash()}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("validation passed")
    return 0


def cmd_extract(args) -> int:
    from joblib import Parallel, delayed

    from .activation import process_run
    from .io import atlas_labels_path, write_manifest

    cfg = _load(args.config)
    refs, _ = _refs(cfg, strict=not args.no_strict)
    refs = _attach_confounds(cfg, _attach_censor(cfg, refs))
    atlases = _atlases(cfg)
    if args.limit:
        refs = refs[: args.limit]

    if not args.shard or args.shard.startswith("0/"):
        # Shared file, single writer -- the same rule as _metadata.
        for atlas in atlases:
            atlas.write_labels(atlas_labels_path(cfg.output_root, atlas.name))

    jobs, shard = _shard([(ref, atlas) for atlas in atlases for ref in refs], args.shard)
    tag = f" [shard {shard[0]}/{shard[1]}]" if shard else ""
    print(f"stage 2{tag}: {len(refs)} run(s) x {len(atlases)} atlas(es) "
          f"-> {len(jobs)} shard file(s) here")
    entries = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(process_run)(ref, atlas, cfg, args.overwrite) for ref, atlas in jobs
    )
    write_manifest(cfg.output_root, entries, cfg.to_dict(),
                   name=_manifest_name("activation", shard), cohort=cfg.cohort)
    return _report(entries)


def cmd_dfc(args) -> int:
    from joblib import Parallel, delayed

    from .dfc import process_activation_file
    from .io import activation_root, write_manifest

    cfg = _load(args.config)
    atlases = _atlases(cfg)
    sizes = [float(w) for w in (args.window_s or cfg.windows.sizes_s)]

    jobs = []
    for atlas in atlases:
        root = activation_root(cfg.output_root, atlas.name, cfg.cohort)
        files = sorted(root.rglob("*.parquet"))
        if not files:
            print(f"  no activation shards for atlas {atlas.name!r} under {root}")
        for f in files:
            for w in sizes:
                jobs.append((f, atlas, w))
    if not jobs:
        print("nothing to do -- run `extract` first")
        return 1

    jobs, shard = _shard(jobs, args.shard)
    tag = f" [shard {shard[0]}/{shard[1]}]" if shard else ""
    print(f"stage 3{tag}: {len(jobs)} shard file(s) across window sizes {sizes}")
    entries = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(process_activation_file)(f, a, cfg, w, None, args.overwrite)
        for f, a, w in jobs
    )
    write_manifest(cfg.output_root, entries, cfg.to_dict(),
                   name=_manifest_name("dfc", shard), cohort=cfg.cohort)
    return _report(entries)


def cmd_diagnose(args) -> int:
    import pandas as pd

    from .validate import (coverage_table, isc_alignment, isc_gate,
                           lr_correlation_diagnostic, write_diagnostic)

    from .io import activation_root

    cfg = _load(args.config)
    atlases = _atlases(cfg)
    atlas = atlases[0]
    files = sorted(activation_root(cfg.output_root, atlas.name, cfg.cohort).rglob("*.parquet"))
    if not files:
        print("no activation shards found")
        return 1

    cov = coverage_table(files)
    print("coverage ->", write_diagnostic(cov, cfg.output_root, "coverage.parquet",
                                          cohort=cfg.cohort))

    lr = lr_correlation_diagnostic(files, atlas)
    if len(lr):
        print("L-R diagnostic ->", write_diagnostic(
            lr, cfg.output_root, f"atlas-{atlas.name}_lr_diagnostic.csv",
            cohort=cfg.cohort))
        print(lr.head(5).to_string(index=False))

    try:
        parcels = tuple(args.isc_parcels) if args.isc_parcels else (atlas.columns[0],)
        isc = isc_alignment(files, parcels=parcels)
        print("ISC alignment ->", write_diagnostic(
            isc, cfg.output_root, "isc_alignment.csv", cohort=cfg.cohort))
        ok, msg = isc_gate(isc, cfg.stimulus.isc_gate_tr)
        print(("PASS " if ok else "FAIL ") + msg)
        if not ok:
            return 2
    except Exception as exc:                                     # noqa: BLE001
        print(f"ISC skipped: {type(exc).__name__}: {exc}")
    with pd.option_context("display.width", 120):
        pass
    return 0


def cmd_merge_manifests(args) -> int:
    """Consolidate per-array-task manifests into one, serially, after the array."""
    import json

    from .io import cohort_meta_dir, write_manifest, ManifestEntry

    cfg = _load(args.config)
    shard_dir = cohort_meta_dir(cfg.output_root, cfg.cohort) / "shards"
    files = sorted(shard_dir.glob(f"manifest_{args.stage}_shard-*.json"))
    if not files:
        print(f"no shard manifests for stage {args.stage!r} under {shard_dir}")
        return 1
    entries = []
    for f in files:
        for e in json.loads(f.read_text())["entries"]:
            entries.append(ManifestEntry(**e))
    write_manifest(cfg.output_root, entries, cfg.to_dict(),
                   name=f"manifest_{args.stage}.json", cohort=cfg.cohort,
                   extra={"merged_from": [f.name for f in files]})
    print(f"merged {len(files)} shard manifest(s), {len(entries)} entries")
    return _report(entries)


def cmd_fixture(args) -> int:
    from .fixtures import make_fixture

    info = make_fixture(args.root, n_subs=args.n_subs, n_tr=args.n_tr)
    print(f"fixture written to {info['root']}")
    print(f"config: {info['config']}")
    return 0


def _report(entries) -> int:
    from collections import Counter

    counts = Counter(e.status for e in entries)
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for e in entries:
        if e.status == "error":
            print(f"  ERROR {e.sub}/{e.task}/{e.atlas}: {e.detail}")
        elif e.status == "empty":
            print(f"  empty {e.sub}/{e.task}/{e.atlas}: {e.detail}")
    return 1 if counts.get("error") else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fmri-decomp", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("config", help="path to the cohort YAML")
        sp.add_argument("--n-jobs", type=int, default=1,
                        help="workers within this task; match --cpus-per-task")
        sp.add_argument("--shard", default=None, metavar="I/N",
                        help="process only slice I of N (one SLURM array task)")
        sp.add_argument("--overwrite", action="store_true")
        return sp

    common(sub.add_parser("extract", help="stage 2: NIfTI -> parcel timeseries")).set_defaults(
        func=cmd_extract)
    sub.choices["extract"].add_argument("--limit", type=int, default=None)
    sub.choices["extract"].add_argument("--no-strict", action="store_true",
                                        help="warn instead of failing on validation problems")

    common(sub.add_parser("dfc", help="stage 3: parcel timeseries -> windowed DFC")).set_defaults(
        func=cmd_dfc)
    sub.choices["dfc"].add_argument("--window-s", type=float, nargs="*", default=None)

    v = sub.add_parser("validate", help="check config, discovery and participants.csv")
    v.add_argument("config")
    v.set_defaults(func=cmd_validate)

    d = sub.add_parser("diagnose", help="coverage, L-R and ISC alignment artifacts")
    d.add_argument("config")
    d.add_argument("--isc-parcels", nargs="*", default=None)
    d.set_defaults(func=cmd_diagnose)

    m = sub.add_parser("merge-manifests", help="consolidate per-array-task manifests")
    m.add_argument("config")
    m.add_argument("--stage", choices=["activation", "dfc"], required=True)
    m.set_defaults(func=cmd_merge_manifests)

    f = sub.add_parser("fixture", help="write a synthetic cohort for testing")
    f.add_argument("root")
    f.add_argument("--n-subs", type=int, default=4)
    f.add_argument("--n-tr", type=int, default=240)
    f.set_defaults(func=cmd_fixture)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
