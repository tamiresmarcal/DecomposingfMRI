#!/usr/bin/env python3
"""Assemble a BIDS root fMRIPrep can read, out of Cam-CAN's split layout.

    python3 preprocessing/camcan/01_build_bids.py --limit 10

RUN ON A LOGIN NODE. Pure stdlib, no container needed -- it only makes
symlinks, so it costs no disk and takes seconds.

WHY THIS IS NEEDED
------------------
Cam-CAN ships anatomy and function as SEPARATE BIDS datasets:

    BIDSsep/anat/sub-CC110033/anat/sub-CC110033_T1w.nii.gz
    BIDSsep/func_movie/sub-CC110033/func/sub-CC110033_task-Movie_echo-01_bold.nii.gz

fMRIPrep needs one root with both under a single `sub-CC110033/`. It will not
find the anatomy otherwise, and without a T1w it cannot normalise anything.

Symlinks, not copies. The echoes alone are ~140 MB per subject and ~91 GB
across all 649; there is no reason to duplicate that to change a directory
name. fMRIPrep reads through symlinks without complaint.

WHAT COUNTS AS A USABLE SUBJECT
-------------------------------
A T1w, plus all five echoes with their sidecars. Anything missing a piece is
skipped and REPORTED -- never silently dropped. A subject with four echoes is
not a subject with slightly less data; multi-echo combination needs the set,
and a partial subject that slipped through would fail deep inside fMRIPrep
where the message is much harder to read.

Ordering is sorted and stable, so `subjects.txt` line N means the same subject
on every run. 02_fmriprep.sbatch indexes into that file, so a re-submitted
array maps to the same subjects as the first attempt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CAMCAN_BIDSSEP = Path(
    "/project/6008063/tamires/cohorts/camcan/cc700/mri/pipeline/release004/BIDSsep"
)
DEFAULT_OUT = Path("/project/6008063/tamires/cohorts/camcan_bids")

N_ECHOES = 5
TASK = "Movie"

# T2w is optional: fMRIPrep will use one if present to refine the brain mask,
# and 653 of 739 subjects have one. Its absence is not a reason to skip.
ANAT_REQUIRED = ["T1w"]
ANAT_OPTIONAL = ["T2w"]


def _echo_names(sub: str) -> list[str]:
    return [
        f"sub-{sub}_task-{TASK}_echo-{i:02d}_bold{ext}"
        for i in range(1, N_ECHOES + 1)
        for ext in (".nii.gz", ".json")
    ]


def survey(bidssep: Path) -> tuple[list[str], dict[str, str]]:
    """Return (usable subjects, {subject: reason skipped})."""
    anat_root, func_root = bidssep / "anat", bidssep / "func_movie"
    for d in (anat_root, func_root):
        if not d.is_dir():
            raise SystemExit(f"not a directory: {d}\nIs --camcan-bidssep correct?")

    # Union, so a subject present in only one half is reported rather than
    # invisible. Intersecting here would hide exactly the cases worth seeing.
    subs = sorted(
        {p.name[4:] for p in anat_root.glob("sub-*") if p.is_dir()}
        | {p.name[4:] for p in func_root.glob("sub-*") if p.is_dir()}
    )

    usable: list[str] = []
    skipped: dict[str, str] = {}
    for sub in subs:
        adir, fdir = anat_root / f"sub-{sub}" / "anat", func_root / f"sub-{sub}" / "func"
        missing_anat = [
            s for s in ANAT_REQUIRED if not (adir / f"sub-{sub}_{s}.nii.gz").is_file()
        ]
        missing_echo = [n for n in _echo_names(sub) if not (fdir / n).is_file()]

        # "no movie at all" and "movie missing an echo" are different findings
        # and must not share a bucket. On the real archive ~90 subjects have a
        # T1w but never did the movie task -- that is normal, and reporting it
        # as "incomplete" would read as data corruption. A subject genuinely
        # missing one echo of five is rare and worth looking at.
        no_movie = len(missing_echo) == N_ECHOES * 2
        if missing_anat and no_movie:
            skipped[sub] = "no T1w and no movie"
        elif missing_anat:
            skipped[sub] = f"missing anat: {', '.join(missing_anat)}"
        elif no_movie:
            skipped[sub] = "no movie data (subject did not do the task)"
        elif missing_echo:
            n_nii = sum(1 for n in missing_echo if n.endswith(".nii.gz"))
            skipped[sub] = (
                f"incomplete movie: {len(missing_echo)} file(s) absent "
                f"({n_nii} image(s)) of {N_ECHOES * 2}"
            )
        else:
            usable.append(sub)
    return usable, skipped


def link(src: Path, dst: Path, force: bool) -> None:
    """Symlink src -> dst, with an ABSOLUTE target.

    A relative target resolves against the LINK's directory, not the cwd, so
    `--camcan-bidssep some/relative/path` would silently produce links
    pointing at `<out>/sub-X/func/some/relative/path/...` -- every one of them
    dangling, with nothing to notice until fMRIPrep reports a subject with no
    images hours later. src is resolved by the caller; this asserts it.
    """
    if not src.is_absolute():
        raise ValueError(f"link source must be absolute, got {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if not force:
            return
        dst.unlink()
    dst.symlink_to(src)


def build(subs: list[str], bidssep: Path, out: Path, force: bool) -> None:
    anat_root, func_root = bidssep / "anat", bidssep / "func_movie"
    for sub in subs:
        s = f"sub-{sub}"
        for suffix in ANAT_REQUIRED + ANAT_OPTIONAL:
            for ext in (".nii.gz", ".json"):
                src = anat_root / s / "anat" / f"{s}_{suffix}{ext}"
                if src.is_file():
                    link(src, out / s / "anat" / src.name, force)
        for name in _echo_names(sub):
            link(func_root / s / "func" / name, out / s / "func" / name, force)

    # subjects.txt and skipped_subjects.tsv are OUR bookkeeping, not BIDS. The
    # validator rejects any file it does not recognise (NOT_INCLUDED), which is
    # a hard error -- fMRIPrep runs bids-validator before it does anything and
    # refuses to start when it fails. .bidsignore is the spec's own mechanism
    # for this, so the files can stay next to the data they describe.
    #
    # Keeping them here rather than dropping --skip-bids-validation is
    # deliberate: validation is worth having. It is what confirmed this tree
    # holds 10 subjects, one session and task Movie.
    (out / ".bidsignore").write_text("subjects.txt\nskipped_subjects.tsv\n")

    # BIDS wants a README. Only a warning, not an error, but a dataset that
    # cannot say what it is invites exactly the confusion this whole directory
    # exists to resolve.
    (out / "README").write_text(
        "Cam-CAN CC700 movie-watching, assembled as a single BIDS root.\n"
        "\n"
        "Symlinks only -- no data is copied. Cam-CAN distributes anatomy and\n"
        "function as two separate BIDS datasets (BIDSsep/anat and\n"
        "BIDSsep/func_movie); fMRIPrep needs both under one sub-<label>/.\n"
        "\n"
        "Built by preprocessing/camcan/01_build_bids.py in the DecomposingfMRI\n"
        "repository. Do not edit by hand -- re-run that script instead.\n"
        "\n"
        "subjects.txt          subjects included, sorted; the SLURM array\n"
        "                      indexes into this file.\n"
        "skipped_subjects.tsv  subjects excluded, with the reason for each.\n"
    )

    # The source dataset_description.json is a blank template -- every field an
    # empty string, BIDSVersion 1.0.1 -- which the validator rejects. This file
    # describes the VIEW we just assembled, so it is ours to write.
    (out / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "Cam-CAN CC700 movie-watching (BIDS view for fMRIPrep)",
                "BIDSVersion": "1.8.0",
                "DatasetType": "raw",
                "Authors": ["Cam-CAN consortium"],
                "HowToAcknowledge": (
                    "Cite Shafto et al. 2014 (BMC Neurology) and "
                    "Taylor et al. 2017 (NeuroImage)."
                ),
                "ReferencesAndLinks": [
                    "https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/"
                ],
            },
            indent=2,
        )
        + "\n"
    )


def verify(selected: list[str], out: Path) -> tuple[list[str], list[str]]:
    """Check the BIDS root as it now stands. Returns (dangling, stale).

    Deliberately walks every `sub-*` on disk rather than only what this run
    selected. Two reasons, both found by testing rather than reasoning:

      * A dangling symlink looks identical to a working one in `ls`, costs
        nothing to create, and surfaces as a baffling fMRIPrep failure hours
        into a compute job. One stat() per file catches it for free.
      * A previous run with a larger --limit, or against data that has since
        changed, leaves `sub-*` directories behind. fMRIPrep globs `sub-*`;
        it does not read subjects.txt. So a subject this run correctly
        rejected is still sitting in the root waiting to break the job.
        Verifying only what we just linked would report success on a root
        that is broken.
    """
    dangling, stale = [], []
    keep = {f"sub-{s}" for s in selected}
    for sdir in sorted(out.glob("sub-*")):
        if not sdir.is_dir():
            continue
        if sdir.name not in keep:
            stale.append(sdir.name)
        for d in ("anat", "func"):
            if not (sdir / d).is_dir():
                continue
            for p in sorted((sdir / d).iterdir()):
                if not p.exists():  # follows the link
                    dangling.append(str(p.relative_to(out)))
    return dangling, stale


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--camcan-bidssep", type=Path, default=CAMCAN_BIDSSEP)
    p.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--limit",
        type=int,
        help="only the first N usable subjects, for a pilot. Selection is from "
        "the sorted list, so --limit 10 twice gives the same 10.",
    )
    p.add_argument(
        "--subjects",
        help="comma-separated CCIDs (e.g. CC110033,CC110037), overriding --limit",
    )
    p.add_argument("--force", action="store_true", help="replace existing symlinks")
    p.add_argument(
        "--dry-run", action="store_true", help="report what would be linked, link nothing"
    )
    args = p.parse_args(argv)

    # Before anything else. Symlink targets must be absolute (see link()), and
    # a relative --out would be interpreted against whatever cwd the caller
    # happened to be in.
    args.camcan_bidssep = args.camcan_bidssep.resolve()
    args.out = args.out.resolve()

    usable, skipped = survey(args.camcan_bidssep)
    print(f"usable subjects (T1w + all {N_ECHOES} echoes): {len(usable)}")
    print(f"skipped:                                      {len(skipped)}")

    reasons: dict[str, int] = {}
    for reason in skipped.values():
        key = reason.split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    for key, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {n:4d}  {key}")

    if args.subjects:
        want = [s.strip() for s in args.subjects.split(",") if s.strip()]
        unknown = [s for s in want if s not in usable]
        if unknown:
            raise SystemExit(
                f"not usable (missing files, see above): {', '.join(unknown)}"
            )
        selected = want
    else:
        selected = usable[: args.limit] if args.limit else usable

    print(f"\nselecting {len(selected)} subject(s) -> {args.out}")
    if args.dry_run:
        print("(dry run -- nothing written)")
        print("first few:", ", ".join(selected[:5]))
        return 0

    if args.out.exists() and not args.force and any(args.out.glob("sub-*")):
        print(
            f"\nrefusing: {args.out} already has sub-* directories.\n"
            "Re-run with --force to relink, or pick a different --out.",
            file=sys.stderr,
        )
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    build(selected, args.camcan_bidssep, args.out, args.force)

    dangling, stale = verify(selected, args.out)
    if stale:
        print(
            f"\nERROR: {len(stale)} subject dir(s) in {args.out} are not in this "
            f"selection:\n    {', '.join(stale[:8])}"
            f"{' ...' if len(stale) > 8 else ''}\n"
            "fMRIPrep globs sub-* and does not read subjects.txt, so these WILL "
            "be picked up.\nThey are left over from an earlier run, or were "
            "rejected this time round.\nRemove them, then re-run:\n"
            + "".join(f"    rm -rf {args.out / s}\n" for s in stale[:8]),
            file=sys.stderr,
        )
    if dangling:
        print(
            f"\nERROR: {len(dangling)} link(s) do not resolve, e.g.:",
            file=sys.stderr,
        )
        for p in dangling[:5]:
            print(f"    {p}", file=sys.stderr)
    if stale or dangling:
        print("\nThe BIDS root is unusable. Do NOT submit the array job.",
              file=sys.stderr)
        return 1
    print(f"verified: {len(selected)} subject(s), all links resolve, "
          "no stale directories")

    # The array job reads this. Sorted and newline-terminated, so line N is
    # stable across re-submissions.
    (args.out / "subjects.txt").write_text("\n".join(selected) + "\n")

    # A record of what was left out and why. Without it, "649 on disk, 640
    # processed" is a mystery six months from now.
    with (args.out / "skipped_subjects.tsv").open("w") as fh:
        fh.write("sub\treason\n")
        for sub in sorted(skipped):
            fh.write(f"{sub}\t{skipped[sub]}\n")

    print(f"wrote {args.out}/subjects.txt        ({len(selected)} lines)")
    print(f"wrote {args.out}/skipped_subjects.tsv ({len(skipped)} rows)")
    print(f"\nNext:  sbatch --array=0-{len(selected) - 1} "
          f"preprocessing/camcan/02_fmriprep.sbatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
