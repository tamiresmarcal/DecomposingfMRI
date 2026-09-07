#!/usr/bin/env python3
"""Consolidate Cam-CAN's `cc700-scored/` test summaries into one table.

    python preprocessing/camcan/03_build_participants_scores.py \
        --scored-root /project/6008063/tamires/cohorts/camcan/cc700-scored \
        --output-root /project/6008063/tamires/DecomposingfMRI/outputs

writes

    outputs/meta/cohorts/cohort=camcan/participants_scores.csv
    outputs/meta/cohorts/cohort=camcan/participants_scores_manifest.json

one row per subject, one column block per test, plus a manifest recording which
release each block came from.

WHY THIS IS NOT IN THE PACKAGE
------------------------------
`cc700-scored/` is not BIDS and not derived from the imaging at all: it is the
Cam-CAN archive's own behavioural scoring, one directory per test, each with its
own release numbering and its own summary file written by whoever wrote that
test's analysis script in 2011-2014. Nothing about the layout generalises to
another cohort, so it lives here in `preprocessing/` next to the other
cohort-specific stage-1 scripts rather than in `fmri_decomposition/`, which is
cohort-agnostic by design.

    cc700-scored/<Test>/release00N/summary/<Test>_summary.txt

WHAT THE SUMMARY FILES LOOK LIKE
--------------------------------
A header block, a tab-separated table, a footer of column statistics:

    =========================================
    TOT - release001
    =========================================
    Dir: /imaging/camcan/cc700-scored/TOT/release001/data
    Date: 29-Jul-2014 11:18:16
    Output File: .../summary/TOT_summary.txt
    N: 656
    -----------------------------------------
    Subject  sum_KC  sum_KI  ...  ToT_ratio  ErrorMessages
    CC110033 12.00000 2.00000 ...  0.36842
    CC110098                                 replied dont know on > 80% trials
    -----------------------------------------
    SUMMARY  sum_KC  ...
    MEAN     15.99068 ...
    N        644.00000 ...
    =========================================

Three things that follow from that shape and that this parser does not guess at:

**The footer is not data.** `MEAN`, `STDEV`, `MIN`, `MAX` and `N` sit in the
same column layout as the subjects and would join as five extra "subjects" whose
`sub` happens to be a word. They are dropped by name AND by the separator line
that precedes them, so a file whose footer is spelled differently still stops at
the rule.

**A blank row is a QC exclusion, not a missing subject.** TOT lists 656 subjects
and scores 644; the twelve blanks carry their reason in `ErrorMessages`
("replied dont know on > 80% trials"). The row is kept, the values are NaN, and
the reason is preserved as `<test>_note` -- deleting the row would lose the
distinction between "excluded by the test's own QC" and "never took the test".

**`N:` in the header is the number of rows, not the number of scores.** It is
recorded in the manifest as reported and separately as counted, so a
disagreement is visible rather than inherited.

The `*_cumulative_*.txt` files that sit in the same directory are scoring-run
logs, not the release table -- TOT's has 3,111 rows for 656 subjects. Only
`<Test>_summary.txt` is read.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

# Footer labels that occupy the subject column in these files.
FOOTER_LABELS = {"SUMMARY", "MEAN", "STDEV", "STD", "SD", "MIN", "MAX", "N",
                 "MEDIAN", "COUNT"}

# The id column, whichever the test's author called it.
ID_COLUMNS = ("Subject", "CCID", "SubjectID", "ID")

# Free-text columns kept as text rather than coerced to a number.
NOTE_COLUMNS = ("ErrorMessages", "ErrorMessage", "Notes", "Comments")


def read_text(path: Path) -> list[str]:
    """These files are from 2011-2014 and are not all UTF-8."""
    raw = path.read_bytes()
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding).splitlines()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").splitlines()


def parse_summary(path: Path) -> tuple["pd.DataFrame", dict]:
    """One `<Test>_summary.txt` -> (rows, header metadata).

    Raises rather than returning an empty frame: a summary file that does not
    parse is a fact about the archive worth stopping on, and an empty block
    joined into the output would read as "every subject missing this test".
    """
    import pandas as pd

    lines = read_text(path)

    meta = {}
    for line in lines[:20]:
        m = re.match(r"^(Dir|Date|Output File|N):\s*(.*)$", line.strip())
        if m:
            meta[m.group(1).replace(" ", "_").lower()] = m.group(2).strip()

    header_i = None
    for i, line in enumerate(lines):
        first = line.split("\t", 1)[0].strip()
        if first in ID_COLUMNS and "\t" in line:
            header_i = i
            break
    if header_i is None:
        raise ValueError(
            f"{path}: no header row starting with one of {ID_COLUMNS}. "
            f"First 10 lines:\n  " + "\n  ".join(lines[:10]))

    columns = [c.strip() for c in lines[header_i].split("\t")]
    columns = [c for c in columns if c]                    # trailing tab

    rows = []
    for line in lines[header_i + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if set(stripped) <= {"-", "="}:                    # the rule before the footer
            break
        cells = [c.strip() for c in line.split("\t")]
        if cells[0] in FOOTER_LABELS:                      # belt and braces
            break
        rows.append(cells[:len(columns)] + [""] * (len(columns) - len(cells)))

    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        raise ValueError(f"{path}: header found but no data rows")

    id_col = next(c for c in columns if c in ID_COLUMNS)
    out = pd.DataFrame({"sub": df[id_col].str.strip()})
    for col in columns:
        if col == id_col:
            continue
        if col in NOTE_COLUMNS:
            out[col] = df[col].replace("", pd.NA)
        else:
            out[col] = pd.to_numeric(df[col].replace("", pd.NA), errors="coerce")
    meta["n_rows_counted"] = len(out)
    meta["columns"] = [c for c in columns if c != id_col]
    return out, meta


def pick_release(test_dir: Path, want: str | None) -> Path | None:
    """The newest `release00N` under a test, or the one that was asked for."""
    releases = sorted(d for d in test_dir.iterdir()
                      if d.is_dir() and re.match(r"^release\d+$", d.name))
    if not releases:
        return None
    if want and want != "latest":
        match = [d for d in releases if d.name == want]
        return match[0] if match else None
    return releases[-1]


def find_summary(release_dir: Path, test: str) -> Path | None:
    """`<Test>_summary.txt`, or the single `*_summary.txt` if it is named oddly.

    Never a `*_cumulative_*` file: those are scoring-run logs that live in the
    same directory and carry several rows per subject.
    """
    summary_dir = release_dir / "summary"
    if not summary_dir.is_dir():
        return None
    exact = summary_dir / f"{test}_summary.txt"
    if exact.exists():
        return exact
    candidates = [p for p in sorted(summary_dir.glob("*_summary.txt"))
                  if "cumulative" not in p.name.lower()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"{summary_dir}: {len(candidates)} summary files and none named "
            f"{test}_summary.txt: {[p.name for p in candidates]}")
    return None


def write_atomic(df, path: Path, fmt: str) -> None:
    """Write then rename -- the same guarantee the pipeline's own writes give."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        if fmt == "parquet":
            df.to_parquet(tmp, index=False)
        else:
            df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main(argv=None) -> int:
    import pandas as pd

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scored-root", required=True,
                   help="the cc700-scored directory (one subdirectory per test)")
    p.add_argument("--output-root",
                   help="pipeline output_root; default: read from "
                        "config/camcan_movie.yaml")
    p.add_argument("--cohort", default="camcan",
                   help="cohort key the table is written under (default: camcan)")
    p.add_argument("--tests", default="",
                   help="comma-separated subset of tests; default: every "
                        "directory under --scored-root")
    p.add_argument("--release", default="latest",
                   help="'latest' (default) or a release directory name, e.g. "
                        "release001, applied to every test")
    p.add_argument("--format", choices=("csv", "parquet"), default="csv")
    p.add_argument("--participants",
                   help="participants CSV to report coverage against; default: "
                        "the one matching --cohort in config/")
    args = p.parse_args(argv)

    scored = Path(args.scored_root)
    if not scored.is_dir():
        print(f"--scored-root does not exist: {scored}", file=sys.stderr)
        return 1

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        import yaml
        cfg = yaml.safe_load((REPO / "config" / "camcan_movie.yaml").read_text())
        output_root = Path(cfg["output_root"])

    from fmri_decomposition.io import cohort_meta_dir

    wanted = [t.strip() for t in args.tests.split(",") if t.strip()]
    tests = sorted(d.name for d in scored.iterdir() if d.is_dir())
    if wanted:
        missing = [t for t in wanted if t not in tests]
        if missing:
            print(f"no such test directory under {scored}: {missing}\n"
                  f"  present: {tests}", file=sys.stderr)
            return 1
        tests = wanted

    print(f"{len(tests)} test directory(ies) under {scored}:")
    frames, manifest, skipped = [], [], []
    for test in tests:
        release = pick_release(scored / test, args.release)
        if release is None:
            skipped.append((test, f"no {args.release} release directory"))
            print(f"  {test:<20} SKIPPED -- no {args.release} release directory")
            continue
        summary = find_summary(release, test)
        if summary is None:
            skipped.append((test, f"no summary file under {release.name}/summary"))
            print(f"  {test:<20} SKIPPED -- no summary file in "
                  f"{release.name}/summary")
            continue

        df, meta = parse_summary(summary)
        value_cols = [c for c in df.columns if c != "sub"]
        # Namespace every column by its test: `sum_KC` is not unique across
        # these tests, and a silent overwrite in the join would be invisible.
        df = df.rename(columns={c: f"{test}_{c}" for c in value_cols})
        note_cols = [f"{test}_{c}" for c in NOTE_COLUMNS if f"{test}_{c}" in df]
        numeric = [c for c in df.columns
                   if c != "sub" and c not in note_cols]
        scored_n = int(df[numeric].notna().any(axis=1).sum()) if numeric else 0

        dupes = df["sub"].duplicated()
        if dupes.any():
            print(f"  {test:<20} {int(dupes.sum())} duplicate subject row(s), "
                  f"keeping the first")
            df = df[~dupes]

        print(f"  {test:<20} {release.name}  {len(df):>4} rows, "
              f"{scored_n:>4} scored, {len(numeric)} measure(s)")
        frames.append(df)
        manifest.append({
            "test": test, "release": release.name,
            "summary_file": str(summary),
            "n_reported": meta.get("n"), "n_rows": len(df),
            "n_subjects_with_any_score": scored_n,
            "scoring_date": meta.get("date"),
            "columns": list(df.columns[1:]),
        })

    if not frames:
        print("no summary files parsed", file=sys.stderr)
        return 1

    scores = frames[0]
    for df in frames[1:]:
        scores = scores.merge(df, on="sub", how="outer")

    numeric_all = [c for c in scores.columns
                   if c != "sub" and not any(c.endswith(f"_{n}") for n in NOTE_COLUMNS)]
    per_test = {}
    for entry in manifest:
        cols = [c for c in entry["columns"] if c in numeric_all]
        per_test[entry["test"]] = scores[cols].notna().any(axis=1) if cols else False
    scores.insert(1, "n_tests_with_data",
                  sum(per_test.values()).astype(int) if per_test else 0)
    scores = scores.sort_values("sub", ignore_index=True)

    out_dir = cohort_meta_dir(output_root, args.cohort)
    suffix = "csv" if args.format == "csv" else "parquet"
    out_path = out_dir / f"participants_scores.{suffix}"
    write_atomic(scores, out_path, args.format)

    manifest_path = out_dir / "participants_scores_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "cohort": args.cohort,
        "scored_root": str(scored),
        "release_policy": args.release,
        "n_subjects": int(len(scores)),
        "n_measures": int(len(numeric_all)),
        "n_columns": int(scores.shape[1] - 2),
        "tests": manifest,
        "skipped": [{"test": t, "reason": r} for t, r in skipped],
    }, indent=2))
    os.replace(tmp, manifest_path)

    n_notes = scores.shape[1] - 2 - len(numeric_all)
    print(f"\nwrote {out_path}")
    print(f"      {len(scores)} subject(s) x {len(numeric_all)} measure(s) "
          f"(+{n_notes} note column(s)) from {len(manifest)} test(s)")
    print(f"      {manifest_path.name}")
    print("\ntests per subject:")
    print(scores["n_tests_with_data"].value_counts().sort_index()
          .rename_axis("n_tests").rename("subjects").to_string())

    # Coverage against the cohort, which is the number that decides whether
    # these scores can carry an analysis.
    part = Path(args.participants) if args.participants else None
    if part is None:
        for cand in sorted((REPO / "config").glob("*_participants.csv")):
            t = pd.read_csv(cand, dtype=str)
            if args.cohort in set(t.get("cohort", pd.Series(dtype=str)).dropna()):
                part = cand
                break
    if part and part.exists():
        t = pd.read_csv(part, dtype=str)
        cohort_subs = set(t.loc[t["cohort"] == args.cohort, "sub"])
        have = set(scores.loc[scores["n_tests_with_data"] > 0, "sub"])
        print(f"\ncoverage against {part.name}: "
              f"{len(cohort_subs & have)}/{len(cohort_subs)} cohort subject(s) "
              f"have at least one score")
        absent = sorted(cohort_subs - have)
        if absent:
            print(f"  {len(absent)} with none: {absent[:10]}"
                  + (" ..." if len(absent) > 10 else ""))
        extra = len(have - cohort_subs)
        if extra:
            print(f"  {extra} scored subject(s) are not in this cohort "
                  f"(the archive is larger than the movie sample); kept")
    else:
        print(f"\nno participants CSV found for cohort={args.cohort!r}; "
              f"coverage not checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
