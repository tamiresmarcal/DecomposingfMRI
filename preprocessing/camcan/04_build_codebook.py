#!/usr/bin/env python3
"""Flatten Cam-CAN's code-book workbook into two TSVs you can grep.

    python preprocessing/camcan/04_build_codebook.py \
        --code-book /path/to/MRC_CamCAN_Code_Book_variables.xlsx \
        --data /path/to/approved_data.tsv \
        --output-root /project/6008063/tamires/DecomposingfMRI/outputs

writes

    outputs/meta/cohorts/cohort=camcan/codebook_variables.tsv
    outputs/meta/cohorts/cohort=camcan/codebook_values.tsv

WHAT PROBLEM THIS SOLVES
------------------------
The home-interview table (`approved_data.tsv`) names its columns
`homeint_v112`, `epaq_DURTV`, `additional_acer`. Nothing in that file says what
`v112` asked. The code book knows, but it is a seven-sheet workbook with the
header three rows down, section titles carried in "blue header rows" that leave
the variable column empty, and the join key sitting in `Variable_name_in_test`
rather than `Variable_name`. That is not a file you can grep mid-analysis.

So: one row per variable, flat, with the data column name it corresponds to.

    column              group          code        question
    homeint_v112        Demographics   DG7         Highest education level
    epaq_DURTV          EPAQ           DURTV       Hours per day watching TV

and a second file for the value codings, one row per allowed value:

    code    value   label
    DG7     1       College or university degree or higher
    DG7     2       A levels/AS levels or equivalent

HOW THE JOIN WORKS, AND WHY IT IS ONLY PARTIAL
----------------------------------------------
`Variable_name` is the semantic code (`DG7`, `EH_total`, `ACE-R_total`);
`Variable_name_in_test` is what the exported data calls it, and holds either a
`v<NNN>` number (238 of them in CC1) or a literal name that the EPAQ columns use
verbatim (`WKDAYSLEEP`, `DURTV`, `CARadj`). So the match is: strip the data
column's `homeint_` / `epaq_` / `additional_` prefix, then look that stem up in
`Variable_name_in_test` first and `Variable_name` second, case-insensitively.

Measured on the 2026-09 workbook against `approved_data.tsv`, that documents
201 of the 351 data columns. The other 150 -- `homeint_v19`, `v20`, `v123`,
`v205` and the rest -- have no entry anywhere in the workbook, not under any
column. They are written into the output anyway, with `in_code_book=no`, so the
file is a complete inventory of the data rather than a complete transcript of
the code book: looking up an undocumented column tells you it is undocumented
instead of returning nothing and leaving you unsure whether you mistyped.

The `PLEASE NOTE` cells in the workbook's first row are leftover template
instructions ("below are only some examples/snippets"), not a statement about
this file's completeness. The 201/351 number is measured here, not claimed
there.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

# Prefixes the exported table puts on its columns, which the code book does not.
DATA_PREFIXES = ("homeint_", "epaq_", "additional_")

VARIABLE_COLUMNS = ["column", "group", "code", "question", "type", "format",
                    "allowable_values", "value_description", "example",
                    "missing", "comments", "source_sheet", "in_data",
                    "in_code_book", "match"]

# Three data columns are documented under a different name, so no string match
# can find them. They are listed rather than guessed at in code, and the output
# records for each row HOW it was matched, so an inference never reads as a
# fact the workbook stated.
#
#   verified -- checked against the data in this repo's session, 2026-09
#   inferred -- name and value range agree, but the workbook gives the code-book
#               variable no export name at all, so nothing states the link
ALIASES = {
    "sex": ("DG1", "verified",
            "identical to homeint_v1 (=DG1, 'gender') on all 2,676 rows: 1=M, 2=F"),
    "handedness": ("EH_total", "inferred",
                   "Edinburgh Handedness total; observed range -100..100 is the "
                   "laterality quotient. EH_total has no Variable_name_in_test"),
    "mmse_i": ("mmse_cal", "inferred",
               "'total mmse (out of 30)'; observed range 11..30. mmse_cal has "
               "no Variable_name_in_test"),
}

# A `Variable_name_in_test` cell is only usable as a join key if it looks like a
# name. Many of them hold the answer text instead ("none of the above", "o level
# / gcse / leaving certificate"), which collide with each other in the hundreds
# and could match a data column by accident.
_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,39}$")
VALUE_COLUMNS = ["code", "value", "label", "comments", "source_sheet"]


def cell(value) -> str:
    return "" if value is None else str(value).strip()


def parse_code_sheet(ws, sheet: str) -> list[dict]:
    """One `*_code` / `*_addnl_Codes` sheet -> variable rows, section carried down.

    The header is not row 1 and not at the same column offset on every sheet
    (CC1 has a leading `Category/variable` column that CC3 does not), so it is
    found by looking for the row containing `Variable_name` and then indexing
    by header text rather than by position.

    A row with no `Variable_name` but a test id or name is a section header --
    the workbook's "blue header rows" -- and names the block that follows.
    """
    rows = [[cell(c) for c in r] for r in ws.iter_rows(values_only=True)]
    try:
        head_i = next(i for i, r in enumerate(rows) if "Variable_name" in r)
    except StopIteration:
        raise ValueError(f"{sheet}: no header row containing 'Variable_name'")
    index = {h: j for j, h in enumerate(rows[head_i]) if h}

    out, section = [], ""
    for row in rows[head_i + 1:]:
        def get(key: str) -> str:
            j = index.get(key)
            return row[j] if j is not None and j < len(row) else ""

        name = get("Variable_name")
        if not name:
            header = get("Test_name") or get("Test_ID")
            if header:
                section = header
            continue
        out.append({
            "group": section,
            "code": name,
            "data_name": get("Variable_name_in_test"),
            "question": get("Variable_description"),
            "type": get("Type"),
            "format": get("Format /length"),
            "allowable_values": get("Allowable values"),
            "value_description": get("Value description"),
            "example": get("Examples"),
            "missing": get("Missing data"),
            "comments": get("Comments"),
            "source_sheet": sheet,
        })
    return out


def parse_lookup_sheet(ws, sheet: str) -> list[dict]:
    """One `*_LOOKUP` sheet -> value codings, one row per allowed value.

    Same shape as the code sheets: a variable name starts a block and the value
    rows under it leave that cell empty, so the name is carried down.
    """
    rows = [[cell(c) for c in r] for r in ws.iter_rows(values_only=True)]
    try:
        head_i = next(i for i, r in enumerate(rows) if "Allowable values" in r)
    except StopIteration:
        return []
    index = {h: j for j, h in enumerate(rows[head_i]) if h}

    out, code = [], ""
    for row in rows[head_i + 1:]:
        def get(key: str) -> str:
            j = index.get(key)
            return row[j] if j is not None and j < len(row) else ""

        if get("Variable"):
            code = get("Variable")
        value, label = get("Allowable values"), get("Value description")
        if not code or not (value or label):
            continue
        out.append({"code": code, "value": value, "label": label,
                    "comments": get("comments"), "source_sheet": sheet})
    return out


def stem(column: str) -> str:
    for prefix in DATA_PREFIXES:
        if column.startswith(prefix):
            return column[len(prefix):]
    return column


def write_tsv(rows: list[dict], columns: list[str], path: Path) -> None:
    """Write then rename, and escape nothing -- so strip tabs and newlines.

    A description containing a literal tab would silently shift every later
    field of that row, which is exactly the kind of corruption a dictionary
    file must not have.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("\t".join(columns) + "\n")
            for row in rows:
                fh.write("\t".join(
                    re.sub(r"\s+", " ", str(row.get(c, ""))).strip()
                    for c in columns) + "\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main(argv=None) -> int:
    import openpyxl

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--code-book", required=True, help="MRC_CamCAN_Code_Book_variables.xlsx")
    p.add_argument("--data", help="approved_data.tsv, to match columns and report "
                                  "coverage. Without it the code book is flattened "
                                  "but nothing is matched.")
    p.add_argument("--output-root", help="default: output_root from config/camcan_movie.yaml")
    p.add_argument("--cohort", default="camcan")
    p.add_argument("--out-dir", help="write here instead of the cohort meta directory")
    args = p.parse_args(argv)

    book = Path(args.code_book)
    if not book.exists():
        print(f"code book not found: {book}", file=sys.stderr)
        return 1

    wb = openpyxl.load_workbook(book, read_only=True, data_only=True)
    variables, values = [], []
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        if sheet.upper().endswith("LOOKUP"):
            got = parse_lookup_sheet(ws, sheet)
            values += got
            print(f"  {sheet:<18} {len(got):>4} value coding(s)")
        elif "code" in sheet.lower():
            got = parse_code_sheet(ws, sheet)
            variables += got
            with_key = sum(1 for r in got if r["data_name"])
            print(f"  {sheet:<18} {len(got):>4} variable(s), {with_key} with a data name, "
                  f"{len({r['group'] for r in got})} section(s)")
        else:
            print(f"  {sheet:<18} skipped (neither a code nor a LOOKUP sheet)")

    # Two lookups, tried in this order. `Variable_name_in_test` is the export's
    # own name (v112, DURTV); `Variable_name` is the semantic code, which some
    # columns use instead.
    by_data_name, by_code, collisions, junk = {}, {}, 0, 0
    for row in variables:
        for key, table in ((row["data_name"], by_data_name), (row["code"], by_code)):
            k = key.lower()
            if not k:
                continue
            if not _KEY.match(k):
                junk += 1                     # answer text, not a variable name
                continue
            if k in table:
                collisions += 1
                continue                      # first wins; reported below
            table[k] = row
    if junk:
        print(f"  {junk} cell(s) that are answer text rather than a name, not indexed")
    if collisions:
        print(f"  {collisions} duplicate name(s) in the code book, first kept")

    matched = 0
    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"--data not found: {data_path}", file=sys.stderr)
            return 1
        with open(data_path, encoding="utf-8") as fh:
            columns = [c.strip() for c in fh.readline().rstrip("\n").split("\t")]
        columns = [c for c in columns if c and c.upper() != "CCID"]

        undocumented, aliased = [], 0
        for column in columns:
            key = stem(column).lower()
            how = "data_name"
            row = by_data_name.get(key)
            if row is None:
                row, how = by_code.get(key), "code"
            if row is None and key in ALIASES:
                target, confidence, note = ALIASES[key]
                row = by_code.get(target.lower())
                how = f"alias({confidence})"
                if row is not None:
                    row["comments"] = (f"{row['comments']} | alias: {note}"
                                       if row["comments"] else f"alias: {note}")
                    aliased += 1
            if row is None:
                undocumented.append(column)
                continue
            row.setdefault("column", "")
            # A code-book row can serve only one data column; a second match
            # would overwrite the first silently.
            if not row["column"]:
                row["column"], row["match"] = column, how
            matched += 1
        for column in undocumented:
            variables.append({"column": column, "group": "(not in code book)",
                              "code": stem(column), "question": "",
                              "source_sheet": "", "in_code_book": "no",
                              "match": "none"})
        if aliased:
            print(f"  {aliased} column(s) matched through the alias table "
                  f"(see the `match` column)")
        print(f"\n{matched}/{len(columns)} data column(s) documented; "
              f"{len(undocumented)} with no code-book entry")
        if undocumented:
            print(f"  e.g. {undocumented[:10]}")

    for row in variables:
        row.setdefault("column", "")
        row.setdefault("in_code_book", "yes")
        row.setdefault("match", "" if not args.data else "unmatched")
        row["in_data"] = "yes" if row["column"] else ("no" if args.data else "")

    variables.sort(key=lambda r: (r["group"] or "~", r["code"] or ""))

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        if args.output_root:
            output_root = Path(args.output_root)
        else:
            import yaml
            cfg = yaml.safe_load((REPO / "config" / "camcan_movie.yaml").read_text())
            output_root = Path(cfg["output_root"])
        from fmri_decomposition.io import cohort_meta_dir
        out_dir = cohort_meta_dir(output_root, args.cohort)

    var_path = out_dir / "codebook_variables.tsv"
    val_path = out_dir / "codebook_values.tsv"
    write_tsv(variables, VARIABLE_COLUMNS, var_path)
    write_tsv(values, VALUE_COLUMNS, val_path)

    print(f"\nwrote {var_path}  ({len(variables)} row(s))")
    print(f"      {val_path}  ({len(values)} row(s), "
          f"{len({v['code'] for v in values})} coded variable(s))")
    print("\nlook something up:")
    print(f"  grep -P '^homeint_v112\\t' {var_path.name}")
    print(f"  grep -P '^DG7\\t' {val_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
