#!/usr/bin/env python3
"""Symptom severity in Cam-CAN and HCP, side by side -- and why the bands do not line up.

    python3 tools/symptom_distribution.py \
        --hcp-pheno    /project/6008063/tamires/cohorts/hcp_pheno/participants_pheno.csv \
        --hcp-subjects config/hcp7t_fix_participants.csv \
        --camcan-dir   "$CAMCAN" \
        --camcan-subjects config/camcan_participants.csv

Stdlib only -- no pandas, so it runs on a login node without $RUN23. That is
deliberate: this is the one table you want to be able to produce without
remembering to source anything.

THE THING TO READ BEFORE THE NUMBERS
------------------------------------
The two cohorts are measured with instruments whose published cutoffs were
drawn for DIFFERENT purposes, so the BAND PERCENTAGES ARE NOT COMPARABLE even
though both read as "% mild or worse".

  HADS (Cam-CAN), each subscale 0-21, bands 0-7 / 8-10 / 11-14 / 15-21
  = Normal / Mild / Moderate / Severe (Zigmond & Snaith 1983).
  The 8+ cutoff is a SCREENING threshold, chosen for sensitivity: in general
  population samples something like a fifth to a third of people clear it.

  ASR (HCP), reported as T-scores normed on a reference sample.
  DSM-oriented and syndrome scales: <65 Normal, 65-69 Borderline, >=70
  Clinical. Broad-band scales (Internalizing, Externalizing, Total):
  <60 Normal, 60-63 Borderline, >=64 Clinical (Achenbach & Rescorla 2003).
  T>=65 IS THE 93RD PERCENTILE BY CONSTRUCTION -- about 7% of the reference
  population, because that is what a T-score means.

So "7% clinical in HCP" and "25% mild-or-worse in Cam-CAN" are not a finding
about the two samples; they are mostly a fact about where two different
committees drew a line. Reading them as a difference in symptom burden is the
error this header exists to prevent.

WHAT IS ACTUALLY COMPARABLE
---------------------------
Three things, and the output gives all three:

  1. WITHIN-COHORT PERCENTILES of the raw score. Scale-free. If the top decile
     of Cam-CAN sits at HADS 11 and the top decile of HCP at DSM_Anxi 6, that
     says something about spread without needing the two scales to agree.

  2. THE FLOOR. The fraction scoring the minimum. This is the number that
     decides whether a regression target has usable variance: Spearman over a
     column where 60% of subjects tie at zero has far less power than its n
     suggests, and no amount of extra subjects fixes a target that is mostly
     floor.

  3. THE SPREAD AS A FRACTION OF RANGE (sd / scale max). Also scale-free, and
     the quantity that enters a power calculation.

RAW VS T, AND WHY BOTH APPEAR HERE
----------------------------------
The bands only exist on T. The MODELLING target should be RAW: T-scores are
normed by age and sex in the reference sample, so regressing a T-score against
brain states silently removes age and sex in a way Cam-CAN's HADS never had,
and the two cohorts stop being comparable in exactly the dimension the design
cares about. So: T for describing the sample, raw for fitting. This tool
prints both and labels which is which.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

# ------------------------------------------------------------------ bands ---
HADS_BANDS = [(0, 7, "Normal"), (8, 10, "Mild"), (11, 14, "Moderate"), (15, 21, "Severe")]
HADS_MAX = 21

# ASR T-score cutoffs. Two sets, because the broad-band scales use a lower one.
ASR_T_NARROW = [(0, 64, "Normal"), (65, 69, "Borderline"), (70, 200, "Clinical")]
ASR_T_BROAD = [(0, 59, "Normal"), (60, 63, "Borderline"), (64, 200, "Clinical")]

# The HCP columns worth looking at, with which cutoff applies and what each is.
HCP_SCALES = [
    ("DSM_Anxi", ASR_T_NARROW, "DSM-oriented anxiety      <- pair with HADS anxiety"),
    ("DSM_Depr", ASR_T_NARROW, "DSM-oriented depression   <- pair with HADS depression"),
    ("ASR_Anxd", ASR_T_NARROW, "syndrome anxious/depressed (mixes the two)"),
    ("ASR_Intn", ASR_T_BROAD, "broad internalizing (Anxd+Witd+Soma; more items, less floor)"),
]


def band_of(x: float, bands) -> str:
    for lo, hi, name in bands:
        if lo <= x <= hi:
            return name
    return "out-of-range"


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:5.1f}%" if total else "    --"


def nums(rows, col):
    out = []
    for r in rows:
        v = (r.get(col) or "").strip()
        if v in ("", "NA", "NaN", "nan", "."):
            continue
        try:
            out.append(float(v))
        except ValueError:
            pass
    return out


def quantiles(v: list[float]) -> dict:
    v = sorted(v)
    n = len(v)
    q = lambda p: v[min(n - 1, int(p * n))]
    return {"min": v[0], "p25": q(.25), "med": q(.50), "p75": q(.75),
            "p90": q(.90), "p95": q(.95), "max": v[-1]}


def read_csv_any(path: Path, sep=None):
    """Read a csv/tsv, sniffing the separator if not given. utf-8-sig strips a BOM."""
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        head = f.readline()
        f.seek(0)
        if sep is None:
            sep = "\t" if head.count("\t") > head.count(",") else ","
        return list(csv.DictReader(f, delimiter=sep))


class MissingTable(FileNotFoundError):
    """A participants table that is derived, not versioned, and has not been built yet.

    Raised with the command that builds it, because the traceback from
    `Path.open` seven frames down says only that a file is absent -- not that
    it is a GENERATED file, nor which of several candidates in config/ the
    caller probably meant.
    """


def subject_ids(path) -> set[str]:
    """Subject ids from a participants table, whatever the id column is called.

    `path` may be SEVERAL tables (a list/tuple), in which case the ids are
    unioned. That is not a convenience: Cam-CAN's movie and rest runs were
    discovered as two cohorts before they were merged into one, so the subjects
    with imaging are the union of `camcan_movie_participants.csv` and
    `camcan_rest_participants.csv`. Taking only one of them would silently drop
    everyone scanned under the other condition.
    """
    if isinstance(path, (list, tuple, set)):
        out: set[str] = set()
        for one in path:
            out |= subject_ids(one)
        return out
    path = Path(path)
    if not path.exists():
        near = sorted(p.name for p in path.parent.glob("*participants*.csv")) \
            if path.parent.is_dir() else []
        cfg = path.name.replace("_participants.csv", ".yaml")
        raise MissingTable(
            f"{path} does not exist.\n"
            f"  Participants tables are DERIVED -- built where the data is, not committed.\n"
            f"  Build it:   python3 tools/make_participants.py config/{cfg} -o {path}\n"
            + (f"  Or pass one of these, which do exist in {path.parent}:\n"
               + "".join(f"    {n}\n" for n in near) if near else
               f"  No *participants*.csv found in {path.parent} at all.\n")
            + "  Or pass camcan_subjects=None to use every phenotype row, unmatched to imaging.")
    rows = read_csv_any(path)
    if not rows:
        sys.exit(f"{path}: no rows")
    for cand in ("sub", "subject", "participant_id", "CCID"):
        col = next((c for c in rows[0] if c.strip().lower() == cand.lower()), None)
        if col:
            break
    else:
        sys.exit(f"{path}: no subject column among {list(rows[0])}")
    return {re.sub(r"^sub-", "", (r[col] or "").strip()).upper()
            for r in rows if (r[col] or "").strip()}


def merge_on(frames, id_col):
    """Outer-merge a list of row-lists on id_col, first frame's value winning."""
    out: dict[str, dict] = {}
    for rows in frames:
        for r in rows:
            k = (r.get(id_col) or "").strip().upper()
            if not k:
                continue
            tgt = out.setdefault(k, {})
            for c, v in r.items():
                if c not in tgt or not (tgt[c] or "").strip():
                    tgt[c] = v
    return out


# ------------------------------------------------- the whole HCP instrument ---
# Everything the ASR/SSAGA battery offers, not just the four scales that pair
# with HADS. Grouped by family, because the families mean different things and
# should not be read as one ranked list:
#
#   DSM-oriented   the scales built to line up with DSM categories. These are
#                  what pairs with a HADS subscale.
#   syndrome       the empirically derived ASR syndromes. Narrower, and
#                  `Anxd` deliberately mixes anxiety with depression.
#   broad-band     sums over several syndromes. MORE ITEMS, so a lower floor
#                  and more resolution -- often the better regression target
#                  even with no HADS counterpart.
#   SSAGA          diagnostic-interview counts and flags, a different kind of
#                  measurement: some are 0/1, where a Spearman is a point-
#                  biserial and the floor fraction means something else.
HCP_FAMILIES = {
    "DSM-oriented": ["DSM_Depr", "DSM_Anxi", "DSM_Somp", "DSM_Avoid",
                     "DSM_Adh", "DSM_Inat", "DSM_Hype", "DSM_Antis"],
    "syndrome": ["ASR_Anxd", "ASR_Witd", "ASR_Soma", "ASR_Thot", "ASR_Attn",
                 "ASR_Aggr", "ASR_Rule", "ASR_Intr", "ASR_Oth", "ASR_Crit"],
    "broad-band": ["ASR_Intn", "ASR_Extn", "ASR_TAO_Sum", "ASR_Totp"],
    "SSAGA": ["SSAGA_ChildhoodConduct", "SSAGA_PanicDisorder",
              "SSAGA_Agoraphobia", "SSAGA_Depressive_Ep",
              "SSAGA_Depressive_Sx"],
}

HCP_LABEL = {
    "DSM_Depr": "depressive", "DSM_Anxi": "anxiety", "DSM_Somp": "somatic",
    "DSM_Avoid": "avoidant personality", "DSM_Adh": "ADH problems",
    "DSM_Inat": "inattention", "DSM_Hype": "hyperactivity",
    "DSM_Antis": "antisocial personality",
    "ASR_Anxd": "anxious/depressed", "ASR_Witd": "withdrawn",
    "ASR_Soma": "somatic complaints", "ASR_Thot": "thought problems",
    "ASR_Attn": "attention problems", "ASR_Aggr": "aggressive behaviour",
    "ASR_Rule": "rule-breaking", "ASR_Intr": "intrusive",
    "ASR_Oth": "other problems", "ASR_Crit": "critical items",
    "ASR_Intn": "internalizing", "ASR_Extn": "externalizing",
    "ASR_TAO_Sum": "total adaptive/other", "ASR_Totp": "total problems",
    "SSAGA_ChildhoodConduct": "childhood conduct",
    "SSAGA_PanicDisorder": "panic disorder", "SSAGA_Agoraphobia": "agoraphobia",
    "SSAGA_Depressive_Ep": "depressive episode",
    "SSAGA_Depressive_Sx": "depressive symptoms",
}

# The two that pair with a HADS subscale, marked so a figure can say so.
HCP_HADS_PAIR = {"DSM_Anxi": "HADS anxiety", "DSM_Depr": "HADS depression"}


def hcp_skipped(hcp_pheno, hcp_subjects) -> list[dict]:
    """Symptom-looking columns the inventory did NOT include, and why.

    `hcp_scales` keeps only what parses as a number, which is the right rule
    and a silent one: a column of "Yes"/"No", or one this module's family
    lists do not name, simply never appears, and a reader cannot tell a
    deliberate omission from an oversight. This says which, so the absence of
    a scale from a figure is answerable rather than mysterious.
    """
    rows_all, cols, _ = _hcp_rows(Path(hcp_pheno), Path(hcp_subjects))
    known = {f"{s}_Raw" for v in HCP_FAMILIES.values() for s in v} \
        | {s for v in HCP_FAMILIES.values() for s in v}
    out = []
    for c in cols:
        if c in known or c.endswith("_T"):
            continue
        if not any(k in c.upper() for k in ("ASR", "DSM", "SSAGA")):
            continue
        vals = [(r.get(c) or "").strip() for r in rows_all]
        present = [v for v in vals if v not in ("", "NA", "NaN", ".")]
        numeric = len(nums(rows_all, c))
        out.append({"column": c, "n_present": len(present),
                    "n_numeric": numeric,
                    "why": ("not in any family list" if numeric == len(present)
                            else f"{len(present) - numeric} non-numeric value(s)"),
                    "examples": sorted(set(present))[:5]})
    return out


def hcp_scales(hcp_pheno, hcp_subjects) -> list[dict]:
    """Every symptom scale in the HCP table, summarised and sorted by family.

    RAW only. The T-scores exist for the ASR scales, but a T-score has age and
    sex normed out of it in the reference sample, so fitting one removes age
    and sex from this cohort in a way Cam-CAN's HADS never had. T is for
    describing the sample; raw is for fitting, and this function is for
    choosing what to fit.

    Sorted within each family by floor fraction, ascending -- i.e. the most
    usable target first. That ordering is the point of the table: a scale
    where most subjects tie at the minimum has far less power than its n
    suggests, and no extra subjects fix it.

    A binary column (two distinct values) is FLAGGED rather than ranked
    alongside the counts: its floor fraction is just its base rate, which is
    not the same quantity, and Spearman on it is a point-biserial.
    """
    rows_all, cols, _ = _hcp_rows(Path(hcp_pheno), Path(hcp_subjects))
    out = []
    for family, stems in HCP_FAMILIES.items():
        got = []
        for stem in stems:
            col = f"{stem}_Raw" if f"{stem}_Raw" in cols else (
                stem if stem in cols else None)
            if col is None:
                continue
            v = nums(rows_all, col)
            if not v:
                continue
            s = _summary(v)
            n_distinct = len(set(v))
            got.append({
                "family": family, "stem": stem, "column": col,
                "label": HCP_LABEL.get(stem, stem),
                "pairs_with": HCP_HADS_PAIR.get(stem, ""),
                "n": s["n"], "median": s["med"], "p75": s["p75"],
                "p90": s["p90"], "p95": s["p95"], "max": s["max"],
                "sd": round(s["sd"], 2), "n_distinct": n_distinct,
                "binary": n_distinct <= 2,
                "pct_at_floor": round(100 * s["floor_frac"], 1),
                # For a 0/1 flag the floor fraction reads backwards to anyone
                # scanning the figure: a 93% bar on "agoraphobia" looks like
                # 93% HAVE it, when it means 93% scored 0 and 7% have it. So
                # the positive rate is carried explicitly rather than left for
                # the reader to subtract.
                "pct_above_floor": round(100 * (1 - s["floor_frac"]), 1),
                "floor_value": s["floor_value"],
                "values": v,
            })
        out += sorted(got, key=lambda r: (r["binary"], r["pct_at_floor"]))
    return out


# -------------------------------------------------------------------- load ---
# The notebook (notebooks/12_symptom_distributions.ipynb) imports `load`, so the
# band definitions and the id matching live in exactly one place. Returning
# plain lists and Counters keeps this module stdlib-only; the notebook is free
# to put them in a DataFrame.

def _camcan_rows(camcan_dir: Path, camcan_subjects=None):
    files = [p for p in (camcan_dir / "approved_data.tsv",
                         camcan_dir / "standard_data.csv") if p.exists()]
    if not files:
        return [], []
    ph = merge_on([read_csv_any(p) for p in files], "CCID")
    if camcan_subjects:
        keep = subject_ids(camcan_subjects)
    else:
        keep = None
        print("!! Cam-CAN not restricted to an imaging cohort: every phenotype row is "
              "included.\n   The two cohorts' n are then not matched to who actually has "
              "scans.", file=sys.stderr)
    rows = [r for k, r in ph.items() if keep is None or k in keep]
    cols = list(next(iter(ph.values()))) if ph else []
    return rows, cols


def _hcp_rows(hcp_pheno: Path, hcp_subjects: Path):
    ph = {(r.get("Subject") or "").strip().upper(): r for r in read_csv_any(hcp_pheno)}
    keep = subject_ids(hcp_subjects)
    rows = [r for k, r in ph.items() if k in keep]
    cols = list(next(iter(ph.values()))) if ph else []
    return rows, cols, sorted(keep - set(ph))


def _summary(v: list[float], scale_max=None) -> dict:
    if not v:
        return {}
    q = quantiles(v)
    floor_v = min(v)
    return {**q, "n": len(v), "sd": st.pstdev(v) if len(v) > 1 else 0.0,
            "floor_value": floor_v,
            "floor_frac": sum(1 for x in v if x == floor_v) / len(v),
            "scale_max": scale_max,
            "sd_over_max": (st.pstdev(v) / scale_max) if (scale_max and len(v) > 1) else None}


def load(hcp_pheno, hcp_subjects, camcan_dir=None, camcan_subjects=None) -> dict:
    """Both cohorts' symptom scores, bands and summaries, in one nested dict.

    {cohort: {"n": int, "instrument": str, "missing_pheno": [ids],
              "scales": {construct: {"raw": [...], "raw_col": str,
                                     "band": Counter, "band_order": [...],
                                     "cutoffs": [...], "scale_max": int|None,
                                     "banded_on": "raw"|"T"|"category",
                                     "summary": {...}, "t": [...]|None,
                                     "what": str}}}}

    `zero_n` is how many subjects score the scale minimum. A figure may subtract
    it from the LOWEST band to show "zero" as its own segment, and that
    arithmetic needs no per-subject join: a raw score of 0 is below every
    cutoff on every scale here, so a zero-scoring subject is necessarily in the
    lowest band. It is only valid when the raw and band counts have the same n
    (different missingness would break it), which is why both are returned.

    `cutoffs` are the lower edges of every band above the first, so a figure can
    draw them as rules without re-deriving them. `banded_on` records WHICH
    quantity the bands were applied to -- "category" when the cohort ships its
    own banding, "T" for the ASR, which has no published raw cutoff. That
    distinction is the whole caveat of this module, so it travels with the data.
    """
    out: dict = {}

    if camcan_dir:
        rows, cols = _camcan_rows(Path(camcan_dir), camcan_subjects or None)
        hads = sorted(c for c in cols if "hads" in c.lower())
        scales = {}
        for which, pat in (("anxiety", "anx"), ("depression", "dep")):
            cat = next((c for c in hads if pat in c.lower() and "categ" in c.lower()), None)
            raw = next((c for c in hads if pat in c.lower() and "categ" not in c.lower()), None)
            v = nums(rows, raw) if raw else []
            if cat:
                band = Counter((r.get(cat) or "").strip() for r in rows
                               if (r.get(cat) or "").strip() not in ("", "NA"))
                banded_on = "category"
            elif v:
                band = Counter(band_of(x, HADS_BANDS) for x in v)
                banded_on = "raw"
            else:
                band, banded_on = Counter(), None
            scales[which] = {
                "raw": v, "raw_col": raw, "t": None,
                "band": band, "band_order": ["Normal", "Mild", "Moderate", "Severe"],
                "cutoffs": [8, 11, 15], "scale_max": HADS_MAX, "banded_on": banded_on,
                "summary": _summary(v, HADS_MAX),
                "zero_n": sum(1 for x in v if x == 0),
                "what": f"HADS {which} (0-{HADS_MAX})"}
        out["camcan"] = {"n": len(rows), "instrument": "HADS",
                         "missing_pheno": [], "scales": scales}

    rows, cols, missing = _hcp_rows(Path(hcp_pheno), Path(hcp_subjects))
    scales = {}
    name_of = {"DSM_Anxi": "anxiety", "DSM_Depr": "depression",
               "ASR_Anxd": "anxious/depressed", "ASR_Intn": "internalizing"}
    for stem, bands, what in HCP_SCALES:
        tcol, rcol = f"{stem}_T", f"{stem}_Raw"
        if tcol not in cols and rcol not in cols:
            continue
        tv = nums(rows, tcol) if tcol in cols else []
        rv = nums(rows, rcol) if rcol in cols else []
        scales[name_of.get(stem, stem)] = {
            "raw": rv, "raw_col": rcol if rcol in cols else None, "t": tv or None,
            "band": Counter(band_of(x, bands) for x in tv) if tv else Counter(),
            "band_order": ["Normal", "Borderline", "Clinical"],
            "cutoffs": [bands[1][0], bands[2][0]], "scale_max": None,
            "banded_on": "T" if tv else None,
            "summary": _summary(rv), "zero_n": sum(1 for x in rv if x == 0),
            "what": what, "stem": stem}
    out["hcp"] = {"n": len(rows), "instrument": "ASR/DSM-oriented",
                  "missing_pheno": missing, "scales": scales}
    return out


# ----------------------------------------------------------------- report ---
def banded(title, counts, order, total, note=""):
    print(f"\n  {title}   n={total}")
    if note:
        print(f"    {note}")
    for name in order:
        n = counts.get(name, 0)
        bar = "#" * round(40 * n / total) if total else ""
        print(f"    {name:11s} {n:5d}  {pct(n, total)}  {bar}")
    extra = set(counts) - set(order)
    for name in sorted(extra):
        print(f"    {name:11s} {counts[name]:5d}  {pct(counts[name], total)}  <-- UNEXPECTED")
    atleast = sum(counts.get(x, 0) for x in order[1:])
    print(f"    {'>= ' + order[1]:11s} {atleast:5d}  {pct(atleast, total)}")


def describe_raw(label, v, scale_max=None):
    if not v:
        print(f"    {label:24s} EMPTY")
        return
    q = quantiles(v)
    floor = sum(1 for x in v if x == min(v)) / len(v)
    sd = st.pstdev(v) if len(v) > 1 else 0.0
    frac = f"  sd/max={sd / scale_max:.3f}" if scale_max else ""
    print(f"    {label:24s} n={len(v):4d}  med={q['med']:5.1f}  sd={sd:5.2f}{frac}")
    print(f"    {'':24s} min={q['min']:5.1f}  p25={q['p25']:5.1f}  p75={q['p75']:5.1f} "
          f"p90={q['p90']:5.1f}  p95={q['p95']:5.1f}  max={q['max']:5.1f}")
    print(f"    {'':24s} at floor ({q['min']:.0f}): {100 * floor:.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Symptom severity bands and raw distributions, Cam-CAN vs HCP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Read the module docstring first: the band PERCENTAGES are not "
               "comparable between the two instruments, and the output says so too.")
    ap.add_argument("--hcp-pheno", type=Path, required=True,
                    help="HCP restricted+unrestricted phenotype csv (keyed by Subject).")
    ap.add_argument("--hcp-subjects", type=Path, required=True,
                    help="participants csv of the imaging cohort, to restrict to.")
    ap.add_argument("--camcan-dir", type=Path,
                    help="Cam-CAN access directory holding approved_data.tsv and standard_data.csv.")
    ap.add_argument("--camcan-subjects", type=Path, nargs="+",
                    help="Cam-CAN participants csv/s to restrict to; several are unioned "
                         "(movie and rest were separate cohorts before the merge).")
    a = ap.parse_args()

    # ----------------------------------------------------------- Cam-CAN ---
    if a.camcan_dir:
        files = [p for p in (a.camcan_dir / "approved_data.tsv",
                             a.camcan_dir / "standard_data.csv") if p.exists()]
        if not files:
            print(f"!! no approved_data.tsv / standard_data.csv under {a.camcan_dir}",
                  file=sys.stderr)
        else:
            ph = merge_on([read_csv_any(p) for p in files], "CCID")
            keep = subject_ids(a.camcan_subjects) if a.camcan_subjects else None
            rows = [r for k, r in ph.items() if keep is None or k in keep]
            cols = list(next(iter(ph.values()))) if ph else []
            hads = sorted(c for c in cols if "hads" in c.lower())

            print("=" * 78)
            print("CAM-CAN -- HADS")
            print("=" * 78)
            print(f"  phenotype rows: {len(ph)}"
                  + (f"   restricted to imaging cohort: {len(rows)} of {len(keep)}" if keep else ""))
            print(f"  HADS columns found: {hads}")

            for which, pat in (("anxiety", "anx"), ("depression", "dep")):
                cat = next((c for c in hads if pat in c.lower() and "categ" in c.lower()), None)
                raw = next((c for c in hads if pat in c.lower() and "categ" not in c.lower()), None)
                if cat:
                    cc = Counter((r.get(cat) or "").strip() for r in rows
                                 if (r.get(cat) or "").strip() not in ("", "NA"))
                    banded(f"HADS {which}  [{cat}]  -- Cam-CAN's own bands",
                           cc, ["Normal", "Mild", "Moderate", "Severe"], sum(cc.values()),
                           note="cutoffs 0-7 / 8-10 / 11-14 / 15-21 (Zigmond & Snaith); "
                                "8+ is a SCREENING threshold, deliberately sensitive")
                else:
                    print(f"\n  HADS {which}: no *_category column found")
                if raw:
                    print(f"\n    raw score  [{raw}]   <-- THIS is the modelling target")
                    describe_raw(raw, nums(rows, raw), HADS_MAX)

    # --------------------------------------------------------------- HCP ---
    ph = {(r.get("Subject") or "").strip().upper(): r
          for r in read_csv_any(a.hcp_pheno)}
    keep = subject_ids(a.hcp_subjects)
    rows = [r for k, r in ph.items() if k in keep]
    missing = sorted(keep - set(ph))

    print()
    print("=" * 78)
    print("HCP -- ASR / DSM-oriented")
    print("=" * 78)
    print(f"  phenotype rows: {len(ph)}   imaging subjects: {len(keep)}   "
          f"with both: {len(rows)}")
    if missing:
        print(f"  !! {len(missing)} imaging subjects have NO phenotype row: {missing[:10]}")

    for stem, bands, what in HCP_SCALES:
        tcol, rcol = f"{stem}_T", f"{stem}_Raw"
        cols = list(next(iter(ph.values()))) if ph else []
        if tcol not in cols and rcol not in cols:
            print(f"\n  {stem}: neither {tcol} nor {rcol} in the file")
            continue
        print()
        print("-" * 78)
        print(f"  {stem}  --  {what}")
        if tcol in cols:
            v = nums(rows, tcol)
            cc = Counter(band_of(x, bands) for x in v)
            lo = bands[1][0]
            banded(f"bands on {tcol}", cc, ["Normal", "Borderline", "Clinical"], len(v),
                   note=f"cutoffs T<{lo} / {lo}-{bands[1][1]} / >={bands[2][0]}  "
                        f"(Achenbach & Rescorla). T>={lo} is roughly the "
                        f"{'93rd' if lo == 65 else '84th'} percentile BY CONSTRUCTION")
            print()
            describe_raw(f"{tcol} (normed by age/sex)", v)
        if rcol in cols:
            print(f"\n    raw score  [{rcol}]   <-- THIS is the modelling target")
            describe_raw(rcol, nums(rows, rcol))
            v = nums(rows, rcol)
            if v:
                cc = Counter(int(x) for x in v)
                top = dict(sorted(cc.items())[:12])
                print(f"    {'':24s} value counts (low end): {top}")

    # ---------------------------------------------------------- families ---
    fam = Counter((r.get("Family_ID") or "").strip() for r in rows
                  if (r.get("Family_ID") or "").strip())
    if fam:
        sizes = Counter(fam.values())
        print()
        print("-" * 78)
        print(f"  families: {len(fam)} for {sum(fam.values())} subjects")
        print(f"  cluster sizes: {dict(sorted(sizes.items()))}")
        print(f"  -> GroupKFold by Family_ID has {len(fam)} groups; largest cluster "
              f"{max(sizes)} subject(s)")

    print()
    print("=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("""  Do NOT compare "% mild-or-worse" against "% borderline-or-worse". HADS 8+
  is a sensitive screening line; ASR T>=65 is the 93rd percentile by
  construction. A gap between those two numbers is mostly a gap between two
  conventions.

  Compare instead:
    - the percentile rows (p75 / p90 / p95) of the RAW scores, each within its
      own cohort and scale;
    - the floor fraction, which is what limits a Spearman target;
    - sd/max where both scales have a known maximum.

  And fit on RAW, not T: a T-score has age and sex normed out of it, which
  HADS does not, so a T-score target quietly makes the two cohorts
  incomparable in the dimension the design is about.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
