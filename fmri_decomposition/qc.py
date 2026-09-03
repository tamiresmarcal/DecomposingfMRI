"""Per-subject QC metrics, computed from stage-2 output.

Four measurements, chosen to be about four DIFFERENT failures. A long list of
correlated criteria costs sample size while only looking rigorous; these are
the axes that catch things the others cannot:

    mean_fd                 head movement            -> motion.py
    best_lag_tr             wrong stimulus timing    -> validate.isc_alignment
    frac_stimulus_covered   incomplete scan          -> here
    frac_parcels_empty      registration failure     -> here
    frac_good_frames        scrubbing survival       -> here (cohort-dependent)

Everything here reads the activation parquet, never a NIfTI, and nothing here
writes anything. `tools/make_participants.py` is the sole writer; this module
only computes. That split is deliberate: exclusions live in one file, under one
marker convention, with one writer.

WHY frac_stimulus_covered EXISTS
--------------------------------
Neither cohort configures `stimulus.durations_s`, so `dfc.py` builds each
subject's window grid from THAT SUBJECT'S OWN file length. Same origin, same
stride -- so window ids still line up -- but a subject whose scan stopped early
simply has no rows in the late windows. No error, no flag, no manifest entry.
The group mean for late-movie windows is then computed over fewer people than
for early ones, and nothing in the outputs says so. This is the one metric here
that is about correctness rather than quality.

WHY frac_parcels_empty IS THE REGISTRATION CHECK
------------------------------------------------
`extract_parcels` writes NaN for any parcel with no voxels under the subject's
brain mask. A handful is normal at the edges. Thirty out of Harvard-Oxford's
111 means the brain is not where MNI space says it should be -- which is what a
failed normalisation looks like, measured by its consequence rather than by
re-inspecting a transform we did not compute.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_shard

# Auditory cortex tracks a film's soundtrack more reliably than anything else,
# which is what makes it the conventional ISC seed. Matched case-insensitively
# against the QC atlas's column names, in order, first hit wins.
ISC_PARCEL_PATTERNS = (r"heschl", r"transverse.*temporal", r"\bauditory\b",
                       r"sommot|somatomotor", r"\bvisual\b|\bvis\b")


@dataclass
class SubjectQC:
    """One (sub, task). Scalars only, so it joins straight onto a CSV row."""

    n_tr_total: int = 0
    n_tr_good: int = 0
    frac_good_frames: float = float("nan")
    stimulus_duration_s: float = float("nan")
    frac_stimulus_covered: float = float("nan")
    n_parcels: int = 0
    n_parcels_empty: int = 0
    frac_parcels_empty: float = float("nan")
    qc_atlas: str = ""
    peak_isc: float = float("nan")
    best_lag_tr: float = float("nan")
    n_isc_subjects: int = 0
    qc_note: str = ""

    def as_row(self) -> dict:
        return asdict(self)


def _note(existing: str, addition: str) -> str:
    return "; ".join(x for x in (existing, addition) if x)


# ------------------------------------------------------------ activation ---
def activation_qc(files, atlas, tr: float) -> dict[tuple[str, str], dict]:
    """Walk one atlas's activation shards -> per (sub, task) QC.

    A (sub, task) with several ses/run shards is aggregated: frames sum, the
    duration is the longest shard's, and the empty-parcel count is the WORST
    shard's. A single badly-registered session is a finding, not something to
    average away.
    """
    cols = list(atlas.columns)
    acc: dict[tuple[str, str], dict] = {}
    for path in files:
        df = read_shard(path)
        key = (str(df["sub"].iloc[0]), str(df["task"].iloc[0]))
        present = [c for c in cols if c in df.columns]
        good = (df["good_frame"].to_numpy(bool) if "good_frame" in df
                else np.ones(len(df), bool))
        duration = (float(df["stimulus_time_s"].max()) + float(tr)
                    if "stimulus_time_s" in df and len(df) else len(df) * float(tr))
        # A parcel is empty exactly when extract_parcels wrote it as all-NaN,
        # which it does for `counts == 0` -- no voxel of this brain inside it.
        empty = int(sum(1 for c in present
                        if not np.isfinite(df[c].to_numpy(dtype=float)).any()))

        cur = acc.setdefault(key, {"n_tr_total": 0, "n_tr_good": 0,
                                   "stimulus_duration_s": 0.0, "n_parcels": len(present),
                                   "n_parcels_empty": 0, "qc_atlas": atlas.name,
                                   "qc_note": ""})
        cur["n_tr_total"] += int(len(df))
        cur["n_tr_good"] += int(good.sum())
        cur["stimulus_duration_s"] = max(cur["stimulus_duration_s"], duration)
        cur["n_parcels_empty"] = max(cur["n_parcels_empty"], empty)
        if len(present) != len(cols):
            cur["qc_note"] = _note(cur["qc_note"],
                                   f"{len(cols) - len(present)} parcel column(s) absent "
                                   f"from the shard")
    for cur in acc.values():
        cur["frac_good_frames"] = (cur["n_tr_good"] / cur["n_tr_total"]
                                   if cur["n_tr_total"] else float("nan"))
        cur["frac_parcels_empty"] = (cur["n_parcels_empty"] / cur["n_parcels"]
                                     if cur["n_parcels"] else float("nan"))
    return acc


def add_stimulus_coverage(qc: dict[tuple[str, str], dict]) -> None:
    """frac_stimulus_covered, relative to the longest scan of the SAME stimulus.

    Per task, not per cohort: ds002837's films run 95 to 154 minutes, so a
    cohort-wide reference would mark every Pulp Fiction viewer as complete and
    every 500 Days viewer as 60% covered. The reference is the longest observed
    scan rather than a configured duration because no duration is configured --
    it is the same fallback `dfc.py` uses to build the grid, so the two agree
    by construction.
    """
    longest: dict[str, float] = {}
    for (_, task), cur in qc.items():
        d = float(cur.get("stimulus_duration_s") or 0.0)
        longest[task] = max(longest.get(task, 0.0), d)
    for (_, task), cur in qc.items():
        ref = longest.get(task, 0.0)
        d = float(cur.get("stimulus_duration_s") or 0.0)
        cur["frac_stimulus_covered"] = (d / ref) if ref > 0 else float("nan")


# ------------------------------------------------------------------- ISC ---
def default_isc_parcels(atlas) -> tuple[list[str], str]:
    """Pick an ISC seed from the QC atlas. Returns (columns, how)."""
    for pattern in ISC_PARCEL_PATTERNS:
        hits = [c for c in atlas.columns if re.search(pattern, c, flags=re.I)]
        if hits:
            return hits, f"matched /{pattern}/ in {atlas.name}"
    return list(atlas.columns[:1]), (
        f"NO auditory or visual parcel in {atlas.name}; fell back to "
        f"{atlas.columns[0]!r}, which is arbitrary -- pass --isc-parcels"
    )


def isc_qc(files, atlas, parcels=None, max_lag_tr: int = 30):
    """(sub, task) -> peak_isc / best_lag_tr, or an empty dict with a reason."""
    from .validate import isc_alignment

    how = "supplied by the caller"
    if not parcels:
        parcels, how = default_isc_parcels(atlas)
    try:
        isc = isc_alignment(files, parcels=tuple(parcels), max_lag_tr=max_lag_tr)
    except (ValueError, KeyError) as exc:                        # noqa: BLE001
        return {}, f"ISC skipped: {type(exc).__name__}: {exc}"
    out = {}
    for row in isc.to_dict("records"):
        out[(str(row["sub"]), str(row["movie"]))] = {
            "peak_isc": float(row["peak_isc"]),
            "best_lag_tr": float(row["best_lag_tr"]),
            "n_isc_subjects": int(row.get("n_subjects") or 0),
            "qc_note": str(row.get("isc_note") or ""),
        }
    return out, f"ISC seed: {', '.join(parcels[:3])} ({how})"


# ------------------------------------------------------------ collection ---
def pick_qc_atlas(cfg, atlases):
    """The atlas the QC is measured on: the finest one that has shards.

    Empty parcels are the point. Yeo's 7 networks are so large that a
    normalisation would have to fail catastrophically to empty one, while
    Harvard-Oxford's 111 register the same failure as a count you can threshold.
    """
    from .io import activation_root

    with_shards = [a for a in atlases
                   if any(activation_root(cfg.output_root, a.name, cfg.cohort)
                          .rglob("*.parquet"))]
    if not with_shards:
        return None
    return max(with_shards, key=lambda a: a.n_nodes)


def collect_qc(cfg, atlases, isc_parcels=None, max_lag_tr: int = 30):
    """Everything above, for one cohort. Returns (rows_by_key, messages)."""
    from .io import activation_root

    messages: list[str] = []
    atlas = pick_qc_atlas(cfg, atlases)
    if atlas is None:
        return {}, ["no activation shards found -- run `extract` first"]
    files = sorted(activation_root(cfg.output_root, atlas.name, cfg.cohort)
                   .rglob("*.parquet"))
    messages.append(f"QC atlas: {atlas.name} ({atlas.n_nodes} nodes), {len(files)} shard(s)")

    qc = activation_qc(files, atlas, cfg.tr)
    add_stimulus_coverage(qc)

    isc, isc_msg = isc_qc(files, atlas, isc_parcels, max_lag_tr)
    messages.append(isc_msg)
    for key, row in isc.items():
        cur = qc.setdefault(key, {})
        note = row.pop("qc_note", "")
        cur.update(row)
        if note:
            cur["qc_note"] = _note(cur.get("qc_note", ""), note)

    # Fill every key to the full schema so the CSV columns are never ragged.
    blank = SubjectQC().as_row()
    return {k: {**blank, **v} for k, v in qc.items()}, messages


QC_COLUMNS = list(SubjectQC().as_row().keys())
