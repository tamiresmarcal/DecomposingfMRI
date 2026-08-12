"""Discovery, the participants join, and validation.

This module and `config.py` are the only places that know a filesystem path,
an entity layout, or a TR. Swapping cohorts means writing a config and, at
worst, a discovery function -- never touching stages 2-4.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .activation import RunRef
from .config import CohortConfig, ConfigError
from .timing import RunSegment

PARTICIPANT_REQUIRED = ["participant_id", "sub", "cohort", "task", "excluded"]


class ValidationError(RuntimeError):
    pass


# ------------------------------------------------------------ discovery ---
def discover_runs(cfg: CohortConfig) -> list[RunRef]:
    if cfg.discovery.backend == "pybids":
        try:
            return _discover_pybids(cfg)
        except ImportError:
            pass                       # documented fallback, not a silent one
    return _discover_glob(cfg)


def _discover_glob(cfg: CohortConfig) -> list[RunRef]:
    sub_re = re.compile(cfg.discovery.subject_pattern)
    task_re = re.compile(cfg.discovery.task_pattern)
    ses_re, run_re, acq_re = (re.compile(f"{k}-([A-Za-z0-9]+)") for k in ("ses", "run", "acq"))

    refs: list[RunRef] = []
    for path in sorted(Path(cfg.derivatives_root).glob(cfg.discovery.bold_glob)):
        name = path.name
        m_sub, m_task = sub_re.search(str(path)), task_re.search(name)
        if not (m_sub and m_task):
            continue
        task = m_task.group(1)
        if cfg.discovery.include_tasks and task not in cfg.discovery.include_tasks:
            continue
        if task in cfg.discovery.exclude_tasks:
            continue
        mask = None
        if cfg.discovery.mask_glob:
            hits = sorted(path.parent.glob(cfg.discovery.mask_glob.format(
                sub=m_sub.group(1), task=task)))
            mask = hits[0] if hits else None
        refs.append(RunRef(
            cohort=cfg.cohort, sub=m_sub.group(1), task=task, bold=path, mask=mask,
            ses=(ses_re.search(name).group(1) if ses_re.search(name) else None),
            run=(run_re.search(name).group(1) if run_re.search(name) else None),
            acq=(acq_re.search(name).group(1) if acq_re.search(name) else None),
            timing_source=cfg.stimulus.timing_source,
        ))
    return refs


def _discover_pybids(cfg: CohortConfig) -> list[RunRef]:
    from bids import BIDSLayout

    layout = BIDSLayout(str(cfg.derivatives_root), validate=False, derivatives=True)
    refs = []
    for f in layout.get(suffix="bold", extension=".nii.gz"):
        ent = f.get_entities()
        task = ent.get("task")
        if cfg.discovery.include_tasks and task not in cfg.discovery.include_tasks:
            continue
        if task in cfg.discovery.exclude_tasks:
            continue
        refs.append(RunRef(
            cohort=cfg.cohort, sub=str(ent.get("subject")), task=str(task),
            bold=Path(f.path),
            ses=_opt(ent.get("session")), run=_opt(ent.get("run")),
            acq=_opt(ent.get("acquisition")),
            timing_source=cfg.stimulus.timing_source,
        ))
    return refs


def _opt(v):
    return None if v is None else str(v)


# ---------------------------------------------------------- participants ---
def load_participants(cfg: CohortConfig) -> pd.DataFrame:
    if cfg.participants is None:
        raise ConfigError("participants.csv path is not configured")
    df = pd.read_csv(cfg.participants, dtype={"sub": str})
    if "movie" in df.columns and "task" not in df.columns:
        df = df.rename(columns={"movie": "task"})
    missing = [c for c in PARTICIPANT_REQUIRED if c not in df.columns]
    if missing:
        raise ValidationError(f"participants.csv missing required column(s): {missing}")
    df["sub"] = df["sub"].astype(str)
    df["excluded"] = df["excluded"].astype(bool)
    return df


def attach_participants(refs: list[RunRef], participants: pd.DataFrame,
                        cfg: CohortConfig) -> list[RunRef]:
    """Join subject metadata onto discovered runs, dropping excluded subjects."""
    idx = participants.set_index(["sub", "task"], drop=False)
    out = []
    for ref in refs:
        key = (str(ref.sub), ref.task)
        if key not in idx.index:
            continue
        row = idx.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if bool(row["excluded"]):
            continue
        if cfg.trim.column and cfg.trim.column in row.index and pd.notna(row[cfg.trim.column]):
            ref.trim_end_s = float(row[cfg.trim.column])
            if cfg.trim.unit == "tr":
                ref.trim_end_s *= cfg.tr
        ref.extra.update(row.to_dict())
        out.append(ref)
    return out


def segments_for_run(ref: RunRef, scans_table: pd.DataFrame | None, cfg: CohortConfig):
    """Per-run segments from a scans.tsv-derived table, if one exists."""
    if scans_table is None:
        return []
    rows = scans_table[(scans_table["sub"].astype(str) == str(ref.sub))
                       & (scans_table["task"] == ref.task)].sort_values("run")
    if rows.empty:
        return []
    from .timing import segments_from_scans

    return segments_from_scans(
        rows["acq_time_s"].astype(float).tolist(),
        rows["n_vols"].astype(int).tolist(),
        cfg.tr,
        drop_after_restart=int(rows.get("drop_after_restart", pd.Series([0])).iloc[0]),
    )


# ------------------------------------------------------------ validation ---
def validate_cohort(cfg: CohortConfig, refs: list[RunRef],
                    participants: pd.DataFrame, strict: bool = True) -> list[str]:
    """Run before any extraction. Reports both directions of the disk/table join.

    Exclusions must be *recorded*, not omitted: if an excluded subject is
    simply absent from the table, "is this excluded or did someone forget?"
    becomes unanswerable and the validator cannot tell a curation decision
    from a data-entry slip.
    """
    problems: list[str] = []

    if participants["cohort"].nunique() != 1:
        problems.append(f"participants.csv spans multiple cohorts: "
                        f"{sorted(participants['cohort'].unique())}")
    elif participants["cohort"].iloc[0] != cfg.cohort:
        problems.append(f"participants.csv cohort {participants['cohort'].iloc[0]!r} "
                        f"!= config cohort {cfg.cohort!r}")

    if not pd.api.types.is_string_dtype(participants["sub"]):
        problems.append("participants.csv 'sub' must be string dtype "
                        "(Cam-CAN uses ids like CC110033)")

    dupes = participants.duplicated(subset=["sub", "task"], keep=False)
    if dupes.any():
        problems.append(f"duplicate (sub, task) keys: "
                        f"{participants.loc[dupes, ['sub', 'task']].to_dict('records')}")

    # Both directions are reported, but an *excluded* row is allowed to have no
    # file: that is exactly the NNDb case of 86 runs on disk and 85 analysable
    # rows, where the corrupted run is recorded rather than omitted.
    on_disk = {(str(r.sub), r.task) for r in refs}
    analysed = set(zip(participants.loc[~participants["excluded"], "sub"],
                       participants.loc[~participants["excluded"], "task"]))
    all_rows = set(zip(participants["sub"], participants["task"]))
    only_disk = sorted(on_disk - all_rows)
    only_table = sorted(analysed - on_disk)
    if only_disk:
        problems.append(f"{len(only_disk)} run(s) on disk with no table row: {only_disk[:5]} "
                        "-- record them as excluded rather than leaving them unlisted")
    if only_table:
        problems.append(f"{len(only_table)} analysable table row(s) with no run on disk: "
                        f"{only_table[:5]}")

    excluded_no_reason = participants[
        participants["excluded"]
        & participants.get("exclusion_reason", pd.Series(dtype=object)).isna()
    ]
    if len(excluded_no_reason):
        problems.append(f"{len(excluded_no_reason)} excluded row(s) with no "
                        "exclusion_reason -- no third category is allowed")

    if cfg.trim.column and cfg.trim.column in participants.columns:
        vals = participants[cfg.trim.column].dropna()
        if cfg.trim.unit == "seconds":
            n_tr = vals / cfg.tr
            bad = vals[(n_tr - n_tr.round()).abs() > 1e-6]
            if len(bad):
                problems.append(f"{cfg.trim.column} does not convert to whole TRs "
                                f"at TR={cfg.tr} for {len(bad)} row(s)")

    for task in sorted({r.task for r in refs}):
        if task not in cfg.stimulus.durations_s and not cfg.trim.column:
            problems.append(f"no stimulus duration configured for task {task!r}; "
                            "the window grid is defined per stimulus")

    if problems and strict:
        raise ValidationError("cohort validation failed:\n  - " + "\n  - ".join(problems))
    return problems
