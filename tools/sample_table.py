#!/usr/bin/env python3
"""Supplementary Table: samples, acquisition and quality control, per cohort.

    source slurm/env.sh
    apptainer exec --bind /project,/scratch,/home --pwd $PWD $FMRIDECOMP_SIF45 \
        python3 tools/sample_table.py -o figures/supplementary/sample_table

writes <out>.csv (every number, for checking) and <out>.tex (a booktabs table
to paste into the manuscript). Read-only on outputs/.

ONE ROW PER (cohort, condition). Everything is rebuilt from what is on disk, so
a cohort added later appears on the next run with nothing to edit here:

  role          train / projected          the latents' `role` column
  tasks         number of runs/episodes    distinct task= of that condition
  TR            seconds                    config/<cohort>.yaml (tr, tr_by_task)
  space         template                   config/<cohort>.yaml
  scanned       participants with data     the censor table (before gating)
  excluded      excluded by QC             censor `keep == False`, by reason
  final         participants analysed      the latents of the state space
  windows       30-s windows analysed      the latents
  mean FD       mean (s.d.) of kept        the censor table, `mean_fd`

The censor policy is read from the fit itself (decompose's description), so
the table describes the QC the states were actually fitted under.

Condition is `rest` when the task name matches --rest-pattern (default: any
task containing "rest", any case), otherwise `movie`.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd                                                  # noqa: E402
import pyarrow.parquet as pq                                         # noqa: E402
import yaml                                                          # noqa: E402

from fmri_decomposition.io import (censor_root, default_output_root,  # noqa: E402
                                   latents_root)

#: How a cohort directory is named in the manuscript. A cohort missing here is
#: printed under its directory name -- add a line rather than editing the table.
LABELS = {"ds002837": "NNDb", "cneuromod": "CNeuroMod Friends",
          "camcan": "Cam-CAN", "hcp7t_fix": "HCP 7T"}


def hive(path: Path, key: str) -> str:
    return next((p.split("=", 1)[1] for p in path.parts
                 if p.startswith(f"{key}=")), "")


def cohort_configs() -> dict:
    """cohort name -> its YAML, from config/*.yaml (the `cohort:` key)."""
    out = {}
    for f in sorted((REPO / "config").glob("*.yaml")):
        raw = yaml.safe_load(f.read_text()) or {}
        if isinstance(raw, dict) and raw.get("cohort"):
            out[str(raw["cohort"])] = raw
    return out


def tr_of(cfg: dict, tasks) -> str:
    by_task = cfg.get("tr_by_task") or {}
    trs = sorted({float(by_task.get(t, cfg.get("tr"))) for t in tasks
                  if by_task.get(t, cfg.get("tr")) is not None})
    return " / ".join(f"{t:g}" for t in trs) if trs else ""


def fit_policy(root: Path, atlas: str, window_s: str):
    from fmri_decomposition.unsupervised_model.decompose import (
        latents_fit_description)
    files = sorted(latents_root(root, atlas, window_s)
                   .glob("cohort=*/task=*/data.parquet"))
    if not files:
        raise SystemExit(f"no latents for atlas={atlas} window_s={window_s}")
    return latents_fit_description(files[0])[0].get("censor_policy"), files


def build(root: Path, atlas: str, window_s: str, rest_pattern: str,
          cohorts: list | None) -> pd.DataFrame:
    policy, files = fit_policy(root, atlas, window_s)
    cfgs = cohort_configs()
    is_rest = re.compile(rest_pattern, re.I)

    lat = []
    for f in files:
        cohort, task = hive(f, "cohort"), hive(f, "task")
        if cohorts and cohort not in cohorts:
            continue
        cols = [c for c in ("sub", "role") if c in pq.read_schema(f).names]
        t = pd.read_parquet(f, columns=cols)
        lat.append(t.assign(cohort=cohort, task=task, sub=t["sub"].astype(str)))
    lat = pd.concat(lat, ignore_index=True)
    lat["condition"] = ["rest" if is_rest.search(t) else "movie"
                        for t in lat["task"]]

    cen = []
    if policy:
        for f in sorted(censor_root(root, policy).glob(
                "cohort=*/task=*/subjects.parquet")):
            cohort = hive(f, "cohort")
            if cohorts and cohort not in cohorts:
                continue
            t = pd.read_parquet(f)
            cen.append(t.assign(cohort=cohort, task=hive(f, "task"),
                                sub=t["sub"].astype(str)))
    cen = pd.concat(cen, ignore_index=True) if cen else pd.DataFrame()
    if len(cen):
        cen["condition"] = ["rest" if is_rest.search(t) else "movie"
                            for t in cen["task"]]

    rows = []
    for (cohort, cond), g in lat.groupby(["cohort", "condition"], sort=True):
        cfg = cfgs.get(cohort, {})
        tasks = sorted(g["task"].unique())
        row = dict(cohort=LABELS.get(cohort, cohort), cohort_dir=cohort,
                   condition=cond,
                   role=(g["role"].iloc[0] if "role" in g else ""),
                   tasks=len(tasks), TR_s=tr_of(cfg, tasks),
                   space=cfg.get("space", ""),
                   final=g["sub"].nunique(), windows=len(g))
        c = (cen[(cen["cohort"] == cohort) & (cen["condition"] == cond)]
             if len(cen) else pd.DataFrame())
        if len(c):
            # A participant is excluded from a condition when every one of
            # their runs in it was gated out.
            per_sub = c.groupby("sub")["keep"].any()
            row["scanned"] = int(per_sub.size)
            row["excluded"] = int((~per_sub).sum())
            reasons = (c.loc[~c["keep"], "reason"].str.split("; ").explode()
                        .str.extract(r"^\|?([a-z_]+)")[0].value_counts())
            row["exclusion_reasons"] = "; ".join(f"{k} ({v})"
                                                 for k, v in reasons.items())
            if "mean_fd" in c:
                fd = c.loc[c["keep"]].groupby("sub")["mean_fd"].mean()
                row["mean_fd"] = f"{fd.mean():.3f} ({fd.std():.3f})"
        rows.append(row)

    out = pd.DataFrame(rows)
    order = ["cohort", "condition", "role", "tasks", "TR_s", "space",
             "scanned", "excluded", "final", "windows", "mean_fd",
             "exclusion_reasons", "cohort_dir"]
    out = out[[c for c in order if c in out]]
    out.attrs.update(policy=policy, atlas=atlas, window_s=window_s)
    return out.sort_values(["role", "cohort", "condition"],
                           ascending=[False, True, True]).reset_index(drop=True)


#: How each subject rule of a censor policy reads in a caption.
RULE_TEXT = {"max_mean_fd": "mean framewise displacement $>{}$~mm",
             "min_frac_good_frames": "fewer than {:.0%} usable frames",
             "max_frac_parcels_empty": "more than {:.0%} of regions outside the brain mask",
             "max_abs_best_lag_tr": "a stimulus lag of more than {} TR",
             "min_frac_stimulus_covered": "less than {:.0%} of the stimulus covered"}


def policy_text(root: Path, policy: str | None) -> str:
    """The exclusion rules of the policy, from its own record on disk."""
    import json
    from fmri_decomposition.io import meta_dir
    if not policy:
        return "No participant was excluded (no censor policy)."
    f = meta_dir(root, "extractions") / "censor" / f"policy={policy}.json"
    try:
        s = json.loads(f.read_text())
        rules = (s.get("policy", s).get("subject") or {})
    except (OSError, ValueError, AttributeError):
        return f"Exclusions follow the \\texttt{{{policy}}} censor policy."
    parts = [RULE_TEXT[k].format(v) for k, v in rules.items()
             if v is not None and k in RULE_TEXT]
    return (f"Participants were excluded for "
            + " or ".join(parts).replace("%", r"\%") + "." if parts else
            "No subject-level exclusion rule was applied.")


def to_latex(t: pd.DataFrame) -> str:
    def cell(v):
        if pd.isna(v):
            return "--"
        return str(v).replace("_", r"\_").replace("&", r"\&")

    cols = [("cohort", "Cohort"), ("condition", "Condition"), ("role", "Role"),
            ("tasks", "Runs"), ("TR_s", "TR (s)"), ("scanned", "Scanned"),
            ("excluded", "Excluded"), ("final", "Final $n$"),
            ("windows", "Windows"), ("mean_fd", "Mean FD (s.d.), mm")]
    cols = [(k, h) for k, h in cols if k in t]
    lines = [r"\begin{table}[h]", r"\centering\small",
             r"\caption{Samples, acquisition and quality control. "
             + t.attrs.get("policy_text", "") +
             f" Windows: {t.attrs.get('window_s')}-s windows entering the "
             r"brain-state analysis.}",
             r"\label{tab:samples}",
             r"\begin{tabular}{" + "l" * 3 + "r" * (len(cols) - 3) + "}",
             r"\toprule", " & ".join(h for _, h in cols) + r" \\", r"\midrule"]
    for _, r in t.iterrows():
        lines.append(" & ".join(cell(r[k]) if k != "windows"
                                else f"{int(r[k]):,}".replace(",", "{,}")
                                for k, _ in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", default="figures/supplementary/sample_table",
                   help="output stem; writes <stem>.csv and <stem>.tex")
    p.add_argument("--output-root", default=None)
    p.add_argument("--atlas", default="harvardoxford")
    p.add_argument("--window-s", default="30")
    p.add_argument("--cohorts", nargs="*", default=None,
                   help="cohort directories to include (default: every cohort "
                        "in the latents). The manuscript uses: ds002837 "
                        "cneuromod camcan hcp7t_fix")
    p.add_argument("--rest-pattern", default=r"rest")
    a = p.parse_args(argv)

    root = Path(a.output_root) if a.output_root else default_output_root()
    t = build(root, a.atlas, a.window_s, a.rest_pattern, a.cohorts)
    t.attrs["policy_text"] = policy_text(root, t.attrs.get("policy"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".tex").write_text(to_latex(t))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t.drop(columns=["exclusion_reasons"], errors="ignore")
              .to_string(index=False))
    print(f"\ncensor policy: {t.attrs['policy']}\n"
          f"wrote {out.with_suffix('.csv')} and {out.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
