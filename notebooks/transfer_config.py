"""The two cohorts' Settings for the Cam-CAN -> HCP transfer, in ONE place.

Two callers need them and they must not drift apart:

    13_camcan_to_hcp.ipynb   draws the figures
    tools/transfer_grid.py   computes the rows, sharded, on the cluster

If each built its own `Settings`, the figures would eventually be drawn from
rows produced under a different target range, a different covariate spelling
or a different task map, and nothing in the figure would say so. So both
import `build()` from here, and the provenance written beside the rows records
what it returned.

Paths come from the environment where there is one, so a job script never has
to carry somebody's scratch path:

    FMRIDECOMP_OUTPUTS   the outputs tree           (hads_tables.Settings)
    CAMCAN_PHENO         Cam-CAN's phenotype dir    (hads_tables.Settings)
    HCP_PHENO            HCP's phenotype CSV        (here)
"""
from __future__ import annotations

import os
from pathlib import Path

import hads_tables as ht
import transfer_tables as tt

HCP_PHENO = Path(os.environ.get(
    "HCP_PHENO", "/project/6008063/tamires/cohorts/hcp_pheno/participants_pheno.csv"))

# Cam-CAN's two target columns, in both forms. The raw has 22 levels against
# the category's 4, so it gives a model far more to rank -- at the cost of not
# being what the existing results used. Run both and compare; a signal that is
# real should rank BETTER on the raw.
CC_TARGETS = {
    "raw": {"anxiety": "additional_HADS_anxiety",
            "depression": "additional_HADS_depression"},
    "category": {"anxiety": "additional_HADS_anx_category",
                 "depression": "additional_HADS_dep_category"},
}

# HCP's own instrument. DSM_Anxi pairs with HADS anxiety and DSM_Depr with HADS
# depression; the LABELS must match between the two Settings, the columns need
# not -- Spearman only needs the two to order people the same way.
HCP_TARGETS = {"anxiety": "DSM_Anxi_Raw", "depression": "DSM_Depr_Raw"}

# HCP spells the covariates differently: Gender for Sex, Age_in_Yrs for Age.
# Age_in_Yrs and NOT Age, because the same file also carries an `Age` holding
# the open-access age BAND ("26-30"), which is not a number. The alias wins
# over a column that already has the canonical name, and says so when it does.
HCP_ALIASES = {"Sex": "Gender", "Age": "Age_in_Yrs"}

# And the VALUES, not only the column name. `matrix` one-hots with
# get_dummies, which names its columns after the values each cohort happens to
# hold -- Cam-CAN's MALE/FEMALE and HCP's M/F give DIFFERENT dummy columns
# from the same variable, which align_columns then refuses (correctly) and no
# row is ever produced. Mapped onto one spelling for BOTH cohorts, with the
# unambiguous strings tools/make_phenotype.py already uses. A value neither
# covers is refused rather than guessed: `1` is male in Cam-CAN's tables and
# female in others, and backwards flips every subject with nothing looking
# wrong.
SEX = {"Sex": {"M": "M", "MALE": "M", "MAN": "M",
               "F": "F", "FEMALE": "F", "WOMAN": "F"}}

# The hive `task=` values each cohort actually writes. The KEYS are what must
# match between the two Settings; the value each maps to is that cohort's own
# spelling. HCP's four movies are task=MOVIE1..MOVIE4 and its four rests
# task=REST1..REST4 -- four partitions of ONE condition, which only a prefix
# names, and which `pool_fn` then turns into one row per subject because the
# models score one row per subject and will not average them.
CC_TASKS = {"rest": "Rest", "movie": "Movie"}
HCP_TASKS = {"rest": "REST", "movie": "MOVIE"}

COVARIATES = ["Age", "Sex", "n_transitions"]
BASELINE = ["Age", "Sex"]
MODELS = ["ridge", "hgb"]
# ONE cohort, two conditions. Cam-CAN's movie and rest are the same people, so
# two cohort values would make the output tree claim they were two samples --
# README's output structure says so and tests/test_camcan_one_cohort.py pins
# it. The rest data is `cohort=camcan task=Rest`; CC_TASKS selects it.
#
# A `camcan_rest` here matched nothing and was harmless for years, because
# `load_set` filters cohorts with `in` and an absent name simply never hits.
# It became a hard failure the moment the fcm precondition started REQUIRING
# every listed cohort to have a static-FC table, and it sent two cluster jobs
# and a long diagnosis after a cohort that does not exist.
TRAIN_COHORTS = ["camcan"]
TEST_COHORTS = ["hcp7t_fix"]


def _pool(record: list | None = None):
    """`pool_tasks` with its per-cell log silenced, optionally recording sizes.

    It prints one line per cell it pools, which is 564 lines over the
    multiverse and says nothing the first one did not. The notebook verifies
    the pooling ONCE instead, from `record`.
    """
    def pool(d, K):
        out = tt.pool_tasks(d, K, log=lambda *_: None)
        if record is not None:
            record.append((len(d), len(out)))
        return out
    return pool


def build(target: str = "raw", root=None, camcan=None, hcp_pheno=None,
          save_dir=None, models=None, pool_record=None):
    """(train_cfg, test_cfg) for one choice of Cam-CAN target column."""
    if target not in CC_TARGETS:
        raise SystemExit(f"target must be one of {sorted(CC_TARGETS)}, not {target!r}")
    common = dict(covariates=COVARIATES, baseline_covariates=BASELINE,
                  models=list(models or MODELS), covariate_values=SEX,
                  save_dir=save_dir)
    if root is not None:
        common["root"] = Path(root)
    train_cfg = ht.Settings(
        cohorts=list(TRAIN_COHORTS), tasks=dict(CC_TASKS),
        targets=dict(CC_TARGETS[target]),
        # only the raw columns have a scale to be outside of. Cam-CAN codes
        # "no answer" as 9999 in additional_HADS_depression while HADS runs
        # 0-21: invisible while the target was the *_category column (text
        # never codes) and an enormous outlier the moment the raw one is used.
        target_range=({"anxiety": (0, 21), "depression": (0, 21)}
                      if target == "raw" else None),
        **({"camcan": Path(camcan)} if camcan else {}),
        # B1 / B2 / B2.K are the report's figures, and their default captions
        # say "CV score" and "sd over seeds". Neither is true here: one fit on
        # every Cam-CAN subject, one score on HCP, and an interval from
        # resampling the HCP subjects.
        score_name="transfer score, HCP (Spearman)",
        spread_name="sd of the bootstrap over the test subjects",
        **common)
    test_cfg = ht.Settings(
        cohorts=list(TEST_COHORTS), tasks=dict(HCP_TASKS), targets=dict(HCP_TARGETS),
        pheno_files=[(Path(hcp_pheno or HCP_PHENO), ",")], pheno_id="Subject",
        covariate_aliases=dict(HCP_ALIASES),
        task_match="prefix", pool_fn=_pool(pool_record), **common)
    return train_cfg, test_cfg


def studies(target: str = "raw", **kw):
    """(train, test) Studies, plus the (train_cfg, test_cfg) they came from."""
    train_cfg, test_cfg = build(target, **kw)
    return ht.Study(train_cfg), ht.Study(test_cfg), train_cfg, test_cfg
