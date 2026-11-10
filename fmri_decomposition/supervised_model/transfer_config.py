"""The two cohorts' Settings for a transfer, read from a YAML.

Two callers need them and they must not drift apart:

    notebooks/13_camcan_to_hcp.ipynb   draws the figures
    fmri-decomp transfer               computes the rows, sharded, on the
                                       cluster

If each built its own `Settings`, the figures would eventually be drawn from
rows produced under a different target range, a different covariate spelling
or a different task map, and nothing in the figure would say so. So both call
`build()` here, and the provenance written beside the rows records the config
file and what it returned.

THE PAIR IS A CONFIG FILE, NOT PYTHON. `config/transfer/camcan_to_hcp.yaml`
holds which cohorts, which columns, which task maps, which covariates. That is
what makes swapping the roles -- Cam-CAN tested on rather than trained on,
which is the `v2` modelling version -- a second YAML instead of an edit to a
module every other analysis imports. It is also the same shape the rest of the
repo already uses: `config/<cohort>.yaml`, `config/censor/<policy>.yaml`.

Paths come from the environment where there is one, so a config never has to
carry somebody's scratch path. `${VAR}` in any path in the YAML is expanded:

    FMRIDECOMP_OUTPUTS   the outputs tree
    CAMCAN_PHENO         Cam-CAN's phenotype dir
    HCP_PHENO            HCP's phenotype CSV
"""
from __future__ import annotations

import os
from pathlib import Path

from fmri_decomposition.supervised_model import hads as ht
from fmri_decomposition.supervised_model import transfer_tables as tt

#: Where a transfer config lives when only its NAME was given, so a job script
#: can say `--config camcan_to_hcp` and not carry a path.
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "transfer"

#: Every key `build()` understands, so a typo in a YAML is caught by name
#: instead of silently doing nothing. A key that only changes a figure's
#: caption is as load-bearing as one that changes a column: the caption is a
#: claim about what the number is.
TOP_KEYS = {"name", "train", "test", "covariates", "baseline_covariates",
            "models", "covariate_values", "score_name", "spread_name"}
SIDE_KEYS = {"cohorts", "tasks", "task_match", "targets", "target_range",
             "pheno_files", "pheno_id", "covariate_aliases", "pool_runs"}


def config_path(name_or_path) -> Path:
    """A path as given, or a bare name resolved under `config/transfer/`."""
    p = Path(str(name_or_path))
    if p.suffix in (".yaml", ".yml") or p.exists():
        return p
    for ext in (".yaml", ".yml"):
        cand = CONFIG_DIR / f"{p.name}{ext}"
        if cand.exists():
            return cand
    have = sorted(f.stem for f in CONFIG_DIR.glob("*.y*ml"))
    raise SystemExit(
        f"no transfer config {name_or_path!r}. Give a path, or one of these "
        f"names from {CONFIG_DIR}: {', '.join(have) or '(none)'}")


def _expand(v):
    """`${VAR}` in any string, so a config carries no scratch path.

    An unset variable is LEFT IN PLACE, not silently emptied and not an error
    here. Both of the other options were wrong:

    * Emptying it gives `/participants_pheno.csv`, which fails much later
      somewhere that has no idea a variable was missing.
    * Raising at load time breaks a caller that is about to override the path
      anyway -- `--hcp-pheno` and notebook 13 both do, and every test does.
      A config could then not be loaded at all on a machine where the variable
      is irrelevant.

    So the literal survives, and `_pheno_files` refuses it at the point the
    path is actually used. A path nobody uses never raises.
    """
    if isinstance(v, str) and "${" in v:
        import re
        return re.sub(r"\$\{(\w+)\}",
                      lambda m: os.environ.get(m.group(1), m.group(0)), v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def load(name_or_path) -> dict:
    """The YAML, expanded and checked for keys this module understands."""
    import yaml

    path = config_path(name_or_path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: expected a mapping at the top level")
    for side in ("train", "test"):
        if side not in raw:
            raise SystemExit(f"{path}: no `{side}:` section. A transfer needs "
                             f"both -- which cohort is fitted and which is "
                             f"scored is the whole claim.")
    bad = sorted(set(raw) - TOP_KEYS)
    if bad:
        raise SystemExit(
            f"{path}: unknown top-level key(s) {bad}. Known: "
            f"{', '.join(sorted(TOP_KEYS))}.\n"
            f"  Refused rather than ignored: a key nothing reads looks like a "
            f"setting that is in force.")
    for side in ("train", "test"):
        bad = sorted(set(raw[side]) - SIDE_KEYS)
        if bad:
            raise SystemExit(f"{path}: unknown key(s) {bad} under `{side}:`. "
                             f"Known: {', '.join(sorted(SIDE_KEYS))}.")
    cfg = _expand(raw)
    cfg["_path"] = str(path)
    return cfg


def targets_for(cfg: dict, target: str) -> dict:
    """The train side's target columns for one `--target` choice.

    The test side has ONE set: its instrument does not change because the
    training column did.
    """
    t = cfg["train"]["targets"]
    # A single mapping (label -> column) means this config offers one choice
    # and `--target` is not a question it asks.
    if t and all(isinstance(v, str) for v in t.values()):
        return dict(t)
    if target not in t:
        raise SystemExit(
            f"{cfg['_path']}: --target {target!r} is not one of "
            f"{sorted(t)}. Those are the target SETS this config offers; each "
            f"is a different claim about the same answers.")
    return dict(t[target])


def target_names(cfg: dict) -> list:
    t = cfg["train"]["targets"]
    if t and all(isinstance(v, str) for v in t.values()):
        return []
    return sorted(t)


def _pheno_files(spec):
    """[(path, sep)], refusing a `${VAR}` that was never set.

    This is where an unexpanded variable becomes an error: the path is about
    to be read, so the variable really is needed. A caller that overrides the
    path never reaches here with it.
    """
    if not spec:
        return None
    out = []
    for e in spec:
        path, sep = ((e, ",") if isinstance(e, str)
                     else (e["path"], e.get("sep", ",")))
        if "${" in str(path):
            import re
            missing = ", ".join(sorted(set(re.findall(r"\$\{(\w+)\}",
                                                     str(path)))))
            raise SystemExit(
                f"the transfer config needs this phenotype file:\n"
                f"  {path}\n"
                f"and {missing} is not set. Export it, write the path into "
                f"the config, or pass the path on the command line "
                f"(--hcp-pheno / --camcan-pheno).")
        out.append((Path(path), sep))
    return out


def _pool(record: list | None = None, root=None):
    """`pool_tasks` with its per-cell log silenced, optionally recording sizes.

    It prints one line per cell it pools, which is 564 lines over the
    multiverse and says nothing the first one did not. The notebook verifies
    the pooling ONCE instead, from `record`.

    `root` turns on the EXACT dispersion. Every other dynamic pools exactly
    from the transition table; dispersion cannot, because the distance from
    each run's centroid is not the distance from the centroid of all four, and
    HCP has four runs per condition. So the coordinates are read back from the
    latents -- once per (atlas, aperture, runs) cell, memoised, because they do
    not depend on the state set and all 282 of a cell share them.

    Without `root` it falls back to the weighted mean and the frame says so in
    `dispersion_pooled_approximately`; the fallback is kept because a tree
    whose latents were moved or pruned should still produce rows.
    """
    said = []                        # the NOTE below, once per process

    def pool(d, K):
        disp = None
        if root is not None:
            try:
                # The cell comes off the frame's OWN provenance columns, which
                # `transitions` writes into every row -- not from a setting, so
                # the latents that get read are necessarily the ones these
                # transitions were computed from.
                pairs = {(c, t) for c, t in zip(d["cohort"].astype(str),
                                                d["task"].astype(str))}
                disp = tt.exact_dispersion(root, str(d["atlas"].iloc[0]),
                                           d["window_s"].iloc[0], pairs)
            except (KeyError, SystemExit) as e:
                # A tree whose latents were pruned, or a transition table
                # written before the provenance columns existed. Degrade to the
                # weighted mean rather than refuse: every other column still
                # pools exactly, and `dispersion_pooled_approximately` marks
                # the rows, so nothing silently claims to be the real thing.
                # Said ONCE -- this runs per state set, which is 282 times.
                if not said:
                    said.append(True)
                    print("NOTE: dispersion is being pooled by the "
                          "n_windows-weighted mean, not read from the latents "
                          f"({type(e).__name__}: {str(e).splitlines()[0]}). "
                          "The rows are flagged "
                          "`dispersion_pooled_approximately`.")
        out = tt.pool_tasks(d, K, log=lambda *_: None, dispersion=disp)
        if record is not None:
            record.append((len(d), len(out)))
        return out
    return pool


#: The config `build()` reads when none is named. Cam-CAN -> HCP is this
#: project's transfer, and naming it here keeps every existing caller working
#: without a flag -- but it is a DEFAULT and not a constant: pass `config=` to
#: read another pair, which is how a role swap becomes a second YAML.
DEFAULT_CONFIG = "camcan_to_hcp"


def build(target: str = "raw", root=None, camcan=None, hcp_pheno=None,
          save_dir=None, models=None, pool_record=None, config=None,
          modelling_version=None):
    """(train_cfg, test_cfg) for one transfer config and one target choice.

    `config` is a path or a bare name under `config/transfer/`; omitted, it is
    DEFAULT_CONFIG. `camcan` and `hcp_pheno` override the phenotype locations
    the config gives, for a caller that has them somewhere else -- they are
    kept because notebook 13 and the job script both expose them.
    """
    cfg = config if isinstance(config, dict) else load(config or DEFAULT_CONFIG)
    tr, te = cfg["train"], cfg["test"]

    common = dict(covariates=list(cfg.get("covariates",
                                          ["Age", "Sex", "n_transitions"])),
                  baseline_covariates=list(cfg.get("baseline_covariates",
                                                   ["Age", "Sex"])),
                  models=list(models or cfg.get("models", ["ridge", "hgb"])),
                  covariate_values=dict(cfg.get("covariate_values") or {}),
                  modelling_version=modelling_version,
                  save_dir=save_dir)
    if root is not None:
        common["root"] = Path(root)

    want = targets_for(cfg, target)

    def range_for(side, labels):
        """`target_range` for one side, keyed per LABEL.

        The range applies to the target SET, so it survives `--target`: `raw:
        [0, 21]` means the raw columns have a scale and the category ones do
        not. BOTH SIDES can have one -- v0 only needed it on the train side,
        because that is where the raw HADS columns were; swap the roles and
        the 9999 sentinel is on the TEST side, where an unfiltered one is
        scored as a real value and drags Spearman toward zero.
        """
        rng = (side.get("target_range") or {}).get(target)
        return {lab: tuple(rng) for lab in labels} if rng is not None else None

    train_kw = {}
    rng = range_for(tr, want)
    if rng:
        train_kw["target_range"] = rng
    if tr.get("pheno_files"):
        train_kw["pheno_files"] = _pheno_files(tr["pheno_files"])
    if tr.get("pheno_id"):
        train_kw["pheno_id"] = tr["pheno_id"]
    if camcan:
        train_kw["camcan"] = Path(camcan)
    # POOLING IS PER SIDE, not a property of the transfer. v0 needed it only on
    # the test side, because HCP is the cohort with four runs per condition --
    # so `pool_fn` was wired on the test Settings and nowhere else. Swap the
    # roles and HCP is the TRAINING cohort: without this its subjects appear
    # four times each and `load_set` refuses, which is the right refusal to a
    # question nobody should have had to ask.
    if tr.get("pool_runs"):
        train_kw["pool_fn"] = _pool(
            pool_record, Path(root) if root is not None else ht.Settings.root)

    train_cfg = ht.Settings(
        cohorts=list(tr["cohorts"]), tasks=dict(tr["tasks"]), targets=want,
        task_match=tr.get("task_match", "exact"),
        covariate_aliases=dict(tr.get("covariate_aliases") or {}) or None,
        # The report's figures default to "CV score" and "sd over seeds", and
        # neither is true here: one fit on every training subject, one score on
        # the test cohort, and an interval from resampling the TEST subjects.
        score_name=cfg.get("score_name", "transfer score (Spearman)"),
        spread_name=cfg.get("spread_name",
                            "sd of the bootstrap over the test subjects"),
        **train_kw, **common)

    # The OVERRIDE FIRST: _pheno_files refuses an unexpanded ${VAR}, and a
    # caller that supplied the path has no need of the variable. Resolving the
    # config's path only to throw it away would refuse a run that is fine.
    if hcp_pheno:
        spec = te.get("pheno_files") or []
        sep = (spec[0].get("sep", ",") if spec and isinstance(spec[0], dict)
               else ",")
        pheno = [(Path(hcp_pheno), sep)]
    else:
        pheno = _pheno_files(te.get("pheno_files"))
    te_rng = range_for(te, te["targets"])
    test_cfg = ht.Settings(
        cohorts=list(te["cohorts"]), tasks=dict(te["tasks"]),
        targets=dict(te["targets"]),
        task_match=te.get("task_match", "exact"),
        **({"target_range": te_rng} if te_rng else {}),
        covariate_aliases=dict(te.get("covariate_aliases") or {}) or None,
        **({"pheno_files": pheno} if pheno else {}),
        **({"pheno_id": te["pheno_id"]} if te.get("pheno_id") else {}),
        # Path(root) when build() was given one, else Settings' own default --
        # the tree the test cfg is about to read its transitions from either
        # way, so the latents read for dispersion come from beside them.
        pool_fn=(_pool(pool_record,
                       Path(root) if root is not None else ht.Settings.root)
                 if te.get("pool_runs") else None),
        **common)
    return train_cfg, test_cfg


def studies(target: str = "raw", **kw):
    """(train, test) Studies, plus the (train_cfg, test_cfg) they came from."""
    train_cfg, test_cfg = build(target, **kw)
    return ht.Study(train_cfg), ht.Study(test_cfg), train_cfg, test_cfg
