"""Cohort YAML -> validated dataclasses.

All cohort-specific knowledge enters the pipeline here and in `cohort.py`.
Nothing downstream reads a path or a TR from anywhere else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_WINDOWS_S: tuple[float, ...] = (30.0, 60.0, 120.0, 300.0)


class ConfigError(ValueError):
    """Raised for a config that cannot produce a well-defined pipeline."""


@dataclass
class DiscoveryConfig:
    backend: str = "glob"                 # glob | pybids
    bold_glob: str = "sub-*/func/*_bold.nii.gz"
    mask_glob: str | None = None
    include_tasks: list[str] = field(default_factory=list)
    exclude_tasks: list[str] = field(default_factory=lambda: ["localizer", "retinotopy", "rest"])
    subject_pattern: str = r"sub-([A-Za-z0-9]+)"
    task_pattern: str = r"task-([A-Za-z0-9]+)"


@dataclass
class ConfoundsConfig:
    format: str = "none"                  # none | afni_1D | fmriprep_tsv
    strategy: str = "none"                # none | 24hmp | custom (use `columns`)
    columns: list[str] = field(default_factory=list)
    confounds_glob: str | None = None     # fMRIPrep *_desc-confounds_timeseries.tsv
    censor_glob: str | None = None        # AFNI censored_timepoints.1D
    # Read for SUBJECT-LEVEL motion QC only -- `tools/make_participants.py --fd`
    # turns it into mean_fd in participants.csv. Stages 2 and 3 never open it.
    # It is separate from censor_glob because the two answer different
    # questions: censoring needs frame-accurate alignment to the images, a mean
    # over ~5,470 frames does not. On ds002837 the first is impossible and the
    # second is fine, which is precisely why this key exists.
    motion_glob: str | None = None        # AFNI motion / nuisance regressor .1D
    dilate_tr: int = 1                    # addendum §3: dilate censor mask by +/-1 TR
    # Frame censoring derived from a confounds column rather than a censor file.
    # AFNI-preprocessed cohorts ship an explicit 1D censor; fMRIPrep ones do not
    # -- they ship framewise displacement and leave the threshold to the user.
    # Without this, an fMRIPrep cohort silently gets no censoring at all, which
    # is precisely the zero-filling bias stage 3 is built to avoid.
    fd_column: str = "framewise_displacement"
    fd_threshold: float | None = None     # mm; None disables FD-based censoring

    def __post_init__(self) -> None:
        if self.strategy not in ("none", "24hmp", "custom"):
            raise ConfigError(
                f"confounds.strategy must be 'none', '24hmp' or 'custom', got "
                f"{self.strategy!r}"
            )
        if self.strategy == "custom" and not self.columns:
            raise ConfigError("confounds.strategy='custom' requires confounds.columns")
        if self.fd_threshold is not None and self.fd_threshold <= 0:
            raise ConfigError(
                f"confounds.fd_threshold must be positive, got {self.fd_threshold!r}"
            )


@dataclass
class FilteringConfig:
    already_applied: bool = True
    bandpass: tuple[float, float] | None = (0.01, 1.0)
    detrend: bool = False
    standardize: bool = False             # correlation is invariant; off by default


@dataclass
class RunsConfig:
    mode: str = "concat"                  # concat | separate
    drop_boundary_windows: bool = False   # flag instead, filter at stage 4


@dataclass
class TrimConfig:
    column: str | None = None             # e.g. end_movie in participants.csv
    unit: str = "seconds"                 # MUST be declared -- see handoff §6
    mode: str = "end"                     # end | start | both

    def __post_init__(self) -> None:
        if self.column is not None and self.unit not in ("seconds", "tr"):
            raise ConfigError(
                f"trim.unit must be 'seconds' or 'tr', got {self.unit!r}. "
                "It is not optional: on a TR!=1 cohort the ambiguity silently "
                "halves or doubles every run."
            )


@dataclass
class StimulusConfig:
    """Stimulus duration is cohort metadata now, alongside TR (addendum §4)."""

    durations_s: dict[str, float] = field(default_factory=dict)
    timing_source: str = "identity"       # identity | from_events | from_scans |
    #                                       from_log | from_paper | from_isc
    unit_of_analysis: str = "run"         # run | clip (HCP is clip-based)
    isc_gate_tr: float = 1.0              # refuse stage 3 if median |lag| exceeds


@dataclass
class WindowSizeSpec:
    """Per-window-size overrides. Everything unset falls back to WindowConfig.

    `atlases` exists because a window size is not equally meaningful at every
    parcellation. A 15 s aperture at TR=1 gives 15 samples: fine for yeo7's 7
    nodes / 21 edges, arithmetically hopeless for Harvard-Oxford's 111 nodes /
    6,105 edges, where every window comes back flagged `rank_deficient` and the
    only thing produced is storage. Restricting the size to the atlases it can
    actually support is cheaper than filtering the flag at stage 4, because the
    compute is never spent.
    """

    atlases: list[str] | None = None       # None = every atlas in cfg.atlases
    n_overlaps: int | None = None          # None = WindowConfig.n_overlaps


@dataclass
class WindowConfig:
    sizes_s: list[float] = field(default_factory=lambda: list(DEFAULT_WINDOWS_S))
    n_overlaps: int = 5
    drop_incomplete: bool = True
    min_n_tr_effective: int = 2           # arithmetically impossible below this
    edge_column_threshold: int = 20_000   # above this, pack edges into a list column
    # size -> WindowSizeSpec. Written in YAML as, e.g.
    #     by_size:
    #       15: {atlases: [yeo7, networks]}
    by_size: dict[float, WindowSizeSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_overlaps < 1:
            raise ConfigError(f"windows.n_overlaps must be >= 1, got {self.n_overlaps}")
        sizes = {float(w) for w in self.sizes_s}
        parsed: dict[float, WindowSizeSpec] = {}
        for key, spec in (self.by_size or {}).items():
            try:
                size = float(key)
            except (TypeError, ValueError):
                raise ConfigError(f"windows.by_size key {key!r} is not a window size")
            if isinstance(spec, WindowSizeSpec):
                parsed[size] = spec
                continue
            if spec is None:
                spec = {}
            if not isinstance(spec, dict):
                raise ConfigError(f"windows.by_size[{key}] must be a mapping, got {type(spec)}")
            unknown = set(spec) - set(WindowSizeSpec.__dataclass_fields__)
            if unknown:
                raise ConfigError(f"unknown key(s) in windows.by_size[{key}]: {sorted(unknown)}")
            parsed[size] = WindowSizeSpec(**spec)
        for size, spec in parsed.items():
            # An override for a size that is never run is dead config, and the
            # commonest way to write one is a typo in the size. Silence there
            # would mean the restriction you thought you applied did nothing.
            if size not in sizes:
                raise ConfigError(
                    f"windows.by_size has an entry for {size:g}s, which is not in "
                    f"windows.sizes_s ({sorted(sizes)}). An override for a size that "
                    "never runs has no effect; add the size or drop the override."
                )
            if spec.atlases is not None and not spec.atlases:
                raise ConfigError(
                    f"windows.by_size[{size:g}].atlases is empty -- that runs nothing. "
                    "Remove the window size from sizes_s instead."
                )
            if spec.n_overlaps is not None and spec.n_overlaps < 1:
                raise ConfigError(
                    f"windows.by_size[{size:g}].n_overlaps must be >= 1, "
                    f"got {spec.n_overlaps}"
                )
        self.by_size = parsed

    # -- resolution ------------------------------------------------------
    def spec_for(self, window_s: float) -> WindowSizeSpec:
        return self.by_size.get(float(window_s), WindowSizeSpec())

    def overlaps_for(self, window_s: float) -> int:
        spec = self.spec_for(window_s)
        return int(spec.n_overlaps if spec.n_overlaps is not None else self.n_overlaps)

    def atlases_for(self, window_s: float, available: list[str]) -> list[str]:
        """Which of `available` this window size runs on, in `available` order."""
        spec = self.spec_for(window_s)
        if spec.atlases is None:
            return list(available)
        wanted = {a for a in spec.atlases}
        return [a for a in available if a in wanted]


@dataclass
class CohortConfig:
    cohort: str
    tr: float
    derivatives_root: Path
    output_root: Path
    space: str = "MNI152NLin2009cAsym"
    smoothing_fwhm: float | None = None
    denoising_method: str | None = None
    participants: Path | None = None
    atlases: list[str] = field(default_factory=lambda: ["harvardoxford", "yeo7", "networks"])
    atlas_params: dict[str, dict] = field(default_factory=dict)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    confounds: ConfoundsConfig = field(default_factory=ConfoundsConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    runs: RunsConfig = field(default_factory=RunsConfig)
    trim: TrimConfig = field(default_factory=TrimConfig)
    stimulus: StimulusConfig = field(default_factory=StimulusConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)

    def __post_init__(self) -> None:
        if self.tr is None or self.tr <= 0:
            raise ConfigError(
                f"tr must be a positive number, got {self.tr!r}. NNDb's "
                "task-*_bold.json omits RepetitionTime (a BIDS violation), so "
                "the config value is mandatory, not a convenience."
            )
        self.derivatives_root = Path(self.derivatives_root)
        self.output_root = Path(self.output_root)
        if self.participants is not None:
            self.participants = Path(self.participants)
        if not self.atlases:
            raise ConfigError("at least one atlas must be configured")
        for size, spec in self.windows.by_size.items():
            if spec.atlases is None:
                continue
            unknown = [a for a in spec.atlases if a not in self.atlases]
            if unknown:
                raise ConfigError(
                    f"windows.by_size[{size:g}].atlases names {unknown}, which is not "
                    f"in the cohort's atlases ({self.atlases}). A restriction that "
                    "matches nothing runs nothing, silently."
                )

    # -- derived ---------------------------------------------------------
    def window_tr(self, window_s: float) -> int:
        from .windows import window_tr_from_seconds

        return window_tr_from_seconds(window_s, self.tr)

    def stimulus_duration_s(self, task: str, fallback: float | None = None) -> float:
        d = self.stimulus.durations_s.get(task)
        if d is None:
            if fallback is not None:
                return float(fallback)
            raise ConfigError(
                f"no stimulus duration for task {task!r}. Since the window grid "
                "is defined per stimulus, this is required; add it under "
                "stimulus.durations_s or supply end_movie in participants.csv."
            )
        return float(d)

    def hash(self) -> str:
        """Stable config hash for the manifest."""
        payload = json.dumps(_jsonify(asdict(self)), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


_SECTIONS = {
    "discovery": DiscoveryConfig,
    "confounds": ConfoundsConfig,
    "filtering": FilteringConfig,
    "runs": RunsConfig,
    "trim": TrimConfig,
    "stimulus": StimulusConfig,
    "windows": WindowConfig,
}


def config_from_dict(raw: dict[str, Any]) -> CohortConfig:
    raw = dict(raw)
    kwargs: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        section = raw.pop(name, None) or {}
        if not isinstance(section, dict):
            raise ConfigError(f"section {name!r} must be a mapping, got {type(section)}")
        unknown = set(section) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ConfigError(f"unknown key(s) in {name}: {sorted(unknown)}")
        kwargs[name] = cls(**section)

    unknown = set(raw) - {f for f in CohortConfig.__dataclass_fields__}
    if unknown:
        raise ConfigError(f"unknown top-level key(s): {sorted(unknown)}")
    kwargs.update(raw)
    missing = [k for k in ("cohort", "tr", "derivatives_root", "output_root") if k not in kwargs]
    if missing:
        raise ConfigError(f"missing required key(s): {missing}")
    return CohortConfig(**kwargs)


def load_config(path: str | Path) -> CohortConfig:
    import yaml

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return config_from_dict(raw)
