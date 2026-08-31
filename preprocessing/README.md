# `preprocessing/` — stage 1, deliberately outside the pipeline

**Nothing in this directory is part of `fmri_decomposition`.**

Stages 2–4 (`extract`, `dfc`, `finalize`) take *already-preprocessed* images and
turn them into parcel timeseries and windowed connectivity. That boundary is
the reason the pipeline is cohort-agnostic: ds002837 arrives as AFNI residuals,
CNeuroMod as fMRIPrep derivatives, and neither required a line of pipeline code
to be changed.

This directory exists because Cam-CAN arrives as **neither**. What the Cam-CAN
archive distributes is raw, five-echo BIDS — no echo combination, no
realignment, no normalisation, no motion parameters. Verified on disk
(2026-08): every file under `cc700/mri/pipeline/release004/BIDSsep` matches
`*_T1w.nii.gz`, `*_T2w.nii.gz`, `*_task-Rest_bold.nii.gz` or
`*_task-Movie_echo-0N_bold.nii.gz`, and a search for `*preproc*`, `*space-*`,
`rp_*` and `*confound*` across the whole tree returns nothing. The
`func_movie/derivatives/` folder exists but is empty — the placeholder BIDS
conversion creates, never filled.

So Cam-CAN cannot enter the pipeline without a preprocessing step first, and
that step lives here rather than pretending to be part of stages 2–4.

## What this means for you

- **Not covered by `run_tests.sh`.** The 159 tests exercise the pipeline, not
  this. Treat anything here as scripts you must read before trusting.
- **Not cohort-agnostic.** `camcan/` is written for Cam-CAN's exact layout.
  Another raw cohort needs its own directory, not a flag added here.
- **Outputs land in a normal derivatives tree**, so from the pipeline's point
  of view Cam-CAN afterwards looks exactly like CNeuroMod: fMRIPrep output that
  `config/camcan_movie.yaml` points at. The pipeline never learns that this
  directory exists.

## Layout

```
preprocessing/camcan/
  00_prefetch_templateflow.sh   LOGIN NODE. Templates, or every job dies offline.
  01_build_bids.py              LOGIN NODE. Symlinks anat+func into one BIDS root.
  02_fmriprep.sbatch            COMPUTE. One SLURM array task per subject.
```

Run them in that order. `00` and `01` are cheap and need the network / a shell;
only `02` costs compute.
