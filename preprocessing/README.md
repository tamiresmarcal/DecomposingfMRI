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
  03_build_participants_scores.py  LOGIN NODE. cc700-scored/ -> one table.
  04_build_codebook.py          LOGIN NODE. Code-book xlsx -> two greppable TSVs.
```

Run `00`–`02` in that order. `00` and `01` are cheap and need the network / a
shell; only `02` costs compute.

`03` is independent of the other three and touches no images. It consolidates
the Cam-CAN archive's **behavioural** scoring — `cc700-scored/<Test>/release00N/
summary/<Test>_summary.txt`, one directory per test — into

    outputs/meta/cohorts/cohort=camcan/participants_scores.csv

one row per subject, columns namespaced by test, alongside a manifest recording
which release each block came from. It belongs here for the same reason the
rest of this directory does: the format is Cam-CAN's own, written by ten
different analysis scripts between 2011 and 2014, and nothing about it
generalises to another cohort.

Two facts worth knowing before using it:

- **The battery is CC700's, not ccfrail's.** Of the 648 subjects in
  `camcan_movie_participants.csv`, 617 have a TOT score; of the 55 in
  `camcan_ccfrail_movie_participants.csv`, **none** do. ccfrail's own
  phenotype — MMSE and case/control group, per the Cam-CAN Phase 4/5 protocol —
  is in that study's release002 files, not in `cc700-scored/`.
- **A blank row is an exclusion, not an absence.** Each test's own QC blanks
  the scores and states a reason in `ErrorMessages` (TOT: "replied dont know on
  > 80% trials", 12 subjects). The row is kept and the reason travels with it as
  `<test>_ErrorMessages`.

`04` is likewise independent and reads no images. The home-interview export
(`approved_data.tsv`) names its columns `homeint_v112`, `epaq_DURTV`,
`additional_acer` and says nowhere what `v112` asked; the answer is in
`MRC_CamCAN_Code_Book_variables.xlsx`, a seven-sheet workbook with its header
three rows down, section titles in "blue header rows" that leave the variable
column empty, and the join key in `Variable_name_in_test` rather than
`Variable_name`. `04` flattens it into two files you can grep mid-analysis:

    outputs/meta/cohorts/cohort=camcan/codebook_variables.tsv   one row per variable
    outputs/meta/cohorts/cohort=camcan/codebook_values.tsv      one row per coded value

```
column          group          code          question
homeint_v112    Verbal fluency verbal fluency I'm going to give you a letter ...
homeint_v144    ACE-R          ACE-R_L_06     Repeat Hippopotamus
epaq_DURTV      EPAQ           EPAQ_010       Reported total TV viewing time (hrs/day)

code   value  label
DG7    1      College or university degree or higher
DG7    2      A levels/AS levels or equivalent
```

Pass `--data` to match against a real export. Measured on the 2026-09 workbook
against `approved_data.tsv`: **204 of 351 data columns documented**. The other
147 appear nowhere in the workbook under any name, and are written out with
`group=(not in code book)` so a lookup returns "undocumented" rather than
nothing. Three columns are documented under a different name and are matched
through a small alias table; the `match` column records which rows those are
and whether the link was verified against the data (`homeint_sex` = `DG1`,
identical on all 2,676 rows) or only inferred from name and range
(`homeint_handedness` = `EH_total`, `homeint_mmse_i` = `mmse_cal`).

Some groups are fully documented but absent from this export — LEQ has 254
code-book rows and no data columns, and the same holds for PSQI, Cattell, VSTM
and the reaction-time tasks. Those are scanner-visit measures that live in
other files, so a group with `with_data_column = 0` means "wrong file", not
"missing data".
