# Phenotype tables

`<cohort>_phenotype.csv` — one row per subject, age / sex / clinical scores.

**These files do not exist until someone builds them.** Nothing in the pipeline
writes a phenotype, and nothing in the pipeline reads one; they exist for the
analysis, on the far side of stage 3.

That is not an oversight, it is the same ownership split the rest of the
metadata follows:

| file | owner | carries |
|---|---|---|
| `config/*_participants.csv` | human | curation — who is in, who was removed, why |
| `outputs/meta/cohorts/cohort=*/participants_qc.csv` | pipeline | measurement — motion, coverage, scrubbing |
| `config/phenotype/*_phenotype.csv` | human | **who the person is** — age, sex, clinical status |
| `outputs/meta/cohorts/cohort=camcan/participants_scores.csv` | script | Cam-CAN's behavioural battery, consolidated from `cc700-scored/` |

`make_participants.py` builds the first from what discovery finds on disk, so
it can only ever contain facts about files. Age is not a fact about a file.

## Schema

```
sub, cohort, age, sex, <any extra columns, verbatim>
```

* `sub` — the bare id, matching the `sub=` partition key: `CC110033`, `01`, `1`.
  A string, always. Never let a reader parse it as an integer.
* `sex` — `M` / `F` after normalisation, or the source's own value if it was a
  code the importer was not told how to read.
* extra columns are whatever the cohort has — `frailty_index`, `gait_speed`,
  `MMSE`, a group label. They are carried through under their source names.

Partial coverage is expected and is not an error. Cam-CAN's scored tables do
not cover every subject with a movie run; `make_phenotype.py` prints exactly
who is missing, and `nbtools.subject_table()` marks the joined rows with
`has_pheno` so a gap never reads as a value.

## Building them

```bash
python tools/make_phenotype.py --help
```

### `ds002837`

OpenNeuro ships `participants.tsv` at the root of the raw dataset (not the
derivatives this pipeline reads).

```bash
python tools/make_phenotype.py --cohort ds002837 \
    --source /path/to/ds002837/participants.tsv \
    --sub-column participant_id --age-column age --sex-column sex
```

### `cneuromod`

Check the raw `friends` dataset — the fMRIPrep derivatives under
`friends.fmriprep/` do not carry participant demographics. Five subjects, so
verify the result by eye rather than trusting a column name.

### `camcan` (CC700) and `camcan_ccfrail`

The phenotype is in the Cam-CAN archive, not in the imaging release: CC700
demographics in `cc700/participants.tsv`, and the ccfrail frailty assessment in
that study's own release002 phenotype files. They cover the whole archive, so
pass `--restrict-to-participants` to keep only the subjects this cohort has.

**`cc700-scored/` is not demographics and not ccfrail's.** It holds the
behavioural battery — ten tests (CardioMeasures, Cattell, EkmanEmHex,
EmotionalMemory, EmotionRegulation, FamousFaces, MotorLearning, Proverbs,
Synsem, TOT), each with its own release and summary file. Measured against the
uploaded `TOT/release001` summary: 617 of the 648 subjects in `camcan` have a
TOT score, and **0 of the 55 in `camcan_ccfrail` do**. So the cognitive scores
belong to CC700, and `preprocessing/camcan/03_build_participants_scores.py`
consolidates them into
`outputs/meta/cohorts/cohort=camcan/participants_scores.csv` rather than into a
phenotype file — they are a battery, not a demographic, and they are
regenerated from the archive rather than curated by hand.

Sex is coded numerically in several of these tables. The importer refuses to
guess a numeric coding — read the source's data dictionary and state it:

```bash
python tools/make_phenotype.py --cohort camcan_ccfrail \
    --source /path/to/ccfrail_phenotype.tsv \
    --sub-column CCID --age-column Age --sex-column Sex \
    --sex-map "1=M,2=F" --keep frailty_index \
    --restrict-to-participants
```

Scores split across several tables are joined one at a time with `--merge`.

## Before comparing `camcan` with `camcan_ccfrail`

`camcan_ccfrail` is the cohort with a frailty assessment, and CC700 is the
obvious healthy comparison — but the two differ in acquisition, not only in
frailty (and note that the *behavioural* battery runs the other way: it exists
for CC700 and not for ccfrail):

|  | `camcan` (CC700) | `camcan_ccfrail` |
|---|---|---|
| TR | 2.47 s | 1.12 s |
| echoes | 5 | 1 |

A window of fixed duration holds ~2.2× more samples in ccfrail, so its edges
are measurably less noisy — in the direction that makes the **frail** group
look **less variable**. `n_tr_effective` is the column that lets a model
account for it (Fisher-z sampling SD is `1/sqrt(n-3)`), and censoring makes it
worse rather than better: motion is high in ccfrail, so the frailest
participants lose the most frames. `config/camcan_ccfrail_movie.yaml` has the
full comparison in its header comments; read it before running the contrast.
