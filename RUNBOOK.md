# Running the pipeline from zero

How to go from preprocessed images to a ranked list of brain-state definitions,
with nothing on disk to start from. For *what* each stage is and how the outputs
are laid out, see `README.md`; this file is only the order to do things in.

Two situations it covers, because they are the same sequence:

* **you lost `outputs/`** and want it back
* **a new cohort arrives** and you want it in the analysis

---

## 0. The mental model

Six stages. Each reads the one before and writes its own directory under
`outputs/`. Nothing is ever edited in place except stage 4b, which adds columns
to the file stage 4 wrote.

```
     preprocessing          fMRIPrep / afni_proc -- OUTSIDE this repo
         |
  1  activation             NIfTI -> parcel timeseries, one row per TR
         |
  2  dfc                    -> windowed connectivity, one row per window
         |
 2.5 censor                 QC measurements -> a named keep/drop decision
         |
  3  decompose              edges or frames -> PCA + UMAP coordinates
         |                  (fit on the TRAIN cohorts, projected onto the rest)
         |
  4  cluster                coordinates -> brain-state LABELS
         |                  (threshold / MeanShift / HMM x pca3 / umap3)
         |
  5  transitions            labels -> one transition matrix per subject
         |
  6  bstm_selection         rank state sets by how well they predict a phenotype
```

Two things about this shape are worth holding onto:

**Stages 1–2.5 are per cohort. Stages 3–6 span cohorts.** Everything up to
`censor` takes one cohort YAML and can run for each cohort independently.
`decompose` fits *across* cohorts by design — that is what makes a state label
mean the same thing in all of them — so from stage 3 on, the unit of work is an
(atlas, aperture) cell, not a cohort.

**Stage 3 can read two different inputs.** `--source dfc` (the default) reads
windowed connectivity and runs once per `--window-s`. `--source activation`
reads per-TR frames and writes to `window_s=-1`. They are the same stage with a
different aperture, and everything downstream treats `-1` as one more window
size.

---

## 1. Before anything: the environment

```bash
cd /project/6008063/tamires/DecomposingfMRI        # wherever your checkout is
cp slurm/env.sh.example slurm/env.sh               # gitignored; yours alone
$EDITOR slurm/env.sh                               # check the three paths
source slurm/env.sh
```

`env.sh` names two container images, because there are two:

| variable | image | stages |
|---|---|---|
| `FMRIDECOMP_SIF` | `fmri_decomp.sif` | 1, 2 — extract, dfc |
| `FMRIDECOMP_SIF45` | `fmri_decomp_stage45.sif` | 3–6 — Stan, hmmlearn, umap, lightgbm |

Build the second if it is missing, **from the repository root** (`%files` copies
the definition into the image and its path resolves against the working
directory):

```bash
module load apptainer/1.3.5            # pin it: a bare `module load` opens a menu
apptainer build --fakeroot \
    /project/6008063/tamires/singularity/fmri_decomp_stage45.sif \
    containers/stage45.def
```

20–40 minutes. The finished `.sif` is about 1 GB — it is compressed squashfs, so
a small file is normal and is **not** evidence that something is missing. Ask the
image what it holds instead:

```bash
apptainer exec "$FMRIDECOMP_SIF45" sh -c \
  'cat /opt/image_provenance.txt | head -3; python3 -c "import lightgbm, hmmlearn, umap"'
```

**Every interactive command below should use `${FMRIDECOMP_SIF45:?}`**, not the
bare variable. If the variable is unset, the bare form passes an empty string and
apptainer silently mounts your *current directory* as the container — you then
get `python3: executable file not found`, which looks like a broken image and is
not one. The `:?` form fails immediately and says so.

```bash
RUN="apptainer exec --cleanenv --bind /project,/scratch,/home ${FMRIDECOMP_SIF45:?source slurm/env.sh}"
```

The sbatch scripts source `env.sh` themselves, so only your interactive commands
need this.

---

## 2. Check what you have, at any time

```bash
$RUN python3 -m fmri_decomposition.cli status
```

This is the one command to reach for whenever you are unsure. It walks
`outputs/`, reads only parquet footers, writes nothing, and prints what each
stage holds — followed by a **PROBLEMS** list of the things that span two stages
and so no single stage can see:

* cohorts in one cell that came from **different fits** (different `model_hash`)
* cohorts, or whole apertures, that disagree on the **censor policy**
* an embedding the plan needs that is absent
* **stale** transition tables — for a state set the latents no longer have
* stage 5 **behind** stage 4 — state sets with no transition table yet

Run it after every stage. It exits non-zero when it finds something.

---

## 3. From zero: the full sequence

### Stage 1–2, per cohort

One cohort at a time. `submit_all.sh` chains extract → finalize → dfc → finalize
with the right dependencies and an ISC gate between them:

```bash
./slurm/submit_all.sh config/ds002837.yaml
./slurm/submit_all.sh config/cneuromod_friends.yaml
./slurm/submit_all.sh config/camcan_movie.yaml
```

Before burning core-hours on a cohort you have not run before:

```bash
$RUN python3 -m fmri_decomposition.cli validate config/<cohort>.yaml
python3 tools/check_cohort.py config/<cohort>.yaml --all --limit 3
```

`submit_all.sh` covers stages 1–2 only, and that is deliberate: it takes one
cohort config, and stages 3–6 span cohorts.

### Stage 2.5 — censor

```bash
$RUN python3 -m fmri_decomposition.cli censor --policy config/censor/motion.yaml
```

Writes `outputs/censor/policy=motion/cohort=<c>/subjects.parquet`. The policy
name is hashed into every fit that uses it, so two runs under different policies
can never be mistaken for each other.

To try a different gate, **copy the YAML** and change its `name` — do not edit
`motion.yaml` in place, or old outputs will claim a policy that no longer means
what it says.

### Stage 3 — decompose

`--censor-policy` is **required**. Pass `none` to fit on everything, which is a
different analysis and is recorded as one.

```bash
# the windowed apertures: one array task per window size
sbatch --array=0-3 slurm/04_decompose.sbatch harvardoxford 30 60 120 300 \
  -- --censor-policy motion
sbatch --array=0-3 slurm/04_decompose.sbatch yeo7         30 60 120 300 \
  -- --censor-policy motion
sbatch --array=0-3 slurm/04_decompose.sbatch networks     30 60 120 300 \
  -- --censor-policy motion

# the frame aperture: one task, because a frame has no window to vary
sbatch --array=0-0 slurm/04_decompose.sbatch harvardoxford -1 \
  -- --source activation --censor-policy motion
```

Check the memory first — it is the training matrix, rows x features x 4 bytes:

```bash
$RUN python3 -m fmri_decomposition.cli decompose \
    --atlas harvardoxford --window-s 30 --censor-policy motion --dry-run
```

**`decompose` skips a cell whose latents already exist.** That is what you want
when resuming and *not* what you want when re-running with changed settings —
pass `--overwrite`, or the job will finish in 20 seconds having done nothing.

### Stage 4 — cluster

```bash
$RUN python3 -m fmri_decomposition.cli cluster --check \
    --atlas harvardoxford yeo7 networks --window-s 30 60 120 300 -1

for A in harvardoxford yeo7 networks; do
  sbatch slurm/04b_cluster.sbatch $A 30 60 120 300 -1
done
```

Run `--check` first. This stage's cost is the HMM — hours at K=27 across a full
grid — and finding a missing embedding after three of those have run is the
expensive way to find out.

**One job per atlas, never an array.** Every clusterer for an (atlas, aperture)
appends to the same parquet file, and parquet cannot append — the file is
rewritten. Two array tasks would each rename over the other's columns.

### Stage 5 — transitions

```bash
$RUN python3 -m fmri_decomposition.cli transitions --check \
    --window-s 30 60 120 300 -1
sbatch slurm/05a_transitions.sbatch -- --window-s 30 60 120 300 -1
```

No arguments needed beyond the apertures: the state sets are **discovered** from
the latents schema, per (atlas, aperture), so a method you add later is picked up
without editing anything.

### Stage 6 — selection

```bash
sbatch --time=06:00:00 slurm/05_select.sbatch additional_HADS_anx_category
sbatch --time=06:00:00 slurm/05_select.sbatch additional_HADS_dep_category
```

Writes `outputs/bstm_selection/target=<t>/` — `summary.csv` (the ranking),
`scores.parquet` (every fit), `DESIGN.md` (what was compared and what was held
fixed) and `figures/`.

**This stage wipes its target directory before writing.** Back up a result you
care about before re-running.

### The whole thing, chained

`sbatch` returns immediately, so stages run concurrently unless you say
otherwise — and stage 3 rewrites the file stage 4 appends to. Let SLURM enforce
the order:

```bash
D=$(for A in harvardoxford yeo7 networks; do
      sbatch --parsable --array=0-0 slurm/04_decompose.sbatch $A -1 \
        -- --source activation --censor-policy motion --overwrite | cut -d';' -f1
    done | paste -sd:)

C=$(for A in harvardoxford yeo7 networks; do
      sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$D \
        slurm/04b_cluster.sbatch $A -1 | cut -d';' -f1
    done | paste -sd:)

T=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$C \
      slurm/05a_transitions.sbatch -- --window-s 30 60 120 300 -1 | cut -d';' -f1)

for Y in additional_HADS_anx_category additional_HADS_dep_category; do
  sbatch --kill-on-invalid-dep=yes --dependency=afterok:$T --time=06:00:00 \
    slurm/05_select.sbatch $Y
done
```

`--kill-on-invalid-dep=yes` cancels the dependents when something upstream fails,
instead of leaving them pending for hours.

---

## 4. Adding a new cohort

Stages 1–2.5 are per cohort, so a new cohort means one new config and one
`submit_all.sh`. Stages 3–6 then re-run **for every cohort together**, because
the fit spans them.

1. **Write `config/<cohort>.yaml`.** Copy the closest existing one.
   `camcan_movie.yaml` for a single short stimulus, `cneuromod_friends.yaml` for
   many episodes per subject, `ds002837.yaml` for one long film per subject with
   preprocessing already applied. The fields that always need attention are
   `tr`, the three absolute paths, `discovery.bold_glob`, `filtering`
   (`already_applied` is the one that silently does nothing if wrong) and
   `confounds`.

2. **Write `config/<cohort>_participants.csv`** if subjects need curating. This
   file is human-owned and load-bearing: rows dropped here never reach stage 1.

3. **Validate before running.**

   ```bash
   $RUN python3 -m fmri_decomposition.cli validate config/<cohort>.yaml
   python3 tools/check_cohort.py config/<cohort>.yaml --all --limit 3
   ```

4. **Stages 1–2.5** for the new cohort only.

5. **Decide its role.** `--train` cohorts are fitted on; `--project` cohorts are
   only labelled. A cohort carrying the phenotype you want to predict should be
   **projected**, never trained on.

6. **Re-run stages 3–6 for every cohort at once**, with `--overwrite` on stage 3.
   A cell whose cohorts came from different `decompose` runs is unusable —
   `status` and `transitions --check` both refuse it — because state 5 is only
   the same state in two cohorts if one fit defined it.

---

## 5. Wiping and starting over

What is safe to delete, cheapest to rebuild first:

| delete | rebuild cost | rebuild with |
|---|---|---|
| `outputs/bstm_selection/` | ~90 min | stage 6 |
| `outputs/transitions/` | minutes | stage 5 |
| `outputs/latents/` | ~1 h per atlas | stages 3 **and** 4 |
| `outputs/censor/` | seconds | stage 2.5 |
| `outputs/dfc/` | hours | stage 2 |
| `outputs/activation/` | many hours | stage 1 |

Deleting `outputs/latents/` also destroys the state columns stage 4 added — they
live **inside** the latents files, so stage 3 and stage 4 always rebuild
together.

Never delete `config/` or `outputs/meta/cohorts/*/`: the first is hand-written,
the second is provenance you cannot regenerate without re-running the stage.

A full wipe is just:

```bash
mv outputs outputs.old.$(date +%F)     # safer than rm until you are sure
mkdir outputs
```

then section 3 from the top.

---

## 6. Things that have actually gone wrong

Each of these cost real time at least once.

| symptom | cause | fix |
|---|---|---|
| job finishes in 10–20 s, "did nothing" | `decompose` skipped a cell whose latents exist | `--overwrite` |
| `python3: executable file not found` | `$FMRIDECOMP_SIF45` unset; apptainer mounted the CWD | `source slurm/env.sh`, use `${VAR:?}` |
| `ModuleNotFoundError: hmmlearn` | ran stage 3–6 in the stage 1–2 image | set `FMRIDECOMP_SIF45` |
| a package the definition has is missing | the image predates that commit | rebuild; `status` and the sbatch scripts compare `/opt/def_sha256` |
| `ArrowTypeError: Unable to merge: Field cohort` | a partition key read as a dataset | use `io.read_file`, never `pq.read_table` on a leaf |
| MeanShift returns K=1 | bandwidth too wide for this data | it now searches the quantile; `--min-k` refuses a degenerate result |
| stage 6 ranks fewer state sets than exist | stage 5 was not re-run after stage 4 | `status` reports it as **behind** |
| the grid spans two censor policies | one aperture built without `--censor-policy` | rebuild that aperture; the flag is required now |

---

## 7. A caveat that is about the data, not the code

The number of transitions a subject contributes depends on the aperture **and on
how long their stimulus is**. For Cam-CAN — one ~8 minute movie — at 80% overlap:

| aperture | windows | transitions per subject |
|---|---|---|
| 30 s | 76 | 74 |
| 60 s | 36 | 35 |
| 120 s | 16 | 15 |
| 300 s | 4 | **2** |
| frame (−1) | ~178 | ~177 |

At 300 s every Cam-CAN subject has exactly two transitions, against 64 cells at
K=8 or 729 at K=27. `switch_rate`, `mean_dwell_s` and `entropy_rate_bits` are
then computed from two events. Nothing errors, and nothing in the output says so
— the column is simply noise.

Check it for any new cohort before trusting a wide aperture:

```bash
$RUN python3 -c "
import pandas as pd
for w in ('30','60','120','300','-1'):
    d = pd.read_parquet(f'outputs/transitions/atlas=yeo7/window_s={w}/'
                        f'states=HMM_pca3_8/cohort=<cohort>/subjects.parquet',
                        columns=['n_transitions'])
    print(f'{w:>4}s  {d.n_transitions.mean():6.1f} +/- {d.n_transitions.std():.2f}')"
```

The same arithmetic explains why `n_transitions`, the nuisance covariate, is
nearly constant within Cam-CAN at the windowed apertures: everyone watches the
same film for the same length of time. It does real work at `-1`, where
frame-level scrubbing makes it vary (177 ± 22), and almost none at 30 s
(74 ± 1.4).
