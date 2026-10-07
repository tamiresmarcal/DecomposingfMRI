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
         |                  (threshold / MeanShift / HMM1 / HMM2
         |                   x pca3 / umap3 / raw<N>)
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

Three cohorts prepared, states defined on two of them and projected onto the
third, then the analysis. Every command below is copy-pasteable in order.

```bash
cd /project/6008063/tamires/DecomposingfMRI
source slurm/env.sh
RUN="apptainer exec --cleanenv --bind /project,/scratch,/home ${FMRIDECOMP_SIF45:?source slurm/env.sh}"
```

### PHASE 1 — PREPARE (per cohort)

One `activation_and_dfc.sh` per cohort. Each chains
extract -> finalize -> dfc -> finalize -> censor with the right dependencies and
an ISC gate between them, so the three can run at the same time without
interfering. When the last job of one exits, that cohort is completely prepared.

```bash
./slurm/activation_and_dfc.sh config/ds002837.yaml
./slurm/activation_and_dfc.sh config/cneuromod_friends.yaml
./slurm/activation_and_dfc.sh config/camcan_movie.yaml
```

The censor policy defaults to `config/censor/motion.yaml`. Override it without
editing the script:

```bash
FMRIDECOMP_CENSOR=config/censor/strict.yaml ./slurm/activation_and_dfc.sh config/camcan_movie.yaml
```

For a cohort you have never run, check it first — this is minutes against hours:

```bash
$RUN python3 -m fmri_decomposition.cli validate config/<cohort>.yaml
python3 tools/check_cohort.py config/<cohort>.yaml --all --limit 3
```

**Wait for all three to finish** before going on. Phase 2 fits across cohorts,
so a cohort that is still being written would silently be left out of the fit:

```bash
squeue -u $USER                       # empty = done
$RUN python3 -m fmri_decomposition.cli status | sed -n '/^1  ACTIV/,/^3 /p'
```

Then read the gate, which is the one number in phase 1 worth looking at by eye —
it is a decision, not a measurement:

```bash
grep -E 'kept|policy' slurm_logs/censor_*.out
```

### Changing your mind about the gate

Re-gating is cheap and needs **only** the censor step — nothing upstream is
touched, because the threshold is the last thing phase 1 does and the stages
that consume it select by name:

```bash
cp config/censor/motion.yaml config/censor/strict.yaml
$EDITOR config/censor/strict.yaml                 # change `name:` too
sbatch slurm/censor.sbatch config/camcan_movie.yaml config/censor/strict.yaml
```

Then rebuild phase 2 onward with `--censor-policy strict`. The old decision
stays on disk beside the new one, so both remain reproducible.

**Never edit a policy in place.** Outputs already written record the policy by
*name*; editing the file they came from leaves them claiming a gate that no
longer means what it says.

### PHASE 2 — DEFINE STATES (across cohorts)

`--train` are the cohorts the fit sees; everything else is projected onto it.
A cohort carrying the phenotype you want to predict should be **projected**,
never trained on.

```bash
TRAIN="--train ds002837 cneuromod --project camcan"
POL="--censor-policy motion"

# dimensionality reduction: PCA + UMAP coordinates
#   windowed apertures -- one array task per window size
for A in harvardoxford yeo7 networks; do
  sbatch --array=0-3 slurm/dimensionality_reduction.sbatch $A 30 60 120 300 \
    -- $TRAIN $POL
done
#   the frame aperture -- one task, a frame has no window to vary.
#   yeo7 and networks also get --passthrough-features, which writes the NAMED
#   parcels beside the PCs so stage 4b can fit on them directly; and the extra
#   --n-latents is the atlas's own width, which makes that PCA a lossless
#   rotation. --umap-latents keeps UMAP at 3, where it belongs.
sbatch --array=0-0 slurm/dimensionality_reduction.sbatch harvardoxford -1 \
  -- --source activation $TRAIN $POL
sbatch --array=0-0 slurm/dimensionality_reduction.sbatch yeo7 -1 \
  -- --source activation $TRAIN $POL \
     --n-latents 3 7 --umap-latents 3 --passthrough-features
sbatch --array=0-0 slurm/dimensionality_reduction.sbatch networks -1 \
  -- --source activation $TRAIN $POL \
     --n-latents 3 14 --umap-latents 3 --passthrough-features
```

Wait for those, then look before spending hours on the HMM:

```bash
$RUN python3 -m fmri_decomposition.cli cluster --check \
    --atlas harvardoxford yeo7 networks --window-s 30 60 120 300 -1 \
    --embeddings pca3 umap3 raw7 raw14
```

`raw7` and `raw14` show as absent in most cells and that is correct — they exist
only where passthrough ran. A missing raw embedding is skipped per cell, not an
error.

**Measure hmm2 before committing to the grid.** It is full covariance, 500
iterations and 15 restarts, which is two orders of magnitude more expensive than
hmm1. One cheap cell tells you what walltime the real run needs:

```bash
sbatch slurm/clustering.sbatch yeo7 30 -- --methods hmm2 --k 10 --hmm2-restarts 5
sacct -X -n -o Elapsed,State --name=fmridecomp_cluster | head -1
```

Cost is roughly linear in `--hmm2-restarts` and `--hmm2-iter`, and steeply worse
in K and in the embedding's width. Then:

```bash
# clustering: coordinates -> brain-state labels, appended to the same files
for A in harvardoxford yeo7 networks; do
  sbatch --time=24:00:00 slurm/clustering.sbatch $A 30 60 120 300 -1 \
    -- --train ds002837 cneuromod --embeddings pca3 umap3 raw7 raw14
done
```

Two state definitions, on purpose. **HMM1** is the incumbent — diagonal
covariance, 50 iterations, one fit — frozen so it stays a fixed baseline.
**HMM2** is the estimator van der Meer et al. 2020 used. At matched K and
embedding the only thing that differs is the estimator, so a difference in the
stage-6 ranking is attributable to it.

**One job per atlas, never an array.** Every clusterer for an (atlas, aperture)
appends to the same parquet, and parquet cannot append — the file is rewritten,
so two array tasks would each rename over the other's columns.

### PHASE 3 — ANALYSE (the cohort with phenotype)

```bash
$RUN python3 -m fmri_decomposition.cli transitions --check --window-s 30 60 120 300 -1
sbatch slurm/brain_states_transitions.sbatch -- --window-s 30 60 120 300 -1
```

Then, once that finishes:

```bash
for Y in additional_HADS_anx_category additional_HADS_dep_category; do
  sbatch --time=06:00:00 slurm/model_selection.sbatch $Y
done
```

The phenotype table defaults to Cam-CAN's release. Everything after `--` goes
straight to `select`, so another cohort's table needs no edit to the script:

```bash
sbatch slurm/model_selection.sbatch severity -- \
  --pheno /path/to/table.tsv:$'\t' --id-col SubjectID --cohort hcp
```

The `--target` column may be numeric, or labelled. Labels are matched against
`Normal Mild Moderate Severe` — Cam-CAN's HADS wording — so any other set has to
be declared, **lowest first**, since the target is fitted as a number and the
order is the claim:

```bash
  --target severity --ordinal-levels low mid high
```

Get that wrong and `select` stops and says so, naming the values it found; it
does not quietly code them to NaN.

`model_selection` **wipes its target directory before writing** — back up a
result you care about first.

### Finally

```bash
$RUN python3 -m fmri_decomposition.cli status | sed -n '/^======/,$p'
column -s, -t outputs/bstm_selection/target=additional_HADS_anx_category/summary.csv | head -12
```

### Looking at a state rather than ranking it

Everything above treats a state as an integer. This is the one command that
says what the state *is* — one row per state, one column per named network:

```bash
$RUN python3 -m fmri_decomposition.cli state-means \
    --atlas networks --window-s -1 --states HMM2_raw14_10 --csv states.csv
```

It works on a `pca<N>` state set too: the rotation is undone from the models
`decompose` saved, exactly. On a reduced embedding like `pca3` it prints a NOTE
saying the means are the state's position in that 3-D subspace rather than its
mean over all the features — true, and the kind of thing a figure should not
hide. UMAP state sets are refused rather than approximated, because UMAP has no
inverse that recovers its input.

This is the table van der Meer et al.'s Fig. 1 plots.

### PHASES 2-3 as one chained submission

Phase 1 stays separate: its cohorts are independent and `activation_and_dfc.sh` already
chains within each. Phases 2-3 are one dependency graph, and `sbatch` returns
immediately, so without `--dependency` they would all start at once — and
dimensionality reduction rewrites the very file clustering appends to.

```bash
source slurm/env.sh
ATLASES="harvardoxford yeo7 networks"
TRAIN="--train ds002837 cneuromod --project camcan"
POL="--censor-policy motion"
APERTURES="30 60 120 300 -1"

D=$(for A in $ATLASES; do
      sbatch --parsable --array=0-3 slurm/dimensionality_reduction.sbatch \
        $A 30 60 120 300 -- $TRAIN $POL | cut -d';' -f1
      sbatch --parsable --array=0-0 slurm/dimensionality_reduction.sbatch \
        $A -1 -- --source activation $TRAIN $POL | cut -d';' -f1
    done | paste -sd:)
[[ -n "$D" ]] || { echo "nothing submitted"; exit 1; }

C=$(for A in $ATLASES; do
      sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$D \
        slurm/clustering.sbatch $A $APERTURES \
        -- --train ds002837 cneuromod | cut -d';' -f1
    done | paste -sd:)

T=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$C \
      slurm/brain_states_transitions.sbatch -- --window-s $APERTURES | cut -d';' -f1)

for Y in additional_HADS_anx_category additional_HADS_dep_category; do
  sbatch --kill-on-invalid-dep=yes --dependency=afterok:$T --time=06:00:00 \
    slurm/model_selection.sbatch $Y
done

squeue -u $USER -o "%.12i %.32j %.9T %.11M %R"
```

`--kill-on-invalid-dep=yes` cancels the dependents when something upstream fails,
rather than leaving them pending for hours on a dependency that will never be
satisfied.

Rough walltimes, measured: dimensionality reduction 8-50 min per atlas (UMAP
transform on the largest cohort is the long pole), clustering ~25 min per atlas,
transitions minutes, model selection ~90 min. End to end for phases 2-3, about
three hours.

## 4. Adding a new cohort

Stages 1–2.5 are per cohort, so a new cohort means one new config and one
`activation_and_dfc.sh`. Stages 3–6 then re-run **for every cohort together**, because
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
                        f'states=HMM1_pca3_8/cohort=<cohort>/subjects.parquet',
                        columns=['n_transitions'])
    print(f'{w:>4}s  {d.n_transitions.mean():6.1f} +/- {d.n_transitions.std():.2f}')"
```

The same arithmetic explains why `n_transitions`, the nuisance covariate, is
nearly constant within Cam-CAN at the windowed apertures: everyone watches the
same film for the same length of time. It does real work at `-1`, where
frame-level scrubbing makes it vary (177 ± 22), and almost none at 30 s
(74 ± 1.4).
