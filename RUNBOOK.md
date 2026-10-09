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
RUN23="apptainer exec --cleanenv --bind /project,/scratch,/home --pwd $PWD ${FMRIDECOMP_SIF:?source slurm/env.sh}"
```

**`$RUN23` is the OTHER image**, and `tools/*.py` need it: a bare `python3` on a
login node has no pandas, so `tools/make_participants.py` and
`tools/check_cohort.py` die on `ModuleNotFoundError: No module named 'pandas'`
before doing anything. They are stage 1-2 tools, so they take the stage 2-3
image -- the same `fmri_decomp.sif` that `slurm/activation_and_dfc.sh` runs its
own pre-flight check in.

The sbatch scripts source `env.sh` themselves, so only your interactive commands
need this.

**The phenotype tables, once.** Every selection run takes `--pheno` and none of
them has a default — the path is a fact about your filesystem, not about this
pipeline, and a cohort's own release table is not in `outputs/`. Cam-CAN splits
what is needed across two files, so it is a list:

```bash
CAMCAN=/project/6008063/tamires/cohorts/camcan/dataman/useraccess/opendata/paule_toussaint_camcan01870
PHENO=("$CAMCAN/approved_data.tsv:"$'\t' "$CAMCAN/standard_data.csv:,")
```

An **array**, and every use below is `--pheno "${PHENO[@]}"`. Two reasons it
cannot be a plain string: the entries are two separate arguments, and the first
one's separator is a real tab, which `$'\t'` produces and `"...\t..."` does not.
Both survive `sbatch ... -- --pheno "${PHENO[@]}"` — the `--` forwarding keeps
each element whole.

Each entry is `path:separator`. They are merged on `--id-col` (default `CCID`).
Nothing in this repository reads them except the two selection stages, and
nothing writes a copy of them into `outputs/` — the covariates go into the fit
and the per-subject values do not come back out.

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
$RUN23 python3 tools/check_cohort.py config/<cohort>.yaml --all --limit 3
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
TRAIN="--train ds002837 cneuromod"
PROJ="--project camcan"
POL="--censor-policy motion"

# dimensionality reduction: PCA + UMAP coordinates
#   windowed apertures -- one array task per window size
for A in harvardoxford yeo7 networks; do
  sbatch --array=0-3 slurm/decomposition.sbatch $A 30 60 120 300 \
    -- $TRAIN $PROJ $POL
done
#   the frame aperture -- one task, a frame has no window to vary.
#   yeo7 and networks also get --passthrough-features, which writes the NAMED
#   parcels beside the PCs so stage 4b can fit on them directly; and the extra
#   --pca-latents is the atlas's own width, which makes that PCA a lossless
#   rotation. --umap-latents keeps UMAP at 3, where it belongs.
sbatch --array=0-0 slurm/decomposition.sbatch harvardoxford -1 \
  -- --source activation $TRAIN $POL
sbatch --array=0-0 slurm/decomposition.sbatch yeo7 -1 \
  -- --source activation $TRAIN $POL \
     --pca-latents 3 7 --umap-latents 3 --passthrough-features
sbatch --array=0-0 slurm/decomposition.sbatch networks -1 \
  -- --source activation $TRAIN $POL \
     --pca-latents 3 14 --umap-latents 3 --passthrough-features
```

Wait for those, then look before spending hours on the HMM:

```bash
$RUN python3 -m fmri_decomposition.cli cluster --check \
    --atlas harvardoxford yeo7 networks --window-s 30 60 120 300 -1
```

`--embeddings` defaults to the families `pca umap raw`, resolved per cell:
`raw` becomes `raw7` on yeo7, `raw14` on networks and nothing on
harvardoxford, read from the files rather than named in the command. A family
that resolves to nothing is skipped and reported, never an error. Pass an exact
name (`raw14`) only when one specific width is the point.

A method that can run on **none** of the chosen embeddings is a hard error, not
a skip — `--methods threshold --embeddings raw` would otherwise produce nothing
anywhere and still report success.

**hmm2 costs two orders of magnitude more than hmm1**, and it is worth knowing
exactly where that goes. Measured on `harvardoxford -1`, 691,434 training rows:

| | time | EM iterations | per iteration |
|---|---|---|---|
| `HMM1_pca3_8` | 77 s | 50 | 1.54 s |
| `HMM1_pca3_27` | 695 s | 50 | 13.9 s |
| `HMM2_pca3_8` | 12,631 s | 7,500 | 1.68 s |
| `HMM2_pca3_10` | 19,032 s | 7,500 | 2.54 s |

Per-iteration cost is the **same** — full covariance in 3-D is cheap. The whole
gap is 15 restarts x 500 iterations against hmm1's single 50. So cost is linear
in `--hmm2-restarts` and `--hmm2-iter`, roughly `K**1.8`, and linear in rows —
which is why the `-1` aperture is the expensive one: it has a row per TR, where
a windowed aperture has a row per window.

The restarts run **in parallel** (`--hmm2-jobs`, defaulting to
`SLURM_CPUS_PER_TASK`), because they are independent fits and a single hmmlearn
fit is sequential over time. At 8 cores that turns a projected 29 h `K=27` cell
into about 4 h for the same core-hours.

**Split by aperture.** A cell is an (atlas, aperture) and different apertures are
different files, so this is safe, schedules faster, and cuts wall-clock about
five-fold against one job per atlas:

```bash
for A in harvardoxford yeo7 networks; do
  for W in 30 60 120 300 -1; do
    sbatch --time=24:00:00 slurm/clustering.sbatch $A $W \
      -- --train ds002837 cneuromod
  done
done
```

Do **not** split one cell by method — an hmm2-only job beside an hmm1 job on the
same cell is two writers on one parquet file, and the second rename wins.

A walltime kill is safe: columns are written per state set with an atomic
rename, so whatever finished is on disk and a resubmit redoes only the rest.

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
  sbatch --time=06:00:00 slurm/model_selection.sbatch $Y -- \
    --cohort camcan --task movie --pheno "${PHENO[@]}"
done
```

`--cohort`, `--task` and `--pheno` are all required and none has a default: a
stage that spans cohorts must not name one, and a path into one person's
scratch space only works for one project. `--task` is the hive partition the
result lands under, so this writes
`outputs/bstm_selection/task=movie/target=$Y/`.

Everything after `--` goes straight to `select-bstm`, so another cohort needs no
edit to the script:

```bash
sbatch slurm/model_selection.sbatch severity -- \
  --pheno /path/to/table.tsv:$'\t' --id-col SubjectID \
  --cohort hcp --task movie
```

The `--target` column may be numeric, or labelled. Labels are matched against
`Normal Mild Moderate Severe` — Cam-CAN's HADS wording — so any other set has to
be declared, **lowest first**, since the target is fitted as a number and the
order is the claim:

```bash
  --target severity --ordinal-levels low mid high
```

Get that wrong and `select-bstm` stops and says so, naming the values it found; it
does not quietly code them to NaN.

`select-bstm` **wipes its own `task=`/`target=` directory before writing** — and
only that one, so a `--task rest` run cannot empty the movie ranking beside it.
Back up a result you care about first.

### Finally

```bash
$RUN python3 -m fmri_decomposition.cli status | sed -n '/^======/,$p'
column -s, -t \
  outputs/bstm_selection/task=movie/target=additional_HADS_anx_category/summary.csv \
  | head -12
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

### PHASE 4 — THE OTHER TWO TREES (is any of it better than the alternatives?)

Phase 3 answers *which movie state set* predicts best. It cannot answer whether
the whole idea beats what it is supposed to improve on. Three more rankings do.
They are not three new trees: there are **two trees, each partitioned by
`task=`**, and phase 3 filled one of their four cells.

| run | what it is | what it costs |
|---|---|---|
| `fcm_selection/task=movie` | one correlation matrix per subject, flattened | one pass over stage 2 |
| `bstm_selection/task=rest` | the same pipeline on the rest scan | fMRIPrep + stage 2 |
| `fcm_selection/task=rest` | the same matrix, from the rest scan | falls out of the two above |

The FC arm on the movie alone needs nothing new preprocessed:

```bash
git pull
sbatch slurm/static_fc.sbatch camcan \
    -- --atlas harvardoxford yeo7 networks
# once that finishes
sbatch slurm/fcm_selection.sbatch additional_HADS_anx_category -- \
    --cohorts camcan --task movie --pheno "${PHENO[@]}"
```

Everything below is the rest arm, which starts at fMRIPrep.

#### 4.0 — pull, and set the environment once

```bash
cd /project/6008063/tamires/DecomposingfMRI
git pull
source slurm/env.sh                 # FMRIDECOMP_SIF, FMRIDECOMP_SIF45, binds, account
```

**Containers.** Two images, one boundary (`containers/README.md`):

| stages | image | env var |
|---|---|---|
| 1 (fMRIPrep) | none — `module load StdEnv/2023 fmriprep/25.1.1` | — |
| 2–3, 3b | `fmri_decomp.sif` | `FMRIDECOMP_SIF` |
| 4–5 | `fmri_decomp_stage45.sif` | `FMRIDECOMP_SIF45` |

Every `slurm/*.sbatch` picks the right one itself and refuses early, by name,
if the image cannot import what it needs. Stage 1 uses no container at all —
it is a cluster module.

`$RUN` below means the stage 2–3 interpreter, for the few read-only commands
that run on a login node:

```bash
RUN="apptainer exec --bind ${FMRIDECOMP_BINDS:-/project,/scratch,/home} --pwd $PWD $FMRIDECOMP_SIF python3"
```

#### 4.1 — confirm the rest acquisition (login node, seconds)

Three of four values are already confirmed in `config/camcan_rest.yaml`
(TR 1.97, `func_rest`, single-echo). This re-checks them on one subject and
costs nothing:

```bash
R=/project/6008063/tamires/cohorts/camcan/cc700/mri/pipeline/release004/BIDSsep
ls $R/func_rest/sub-CC110033/func/
python3 -c "import json;print(json.load(open('$R/func_rest/sub-CC110033/func/sub-CC110033_task-Rest_bold.json'))['RepetitionTime'])"
```

Expect one `_bold.nii.gz` with **no** `echo-` in the name, and `1.97`.

#### 4.2 — build the rest BIDS tree (login node, seconds)

Symlinks only, so it costs no disk. Pilot on 10 first:

```bash
python3 preprocessing/camcan/01_build_bids.py \
    --task Rest --n-echoes 0 --func-subdir func_rest \
    -o /project/6008063/tamires/cohorts/camcan_rest_bids \
    --dataset-name "Cam-CAN CC700 rest (BIDS view for fMRIPrep)" \
    --limit 10
```

It prints `usable subjects` and a breakdown of what it skipped. **0 usable is
never a fact about the data** — it is a wrong `--func-subdir`, `--n-echoes` or
`--task`, and the script refuses rather than exiting 0. When the count looks
right, drop `--limit 10` and re-run to take all of them.

#### 4.3 — fMRIPrep the rest scans (the one slow step)

Prereqs are already satisfied from the movie run — same `TEMPLATEFLOW_HOME`,
same FreeSurfer licence — and the script checks both before the module loads.

```bash
export REST_BIDS=/project/6008063/tamires/cohorts/camcan_rest_bids
export REST_OUT=/project/6008063/tamires/cohorts/camcan_rest_fmriprep
export REST_WORK=/project/6008063/tamires/work/camcan_rest_fmriprep

# pilot: the first 10 lines of subjects.txt
BIDS_ROOT=$REST_BIDS OUT_ROOT=$REST_OUT WORK_ROOT=$REST_WORK TASK_ID=Rest \
  sbatch --array=0-9 preprocessing/camcan/02_fmriprep.sbatch
```

The allocation is in the script: **8 CPUs, 8G/CPU (64G), 8 h**, one array task
per subject. That walltime was sized for the five-echo movie; single-echo rest
has no echo combination to do, so expect well under it — check the pilot's
`finished ... with status 0` lines before scaling up.

```bash
seff <jobid>_0            # what the pilot actually used

# expand subjects.txt from the pilot's 10 to all of them
python3 preprocessing/camcan/01_build_bids.py \
    --task Rest --n-echoes 0 --func-subdir func_rest -o $REST_BIDS \
    --dataset-name "Cam-CAN CC700 rest (BIDS view for fMRIPrep)"

N=$(wc -l < $REST_BIDS/subjects.txt)
BIDS_ROOT=$REST_BIDS OUT_ROOT=$REST_OUT WORK_ROOT=$REST_WORK TASK_ID=Rest \
  sbatch --array=10-$((N-1))%60 preprocessing/camcan/02_fmriprep.sbatch
```

**This is safe to do while the pilot is still running**, and the reason is
worth knowing rather than trusting. Each array task reads `subjects.txt` at
startup (`sed -n "$((IDX+1))p"`), so rewriting the file mid-flight could in
principle hand a pending task a different subject. It cannot here: `--limit`
selects `usable[:limit]`, a sorted PREFIX, so lines 1-10 are byte-identical
before and after. A pilot task that has not started yet still resolves to the
subject it was queued for, and the `IDX >= N_SUBS` guard only loosens.

Re-running the BIDS build leaves the 10 already linked completely alone:
`link()` returns early when the destination exists and `--force` was not
passed. Do NOT reach for `--force` here — it unlinks and re-creates every
symlink, which is a window in which a live job can fail to resolve a path.

What you give up by submitting early is sizing the big array from the pilot's
real `seff`. That is a bounded risk -- the prereq checks already passed, so a
systematic problem would have failed the pilot in seconds -- but if the pilot
turns out to fail for a data reason, cancel and look before re-queueing:

```bash
scancel <big_jobid>
```

`%60` caps concurrent tasks. Raise it if the queue is empty; it is there
because each subject's nipype work dir holds tens of thousands of files and the
inode quota is the real limit, not CPU.

Then the only value still unconfirmed — the volume count:

```bash
$RUN -c "import glob,nibabel as nib,collections; \
print(collections.Counter(nib.load(f).shape for f in \
glob.glob('$REST_OUT/sub-*/func/*task-Rest*desc-preproc_bold.nii.gz')))"
```

**One line of output means one shape**, which is also the corruption check. If
it is 261 everywhere, set `stimulus.durations_s: {Rest: 514.17}` in
`config/camcan_rest.yaml`. If it varies, leave it empty — an assumed fixed
duration where runs differ makes the window grid claim windows some subjects
never acquired.

#### 4.4 — phase 1 for rest (extract, QC, censor)

```bash
$RUN23 python3 tools/make_participants.py config/camcan_rest.yaml \
    -o config/camcan_rest_participants.csv
$RUN23 python3 tools/check_cohort.py config/camcan_rest.yaml
./slurm/activation_and_dfc.sh config/camcan_rest.yaml
```

Two prompts to expect, both from the chain script and both `y` here:

* *"validate reported problem(s) … continue with --no-strict?"* — the missing
  `stimulus.durations_s`, which is expected until 4.3 confirms it. `dfc` falls
  back to each file's observed length.
* a shard-sizing warning, if the default 8 array tasks over-provisions. It
  prints the number to use; re-run with it.

`config/camcan_rest.yaml` sets **`isc_gate_tr: null`**, which is the one line
without which this chain cannot run at all. `diagnose` exits 2 when ISC
alignment fails and stage 3 is chained behind it with `--dependency=afterok`;
at rest there is nothing shared to correlate, so the best lag is noise and the
gate would FAIL on a fact about the design. ISC is still computed and written —
read it the other way round: rest ISC should be near **zero**, and a resting
cohort whose subjects correlate strongly is a finding, not a pass.

This also gives you the windowed apertures (`30 60 120 300`) at rest, since the
chain runs `dfc`. For activation-only, skip `activation_and_dfc.sh` and submit
its first two links by hand:

```bash
E=$(sbatch --parsable --array=0-7 slurm/extract_activations.sbatch \
      config/camcan_rest.yaml --no-strict)
F=$(sbatch --parsable --dependency=afterok:$E slurm/finalize.sbatch \
      config/camcan_rest.yaml activation)
sbatch --dependency=afterok:$F slurm/censor.sbatch \
      config/camcan_rest.yaml config/censor/motion.yaml
```

#### 4.5 — the FC tree (both conditions)

Independent of everything in 4.6, and the cheaper half:

```bash
sbatch slurm/static_fc.sbatch camcan camcan_rest \
    -- --atlas harvardoxford yeo7 networks
# then, once it finishes -- one run per condition, because `task=` is the
# partition the ranking lands under and a run writes exactly one of them
for Y in additional_HADS_anx_category additional_HADS_dep_category; do
  sbatch slurm/fcm_selection.sbatch $Y -- \
    --cohorts camcan      --task movie \
    --atlas harvardoxford yeo7 networks --pheno "${PHENO[@]}"
  sbatch slurm/fcm_selection.sbatch $Y -- \
    --cohorts camcan_rest --task rest \
    --atlas harvardoxford yeo7 networks --pheno "${PHENO[@]}"
done
```

`static-fc` is one run over both cohorts — it writes `cohort=` partitions and
nothing about it is per-condition. `select-fcm` is two, one per `task=`. That
asymmetry is the point: a measurement spans cohorts, a ranking is about one
condition.

`static_fc.sbatch` asks for 4 CPUs / 32G / 3 h; `fcm_selection.sbatch` for
16 CPUs / 32G / 3 h. Both are single jobs, not arrays — each (atlas, cohort)
writes one `subjects.parquet`, so an array would race to write one file.

#### 4.6 — the rest state tree

Rest joins the cell as a **projected** cohort, so the states stay the movie's
and only the dynamics differ. `--project` gains `camcan_rest`; `--train` does
not change:

```bash
source slurm/env.sh
ATLASES="harvardoxford yeo7 networks"
TRAIN="--train ds002837 cneuromod"
PROJ="--project camcan camcan_rest"
POL="--censor-policy motion"

D=$(for A in $ATLASES; do
      sbatch --parsable --array=0-0 slurm/decomposition.sbatch $A -1 \
        -- --source activation $TRAIN $PROJ $POL | cut -d';' -f1
    done | paste -sd:)

C=$(for A in $ATLASES; do
      sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$D \
        --time=24:00:00 slurm/clustering.sbatch $A -1 \
        -- $TRAIN $PROJ | cut -d';' -f1
    done | paste -sd:)

T=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$C \
      slurm/brain_states_transitions.sbatch -- --window-s -1 | cut -d';' -f1)

for Y in additional_HADS_anx_category additional_HADS_dep_category; do
  sbatch --kill-on-invalid-dep=yes --dependency=afterok:$T \
    slurm/model_selection.sbatch $Y -- \
      --cohort camcan_rest --task rest --pheno "${PHENO[@]}"
done

squeue -u $USER -o "%.12i %.32j %.9T %.11M %R"
```

Allocations come from the scripts: decompose 4 CPUs / 64G / 6 h (raised here is
rarely needed at `-1`), clustering 8 CPUs / 32G / 8 h — **override to 24 h as
above**, because HMM2 at K=27 ran ~4 h on 8 cores and three atlases of
`harvardoxford -1` are the long pole. Transitions 4 CPUs / 16G / 2 h.

`--kill-on-invalid-dep=yes` cancels the dependents when something upstream
fails, instead of leaving them pending for hours on a dependency that will
never be satisfied.

#### Adding rest to the shared state space

Rest joins as a **projected** cohort, so it never defines states of its own —
and since `decompose` and `cluster` both reuse what they already have, adding it
costs minutes rather than repeating the fits.

That was not true before this was built, and the two reasons are worth knowing
because they are the shape of the whole stage:

* `cluster` cannot append in place — parquet cannot — so it reads each latents
  file, adds label columns, writes a temp and renames. `decompose` writes the
  same path. A `decompose` run therefore **removes the state columns** from any
  cohort it rewrites.
* `decompose` used to skip only when *every* cohort in `train + project`
  already had a file, and otherwise rewrote all of them. One missing cohort
  rewrote the training cohorts too.

Both now work per cohort:

| | what it skips | what forces it |
|---|---|---|
| `decompose` | a cohort whose file already carries this `model_hash`; and the fit itself, when the saved one matches | `--overwrite` |
| `cluster` | the fit, when one is cached for this `fit_hash`; and a cohort already carrying that `fit_hash` | `--refit` |

Two guards on the caches, both refusing rather than approximating: a cached
clusterer records the `model_hash` of the latents it was fitted on, and both
caches record the library versions they were pickled under. A mismatch means
refit, because a pickled sklearn or hmmlearn estimator is not guaranteed to
behave across versions.

So the sequence is just:

```bash
squeue -u $USER -n fmridecomp_cluster      # wait for any clustering to drain

for A in harvardoxford yeo7 networks; do
  sbatch --array=0-0 slurm/decomposition.sbatch $A -1 \
    -- --source activation --train ds002837 cneuromod \
       --project camcan camcan_rest --censor-policy motion
done
# then clustering, which will reuse every fit and label only camcan_rest
```

You should see `already at this model_hash, left untouched: ...` from
`decompose` and `reusing the cached fit ... labelling only, no refit` from
`cluster`. If you see `loading training cohorts` or a K= line taking hours,
something did not match — check the `model_hash` line before letting it run.

**One caveat that applies to the cohort you add next, not to rest.** The fit
caches are written by the run that fits. A cell clustered *before* this change
has no cached clusterer, so the first run after it still fits once. That is the
re-run rest pays for; everything after it is cheap.

#### 4.7 — read the four tables

Two trees, two `task=` partitions each, one shape:

```bash
Y=additional_HADS_anx_category
for T in bstm_selection fcm_selection; do
  for K in movie rest; do
    echo "== $T / task=$K"
    column -s, -t outputs/$T/task=$K/target=$Y/summary.csv | head -6
  done
done
```

Every table starts with `run` and `target` — `fcm movie`, `bstm rest` — so the
four can be concatenated and still say which is which:

```bash
Y=additional_HADS_anx_category
head -1 outputs/bstm_selection/task=movie/target=$Y/summary.csv > all.csv
for T in bstm_selection fcm_selection; do for K in movie rest; do
  tail -n +2 outputs/$T/task=$K/target=$Y/summary.csv >> all.csv
done; done
column -s, -t all.csv | head -20
```

After those two, every table has `model arm atlas n n_features mean std min max
count`, sorted best-first. What to read:

* `mean` against the `(covariates only)` row **in its own table** — that is the
  floor, and age and sex predict HADS on their own.
* `std` next to every `mean`. A gap smaller than either arm's spread across
  fold seeds is not a result.
* **`n` first, always.** These are four separate runs on whoever each one has,
  so two tables can differ in sample as well as in score. If they differ and
  the gap matters, build one id list and pass it to all four:

  ```bash
  $RUN tools/shared_subjects.py --cohorts camcan camcan_rest \
      -o shared_subjects.txt
  ```

  It intersects the **inputs** — `transitions/` and `static_fc/` — so the list
  can be built before any ranking has run, and it prints a count per
  (stage, cohort) so a source that contributed nothing is visible rather than
  inferred. It refuses rather than silently skipping one. Then re-run all four
  with it:

  ```bash
  for Y in additional_HADS_anx_category additional_HADS_dep_category; do
    sbatch slurm/model_selection.sbatch $Y -- \
        --cohort camcan      --task movie --pheno "${PHENO[@]}" \
        --restrict-subjects shared_subjects.txt
    sbatch slurm/model_selection.sbatch $Y -- \
        --cohort camcan_rest --task rest  --pheno "${PHENO[@]}" \
        --restrict-subjects shared_subjects.txt
    sbatch slurm/fcm_selection.sbatch $Y -- \
        --cohorts camcan      --task movie --pheno "${PHENO[@]}" \
        --restrict-subjects shared_subjects.txt
    sbatch slurm/fcm_selection.sbatch $Y -- \
        --cohorts camcan_rest --task rest  --pheno "${PHENO[@]}" \
        --restrict-subjects shared_subjects.txt
  done
  ```
* In `fcm_selection`, `edges` against `global`. `global` is a subject's mean and
  SD over every edge — two numbers with no topology in them. If it matches
  `edges`, the pattern is not what is being measured; the amount is.
* **Identical `mean`, `std`, `min` and `max` down a whole model's block** means
  that model predicted a constant and never split, so its ordering of the arms
  is noise. Both selection stages now say so by name when it happens. It needs
  a training fold under ~40 subjects to occur, so it is a small-`n` symptom.

None of the four subtracts one row from another, on purpose. The arms share
their folds, so a difference between two scores is dependent and has no
standard error the usual tests supply. A real test belongs in the write-up.

Each run **wipes its own `task=`/`target=` directory before writing** — and only
that one, so a rest run cannot empty the movie ranking beside it. Back up a
result you care about first.

### PHASES 2-3 as one chained submission

Phase 1 stays separate: its cohorts are independent and `activation_and_dfc.sh` already
chains within each. Phases 2-3 are one dependency graph, and `sbatch` returns
immediately, so without `--dependency` they would all start at once — and
dimensionality reduction rewrites the very file clustering appends to.

```bash
source slurm/env.sh
ATLASES="harvardoxford yeo7 networks"
TRAIN="--train ds002837 cneuromod"
PROJ="--project camcan"
POL="--censor-policy motion"
APERTURES="30 60 120 300 -1"

D=$(for A in $ATLASES; do
      sbatch --parsable --array=0-3 slurm/decomposition.sbatch \
        $A 30 60 120 300 -- $TRAIN $PROJ $POL | cut -d';' -f1
      sbatch --parsable --array=0-0 slurm/decomposition.sbatch \
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
    slurm/model_selection.sbatch $Y -- \
      --cohort camcan --task movie --pheno "${PHENO[@]}"
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
   $RUN23 python3 tools/check_cohort.py config/<cohort>.yaml --all --limit 3
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
