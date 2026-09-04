# RUNBOOK — running a cohort end to end

Copy-paste commands for the two cohorts that are ready to run in full:
**Cam-CAN CC700 movie** (648 subjects) and **CNeuroMod *friends* season 1**
(6 subjects × 48 episode segments = 288 runs).

Nothing here is new machinery. It is `submit_all.sh` with the argument values
worked out, plus the one step CNeuroMod needs that Cam-CAN does not: fetching
the data.

Everything runs from the repo root on a **login node**. `submit_all.sh` only
submits; it does not compute.

```bash
cd /project/6008063/tamires/DecomposingfMRI
module load python/3.11 scipy-stack
source ~/venvs/fmridecomp/bin/activate
mkdir -p slurm_logs
```

---

## 1. Cam-CAN — the whole cohort

Already preprocessed by `preprocessing/camcan/`, already configured, already
has its 648-row participants table. There is nothing to prepare.

```bash
python tools/check_cohort.py config/camcan_movie.yaml       # ~1 min, headers only
./slurm/submit_all.sh config/camcan_movie.yaml 81 64
```

That is the entire run. The chain it submits is extract array → finalize +
ISC gate → dfc array → merge, with `afterok` between each, so a failure stops
the chain rather than propagating.

**Where 81 and 64 come from.** The unit of stage-2 work is one *(run × atlas)*
pair, not one subject:

| | stage 2 | stage 3 |
|---|---|---|
| jobs | 648 runs × 3 atlases = **1,944** | 648 × (4 sizes × 3 atlases) = **7,776** |
| at 8 CPUs per array task | 81 tasks × 8 = 648 slots, 3 jobs each | 64 × 8 = 512 slots, ~15 each |

`submit_all.sh` recomputes this from `validate` and warns if you ask for more
slots than there are jobs. Both numbers are safe to lower — fewer, longer array
tasks queue faster on a busy scheduler and lose no work if one is killed.

**No `--no-strict`.** Cam-CAN is the one cohort with `stimulus.durations_s`
set (`Movie: 476.71`, confirmed across all 648 subjects), so validation passes
clean and `submit_all.sh` will not stop to ask you anything.

**The one thing to check first.** `config/camcan_movie_participants.csv` has
648 rows; the cohort had 649 subjects with a complete echo set, and CC221648
failed the first fMRIPrep array and was resubmitted individually. If that
resubmission landed, pick it up before extracting — the table is human-owned
and nothing updates it for you:

```bash
python tools/make_participants.py config/camcan_movie.yaml \
    -o config/camcan_movie_participants.csv --update      # adds new rows, keeps exclusions
```

If it did not land, add the row by hand as `excluded=True` with the reason
rather than leaving it absent — an absent row is indistinguishable from an
oversight, which is exactly the distinction `validate_cohort` exists to keep.

---

## 2. CNeuroMod *friends* — all of season 1, all six subjects

### 2a. Fetch (login node, once)

The clone is datalad: it has every filename and none of the imaging data.

```bash
export AWS_ACCESS_KEY_ID=...        # CNeuroMod credentials, post-DTA
export AWS_SECRET_ACCESS_KEY=...

diskusage_report                                            # you need ~345 GiB free
./preprocessing/cneuromod/fetch_friends_s01.sh --dry-run     # exact size, no download
./preprocessing/cneuromod/fetch_friends_s01.sh
```

It fetches 864 files — the BOLD, brain mask and confounds table for each of the
288 runs, and nothing else — verifies that none is left as a dangling symlink,
and is resumable: rerun it after any interruption and it costs a listing pass.

**Yes, 345 GiB.** ~1.5 GB per 718 s run, because these derivatives were written
with `--output-spaces MNI152NLin2009cAsym` and no `res-` specifier, so fMRIPrep
emitted them on the template's native **1 mm** grid. Cam-CAN asked for `res-2`.
See `preprocessing/cneuromod/README.md` for what that costs at stage 2.

### 2b. Run

```bash
python tools/check_cohort.py config/cneuromod_friends.yaml
./slurm/submit_all.sh config/cneuromod_friends.yaml 36 32
```

| | stage 2 | stage 3 |
|---|---|---|
| jobs | 288 runs × 3 atlases = **864** | 288 × (2 + 3×4) = **4,032** |
| at 8 CPUs per array task | 36 tasks × 8 = 288 slots, 3 jobs each | 32 × 8 = 256 slots, ~16 each |

(Stage 3 is 14 and not 15 per run because `windows.by_size` restricts the 15 s
aperture to `yeo7` and `networks`.)

**`submit_all.sh` will stop and ask you a question here.** `stimulus.durations_s`
is empty for this cohort — episode segments genuinely differ in length, so `dfc`
falls back to each file's own observed duration — and `validate` reports that as
one problem per task. The script prints them and offers `--no-strict`. Answer
`y`. That is the expected path for CNeuroMod, and it is the *only* problem you
should answer `y` to: a participants/disk mismatch listed alongside it means the
fetch was incomplete, and the right response is to rerun the fetch script.

---

## 3. While it runs

```bash
squeue -u $USER -o '%.10i %.20j %.8T %.10M %R'
grep -A40 '^plan:' slurm_logs/dfc_<jobid>_0.out     # rows, edges, size, before writing
```

A timed-out or preempted task is safe to resubmit **unchanged** — atomic rename
plus skip-if-exists means it redoes only what is missing:

```bash
sbatch --array=0-80 slurm/01_extract.sbatch config/camcan_movie.yaml
```

If a CNeuroMod extract task is killed for memory (the 1 mm grid is the reason it
would be), halve the workers instead of editing anything:

```bash
sbatch --cpus-per-task=4 --array=0-35 slurm/01_extract.sbatch \
    config/cneuromod_friends.yaml --no-strict
```

## 4. When it finishes

Both cohorts land in the same tree, separated by the `cohort=` partition key:

```
outputs/activation/atlas=harvardoxford/cohort=camcan/task=Movie/sub=CC110033/data.parquet
outputs/activation/atlas=harvardoxford/cohort=cneuromod/task=s01e01a/sub=01/ses-003.parquet
outputs/dfc/atlas=harvardoxford/window_s=60/cohort=.../task=.../sub=.../
outputs/meta/cohorts/cohort=camcan/participants_qc.csv
outputs/meta/cohorts/cohort=cneuromod/participants_qc.csv
```

`participants_qc.csv` is written by the activation finalize, with no separate
step. It is **measurement only** — `mean_fd`, `best_lag_tr`,
`frac_stimulus_covered`, `frac_parcels_empty`, `frac_good_frames` — and carries
no thresholds. Read it before believing any of the output:

```bash
fmri-decomp diagnose config/camcan_movie.yaml     # prints each metric's spread
```

Two things it will show that are properties of these cohorts, not failures:

- **Cam-CAN's 30 s window holds 12 samples** (TR 2.47). Treat it as unusable
  here rather than merely noisy, and note that a fixed FD threshold removes more
  data from older subjects in an 18–88 lifespan cohort — which manufactures an
  age effect out of a QC rule. `config/camcan_movie.yaml` says more.
- **CNeuroMod ISC is computed per episode segment**, six subjects each, so the
  leave-one-out reference is a mean of five. `peak_isc` is reported and
  deliberately not offered as an exclusion criterion.
