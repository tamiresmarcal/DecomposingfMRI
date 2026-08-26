# Runbook: validating the pipeline on a real cohort

Written for a cluster run. The order matters: every step before step 7 is
free, and step 7 is the one that costs core-hours.

Only **ds002837** has a config today (`config/ds002837.yaml`). CNeuroMod and
Cam-CAN need one written first — see [Adding a cohort](#adding-a-cohort).

---

## Gotchas that will bite you, up front

| | |
|---|---|
| **Atlas fetch needs internet** | `harvardoxford` and `yeo7` are downloaded by nilearn on first use. Most clusters give compute nodes **no** outbound network, so the fetch must happen on a **login node** (step 3). `networks` ships with the package and needs nothing. |
| **Relative paths in the YAML resolve against `$PWD`** | not against the config file. Under a scheduler your CWD is not the repo. **Use absolute paths** for `derivatives_root`, `output_root` and `participants`. |
| **ds002837 needs `--no-strict` on extract** | its `stimulus.durations_s` is deliberately empty (runs vary in length per subject), so validation reports one expected warning. Stage 3 derives duration per file and needs no flag. |
| **`space:` is a label, not a transform** | nothing resamples your data into it. It is recorded for provenance. Set it to whatever your derivatives actually are. |

---

## 1. Get the code

```bash
git clone https://github.com/tamiresmarcal/DecomposingfMRI.git
cd DecomposingfMRI
git checkout claude/real-subjects-testing-5jvi53
```

## 2. Environment (login node)

```bash
module load python/3.11 scipy-stack       # adjust to your cluster
python -m venv ~/venvs/fmridecomp
source ~/venvs/fmridecomp/bin/activate
pip install --upgrade pip
pip install -e ".[test,atlases]"
```

## 3. Pre-fetch the atlases — **login node, before any job**

```bash
export NILEARN_DATA=$HOME/nilearn_data      # a path compute nodes can READ
python -c "
from fmri_decomposition.atlases.registry import get_atlas
for name in ('harvardoxford', 'yeo7', 'networks'):
    a = get_atlas(name)
    print(f'{name}: {a.n_nodes} nodes, {a.n_edges} edges')
"
```

Put the same `NILEARN_DATA` export in your sbatch scripts. If this is skipped,
every array task dies on a network timeout.

## 4. Verify the install

```bash
./run_tests.sh --no-venv        # 142 unit tests + a synthetic end-to-end run
```

All 142 must pass before real data is worth trying.

## 5. Point the config at your data

Edit `config/ds002837.yaml` — **absolute paths only**:

```yaml
derivatives_root: /scratch/$USER/ds002837/derivatives
output_root:      /scratch/$USER/ds002837/outputs
participants:     /home/$USER/DecomposingfMRI/config/ds002837_participants.csv
```

Then confirm the two guesses in that file against your actual tree:

```bash
ls /scratch/$USER/ds002837/derivatives/sub-1/func/       # bold variant
ls /scratch/$USER/ds002837/derivatives/sub-1/anat/       # mask name
ls /scratch/$USER/ds002837/derivatives/sub-1/regressors/ # censor file
```

`discovery.bold_glob` selects `*_bold_no_blur_no_censor.nii.gz` (unblurred, for
ROI analysis, censoring left to this pipeline). `discovery.mask_glob` is my
guess at the EPI-space mask name — if it is wrong, extract fails loudly on a
shape mismatch rather than silently, but check it now anyway.

## 6. Pre-flight checks — free, no core-hours

```bash
fmri-decomp validate config/ds002837.yaml
python tools/check_cohort.py config/ds002837.yaml
```

`check_cohort.py` covers **points 1 and 3** of what you wanted to verify:

- **path uniqueness** — every (run × atlas) maps to a distinct output leaf.
  This is the precondition the lock-free parallelism rests on; a collision
  means two workers race on one file and one silently wins. This is the check
  that catches a new cohort whose `ses`/`run` entities aren't being captured.
- **TR** — compares your config TR against each NIfTI header. Note it will
  *warn* rather than confirm for ds002837: NNDb omits `RepetitionTime`, and a
  header TR of exactly 1.0 is also nibabel's default-when-unset, so it cannot
  corroborate the real value. The 1.0s comes from the paper, not the files.

Do not proceed while `paths` reports FAIL.

## 7. Pilot: two subjects — **this is where spending starts**

```bash
fmri-decomp extract config/ds002837.yaml --n-jobs 4 --no-strict --limit 2
```

Then verify **point 2 (outputs)** and **point 4 (parallelism)**:

```bash
python tools/check_cohort.py config/ds002837.yaml --checks outputs,parallel \
    --limit 2 --n-jobs 4
```

- **outputs** — contract columns present, partition keys *not* duplicated as
  columns (pyarrow reads `sub=01` as int32, which destroys leading zeros and
  Cam-CAN ids), shard metadata matches the current config, no stray `.tmp`.
- **parallel** — extracts the same runs at `n-jobs=1` and `n-jobs=N` into two
  temp dirs and compares values. Catches any worker-count dependence.

Look at a shard yourself too:

```bash
python -c "
from fmri_decomposition.io import read_shard
import glob
df = read_shard(sorted(glob.glob('/scratch/$USER/ds002837/outputs/activation/**/*.parquet', recursive=True))[0])
print(df.shape); print(df.iloc[:3, :8])
print('good frames:', df.good_frame.mean().round(3))
"
```

## 8. Stage 3 on the pilot

```bash
fmri-decomp dfc config/ds002837.yaml --n-jobs 4 --window-s 30 60
fmri-decomp dfc config/ds002837.yaml --n-jobs 4 --window-s 30 60   # must skip everything
```

The second run must report `skipped=` for every shard. That is what makes a
timed-out SLURM task safe to resubmit unchanged.

## 9. Diagnostics — the scientific gate

```bash
fmri-decomp diagnose config/ds002837.yaml
```

Writes coverage, an L–R correlation diagnostic, and ISC alignment. Exit code 2
means the ISC gate failed: subjects are not aligned to a common stimulus clock,
and stage 3 output is not comparable across subjects. **Investigate before
scaling** — the SLURM chain gates stage 3 on this passing, deliberately.

## 10. Scale up

```bash
sbatch --account=YOUR_ACCOUNT --array=0-19 slurm/01_extract.sbatch config/ds002837.yaml
# or the whole dependency chain:
./slurm/submit_all.sh config/ds002837.yaml 20 8
```

Edit the sbatch headers first — they carry `--account=rpp-aevans-ab`, which is
almost certainly not your allocation. Each array task takes
`--shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT` and processes runs
`[i::n]` round-robin, so a 90-minute movie and a 30-minute one don't pile into
one task.

After the array finishes:

```bash
fmri-decomp merge-manifests config/ds002837.yaml --stage activation
fmri-decomp merge-manifests config/ds002837.yaml --stage dfc
```

---

## Adding a cohort

Everything cohort-specific is the YAML. To add CNeuroMod or Cam-CAN, copy
`config/ds002837.yaml` and change:

- `cohort`, `tr`, `derivatives_root`, `output_root`
- `discovery.bold_glob` / `mask_glob` — and `subject_pattern` if ids aren't
  `sub-<alnum>`
- `confounds` — format (`fmriprep_tsv` for CNeuroMod-style derivatives vs
  `afni_1D`), and the censor file convention
- `stimulus.durations_s` if the stimulus has one fixed length
- a participants CSV with `participant_id, sub, cohort, task, excluded,
  exclusion_reason`

Then run step 6. `check_cohort.py` is cohort-agnostic by design — the path
check in particular is what tells you whether a multi-session or multi-run
cohort is going to collide before you spend anything finding out.

**Watch for**: CNeuroMod has `ses` and `run` entities, so confirm they land in
the *filename* (`ses-003_run-01.parquet`), never as new directory levels.
Cam-CAN subject ids like `CC110033` must stay strings — the path check and
`test_io.py` both cover this, but confirm your `subject_pattern` captures them.
