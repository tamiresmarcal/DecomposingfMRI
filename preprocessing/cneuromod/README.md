# `preprocessing/cneuromod/` — fetching, not preprocessing

Unlike `preprocessing/camcan/`, nothing here computes anything. CNeuroMod
arrives already preprocessed by fMRIPrep 20.2.6; the only stage-1 problem is
that the derivatives are a **datalad** dataset, so cloning it gives you 49,516
dangling annex symlinks and zero bytes of imaging data.

```
fetch_friends_s01.sh   LOGIN NODE. datalad get, season 1, sub-01..06.
```

## Why a script rather than one `datalad get`

`datalad get sub-01` works and is wrong. The dataset carries, per run, a T1w-space
BOLD, an fsLR CIFTI, `aseg`/`aparcaseg` segmentations, boldrefs and a dozen
report figures. `config/cneuromod_friends.yaml` reads three files:

| file | read by |
|---|---|
| `*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz` | `discovery.bold_glob` |
| `*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz` | `discovery.mask_glob` |
| `*_desc-confounds_timeseries.tsv` | `confounds.confounds_glob` |

The script fetches those and nothing else — 864 files, 288 runs, six subjects,
all 48 season-1 segments.

## The number to look at before you start

**~345 GiB.** That is ~1.5 GB per 718 s run, which is roughly 40× what Cam-CAN's
runs cost, and it is not a mistake in either direction. These derivatives were
produced with `--output-spaces MNI152NLin2009cAsym` and **no `res-` specifier**
(visible in the dataset's own `code/fmriprep_study-friends_sub-*.sh`), so
fMRIPrep wrote them on the MNI152NLin2009cAsym template's native **1 mm** grid —
~3.2 MB per volume. `config/camcan_movie.yaml` asked for `res-2` and gets 2 mm.

Two consequences worth knowing before, not after:

- **Quota.** Run `diskusage_report` first. `--dry-run` prints the exact figure
  and downloads nothing.
- **Memory at stage 2.** `extract_parcels` streams the 4D file in
  `TIME_CHUNK = 64` volume blocks, so peak memory is bounded — but a 1 mm block
  is 64 × 8.5 M voxels × 8 B ≈ 4.4 GB per worker, against ~0.55 GB at 2 mm.
  `01_extract.sbatch` asks `--mem-per-cpu=8G` with 8 CPUs, so 8 joblib workers
  fit in the 64 GB, with less headroom than the other cohorts. If a task is
  killed for memory, halve the workers rather than editing anything:
  `sbatch --cpus-per-task=4 --array=... slurm/01_extract.sbatch <config>`.

## Access

Content lives in an S3 bucket at `s3.unf-montreal.ca` which git-annex enables
automatically. Reading it needs the credentials CNeuroMod issues after the data
transfer agreement — see <https://docs.cneuromod.ca/en/latest/ACCESS.html>.
Export `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, or leave them in
`~/.aws/credentials`. Bad credentials fail *every* file identically rather than
producing a partial fetch, which is the easy case to recognise.

Compute nodes have no outbound network and the pipeline container has no
datalad, so this must run on a login node, outside any job.

## One upstream correction, recorded so nobody rediscovers it

sub-01's season-1 episodes were once mislabelled: what is now `s01e01` under
`ses-003` was published as `s01e06`, and vice versa. Upstream fixed it
(`friends.fmriprep`, commit *"rename run for stimuli mistake"*). A clone made
before that fix has the two episodes swapped for sub-01, and **nothing
downstream can detect it** — the files are valid, the ISC gate sees six
subjects watching what it is told is the same segment, and the answer is
quietly wrong for two of the 48.

`fetch_friends_s01.sh` runs `datalad update` on an existing clone for exactly
this reason. Do not skip it.
