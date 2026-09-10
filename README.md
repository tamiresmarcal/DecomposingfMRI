# fmri_decomposition

Stages 2 (activation) and 3 (DFC) of the naturalistic-viewing pipeline.
Cohort-specific knowledge lives in `config.py` and `cohort.py`; nothing
downstream touches a filesystem path or a TR.

```bash
pip install -e ".[test,atlases]"
./run_tests.sh                       # unit tests + synthetic end-to-end run
fmri-decomp validate  config/ds002837.yaml
fmri-decomp extract   config/ds002837.yaml --n-jobs 8
fmri-decomp dfc       config/ds002837.yaml --dry-run       # rows before compute
fmri-decomp dfc       config/ds002837.yaml --n-jobs 8 --window-s 15 30 60 120 300
```

### The pipeline, end to end

```
0    fmriprep / afni_proc            outside this repo -- cohorts arrive preprocessed
1.1  01_extract.sbatch    (array)    NIfTI     -> parcel timeseries
     03_finalize.sbatch activation   merge manifests + coverage + L-R
                                     + ISC gate + participants_qc.csv
1.2  02_dfc.sbatch        (array)    parquet   -> windowed connectivity
     03_finalize.sbatch dfc          merge manifests
3.5  fmri-decomp censor              participants_qc.csv + window flags
                                     -> keep/drop, under a named policy
4    fmri-decomp decompose           windowed DFC -> latents, fit on some
                                     cohorts and projected onto others
```

`03_finalize` runs twice, taking the stage as an argument. The activation pass
is where the ISC gate lives, and `--dependency=afterok` on the DFC array is
what stops stage 3 from running on misaligned data.

No threshold appears in stages 1–3: they measure. `censor` is the one place a
measurement becomes a decision, and it does so under a policy that is named,
versioned and hashed into every row it writes — see below.

---

## Output structure

Hive-partitioned parquet. Partition keys are **directory names**; the dataset
exists only at read time, when pyarrow walks the tree and reconstructs the keys.
Nothing is ever appended to — each worker owns one leaf and writes one file.

```
outputs/
├── activation/                                   STAGE 2 — one file per run per atlas
│   └── atlas=harvardoxford/
│       ├── cohort=ds002837/task=500daysofsummer/sub=1/data.parquet
│       ├── cohort=cneuromod/task=s01e01a/sub=01/data.parquet
│       └── cohort=hcp7t/task=MOVIE2/sub=100610/data.parquet
│
├── dfc/                                          STAGE 3 — window_s between atlas and cohort
│   └── atlas=harvardoxford/
│       ├── window_s=30/cohort=ds002837/task=500daysofsummer/sub=1/data.parquet
│       ├── window_s=60/cohort=ds002837/task=500daysofsummer/sub=1/data.parquet
│       └── window_s=120/cohort=cneuromod/task=s01e01a/sub=01/data.parquet
│
├── latents/                                      STAGE 4 — reserved, adds model=
│   └── atlas=.../window_s=.../model=pca50/cohort=.../task=.../sub=.../
│
└── meta/
    ├── atlas-harvardoxford_labels.csv            atlas-level: cohort-independent
    ├── atlas-yeo7_labels.csv
    ├── atlas-networks_labels.csv
    ├── models/                                   STAGE 4 group fits, not partitioned
    └── cohorts/
        └── cohort=ds002837/                      everything scoped to one cohort
            ├── manifest_activation.json
            ├── manifest_dfc.json
            ├── coverage.parquet                  n_good / n_total per stimulus TR
            ├── isc_alignment.csv                 sub, movie, best_lag_tr, peak_isc
            ├── atlas-harvardoxford_lr_diagnostic.csv
            └── shards/                           per-array-task manifests, merged later
```

Three rules the layout encodes:

**Atlas is the outermost key**, because it is the only key that changes *column
count* (111 vs 14 vs 7). A dataset root must be schema-homogeneous, so there has
to be a directory meaning "one atlas, many subjects" — and with atlas any deeper
there would not be one. `window_s` sits directly below it so that "one atlas,
one window size, every cohort" is a single readable path, which is the pooled
query stage 4 runs.

**`task` sits above `sub`**, a fixed convention chosen for the majority case
where a cohort has one stimulus. It costs CNeuroMod roughly 300 thin directories
per atlas per window size; the alternative would make the path shape
cohort-dependent, which is the same class of problem as a per-cohort `run=`
level.

**Directory depth is constant, and so is the leaf name.** Every leaf is
`data.parquet`, in every cohort — a reader never has to know which cohort it is
looking at to guess the filename. `ses`, `run`, `acq` and `run_key` are
*columns* in the table, so the name would only have been a partial second copy.

The cost: two runs of the same `(cohort, atlas, task, sub)` — one subject, one
task, two sessions — now land on the same leaf, and the second would overwrite
the first. `fmri-decomp validate` refuses to pass such a cohort, so it cannot
reach a compute node. If it fires, the answer is not to put the entity back in
the filename; it is that `(task, sub)` is not the unit of analysis for that
cohort and the config has to say so.

Because a cohort no longer owns a subtree of the data, its provenance lives in
`meta/cohorts/cohort=X/`, keyed the same hive way so the same walk finds it.
Atlas label tables stay at `meta/` — they describe the atlas, not the cohort.

### Partition keys are not columns

`cohort`, `atlas`, `task`, `sub` and `window_s` are carried by the path and are
deliberately absent from the files. A key duplicated as a column has to match
its inferred type exactly, and pyarrow reads `sub=01` as **int32** — which both
collides with the string column and silently destroys `CC110033` and every
leading zero. `ses`, `run`, `acq` and `run_key` remain columns, since the
filename rather than a directory carries them.

Use the helpers rather than `partitioning="hive"`, which re-introduces the type
inference:

```python
from fmri_decomposition.io import open_dataset, read_shard, dfc_root

d  = open_dataset(dfc_root(out, "harvardoxford", 30), stage="dfc")  # all cohorts
df = read_shard(path)      # one leaf, partition keys restored as columns
```

One caveat that is quiet rather than loud: a key is recovered from the path
*relative to the dataset root*, so a key at or above the root comes back as a
column of **nulls** — `open_dataset(dfc_root(out, "harvardoxford", 30))` has no
`atlas` and no `window_s`, and a filter on either matches zero rows without
raising. Narrow with a filter, not with a deeper root, or backfill the keys the
root swallowed (`notebooks/nbtools.py` does the latter).

Every file also carries `cohort`, `task`, `sub`, `atlas` in its parquet
key-value metadata, so a shard opened by hand is still self-identifying.

### The coordinate atlas ships with the package

`atlases/data/mni_space_of_networks.csv` is package data, not a test file:
Harvard-Oxford and Yeo are fetched by nilearn from a name, but this atlas has no
fetcher — the CSV *is* the parcellation. Bundling it is what lets it satisfy the
same registry contract as the other two:

```python
get_atlas("networks")                    # 14 networks, 91 edges
get_atlas("networks_nodes")              # 254 nodes, 32,131 edges -> packed storage
get_atlas("networks", csv_path=my_csv)   # override with your own coordinates
```

The `description` column (the citation each network was defined from) is carried
through to `meta/atlas-networks_labels.csv` rather than dropped at load, so the
outputs stay traceable to their source. Note the template caveat applies most
sharply here: a 5 mm sphere is small enough that the few-mm NLin6/NLin2009c
offset matters, and many seeds sit in subcortex where it is largest.

### Reading it

```python
import pyarrow.dataset as ds

from fmri_decomposition.io import dfc_root, open_dataset

# One atlas, one window size, pooled across every cohort.
d = open_dataset(dfc_root("outputs", "harvardoxford", 120), stage="dfc")

# Predicate pushdown: only the matching directories are opened.
df = d.to_table(filter=(ds.field("cohort") == "ds002837")).to_pandas()

# Columnar: selecting QC columns physically reads three columns, not the file.
qc = d.to_table(columns=["window_id", "n_tr_effective", "frac_good_frames"]).to_pandas()
```

`notebooks/` opens both stages this way: `01_activation.ipynb` for stage 2,
`02_dfc.ipynb` for stage 3, sharing the loaders in `notebooks/nbtools.py`
(inventory, partition-pruned reads, footer-based size estimates, and the
participants / QC / phenotype join). `notebooks/README.md` has the container
recipe and the memory arithmetic.

### The window grid is atlas-conditional

`windows.sizes_s` is what to run; `windows.by_size` is where, and how:

```yaml
windows:
  sizes_s: [15, 30, 60, 120, 300]
  n_overlaps: 5
  by_size:
    15:
      atlases: [yeo7, networks]     # not harvardoxford, and never networks_nodes
      # n_overlaps: 3               # optional: a coarser stride at this aperture
  # rank_policy: skip               # the derived form of the same idea (below)
```

Two independent things make a short window size not portable across atlases,
both pure arithmetic:

**Rank.** A window is `round(window_s / TR)` samples, and a correlation matrix
over *p* nodes is singular unless *n − 1 ≥ p*. Inverting that gives a floor
that is a property of the atlas and the TR, not a number anyone picked:

```python
min_window_s_for_nodes(n_nodes, tr)   # == (n_nodes + 0.5) * tr
```

| atlas | nodes | floor at TR = 1 | at TR = 1.49 |
|---|---|---|---|
| `yeo7` | 7 | 7.5 s | 11.2 s |
| `networks` | 14 | 14.5 s | 20.9 s |
| `harvardoxford` | 111 | 111.5 s | 166.1 s |
| `networks_nodes` | 254 | 254.5 s | 379.2 s |

Note the third row: **"short windows are only for the coarse atlases" is the
right instinct with the wrong constant.** A fixed 30 s rule would still admit
Harvard-Oxford, which needs 111.5 s — it is already past the line at 30 s and
60 s in the grid that has been running, and stays there.

Be precise about what rank deficiency costs, because it is easy to overstate:
nothing comes back NaN. Each of Harvard-Oxford's 6,105 edges is an ordinary
two-variable correlation over 15 samples and is finite, just very noisy
(SE ≈ 0.29, so ~6% of edges pass |r| > 0.5 under the null). What is undefined
is the **matrix** — so it matters if you invert or decompose it, and is merely
noise if you analyse edges one at a time.

That is why the enforcement is a choice rather than a default:

| | effect |
|---|---|
| `rank_policy: warn` (default) | print the floor in the plan, run the pair anyway |
| `rank_policy: skip` | do not run any (atlas, size) pair below its floor |

`skip` is the general form of a `by_size.atlases` restriction: derived, so a
window size nobody enumerated is covered too. It would also stop producing
Harvard-Oxford at 30 s and 60 s, which is an analysis decision, not a cleanup.

**Rows.** Window count goes as `1/stride`, so halving the aperture at fixed
`n_overlaps` doubles the rows. On a 5,470 s film:

| window_s | stride (n_overlaps=5) | windows/subject | ×300 s |
|---|---|---|---|
| 300 | 60 s | 87 | 1× |
| 120 | 24 s | 223 | 2.6× |
| 60 | 12 s | 451 | 5.2× |
| 30 | 6 s | 907 | 10× |
| 15 | 3 s | 1,819 | 21× |

At yeo7's 21 edges and `networks`' 91 that 21× is ~90 MB uncompressed for all
86 subjects — nothing. The same grid is ~3.8 GB on `harvardoxford` and ~20 GB
on `networks_nodes`' 32,131 edges. So the row count is not what makes the
configured run affordable; it is what keeps it affordable when someone widens
`atlases:` later. Check before launching, not after:

```bash
fmri-decomp dfc config/ds002837.yaml --dry-run
```

which prints per (atlas, window size) the shard count, estimated rows, edge
count and uncompressed size, from parquet footers only — no table is read.
`n_overlaps: 3` at the fine aperture cuts its rows by ~40% (1,819 → 1,092).
The stride is **not** part of the output path, only `window_s` is, so changing
it for a size that already ran needs `--overwrite`; the value that produced a
shard is in its schema metadata.

### Subject-level QC, and where exclusion happens

Two files, split by **who owns them**:

| file | owner | carries | acted on |
|---|---|---|---|
| `participants.csv` | human | curation — "corrupted run", "consent withdrawn" | stage 2 drops these rows before extraction |
| `meta/cohorts/cohort=<c>/participants_qc.csv` | pipeline | measurement | **nothing** — thresholds live with the models |

`participants_qc.csv` is written by `fmri-decomp diagnose`, which
`03_finalize.sbatch` already runs after the extract array. So the metrics
appear without a separate step, and ISC is computed once for both the gate and
the table. It is regenerated from scratch every run and must never be
hand-edited.

One row per `(sub, task)`, one column block per failure mode — chosen to be
about four **different** things, since a long list of correlated criteria costs
sample size while only looking rigorous:

| column | catches | how it is measured |
|---|---|---|
| `mean_fd` | head movement | Power FD from `confounds.motion_glob`, or fMRIPrep's own column |
| `best_lag_tr` | wrong stimulus timing | ISC peak lag vs. the leave-one-out mean of the same film |
| `frac_stimulus_covered` | scan stopped early | shard length ÷ longest scan of the same film |
| `frac_parcels_empty` | registration failure | all-NaN parcel columns, on the finest atlas with shards |
| `frac_good_frames` | scrubbing survival | `good_frame` — identically 1.0 where censoring is off |

`peak_isc` is written and deliberately **not** offered as a criterion: no
absolute scale, confounded with motion, and with 6 subjects per film for 8 of
ds002837's 10 films the reference is itself a mean of five.

**No threshold appears anywhere in this pipeline.** `mean_fd = 0.52` is a
measurement — deterministic and reproducible. `0.52 > 0.5 → exclude` is a
decision, arguable and part of what you are claiming. The first belongs in the
pipeline; the second belongs with the analysis that rests on it, so a
sensitivity analysis can move a cutoff without re-running any of this.
`diagnose` prints each metric's spread and worst subjects so the distribution
is visible, and stops there.

Since `dfc` walks the filesystem rather than `participants.csv`, it warns when
it finds shards for excluded subjects instead of skipping them. The real filter
is `censor`, below.

### `censor` — stage 3.5, where a measurement becomes a decision

Keeping thresholds out of the pipeline is right, and it leaves a gap: the claim
still has to be made somewhere, and made inline in whatever notebook needs it,
it gets made differently every time and travels with nothing. `censor` closes
that gap with one step between measurement and modelling.

```bash
# Subjects only -- reads participants_qc.csv for every cohort that has one.
fmri-decomp censor --policy config/censor/default.yaml

# Also gate windows, for one atlas x aperture of the DFC path.
fmri-decomp censor --policy config/censor/default.yaml \
    --stage dfc --atlas harvardoxford --window-s 30
```

It reads only QC columns, never imaging data, and runs in seconds — so it is a
login-node command, not a SLURM job. It writes:

```
outputs/censor/policy=<name>/cohort=<c>/subjects.parquet
outputs/censor/policy=<name>/atlas=<a>/window_s=<w>/cohort=<c>/windows.parquet
outputs/meta/censor/policy=<name>.json          # counts + what was skipped
```

Every row carries `keep`, a human-readable `reason` for the drops, and
`policy` / `policy_hash`. Change a number, change the policy `name`, and the
new outputs land **beside** the old ones: a sensitivity analysis is two files,
not a re-run with different constants.

Three things it deliberately does not do:

* **It does not censor frames.** Per-TR censoring already happened at stage 2
  and is in `good_frame`. Where it could not — ds002837, whose regressor and
  image timelines cannot be reconciled — it cannot be recovered here either,
  which is exactly why `max_mean_fd` at the subject level does the work there.
* **It does not edit `participants.csv`.** That file is human curation and is
  not the place for a threshold someone will want to move.
* **It does not drop a cohort for a missing input.** A rule whose column is
  absent or all-NaN is reported as `NOT APPLIED` and skipped rather than
  failing every row. `best_lag_tr` is the live case: it is NaN for any task
  with fewer than three subjects.

Window gating is only for the DFC path — the HMM path runs on stage 2
activation, where `good_frame` is already per-TR. `drop_crosses_run_boundary`
defaults to true because a transition across a run boundary is not a
transition; `drop_rank_deficient` defaults to **false**, because rank
deficiency makes the correlation *matrix* singular while each edge in it stays
an ordinary two-variable correlation (see the aperture section above).

#### ISC is computed per stimulus

`isc_alignment` groups by task. Pooling ds002837's ten films would build a
"group mean" out of ten unrelated soundtracks, truncate everyone to the
shortest film, and report a lag against noise — latent while `include_tasks`
named one film, live once the cohort opened to all ten. A task with fewer than
3 subjects gets `NaN` and a stated reason rather than being silently dropped;
`n_subjects` travels with every row.

### Row contracts

| stage 2 columns | | stage 3 columns | |
|---|---|---|---|
| `t` | int32, TR index in file | `window_id` | int32, index on the **stimulus** grid |
| `time_s` | float32, `t × TR` | `start_tr` | int32, anchor back into stage 2 |
| `stimulus_time_s` | float32, position in stimulus | `stimulus_start_s`, `stimulus_end_s` | float32 |
| `good_frame` | bool, false where censored | `n_tr_nominal` | int16, `round(window_s / TR)` |
| `run_idx` | int8 | `n_tr_effective` | int16, **good frames actually used** |
| `<parcel_1..N>` | float32, NaN if empty | `frac_good_frames` | float32 |
| `ses/run/acq/run_key` | string | `crosses_run_boundary`, `crosses_clip_boundary`, `rank_deficient` | bool |
| | | edges | float32, raw *r* |

Edges are one column per edge (`Left-Amygdala__Right-Amygdala`) below ~20,000,
and a packed `fixed_size_list<float32>` above it — HO-111 gives 6,105 (columns),
Schaefer-1000 gives 499,500 (packed). Use `dfc.read_edges()` to be
storage-mode agnostic.

`n_tr_effective`, not `n_tr`, is the reliability column: pairwise deletion means
subjects contribute different frame counts to the same window.

---

## SLURM

First time on a cluster, once:

```bash
cp slurm/env.sh.example slurm/env.sh   # then edit it
```

`slurm/env.sh` is gitignored and holds the things that are true of *your*
account rather than of the project: which interpreter to use
(`FMRIDECOMP_SIF` for a container, `FMRIDECOMP_VENV` for a venv), the SLURM
account to bill, and where the atlas cache lives. Every script in `slurm/`
sources it, so submitting by hand behaves the same as the driver. Anything
already exported in your shell wins over the file.

Then:

```bash
./slurm/submit_all.sh config/ds002837.yaml 20 8
```

That chains: extract array (20 tasks) → finalize + ISC gate → dfc array
(8 tasks) → merge manifests, with `afterok` between each. It creates
`slurm_logs/` itself.

An interpreter is not optional: a login node's bare `python` cannot import
`fmri_decomposition`, and neither can a compute node's. `submit_all.sh` checks
before submitting anything and refuses rather than treating the ImportError as
a config problem — otherwise the pre-flight `validate` is skipped silently and
the whole chain runs unvalidated. Or submit by hand:

```bash
sbatch --array=0-19 slurm/01_extract.sbatch config/ds002837.yaml
sbatch --array=0-7  slurm/02_dfc.sbatch     config/ds002837.yaml 30 60
```

Each array task takes `--shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT` and
processes runs `[i::n]` — round-robin, so a 90-minute movie and an 8-minute one
don't pile into the same task. A timed-out task is safe to resubmit unchanged:
atomic rename plus skip-if-exists means it redoes only what is missing.

### Differences from `sbatch_processing_15_69.sh`

| legacy | here | why |
|---|---|---|
| `--ntasks=30` | `--cpus-per-task=8` | joblib/loky forks inside **one** task and sees only that task's cores. `--ntasks=30` asks for 30 independent tasks that may land on different nodes, 29 of them idle. |
| `--mem-per-cpu=16G` (480 GB total) | 8G stage 2, 2G stage 3 | stage 3 reads parquet, never a NIfTI. Smaller asks also clear the queue faster. |
| one monolithic job, 48 h | two arrays, 12 h + 3 h | a walltime kill lost everything; now it loses one shard. |
| `array_%A_%a` log names, no `--array` | real array jobs | the legacy names suggest an array was intended. |
| hardcoded paths in the `.py` | config path as `$1` | same script for every cohort. |
| `--account=def-aevans` | `--account=rpp-aevans-ab` | the legacy sbatch and the legacy data paths disagree — **check which allocation you mean to charge.** |

Shared files (`manifest.json`, diagnostics, the atlas label CSVs) have exactly
one writer, in `03_finalize.sbatch`. Array tasks write per-shard manifests into
`meta/shards/`, merged afterwards by `fmri-decomp merge-manifests`. Never let
workers write `_metadata` / `_common_metadata`.
