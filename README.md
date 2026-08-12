# fmri_decomposition

Stages 2 (activation) and 3 (DFC) of the naturalistic-viewing pipeline.
Cohort-specific knowledge lives in `config.py` and `cohort.py`; nothing
downstream touches a filesystem path or a TR.

```bash
pip install -e ".[test,atlases]"
./run_tests.sh                       # unit tests + synthetic end-to-end run
fmri-decomp validate  config/ds002837.yaml
fmri-decomp extract   config/ds002837.yaml --n-jobs 8
fmri-decomp dfc       config/ds002837.yaml --n-jobs 8 --window-s 30 60 120 300
```

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
│       ├── cohort=cneuromod/task=s01e01a/sub=01/ses-003.parquet
│       └── cohort=hcp7t/task=MOVIE2/sub=100610/data.parquet
│
├── dfc/                                          STAGE 3 — window_s between atlas and cohort
│   └── atlas=harvardoxford/
│       ├── window_s=30/cohort=ds002837/task=500daysofsummer/sub=1/data.parquet
│       ├── window_s=60/cohort=ds002837/task=500daysofsummer/sub=1/data.parquet
│       └── window_s=120/cohort=cneuromod/task=s01e01a/sub=01/ses-003.parquet
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

**Directory depth is constant.** Leftover entities go in the *filename*, never a
directory: `sub=01/ses-003_run-01.parquet`.

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

```bash
mkdir -p slurm_logs
./slurm/submit_all.sh config/ds002837.yaml 20 8
```

That chains: extract array (20 tasks) → finalize + ISC gate → dfc array
(8 tasks) → merge manifests, with `afterok` between each. Or submit by hand:

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
