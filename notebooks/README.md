# Analysis notebooks

Exploration of what the pipeline wrote. Nothing here writes to `outputs/`.

| file | what it opens |
|---|---|
| `01_activation.ipynb` | stage 2 — parcel timeseries, one dataset per cohort |
| `02_dfc.ipynb` | stage 3 — windowed connectivity, QC first, edges second |
| `03_qc.ipynb` | one histogram grid: six QC metrics x every cohort, three cells |
| `04_activation_pca.ipynb` | open activation → filter by a `{cohort: [subs]}` dict → PCA |
| `nbtools.py` | the loaders they use: inventory, pruned reads, the participants join |

`03` and `04` are deliberately small: `03` is a load and a histogram grid and
nothing else, `04` is three steps (open, filter, decompose) with the PCA done
by `numpy.linalg.svd` so it needs no scikit-learn.

A blank panel in `03` is never a plotting bug — the cell above the grid names
any metric that is absent from `participants_qc.csv` or present but all-NaN
(`peak_isc` and `best_lag_tr` are NaN for a task with fewer than 3 subjects,
by design).

`nbtools` wraps `fmri_decomposition.io` rather than re-deriving anything —
every path it builds comes from there. What it adds is the part a notebook
needs and the pipeline does not: an inventory of what is on disk, reads that
prune partitions and project columns, a footer-based size estimate, and the
participants / QC / phenotype join.

## Running them

Python — Jupyter included — exists only inside the container. The login node's
bare `python` cannot import `fmri_decomposition`, and nothing heavy should run
there anyway.

```bash
module load apptainer/1.4.5          # without it, `apptainer` opens a menu
salloc --account=rpp-aevans-ab --cpus-per-task=4 --mem=32G --time=3:00:00

export FMRIDECOMP_SIF=/project/6008063/tamires/singularity/fmri_decomp.sif
cd /project/6008063/tamires/DecomposingfMRI
apptainer exec --bind /project,/scratch,/home --pwd $PWD $FMRIDECOMP_SIF \
    jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

Then from your laptop, tunnelling through the login node to the compute node
`salloc` gave you (`squeue -u $USER` names it):

```bash
ssh -L 8888:<node>:8888 <user>@nibi.alliancecan.ca
```

If the image has no `jupyter`, run the notebooks headless instead — same
container, same results, no tunnel:

```bash
apptainer exec --bind /project,/scratch,/home --pwd $PWD $FMRIDECOMP_SIF \
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_activation.ipynb
```

`nbtools` finds `outputs/` from the `output_root:` in the configs. Point it
somewhere else with `FMRIDECOMP_OUTPUTS=/path/to/outputs`.

## Is `--mem=32G --cpus-per-task=4` right?

For these two notebooks, yes, with room to spare — and it is the *edges* that
set it, not the subject count.

Every read below is one atlas, uncompressed, before pandas overhead:

| read | size |
|---|---|
| activation, `yeo7`, all four cohorts at once | ~50 MB |
| activation, `harvardoxford`, one cohort, metadata columns only | a few MB |
| activation, `harvardoxford`, one cohort, all 111 parcels | 0.05–0.3 GB |
| DFC, QC columns only, any atlas, one window size, every cohort | a few MB |
| DFC + edges, `yeo7` (21 edges), one cohort | tens of MB |
| **DFC + edges, `harvardoxford` (6,105 edges), one cohort, 30 s** | **1–2.5 GB** |

The last row is the only one that is large, and it grows as `1/stride`: at
`n_overlaps: 5` a 30 s window steps every 6 s, so ds002837's ~5,500 s films
give ~900 windows per subject — 86 subjects × 900 × 6,105 float32 ≈ 2.3 GB.
Pandas holds a copy during conversion, so budget roughly double.

So: **16 GB is enough** for everything except pooled fine-atlas edges, and
**32 GB** covers one such read with headroom. Going above that is not the fix
for anything in these notebooks — if a read does not fit in 32 GB it is a
full-cohort pass, and that belongs in a batch job writing a summary table.

Cores matter less: these reads are IO- and decompression-bound and nothing here
is parallel. 2–4 is fine; 4 is a reasonable default because parquet decompresses
on multiple threads. Three hours is generous for exploration — the notebooks
themselves run in well under a minute once the filesystem is warm.

## The three things that make a read explode

1. **Reading a whole stage.** `pd.read_parquet("outputs/dfc")` walks and
   materialises everything. Always filter on partition keys (`cohort`,
   `atlas`, `window_s`, `task`, `sub`) — those prune whole directories before a
   file is opened — and always project columns.
2. **A dataset spanning atlases.** `harvardoxford`, `networks` and `yeo7` have
   111, 14 and 7 parcels. One dataset object per atlas, always.
3. **Materialising packed edges.** Above ~20,000 edges the edges are a single
   `fixed_size_list<float32>` column, and selecting it takes all of them. Read
   those per shard with `nb.read_subject_edges`, reduce, discard.

`nb.estimate_gb(dataset, columns=..., filter=...)` prices a read from parquet
footers before you run it, and the loaders carry a `guard_gb` that refuses one
over the limit rather than letting the kernel discover it.

## A gotcha that costs an afternoon

A hive partition key is recovered from the path **relative to the dataset
root**. Open at `dfc/atlas=yeo7/` and the `atlas` column comes back **null**,
because `atlas=yeo7` is the root and not below it — and a filter on it then
matches zero rows without raising. Since one-dataset-per-atlas means the root
always *is* an `atlas=` directory, this always applies.

`nb.dataset()` returns the keys the root swallowed and the loaders backfill
them; `01_activation.ipynb` shows it happening. Filter on `atlas` only by
choosing which dataset you open.

## Participants, QC, and the phenotype that does not exist yet

| file | owner | carries |
|---|---|---|
| `config/*_participants.csv` | human | curation — `excluded`, `exclusion_reason` |
| `outputs/meta/cohorts/cohort=*/participants_qc.csv` | pipeline | measurement — motion, coverage, scrubbing |
| `config/phenotype/*_phenotype.csv` | human | **age, sex, clinical scores** |
| `outputs/meta/cohorts/cohort=camcan/participants_scores.csv` | script | Cam-CAN's behavioural battery (`camcan` only) |

The third is not written by anything and does not exist until it is built:
`tools/make_phenotype.py`, sources documented in `config/phenotype/README.md`.
`nb.subject_table(cohort)` joins all of them and marks every row with
`has_pheno` and `has_scores`, so something missing reads as missing rather than
as a column of NaNs.

The scores table is built by
`preprocessing/camcan/03_build_participants_scores.py` from the archive's
`cc700-scored/` summaries. It exists for `camcan` only: none of the 55
`camcan_ccfrail` subjects appear in that battery.

No threshold is applied anywhere in these notebooks. `mean_fd` is a
measurement; `mean_fd > 0.5 → exclude` is a claim, and it belongs with the
model that rests on it.
