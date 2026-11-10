# Analysis notebooks

**Only notebooks live here.** Every table and figure module moved into the
package, where it is installed, importable without a `sys.path` hack, and
testable:

| was | is now |
|---|---|
| `nbtools.py` | `fmri_decomposition/extractions/inventory.py` |
| `cohort_tables.py` | `fmri_decomposition/unsupervised_model/cohort_tables.py` |
| `state_correlations.py`, `refmaps.py` | `fmri_decomposition/unsupervised_model/contextualization/` |
| `hads_tables.py` | `fmri_decomposition/supervised_model/hads.py` |
| `transfer_tables.py`, `transfer_config.py` | `fmri_decomposition/supervised_model/` |
| `tools/symptom_distribution.py` | `fmri_decomposition/supervised_model/symptoms.py` |
| `*_plots.py`, `state_figures.py` | `fmri_decomposition/figures/` |

Nothing here writes to `outputs/`.

| file | what it opens |
|---|---|
| `03_qc.ipynb` | one histogram grid: six QC metrics x every cohort, three cells |
| `08_state_maps.ipynb` | brain states x reference maps: pick a state set, atlases, families and BrainMap depth; tables then figures, via `unsupervised_model/contextualization/` + `figures/states.py` |
| `09_model_contributions.ipynb` | what the saved bstm models use: out-of-fold SHAP shares across every saved model, one model in depth (beeswarm, predicted vs observed, stability over fold seeds), and where it looks on its state set's graph |
| `10_hads_report.ipynb` | movie-vs-rest and HADS multiverses, every model vs its Age + Sex baseline and what drives the score, the `all` arms' SHAP shares, then one chosen state set and one chosen model in it |
| `11_normal_brain_cohorts.ipynb` | the normal brain across cohorts: rest and movie per cohort and their mean, movie vs rest with the dynamics beside, each cohort's equivalence to the others (TOST), train vs projected cohorts, the transition matrices with a TOST per cell |
| `12_symptom_distributions.ipynb` | how much clinical signal there is to predict: HADS (Cam-CAN) vs ASR/DSM-oriented (HCP). The band PERCENTAGES are not comparable between the two instruments and the figures say so; the percentile table and the Q-Q are the comparison that is |
| `13_camcan_to_hcp.ipynb` | train on every Cam-CAN subject, test once on HCP: the equivalence precondition, the whole multiverse transferred with a bootstrap interval over the test subjects, and one model in depth. Needs no equating of HADS against the ASR -- Spearman compares order, not values |

Retired, and in the git history if they are wanted: `01_activation`,
`02_dfc` (their loaders are `extractions/inventory.py`, which `03` still
uses), `04_activation_pca`, `06_visualize`, `07_transitions_phenotype`
(superseded by `10`), `08_transitions_anxiety_depression` (superseded by `10`).

## Which output version a notebook reads

`outputs/` has three halves, and the fitted two sit under a **version**:

```
outputs/extractions/                                  activation, dfc, censor,
                                                      static_fc, meta
outputs/modelling/version=v0/unsupervised_model/      latents, transitions, meta
outputs/modelling/version=v0/supervised_model/        bstm_selection,
                                                      fcm_selection,
                                                      transferred_models, meta
```

A version is **which cohorts play which role**. `v0` is the current analysis:
the state space fitted on cneuromod + ds002837, the supervised models trained
on Cam-CAN and transferred to HCP 7T. Adding cohorts to the state fit, or
swapping which cohort trains and which is tested, is a new version -- not a
rewrite of this one.

Every `Settings` in the notebooks takes `modelling_version`. It defaults to
`None`, which resolves to `$FMRIDECOMP_MODELLING_VERSION` and then to `v0`:

```python
cfg = ht.Settings(modelling_version="v1")   # read v1 instead
```

What a version holds is recorded in
`outputs/modelling/version=<v>/version.json`, written by the first run under
it and **checked** on every later one: a run whose cohort composition differs
is refused by name rather than filling a tree named for something else.
`extractions/` is outside every version because a parcel timeseries does not
change when a role does.

## Running them

Python -- Jupyter included -- exists only inside the container. The login
node's bare `python` cannot import `fmri_decomposition`, and nothing heavy
should run there anyway.

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

The notebooks import the package, so the repo has to be importable: either
`pip install -e .` in the image, or run Jupyter from the repo root (`--pwd
$PWD` above does that).

`inventory.output_root()` finds `outputs/` from the `output_root:` in the
configs. Point it somewhere else with `FMRIDECOMP_OUTPUTS=/path/to/outputs`.

## Is `--mem=32G --cpus-per-task=4` right?

Yes, with room to spare -- and it is the *edges* that set it, not the subject
count. Every read below is one atlas, uncompressed, before pandas overhead:

| read | size |
|---|---|
| activation, `yeo7`, all four cohorts at once | ~50 MB |
| activation, `harvardoxford`, one cohort, metadata columns only | a few MB |
| activation, `harvardoxford`, one cohort, all 111 parcels | 0.05-0.3 GB |
| DFC, QC columns only, any atlas, one window size, every cohort | a few MB |
| DFC + edges, `yeo7` (21 edges), one cohort | tens of MB |
| **DFC + edges, `harvardoxford` (6,105 edges), one cohort, 30 s** | **1-2.5 GB** |

The last row is the only large one, and it grows as `1/stride`: at
`n_overlaps: 5` a 30 s window steps every 6 s, so ds002837's ~5,500 s films
give ~900 windows per subject -- 86 subjects x 900 x 6,105 float32 ~ 2.3 GB.
Pandas holds a copy during conversion, so budget roughly double.

**16 GB is enough** for everything except pooled fine-atlas edges, and **32
GB** covers one such read with headroom. If a read does not fit in 32 GB it is
a full-cohort pass, and that belongs in a batch job writing a summary table.

Cores matter less: these reads are IO- and decompression-bound. 2-4 is fine.

## The three things that make a read explode

1. **Reading a whole stage.** `pd.read_parquet("outputs/extractions/dfc")`
   walks and materialises everything. Always filter on partition keys
   (`cohort`, `atlas`, `window_s`, `task`, `sub`) -- those prune whole
   directories before a file is opened -- and always project columns.
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
because `atlas=yeo7` is the root and not below it -- and a filter on it then
matches zero rows without raising. Since one-dataset-per-atlas means the root
always *is* an `atlas=` directory, this always applies.

`nb.dataset()` returns the keys the root swallowed and the loaders backfill
them. Filter on `atlas` only by choosing which dataset you open.

## Participants, QC, and the phenotype

| file | owner | carries |
|---|---|---|
| `config/*_participants.csv` | human | curation -- `excluded`, `exclusion_reason` |
| `outputs/extractions/meta/cohorts/cohort=*/participants_qc.csv` | pipeline | measurement -- motion, coverage, scrubbing |
| `config/phenotype/*_phenotype.csv` | human | **age, sex, clinical scores** |
| `outputs/extractions/meta/cohorts/cohort=camcan/participants_scores.csv` | script | Cam-CAN's behavioural battery (`camcan` only) |

The third is not written by anything and does not exist until it is built:
`tools/make_phenotype.py`, sources documented in `config/phenotype/README.md`.
`nb.subject_table(cohort)` joins all of them and marks every row with
`has_pheno` and `has_scores`, so something missing reads as missing rather
than as a column of NaNs.

The scores table is built by
`preprocessing/camcan/03_build_participants_scores.py` from the archive's
`cc700-scored/` summaries. It exists for `camcan` only.

No threshold is applied anywhere in these notebooks. `mean_fd` is a
measurement; `mean_fd > 0.5 -> exclude` is a claim, and it belongs with the
model that rests on it.

A blank panel in `03` is never a plotting bug -- the cell above the grid names
any metric absent from `participants_qc.csv` or present but all-NaN
(`peak_isc` and `best_lag_tr` are NaN for a task with fewer than 3 subjects,
by design).
