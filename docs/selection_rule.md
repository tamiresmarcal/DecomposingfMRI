# Model selection rule — written before any phenotype was touched

**Status: DRAFT until the thresholds marked `[FIX]` are filled in and this file
is re-committed. It is binding from the commit that fills them.**

This file fixes how the state model is chosen, so that the choice cannot be
made — or be seen to have been made — against the outcome. Its authority comes
from being committed and pushed *before* any analysis touching camcan
phenotype exists in this repository. `git log --follow docs/selection_rule.md`
is the record; the push to GitHub is the server-side timestamp.

A git commit is not a third-party registration. If a stronger claim is needed
later, register the same text on OSF and cite both.

## What is being selected: a STATE SET

A **state set** is one complete parcellation of time into states, produced by
one configuration:

    state set = (method, feature, atlas, aperture, K)

e.g. `hmm / activation / harvardoxford / TR-level / K=6`, or
`kmeans / dfc / yeo7 / 30 s / K=12`. Each state set has its own K state maps,
its own transition matrix, its own dwell times, its own everything. Two state
sets are not two views of one thing; they are different objects, and quantities
computed inside one are not comparable to the other unless this file says so.

The grid of candidate state sets is method x feature x atlas x aperture x K.

Two granularities and two apertures are reported, Yeo-style. So "selection"
means: which small-K and which large-K **state set** is promoted to the main
text, with the rest in supplement.

The question this file exists to answer, stated the way the paper asks it:
**which state set has the most stable and most interpretable transitions?**
Criteria 1 and 2 are "stable"; criteria 4 and 5 are "interpretable"; criterion
3 is "usable by someone else".

## Criteria, in this order

Each is evaluated on ds002837 + cneuromod only — the cohorts the model is fit
on. camcan is held out and is used at this stage **only** for criterion 3,
which reads state occupancy and reconstruction error and nothing else.

### 1. Estimability of transitions (primary, and it is close to decisive)

Transitions are the object of the paper; states are infrastructure. K is
therefore chosen by whether a per-subject transition matrix can be estimated,
not by any clustering quality index.

The binding arithmetic, from camcan's 476.7 s clip:

| dwell | state visits / subject | transitions / subject |
|---|---|---|
| 7 s | ~68 | ~67 |
| 14 s | ~34 | ~33 |
| 16 s | ~30 | ~29 |

At ~33 transitions per subject:

| K | cells | transitions per cell per subject | pooled over 648 subjects |
|---|---|---|---|
| 4 | 16 | 2.06 | 1,336 |
| 6 | 36 | 0.92 | 594 |
| 8 | 64 | 0.52 | 334 |
| 12 | 144 | 0.23 | 148 |
| 16 | 256 | 0.13 | 84 |

So a per-subject matrix is *never* estimable without pooling, at any K worth
having. Partial pooling is not a patch, it is the estimator. What K controls
is how much of each subject's matrix is their own data rather than the prior.

Thresholds:

- `[FIX]` median non-overlapping transitions per subject in camcan >= **T1**
- `[FIX]` median posterior SD per off-diagonal cell, under the shrinkage
  estimator, <= **T2** (on the probability scale)
- `[FIX]` shrinkage weight: median fraction of each subject's posterior mass
  attributable to their own counts rather than the prior >= **T3**

A configuration failing any of these is out, whatever it scores below.

### 2a. Split-half stability of the state maps

Yeo-style. Fit independently on two random halves of the training subjects,
100 splits, stratified by cohort.

- `[FIX]` median correlation between matched state maps >= **T4**
- `[FIX]` median adjusted Rand index of hard assignments on held-out
  timepoints >= **T5**

Matching across halves by the Hungarian algorithm on map correlation. Report
the full distribution, not just the median.

### 2b. Split-half stability of the TRANSITIONS

Stable maps do not imply stable transitions, and transitions are the object of
the paper. This criterion was missing from the first version of this file and
is the reason for the amendment; see the note at the end.

On the same 100 splits, after Hungarian matching of states across halves:

- `[FIX]` median correlation between the two halves' group transition
  matrices, off-diagonal cells only, >= **T4b**. Off-diagonal because the
  diagonal is dominated by dwell time and will correlate near 1 for any state
  set with reasonable persistence, which would hide an unstable off-diagonal
  structure.
- `[FIX]` median absolute difference in entropy rate between halves <= **T4c**
  (within a state set, so K is constant and the comparison is legitimate)
- `[FIX]` rank correlation of the stationary distributions >= **T4d**

A state set that passes 2a and fails 2b is out. That combination is the
specific failure this paper would otherwise report as a finding.

### 3. Projectability onto held-out camcan

Transform only. The model is never refit on camcan — enforced in code by the
artifact boundary, not by discipline.

- `[FIX]` fraction of camcan subjects in whom all K states are occupied at
  least once >= **T6**
- `[FIX]` median per-timepoint reconstruction error on camcan, relative to the
  same quantity within training, <= **T7**
- sample-size curve: subsample camcan at n = 25, 50, 100, 200, 400, 648 and
  report where the projected group transition matrix stabilises. This is
  reported, not thresholded.

### 4a. Decoding replication

State maps decoded against Neurosynth/NeuroQuery with spin-test nulls for
spatial autocorrelation.

- `[FIX]` the state -> topic mapping must replicate across cohorts and across
  films: rank correlation of topic loadings between ds002837 and cneuromod,
  and between odd and even films within ds002837, >= **T8**

Post-hoc annotation of whatever came out is not a criterion. A state set whose
anatomical story does not survive being computed twice does not get one.

### 4b. Correspondence with a known parcellation

Each state map is scored against the Yeo-7 networks (Dice on thresholded maps,
and the full loading profile). This is reported for every candidate state set
and is a criterion only in the weak sense:

- `[FIX]` at least **T9** of the K states must have an interpretable
  correspondence -- a dominant Yeo network, or a documented multi-network
  combination -- rather than being spatially diffuse

A state set of K uninterpretable states can have perfectly stable transitions
and still fail the paper's first promise, which is that someone else can use
these states and know what they mean.

### 5. Tie-break, in order

1. Smaller K.
2. Median dwell time closest to the 7-15 s reported by van der Meer et al.
   (2020) and the surrounding literature.
3. Simpler method (k-means before HMM).

## What is explicitly NOT a criterion

Nothing computed against camcan phenotype. Not HADS anxiety, not HADS
depression, not the combined distress score, not age, not sex, not any of the
~290 behavioural measures, not resting FC. No configuration may be compared,
ranked, dropped or promoted on any of them.

Figure 4 — the phenotype panel — is run **once**, on the configurations
criteria 1-5 selected, after this file is final.

## Quantities that must never be compared across K

Entropy rate, mixing time, and the stationary distribution all scale with K.
They are within-configuration descriptors. Any figure showing them carries K
in its title and does not put two K values on one axis.

## Order of operations

1. Fill every `[FIX]`, commit, push. **This is the timestamp.**
2. Fit the grid. Evaluate criteria 1-5. Commit the selection table.
3. Only then write any code that reads a camcan phenotype column.

Step 3 is checkable after the fact: the first commit touching
`config/phenotype/` or a HADS column from analysis code must be later than the
commit that closed step 2.

## If the rule has to change

Amend this file in a new commit that says what changed and why, before running
anything under the new version. Do not edit history. An amendment after seeing
criterion results is legitimate and must be visible as such; an amendment after
seeing phenotype results is not, and the commit order makes that auditable.

### Amendments

**2026-09-10, before any fit.** Added 2b (split-half stability of the
transition matrix) and 4b (correspondence with Yeo-7). The first version tested
stability of the state MAPS and called that stability of the state set -- but
the paper's object is transitions, and a state set can have reproducible maps
with unreproducible off-diagonal transition structure. Also renamed the unit of
selection from "configuration" to **state set** throughout, which is the
paper's own noun. No results of any kind existed when this was written.
