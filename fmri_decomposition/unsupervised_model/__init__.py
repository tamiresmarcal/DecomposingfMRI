"""Stage 3-5a: the state space, fitted with no phenotype in the room.

What belongs here: anything that learns from the imaging data alone. The fit
depends on WHICH COHORTS are given to it and in what role, which is why its
output lives under `outputs/modelling/version=*/unsupervised_model/` -- v0 is
the cneuromod season 1 + ds002837 fit, and a version that adds cohorts to the
fit is a different state space, not a different view of the same one.

    decompose        PCA/UMAP embedding of the dfc (or activation) features
    cluster          brain states: the clusterers and the labels they write
    transitions      per-subject dynamics from a labelled state sequence
    state_means      what a state IS, in the atlas's own units
    cohort_tables    comparing cohorts in the fitted space (notebook 11)
    contextualization  correlating state maps against reference maps and
                     other atlases -- the interpretability arm (notebook 08)

A module here may import from the package root and from extractions, never
from supervised_model. No phenotype column is read anywhere in this package;
if one is, the module belongs next door.
"""
