"""Stage 1-3b: images and confounds -> tables. No model is fitted here.

What belongs in this package: anything whose output does not depend on which
cohort plays which role in the analysis. That is the test, and it is the reason
`outputs/extractions/` sits OUTSIDE `outputs/modelling/version=*/` -- swapping
the training and test cohorts does not change a parcel timeseries, a windowed
correlation matrix, a censor mask or a static-FC matrix, so none of them should
be recomputed or re-stored when the modelling version changes.

    activation   parcel timeseries per (cohort, atlas, task, sub)
    dfc          windowed dynamic functional connectivity
    censor       which frames and which windows survive a motion policy
    static_fc    one correlation matrix per subject -- the window_s=-1 twin
    qc, motion   what fMRIPrep produced and how much of it is usable
    validate     pre-flight: does the config describe what is on disk
    inventory    reading the activation/dfc trees back, for notebook 03

A module here may import from the package root (io, config, cohort, frames,
windows, timing) but NOT from unsupervised_model or supervised_model. The
dependency runs one way: extraction knows nothing about what is fitted on it.
"""
