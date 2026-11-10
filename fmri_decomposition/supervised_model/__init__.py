"""Stage 5b-5d: everything that reads a phenotype.

What belongs here: anything whose answer depends on a symptom score. Its
output lives under `outputs/modelling/version=*/supervised_model/`, and a
version records which cohort trains and which is transferred to -- v0 trains
on Cam-CAN and tests on HCP 7T, and swapping those two is a new version even
though the state space behind it is unchanged.

    bstm_selection   rank state sets by how well their transitions predict
    fcm_selection    the same ranking for static connectivity -- the control
    transfer_tables  fit on one cohort, score on another
    transfer_config  which cohorts, targets, covariates and task maps
    hads             the Study/Settings layer the reports read
    symptoms         the two instruments' scales, side by side

A module here may import from the package root, from extractions and from
unsupervised_model. Nothing imports back.
"""
