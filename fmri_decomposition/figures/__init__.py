"""The figures. One module per report, drawing from any of the three stages.

This is the only package that does not mirror a folder under `outputs/`, and
that is deliberate: a figure is a presentation layer over all three stages
rather than a stage of its own -- notebook 13's panels read the supervised
transfer AND the cohort tables -- and a figure lands wherever the notebook
that asked for it saves it, not in a tree of its own.

    hads       the HADS report's panels (notebooks 10, 13)
    cohorts    cohort comparison in the fitted space (notebook 11)
    states     state cards and correlation maps (notebook 08)
    symptoms   symptom distributions across instruments (notebook 12)

These used to live beside the .ipynb files, where they were importable only
through a sys.path insert, were not installed, and could not be tested. The
notebooks under notebooks/ now hold nothing but cells.
"""
