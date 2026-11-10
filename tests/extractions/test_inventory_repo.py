"""inventory.REPO was dropped when notebooks/nbtools.py moved into the package.

Every config lookup (`output_root`, `cohort_configs`, participants, phenotype)
then raised NameError -- notebook 03 died in its first cell. Pin it.
"""
from pathlib import Path

import pytest

from fmri_decomposition.extractions import inventory as inv


def test_repo_is_the_checkout_root():
    assert inv.REPO == Path(inv.__file__).resolve().parents[2]
    assert (inv.REPO / "config").is_dir()
    assert (inv.REPO / "fmri_decomposition").is_dir()


def test_output_root_without_env_reaches_the_configs(monkeypatch, tmp_path):
    monkeypatch.delenv("FMRIDECOMP_OUTPUTS", raising=False)
    try:
        inv.output_root()
    except FileNotFoundError:
        pass                     # no outputs on this machine: fine, not NameError
    assert isinstance(inv.cohort_configs(), object)
