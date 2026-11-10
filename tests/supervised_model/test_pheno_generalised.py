"""hads_tables._pheno: a second cohort, a numeric target, and the sentinel.

Three things were specific to Cam-CAN and had to open up before a transfer
test could load two cohorts with one class.

1. The files and the id column were constants, so both Studies would have
   read Cam-CAN.
2. The target was coded ONLY by level name, so a raw questionnaire column --
   DSM_Anxi_Raw, or HADS before banding -- coded to NaN for every subject and
   the frame emptied two steps later as "the phenotype join matched nothing".
3. Nothing bounded the target, and Cam-CAN codes "no answer" as 9999 in
   additional_HADS_depression while HADS runs 0-21. Invisible while the target
   was the *_category column; an enormous outlier the moment raw is used.
"""
import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

from fmri_decomposition.supervised_model import hads as ht  # noqa: E402


def write(tmp_path, name, rows, sep=","):
    p = tmp_path / name
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=sep)
        w.writeheader()
        w.writerows(rows)
    return p


def pheno_of(cfg):
    """_pheno without building a Study (which would read transitions)."""
    return ht.Study.__dict__["_pheno"](type("S", (), {"cfg": cfg})())


class TestNumericTargets:
    def test_a_raw_numeric_column_codes_to_itself(self, tmp_path):
        p = write(tmp_path, "ph.csv", [
            {"ID": "A", "score": "0", "Age": "30", "Sex": "F"},
            {"ID": "B", "score": "7", "Age": "40", "Sex": "M"},
            {"ID": "C", "score": "21", "Age": "50", "Sex": "F"}])
        cfg = ht.Settings(pheno_files=[(p, ",")], pheno_id="ID",
                          targets={"anx": "score"}, covariates=["Age", "Sex"],
                          baseline_covariates=["Age", "Sex"])
        got = pheno_of(cfg).set_index("sub")["anx"]
        assert [got["A"], got["B"], got["C"]] == [0.0, 7.0, 21.0]

    def test_named_levels_still_code(self, tmp_path):
        p = write(tmp_path, "ph.csv", [
            {"ID": "A", "score": "Normal", "Age": "30", "Sex": "F"},
            {"ID": "B", "score": "Severe", "Age": "40", "Sex": "M"}])
        cfg = ht.Settings(pheno_files=[(p, ",")], pheno_id="ID",
                          targets={"anx": "score"}, covariates=["Age", "Sex"],
                          baseline_covariates=["Age", "Sex"])
        got = pheno_of(cfg).set_index("sub")["anx"]
        assert [got["A"], got["B"]] == [0.0, 3.0]

    def test_the_level_match_ignores_case(self, tmp_path):
        p = write(tmp_path, "ph.csv", [{"ID": "A", "score": "severe",
                                        "Age": "30", "Sex": "F"}])
        cfg = ht.Settings(pheno_files=[(p, ",")], pheno_id="ID",
                          targets={"anx": "score"}, covariates=["Age", "Sex"],
                          baseline_covariates=["Age", "Sex"])
        assert pheno_of(cfg).set_index("sub")["anx"]["A"] == 3.0


class TestSentinel:
    def rows(self):
        return [{"ID": "A", "score": "3", "Age": "30", "Sex": "F"},
                {"ID": "B", "score": "9999", "Age": "40", "Sex": "M"},
                {"ID": "C", "score": "12", "Age": "50", "Sex": "F"}]

    def test_it_becomes_nan_when_a_range_is_given(self, tmp_path):
        """NaN, not a dropped row. _pheno marks a value unusable and `table()`
        drops it at the join, which is exactly what it already does for a
        level name outside the list. Dropping here instead would make the
        phenotype frame disagree with itself between targets, since a subject
        can be out of range on one scale and fine on the other."""
        p = write(tmp_path, "ph.csv", self.rows())
        cfg = ht.Settings(pheno_files=[(p, ",")], pheno_id="ID",
                          targets={"anx": "score"}, covariates=["Age", "Sex"],
                          baseline_covariates=["Age", "Sex"],
                          target_range={"anx": (0, 21)})
        got = pheno_of(cfg).set_index("sub")["anx"]
        assert got["A"] == 3.0 and got["C"] == 12.0
        assert pd.isna(got["B"])
        assert 9999.0 not in set(got.dropna())

    def test_without_a_range_it_survives(self, tmp_path):
        """The default has to be what it always was: every existing result was
        produced without this setting."""
        p = write(tmp_path, "ph.csv", self.rows())
        cfg = ht.Settings(pheno_files=[(p, ",")], pheno_id="ID",
                          targets={"anx": "score"}, covariates=["Age", "Sex"],
                          baseline_covariates=["Age", "Sex"])
        assert 9999.0 in set(pheno_of(cfg)["anx"])

    def test_a_range_on_one_target_leaves_the_other_alone(self, tmp_path):
        p = write(tmp_path, "ph.csv", [
            {"ID": "A", "a": "3", "d": "9999", "Age": "30", "Sex": "F"},
            {"ID": "B", "a": "9999", "d": "4", "Age": "40", "Sex": "M"}])
        cfg = ht.Settings(pheno_files=[(p, ",")], pheno_id="ID",
                          targets={"anx": "a", "dep": "d"},
                          covariates=["Age", "Sex"],
                          baseline_covariates=["Age", "Sex"],
                          target_range={"dep": (0, 21)})
        got = pheno_of(cfg).set_index("sub")
        assert got.loc["A", "anx"] == 3.0
        assert 9999.0 in set(got["anx"])          # anx was not bounded
        assert got["dep"].max() <= 21             # dep was


class TestASecondCohort:
    def test_a_different_file_and_id_column(self, tmp_path):
        """What the transfer test needs: two Studies, two phenotypes, one
        class. Before this, both would have read Cam-CAN."""
        p = write(tmp_path, "hcp.csv", [
            {"Subject": "100610", "DSM_Anxi_Raw": "2", "Age": "26", "Sex": "M"},
            {"Subject": "102311", "DSM_Anxi_Raw": "5", "Age": "31", "Sex": "F"}])
        cfg = ht.Settings(pheno_files=[(p, ",")], pheno_id="Subject",
                          targets={"anxiety": "DSM_Anxi_Raw"},
                          covariates=["Age", "Sex"],
                          baseline_covariates=["Age", "Sex"])
        got = pheno_of(cfg)
        assert sorted(got["sub"]) == ["100610", "102311"]
        assert sorted(got["anxiety"]) == [2.0, 5.0]

    def test_two_files_are_merged_on_the_id(self, tmp_path):
        a = write(tmp_path, "a.csv", [{"ID": "X", "score": "4"}])
        b = write(tmp_path, "b.csv", [{"ID": "X", "Age": "40", "Sex": "F"}])
        cfg = ht.Settings(pheno_files=[(a, ","), (b, ",")], pheno_id="ID",
                          targets={"anx": "score"}, covariates=["Age", "Sex"],
                          baseline_covariates=["Age", "Sex"])
        got = pheno_of(cfg).set_index("sub")
        assert got.loc["X", "anx"] == 4.0 and got.loc["X", "Age"] == 40
