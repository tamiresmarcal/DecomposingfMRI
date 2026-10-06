"""Stage 5b's design bookkeeping.

The scores are only interpretable beside a correct statement of what varied and
what did not, and that statement is now derived rather than written down. These
tests pin the derivation, including the case that caused the rewrite: an axis
that became varied while the document still called it fixed.
"""
import argparse

import numpy as np
import pytest
import pandas as pd

from fmri_decomposition import bstm_selection as B


def args(**kw):
    d = dict(covariates=["Age", "Sex", "n_transitions"], features=["cells"],
             p_norm=["joint"], models=["ridge"], seeds=[0])
    d.update(kw)
    return argparse.Namespace(**d)


def sets(rows):
    d = pd.DataFrame(rows)
    d["window_s_num"] = d["window_s"].astype(float)
    d["aperture"] = np.where(d["window_s_num"] < 0, "frame (1 TR)",
                             d["window_s"].astype(str) + "s")
    d["source"] = np.where(d["window_s_num"] < 0, "per-TR activation",
                           "windowed DFC edges")
    parts = d["states"].str.rsplit("_", n=2, expand=True)
    d["method"], d["embedding"] = parts[0], parts[1]
    d["K"] = parts[2].astype(int)
    return d


class TestApertureOrder:
    def test_the_frame_aperture_comes_first(self):
        d = sets([{"atlas": "yeo7", "window_s": "30", "states": "HMM_pca3_8"},
                  {"atlas": "yeo7", "window_s": "-1", "states": "HMM_pca3_8"},
                  {"atlas": "yeo7", "window_s": "300", "states": "HMM_pca3_8"}])
        assert B._aperture_order(d) == ["frame (1 TR)", "30s", "300s"]

    def test_accepts_the_score_table_column_name_too(self):
        d = sets([{"atlas": "a", "window_s": "60", "states": "HMM_pca3_8"},
                  {"atlas": "a", "window_s": "-1", "states": "HMM_pca3_8"}])
        d = d.drop(columns=["window_s_num"])
        d["window_s"] = d["window_s"].astype(float)
        assert B._aperture_order(d)[0] == "frame (1 TR)"

    def test_orders_numerically_not_lexically(self):
        d = sets([{"atlas": "a", "window_s": w, "states": "HMM_pca3_8"}
                  for w in ("15", "120", "30", "300")])
        assert B._aperture_order(d) == ["15s", "30s", "120s", "300s"]


class TestLabelAxes:
    def table(self):
        return pd.DataFrame({
            "window_s": [30.0, -1.0, 30.0],
            "states": ["HMM_pca3_8", "MeanShift_umap3_7", "(covariates only)"],
            "kind": ["set", "set", "base"]})

    def test_splits_method_and_embedding_out_of_the_name(self):
        out = B._label_axes(self.table())
        assert out["method"].to_list()[:2] == ["HMM", "MeanShift"]
        assert out["embedding"].to_list()[:2] == ["pca3", "umap3"]

    def test_a_baseline_row_gets_a_label_not_a_guess(self):
        out = B._label_axes(self.table())
        assert out["method"].iloc[2] == "(covariates only)"
        assert out["embedding"].iloc[2] == "(covariates only)"

    def test_a_baseline_row_still_gets_its_real_aperture(self):
        # The aperture is what the figure plots the baseline against, so it must
        # survive even though the baseline has no state set.
        out = B._label_axes(self.table())
        assert out["aperture"].iloc[2] == "30s"

    def test_the_frame_aperture_is_named_not_printed_as_minus_one(self):
        out = B._label_axes(self.table())
        assert out["aperture"].iloc[1] == "frame (1 TR)"
        assert out["source"].iloc[1] == "per-TR activation"
        assert out["source"].iloc[0] == "windowed DFC edges"


class TestHeldFixed:
    def test_an_axis_with_one_value_is_listed_as_fixed(self):
        d = sets([{"atlas": "yeo7", "window_s": "30", "states": "HMM_pca3_8"}])
        rows = "\n".join(B._held_fixed_rows(d, args()))
        assert "**clustering method**" in rows and "HMM" in rows

    def test_an_axis_with_several_values_is_NOT_listed_as_fixed(self):
        # The bug this replaces: `method` stayed in the held-fixed table while
        # stage 4b was adding methods, so the document asserted something false.
        d = sets([{"atlas": "yeo7", "window_s": "30", "states": "HMM_pca3_8"},
                  {"atlas": "yeo7", "window_s": "30",
                   "states": "MeanShift_pca3_7"}])
        rows = "\n".join(B._held_fixed_rows(d, args()))
        assert "**clustering method**" not in rows

    def test_the_input_feature_stops_being_fixed_once_both_apertures_run(self):
        one = sets([{"atlas": "a", "window_s": "30", "states": "HMM_pca3_8"}])
        both = sets([{"atlas": "a", "window_s": "30", "states": "HMM_pca3_8"},
                     {"atlas": "a", "window_s": "-1", "states": "HMM_pca3_8"}])
        assert "**input feature**" in "\n".join(B._held_fixed_rows(one, args()))
        assert "**input feature**" not in "\n".join(
            B._held_fixed_rows(both, args()))

    def test_the_rows_that_are_always_fixed_are_always_there(self):
        d = sets([{"atlas": "a", "window_s": "30", "states": "HMM_pca3_8"}])
        rows = "\n".join(B._held_fixed_rows(d, args()))
        for always in ("PCA components", "censor policy", "cross-validation",
                       "covariates"):
            assert always in rows


class TestPhenoSpecParsing:
    """`--pheno PATH:SEP`. The separator is after the LAST colon, so a path that
    contains one still works -- and a spec with no colon at all must not read the
    path itself as the separator, which is what rpartition does unguarded."""

    def write(self, tmp_path, name, sep):
        p = tmp_path / name
        rows = ["CCID,Age,HADS", "CC01,70,Mild", "CC02,62,Severe"]
        p.write_text("\n".join(r.replace(",", sep) for r in rows) + "\n")
        return p

    def test_explicit_comma(self, tmp_path):
        p = self.write(tmp_path, "t.csv", ",")
        d = B.read_phenotype([f"{p}:,"], "CCID", "HADS", ["Age"], [])
        assert len(d) == 2 and d["y"].to_list() == [1, 3]

    def test_explicit_tab(self, tmp_path):
        p = self.write(tmp_path, "t.tsv", "\t")
        d = B.read_phenotype([f"{p}:\t"], "CCID", "HADS", ["Age"], [])
        assert len(d) == 2

    def test_no_separator_defaults_to_comma(self, tmp_path):
        # Was read as "separator = the whole path", which pandas treats as a
        # regex; it failed later as "id column not found".
        p = self.write(tmp_path, "t.csv", ",")
        d = B.read_phenotype([str(p)], "CCID", "HADS", ["Age"], [])
        assert len(d) == 2

    def test_a_colon_in_the_path_is_not_the_separator(self, tmp_path):
        d = tmp_path / "od:d"
        d.mkdir()
        p = self.write(d, "t.csv", ",")
        assert len(B.read_phenotype([f"{p}:,"], "CCID", "HADS", ["Age"], [])) == 2

    def test_a_multi_character_separator_is_refused_by_name(self, tmp_path):
        p = self.write(tmp_path, "t.csv", ",")
        with pytest.raises(SystemExit, match="not a single character"):
            B.read_phenotype([f"{p}:;;"], "CCID", "HADS", ["Age"], [])

    def test_a_numeric_target_bypasses_ORDINAL_LEVELS(self, tmp_path):
        # A new cohort whose score is 0-21 rather than Normal/Mild/... still
        # works: to_numeric runs first and the ordinal map is only a fallback.
        p = tmp_path / "t.csv"
        p.write_text("CCID,Age,total\nCC01,70,14\nCC02,62,3\n")
        d = B.read_phenotype([f"{p}:,"], "CCID", "total", ["Age"], [])
        assert sorted(d["y"].to_list()) == [3, 14]
