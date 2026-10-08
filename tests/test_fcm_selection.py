"""Stage 5b' -- the static-connectivity tree.

Three trees are read side by side: `bstm_selection` (movie states),
`resting_bstm_selection` (rest states, the same script under another
`--output-name`) and this one. They are separate RUNS, so the thing that can
silently invalidate a comparison between them is the sample, not the code --
which is why `n` is in every summary and why `--restrict-subjects` is shared
rather than reimplemented here.
"""
import argparse

import numpy as np
import pandas as pd
import pytest

from fmri_decomposition import fcm_selection as F
from fmri_decomposition import static_fc as S
from tests.test_static_fc import write_activation


def cell(root, cohort, atlas, subs, task="m", seed=0, nan_edges=0):
    """A stored static-FC table, written the way `static-fc` writes one."""
    rng = np.random.default_rng(seed)
    for i, sub in enumerate(subs):
        write_activation(root, cohort, task, sub, seed=seed + i)
    d, _ = S.fc_for_cohort(root, "mini", cohort, band=None, min_tr=10,
                           log=lambda *a: None)
    cols = S.fc_columns(d)
    for c in cols[:nan_edges]:
        d[c] = np.nan
    S.write_cohort(root, atlas, cohort, d,
                   {"atlas": atlas, "cohort": cohort, "pool": "task"})
    return d


def args(root, **kw):
    d = dict(target="hads", cohorts=["cA"], arms=["edges", "global"],
             models=["ridge"], seeds=[0], fc_transform="z",
             fc_max_missing=0.05, tasks=None, restrict_subjects=None,
             covariates=["Age", "Sex"] + F.QUALITY_COVARIATES,
             categorical=["Sex"], atlas=None,
             output_name="fcm_selection", n_jobs=1, show=20,
             output_root=str(root), pheno=None, id_col="CCID",
             ordinal_levels=["Normal", "Mild", "Moderate", "Severe"])
    d.update(kw)
    return argparse.Namespace(**d)


@pytest.fixture
def one_cell(tmp_path):
    subs = [f"S{i:02d}" for i in range(14)]
    cell(tmp_path, "cA", "mini", subs)
    pheno = pd.DataFrame({"sub": subs, "y": np.arange(14) % 4,
                          "Age": np.arange(30, 44), "Sex": ["M", "F"] * 7})
    return tmp_path, pheno, subs


class TestDiscover:
    def test_finds_the_atlases_a_cohort_has(self, one_cell):
        root, _, _ = one_cell
        assert F.discover(root, ["cA"], None) == [("cA", "mini")]

    def test_an_atlas_filter_narrows_it(self, one_cell):
        root, _, _ = one_cell
        with pytest.raises(SystemExit):
            F.discover(root, ["cA"], ["notanatlas"])

    def test_a_cohort_with_no_table_names_the_command_that_makes_one(
            self, one_cell):
        root, _, _ = one_cell
        with pytest.raises(SystemExit) as e:
            F.discover(root, ["cB"], None)
        assert "fmri-decomp static-fc" in str(e.value)
        assert "--cohorts cB" in str(e.value)


class TestTheTwoArms:
    def frame(self, r):
        names = S.edge_names(["x", "y", "z"])
        d = pd.DataFrame(np.asarray(r, float), columns=names)
        d["sub"] = [f"S{i}" for i in range(len(d))]
        return d

    def test_edges_is_every_pairwise_correlation(self):
        d = self.frame([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]])
        b, info = F.blocks(d, "z", 0.05)
        assert b["edges"].shape == (2, 3)
        assert info["n_edges_used"] == 3

    def test_global_is_two_numbers_with_no_topology(self):
        """The control. If it matches `edges`, the pattern is not what is
        being measured -- the amount is."""
        d = self.frame([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]])
        b, _ = F.blocks(d, "r", 0.05)
        assert b["global"].shape == (2, 2)
        assert b["global"][0, 0] == pytest.approx(0.2)         # mean
        assert b["global"][0, 1] == pytest.approx(np.std([0.1, 0.2, 0.3]))

    def test_the_control_is_computed_from_the_same_matrix_as_the_hypothesis(
            self):
        """After the same transform and the same imputation, so the two arms
        differ only in what is kept."""
        d = self.frame([[np.nan, 0.2, 0.3], [0.2, 0.1, 0.4]])
        b, _ = F.blocks(d, "z", 0.9)
        assert np.allclose(b["global"][:, 0], b["edges"].mean(axis=1))
        assert np.isfinite(b["global"]).all()

    def test_fisher_z_by_default(self):
        d = self.frame([[0.5, 0.2, 0.3], [0.2, 0.1, 0.4]])
        b, _ = F.blocks(d, "z", 0.05)
        assert b["edges"][0, 0] == pytest.approx(np.arctanh(0.5))

    def test_a_perfect_correlation_does_not_become_infinite(self):
        d = self.frame([[1.0, -1.0, 0.3], [0.2, 0.1, 0.4]])
        b, _ = F.blocks(d, "z", 0.05)
        assert np.isfinite(b["edges"]).all()

    def test_an_edge_missing_too_often_is_dropped_and_counted(self):
        d = self.frame([[np.nan, 0.2, 0.3], [np.nan, 0.1, 0.4]])
        b, info = F.blocks(d, "z", 0.05)
        assert info["n_edges_dropped"] == 1
        assert b["edges"].shape[1] == 2

    def test_losing_every_edge_points_at_parcel_coverage(self):
        d = self.frame([[np.nan] * 3, [np.nan] * 3])
        with pytest.raises(SystemExit) as e:
            F.blocks(d, "z", 0.05)
        assert "brain mask" in str(e.value)


class TestBuild:
    def test_one_baseline_per_cohort_not_one_per_atlas(self, one_cell):
        """A second baseline at another atlas would be the identical fit under
        another name: same subjects, same covariates."""
        root, pheno, _ = one_cell
        cell(root, "cA", "second", [f"S{i:02d}" for i in range(14)])
        data, meta = F.build(root, [("cA", "mini"), ("cA", "second")], pheno,
                             args(root))
        base = meta[meta["arm"] == "(covariates only)"]
        assert len(base) == 1

    def test_the_covariates_are_prepended_to_every_arm(self, one_cell):
        root, pheno, _ = one_cell
        data, meta = F.build(root, [("cA", "mini")], pheno, args(root))
        C, _ = data[("cA", "(covariates only)", "-")]
        for key, (X, _) in data.items():
            if key[1] == "(covariates only)":
                continue
            assert np.allclose(X[:, :C.shape[1]], C), key

    def test_quality_covariates_come_from_the_fc_table(self, one_cell):
        root, pheno, _ = one_cell
        _, meta = F.build(root, [("cA", "mini")], pheno, args(root))
        # Age + Sex dummy + frac_good_frames + n_tr_used
        assert meta["n_covariates"].max() == 4

    def test_dropping_them_leaves_only_the_phenotype_covariates(self, one_cell):
        root, pheno, _ = one_cell
        _, meta = F.build(root, [("cA", "mini")], pheno,
                          args(root, covariates=["Age", "Sex"]))
        assert meta["n_covariates"].max() == 2

    def test_a_covariate_in_neither_table_says_where_it_looked(self):
        pheno = pd.DataFrame({"sub": ["A"], "y": [0], "Age": [30]})
        q = pd.DataFrame({"sub": ["A"], "n_tr_used": [1]})
        with pytest.raises(SystemExit) as e:
            F.covariate_block(pheno, q, ["A"], ["mean_fd"], [])
        assert "mean_fd" in str(e.value) and "static FC" in str(e.value)

    def test_y_is_aligned_with_the_features_row_for_row(self, one_cell):
        root, pheno, subs = one_cell
        data, _ = F.build(root, [("cA", "mini")], pheno, args(root))
        want = pheno.set_index("sub").loc[sorted(subs), "y"].to_numpy(float)
        for key, (_, y) in data.items():
            assert np.allclose(y, want), key

    def test_a_phenotype_that_matches_nothing_shows_both_id_styles(
            self, one_cell):
        root, pheno, _ = one_cell
        other = pheno.assign(sub="sub-" + pheno["sub"])
        with pytest.raises(SystemExit) as e:
            F.build(root, [("cA", "mini")], other, args(root))
        assert "matched nothing" in str(e.value)


class TestOneSample:
    def test_restrict_subjects_narrows_every_cell(self, one_cell):
        root, pheno, subs = one_cell
        lst = root / "keep.txt"
        lst.write_text("\n".join(subs[:9]) + "\n")
        _, meta = F.build(root, [("cA", "mini")], pheno,
                          args(root, restrict_subjects=str(lst)))
        assert set(meta["n"]) == {9}

    def test_it_reads_the_same_list_format_select_reads(self, one_cell):
        """Shared with `select`, not reimplemented: the whole purpose is that
        the trees agree on who was scored, and two readers drift on whitespace
        or case."""
        from fmri_decomposition import bstm_selection

        assert F.read_subject_list is bstm_selection.read_subject_list

    def test_a_csv_with_a_sub_column_works_too(self, one_cell, tmp_path):
        from fmri_decomposition.bstm_selection import read_subject_list

        p = tmp_path / "keep.csv"
        p.write_text("sub,note\ns01,x\ns02,y\n")
        assert read_subject_list(str(p)) == {"S01", "S02"}

    def test_a_missing_list_is_refused_by_name(self, tmp_path):
        from fmri_decomposition.bstm_selection import read_subject_list

        with pytest.raises(SystemExit) as e:
            read_subject_list(str(tmp_path / "nope.txt"))
        assert "no such file" in str(e.value)

    def test_an_empty_list_is_refused_rather_than_emptying_the_run(self,
                                                                  tmp_path):
        from fmri_decomposition.bstm_selection import read_subject_list

        p = tmp_path / "keep.txt"
        p.write_text("\n\n")
        with pytest.raises(SystemExit) as e:
            read_subject_list(str(p))
        assert "no ids" in str(e.value)


class TestSeveralTasks:
    def test_several_rows_for_one_subject_is_refused_not_averaged(self):
        d = pd.DataFrame({"sub": ["A", "A"], "task": ["m1", "m2"]})
        with pytest.raises(SystemExit) as e:
            F.one_row_per_subject(d, "static_fc")
        assert "--tasks" in str(e.value)
        assert "--pool subject" in str(e.value)

    def test_a_task_filter_narrows_before_the_check(self):
        d = pd.DataFrame({"sub": ["A", "A", "B"], "task": ["m1", "m2", "m1"]})
        assert len(F.one_row_per_subject(d, "t", tasks=["m1"])) == 2

    def test_a_filter_matching_nothing_is_refused_where_it_happens(self):
        d = pd.DataFrame({"sub": ["A"], "task": ["m1"]})
        with pytest.raises(SystemExit) as e:
            F.one_row_per_subject(d, "t", tasks=["nope"])
        assert "nope" in str(e.value) and "m1" in str(e.value)


class TestTheTable:
    def run_it(self, one_cell, tmp_path, **kw):
        root, pheno, subs = one_cell
        cell(root, "cB", "mini", subs, task="rest", seed=99)
        ph = tmp_path / "pheno.csv"
        pd.DataFrame({"CCID": pheno["sub"], "Age": pheno["Age"],
                      "Sex": pheno["Sex"],
                      "hads": [["Normal", "Mild", "Moderate", "Severe"][int(v)]
                               for v in pheno["y"]]}).to_csv(ph, index=False)
        a = args(root, cohorts=["cA", "cB"], pheno=[f"{ph}:,"], **kw)
        assert F.run(a) == 0
        out = root / "fcm_selection" / "target=hads"
        return pd.read_csv(out / "summary.csv"), out

    def test_movie_and_rest_sit_in_one_table_as_a_cohort_column(
            self, one_cell, tmp_path):
        d, _ = self.run_it(one_cell, tmp_path)
        assert set(d["cohort"]) == {"cA", "cB"}

    def test_it_carries_the_same_columns_bstm_selection_does(self, one_cell,
                                                             tmp_path):
        """The three trees are read side by side; a column missing from one of
        them is a comparison that has to be done by hand."""
        d, _ = self.run_it(one_cell, tmp_path)
        for c in ("model", "arm", "atlas", "n", "n_features", "mean", "std",
                  "min", "max", "count"):
            assert c in d.columns, c

    def test_n_is_present_so_two_tables_can_be_checked_for_one_sample(
            self, one_cell, tmp_path):
        d, _ = self.run_it(one_cell, tmp_path)
        assert d["n"].notna().all()

    def test_sorted_best_first_within_a_model(self, one_cell, tmp_path):
        d, _ = self.run_it(one_cell, tmp_path)
        for _, g in d.groupby("model"):
            assert g["mean"].is_monotonic_decreasing

    def test_no_delta_or_contrast_column(self, one_cell, tmp_path):
        """Deliberate: the arms share folds, so a difference between two rows
        is dependent and has no standard error the usual tests supply."""
        d, out = self.run_it(one_cell, tmp_path)
        assert not [c for c in d.columns if "delta" in c or "contrast" in c]
        assert not (out / "contrasts.csv").exists()

    def test_it_writes_its_own_tree_not_bstm_selections(self, one_cell,
                                                        tmp_path):
        _, out = self.run_it(one_cell, tmp_path)
        root = out.parent.parent
        assert out.exists()
        assert not (root / "bstm_selection").exists()

    def test_design_and_manifest_travel_with_the_table(self, one_cell,
                                                       tmp_path):
        _, out = self.run_it(one_cell, tmp_path)
        note = (out / "DESIGN.md").read_text()
        assert "Read `n` before reading a gap" in note
        assert "NOT a test" in note
        assert (out / "scores.parquet").exists()
        assert (out / "figures" / "arms_by_cohort.png").exists()
