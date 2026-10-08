"""Stage 5c -- the three properties that make this a benchmark.

One subject set, one covariate block per condition, one scoring function. Each
has a test here, because each is the kind of thing that silently stops being
true: an arm quietly scored on more subjects than its rival, a covariate that
exists for one arm only, a second copy of the CV loop that drifts.

And `contrasts` is tested for naming `bstm:cells` explicitly, since a row
reading "the best BSTM arm beat static FC" when the winner was occupancy
reports the opposite of what happened.
"""
import argparse

import numpy as np
import pandas as pd
import pytest

from fmri_decomposition import bstm_benchmark as B
from fmri_decomposition import static_fc as S


def summary(rows):
    """A `summary.csv` frame, with the columns `contrasts` reads filled in."""
    d = pd.DataFrame(rows)
    for c, default in (("std", 0.0), ("min", 0.0), ("max", 0.0),
                       ("count", 5), ("n", 90), ("n_features", 1), ("K", 8)):
        if c not in d.columns:
            d[c] = default
    return d


class TestParseConditions:
    def test_label_equals_cohort(self):
        assert B.parse_conditions(["movie=camcan", "rest=camcan_rest"]) == {
            "movie": "camcan", "rest": "camcan_rest"}

    def test_order_is_preserved_because_it_is_the_figure_order(self):
        assert list(B.parse_conditions(["b=x", "a=y"])) == ["b", "a"]

    @pytest.mark.parametrize("spec", ["camcan", "=camcan", "movie="])
    def test_a_bare_cohort_name_is_refused_with_the_shape(self, spec):
        with pytest.raises(SystemExit) as e:
            B.parse_conditions([spec])
        assert "LABEL=COHORT" in str(e.value)

    def test_a_repeated_label_is_refused_rather_than_overwritten(self):
        with pytest.raises(SystemExit) as e:
            B.parse_conditions(["movie=camcan", "movie=other"])
        assert "twice" in str(e.value)


class TestOneRowPerSubject:
    def test_a_unique_table_passes_through(self):
        d = pd.DataFrame({"sub": ["A", "B"], "task": ["m", "m"]})
        assert len(B.one_row_per_subject(d, "t", "c")) == 2

    def test_several_tasks_for_one_subject_is_refused_not_averaged(self):
        d = pd.DataFrame({"sub": ["A", "A"], "task": ["m1", "m2"]})
        with pytest.raises(SystemExit) as e:
            B.one_row_per_subject(d, "transitions x", "c")
        assert "m1" in str(e.value) and "m2" in str(e.value)

    def test_the_refusal_names_both_ways_out(self):
        d = pd.DataFrame({"sub": ["A", "A"], "task": ["m1", "m2"]})
        with pytest.raises(SystemExit) as e:
            B.one_row_per_subject(d, "t", "c")
        assert "--tasks" in str(e.value)
        assert "--pool subject" in str(e.value)


class TestFcBlock:
    def cell(self, tmp_path, cohort="c", atlas="mini", r=None, subs=("A", "B")):
        names = S.edge_names(["x", "y", "z"])
        r = np.asarray(r if r is not None else
                       [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]], float)
        d = pd.DataFrame(r, columns=names)
        d.insert(0, "sub", list(subs))
        d.insert(0, "task", "m")
        d["n_tr_used"] = 100
        d["frac_good_frames"] = 0.9
        p = tmp_path / "static_fc" / f"atlas={atlas}" / f"cohort={cohort}"
        p.mkdir(parents=True, exist_ok=True)
        d.to_parquet(p / "subjects.parquet", index=False)
        return names

    def test_fisher_z_is_applied_by_default(self, tmp_path):
        self.cell(tmp_path)
        X, cols, _ = B.fc_block(tmp_path, "mini", "c", ["A", "B"], "z", 0.05)
        assert X[0, 0] == pytest.approx(np.arctanh(0.1))

    def test_raw_r_is_available_unchanged(self, tmp_path):
        self.cell(tmp_path)
        X, _, _ = B.fc_block(tmp_path, "mini", "c", ["A", "B"], "r", 0.05)
        assert X[0, 0] == pytest.approx(0.1)

    def test_a_perfect_correlation_does_not_become_infinite(self, tmp_path):
        """arctanh(1) is inf, and one inf survives mean imputation to poison
        the whole column. r = +-1 is reachable on a short window."""
        self.cell(tmp_path, r=[[1.0, -1.0, 0.3], [0.2, 0.1, 0.4]])
        X, _, _ = B.fc_block(tmp_path, "mini", "c", ["A", "B"], "z", 0.05)
        assert np.isfinite(X).all()

    def test_an_edge_missing_too_often_is_dropped_and_counted(self, tmp_path):
        self.cell(tmp_path, r=[[np.nan, 0.2, 0.3], [np.nan, 0.1, 0.4]])
        X, cols, info = B.fc_block(tmp_path, "mini", "c", ["A", "B"], "z", 0.05)
        assert len(cols) == 2 and "x__y" not in cols
        assert info["n_edges_dropped"] == 1

    def test_an_edge_missing_rarely_is_imputed_and_counted(self, tmp_path):
        self.cell(tmp_path, r=[[np.nan, 0.2, 0.3], [0.2, 0.1, 0.4]])
        X, cols, info = B.fc_block(tmp_path, "mini", "c", ["A", "B"], "z", 0.9)
        assert len(cols) == 3
        assert info["n_values_imputed"] == 1
        assert X[0, 0] == pytest.approx(np.arctanh(0.2))

    def test_losing_every_edge_points_at_parcel_coverage(self, tmp_path):
        self.cell(tmp_path, r=[[np.nan] * 3, [np.nan] * 3])
        with pytest.raises(SystemExit) as e:
            B.fc_block(tmp_path, "mini", "c", ["A", "B"], "z", 0.05)
        assert "brain mask" in str(e.value)

    def test_rows_come_back_in_the_shared_subject_order(self, tmp_path):
        """The feature matrix and y are indexed positionally downstream, so a
        table sorted differently from the shared set would silently pair each
        subject's edges with another subject's HADS score."""
        self.cell(tmp_path, subs=("B", "A"))
        X, _, _ = B.fc_block(tmp_path, "mini", "c", ["A", "B"], "r", 0.05)
        assert X[0, 2] == pytest.approx(0.4)      # A is the second stored row


class TestCovariateBlock:
    def test_quality_covariates_come_from_the_fc_table(self):
        pheno = pd.DataFrame({"sub": ["A", "B"], "y": [0, 1],
                              "Age": [30, 40], "Sex": ["M", "F"]})
        q = pd.DataFrame({"sub": ["A", "B"], "frac_good_frames": [0.9, 0.5],
                          "n_tr_used": [100, 50]})
        X = B.covariate_block(pheno, q, ["A", "B"], ["Age", "frac_good_frames"],
                              [])
        assert X.tolist() == [[30.0, 0.9], [40.0, 0.5]]

    def test_a_categorical_covariate_is_dummied(self):
        pheno = pd.DataFrame({"sub": ["A", "B"], "y": [0, 1],
                              "Age": [30, 40], "Sex": ["M", "F"]})
        q = pd.DataFrame({"sub": ["A", "B"], "n_tr_used": [1, 2]})
        X = B.covariate_block(pheno, q, ["A", "B"], ["Age", "Sex"], ["Sex"])
        assert X.shape == (2, 2)
        assert set(np.unique(X[:, 1])) == {0.0, 1.0}

    def test_a_covariate_in_neither_table_says_where_it_looked(self):
        pheno = pd.DataFrame({"sub": ["A"], "y": [0], "Age": [30]})
        q = pd.DataFrame({"sub": ["A"], "n_tr_used": [1]})
        with pytest.raises(SystemExit) as e:
            B.covariate_block(pheno, q, ["A"], ["mean_fd"], [])
        assert "mean_fd" in str(e.value)
        assert "static FC" in str(e.value)


class TestContrasts:
    def rows(self, cells=0.30, occ=0.10, fc=0.20, base=0.05):
        return summary([
            {"model": "ridge", "condition": "movie", "arm": "bstm:cells",
             "atlas": "a", "window_s": "-1", "states": "HMM2_pca3_8",
             "mean": cells, "std": 0.01},
            {"model": "ridge", "condition": "movie", "arm": "bstm:occupancy",
             "atlas": "a", "window_s": "-1", "states": "HMM2_pca3_8",
             "mean": occ, "std": 0.02},
            {"model": "ridge", "condition": "movie", "arm": "fc:edges",
             "atlas": "a", "window_s": "static", "states": "fc_edges",
             "mean": fc, "std": 0.03},
            {"model": "ridge", "condition": "movie", "arm": "(covariates only)",
             "atlas": "-", "window_s": "-", "states": "-", "mean": base,
             "std": 0.01}])

    def test_cells_versus_fc_is_the_first_row(self):
        c = B.contrasts(self.rows(), ["movie"])
        assert c["contrast"].iloc[0] == "cells vs fc"

    def test_the_delta_is_a_minus_b(self):
        c = B.contrasts(self.rows(cells=0.30, fc=0.20), ["movie"])
        r = c[c["contrast"] == "cells vs fc"].iloc[0]
        assert r["delta"] == pytest.approx(0.10)

    def test_the_covariate_floor_is_the_comparison_for_both_arms(self):
        c = B.contrasts(self.rows(), ["movie"])
        got = c.set_index("contrast")["score_b"]
        assert got["cells vs covariates"] == pytest.approx(0.05)
        assert got["fc vs covariates"] == pytest.approx(0.05)

    def test_the_controls_are_compared_against_cells_by_name(self):
        c = B.contrasts(self.rows(), ["movie"])
        assert "cells vs occupancy" in set(c["contrast"])

    def test_no_best_bstm_row_when_cells_already_wins(self):
        """It would duplicate `cells vs fc` under a different name."""
        c = B.contrasts(self.rows(cells=0.30, occ=0.10), ["movie"])
        assert "best bstm vs fc" not in set(c["contrast"])

    def test_a_control_beating_cells_gets_its_own_row_and_is_named(self):
        """The case the explicit naming exists for: if occupancy wins, the
        signal is time spent and the transition claim does not hold."""
        c = B.contrasts(self.rows(cells=0.10, occ=0.40), ["movie"])
        r = c[c["contrast"] == "best bstm vs fc"].iloc[0]
        assert "occupancy" in r["a"]

    def test_conditions_are_compared_within_an_arm(self):
        rows = pd.concat([self.rows(), summary([
            {"model": "ridge", "condition": "rest", "arm": "bstm:cells",
             "atlas": "a", "window_s": "-1", "states": "HMM2_pca3_8",
             "mean": 0.08, "std": 0.04},
            {"model": "ridge", "condition": "rest", "arm": "(covariates only)",
             "atlas": "-", "window_s": "-", "states": "-", "mean": 0.05,
             "std": 0.01}])], ignore_index=True)
        c = B.contrasts(rows, ["movie", "rest"])
        r = c[c["contrast"] == "movie vs rest"]
        assert len(r) == 1
        assert r["delta"].iloc[0] == pytest.approx(0.22)

    def test_both_spreads_travel_with_the_delta(self):
        """A difference of 0.02 between two arms whose seed spread is 0.05 is
        not a result, and the table has to make that visible."""
        c = B.contrasts(self.rows(), ["movie"])
        r = c[c["contrast"] == "cells vs fc"].iloc[0]
        assert r["spread_a"] == pytest.approx(0.01)
        assert r["spread_b"] == pytest.approx(0.03)

    def test_an_fc_only_run_still_reports_against_the_floor(self):
        rows = summary([
            {"model": "ridge", "condition": "movie", "arm": "fc:edges",
             "atlas": "a", "window_s": "static", "states": "fc_edges",
             "mean": 0.2, "std": 0.01},
            {"model": "ridge", "condition": "movie", "arm": "(covariates only)",
             "atlas": "-", "window_s": "-", "states": "-", "mean": 0.05,
             "std": 0.01}])
        c = B.contrasts(rows, ["movie"])
        assert set(c["contrast"]) == {"fc vs covariates"}


class TestWipeRefusesTheWrongTree:
    def test_a_benchmark_path_is_not_cleared_as_a_selection_path(self, tmp_path):
        from fmri_decomposition.bstm_selection import _wipe

        out = tmp_path / "bstm_benchmark" / "target=t"
        out.mkdir(parents=True)
        with pytest.raises(SystemExit) as e:
            _wipe(out)                      # default parent is bstm_selection
        assert "bstm_benchmark" in str(e.value)
        assert out.exists()

    def test_it_clears_the_tree_it_was_told_to_own(self, tmp_path):
        from fmri_decomposition.bstm_selection import _wipe

        out = tmp_path / "bstm_benchmark" / "target=t"
        out.mkdir(parents=True)
        (out / "stale.png").write_text("x")
        _wipe(out, parent="bstm_benchmark")
        assert not out.exists()


# --------------------------------------------------------------------------
# Property 1: ONE SUBJECT SET. The failure it guards against is the one that
# would be easiest to publish by accident -- rest scored on the subjects who
# have a usable rest scan, movie scored on more of them, and the difference
# read as a condition effect.
# --------------------------------------------------------------------------
def build_condition(root, cohort, task, subs, n_tr=120, tr=2.0, K=3, seed=0):
    """A condition with both arms present: static FC and a transition table."""
    from tests.test_static_fc import write_activation

    rng = np.random.default_rng(seed)
    for i, sub in enumerate(subs):
        write_activation(root, cohort, task, sub, n_tr=n_tr, tr=tr, seed=seed + i)
    d, stats = S.fc_for_cohort(root, "mini", cohort, band=None, min_tr=10,
                               log=lambda *a: None)
    S.write_cohort(root, "mini", cohort, d,
                   {"atlas": "mini", "cohort": cohort, "pool": "task"})

    cells = [f"{i}->{j}" for i in range(K) for j in range(K)]
    rows = []
    for sub in subs:
        m = rng.random((K, K)) + 0.1
        m /= m.sum()
        r = {"task": task, "sub": sub, "cohort": cohort, "n_states": K,
             "n_transitions": 99, "switch_rate": float(rng.random()),
             "mean_dwell_s": 5.0, "entropy_rate_bits": 1.2, "dispersion": 0.3}
        r.update({c: float(v) for c, v in zip(cells, m.ravel())})
        r.update({f"occ_{k}": float(v) for k, v in enumerate(m.sum(1))})
        rows.append(r)
    p = (root / "transitions" / "atlas=mini" / "window_s=-1"
         / f"states=HMM2_pca3_{K}" / f"cohort={cohort}")
    p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(p / "subjects.parquet", index=False)


def bench_args(root, **kw):
    d = dict(target="hads", conditions=["movie=cA", "rest=cB"],
             arms=["cells", "fc"], models=["ridge"], seeds=[0],
             p_norm="cond", fc_transform="z", fc_max_missing=0.05,
             intersect_subjects=True, min_subjects=5, tasks=None,
             covariates=["Age", "Sex"] + B.QUALITY_COVARIATES,
             categorical=["Sex"], atlas=None, window_s=None,
             output_root=str(root))
    d.update(kw)
    return argparse.Namespace(**d)


@pytest.fixture
def two_conditions(tmp_path):
    """Movie has 12 subjects, rest only the first 8 of them."""
    subs = [f"S{i:02d}" for i in range(12)]
    build_condition(tmp_path, "cA", "m", subs, seed=0)
    build_condition(tmp_path, "cB", "rest", subs[:8], tr=1.5, seed=50)
    pheno = pd.DataFrame({"sub": subs, "y": np.arange(12) % 4,
                          "Age": np.arange(30, 42),
                          "Sex": ["M", "F"] * 6})
    return tmp_path, pheno, subs


class TestOneSubjectSet:
    def test_every_arm_is_scored_on_the_intersection(self, two_conditions):
        root, pheno, _ = two_conditions
        _, meta, _ = B.assemble(root, {"movie": "cA", "rest": "cB"}, pheno,
                                bench_args(root))
        assert set(meta["n"]) == {8}

    def test_without_it_the_conditions_differ_and_the_table_shows_it(
            self, two_conditions):
        root, pheno, _ = two_conditions
        _, meta, _ = B.assemble(root, {"movie": "cA", "rest": "cB"}, pheno,
                                bench_args(root, intersect_subjects=False))
        n = meta.groupby("condition")["n"].max().to_dict()
        assert n == {"movie": 12, "rest": 8}

    def test_a_subject_without_a_phenotype_is_in_no_arm(self, two_conditions):
        root, pheno, _ = two_conditions
        _, meta, _ = B.assemble(root, {"movie": "cA", "rest": "cB"},
                                pheno.iloc[:6], bench_args(root))
        assert set(meta["n"]) == {6}

    def test_a_shared_set_this_thin_is_refused_with_the_per_arm_counts(
            self, two_conditions):
        root, pheno, _ = two_conditions
        with pytest.raises(SystemExit) as e:
            B.assemble(root, {"movie": "cA", "rest": "cB"}, pheno,
                       bench_args(root, min_subjects=99))
        msg = str(e.value)
        assert "--min-subjects 99" in msg
        assert "per arm" in msg

    def test_the_covariate_floor_is_one_fit_shared_by_every_arm(
            self, two_conditions):
        """Property 2. Every arm in a condition is prepended with the SAME
        covariate columns, so the floor is literally the same fit -- not a
        per-arm baseline that happens to be close."""
        root, pheno, _ = two_conditions
        data, meta, _ = B.assemble(root, {"movie": "cA", "rest": "cB"}, pheno,
                                   bench_args(root))
        base, _ = data[("movie", "(covariates only)", "-", "-", "-")]
        for key, (X, _) in data.items():
            if key[0] != "movie" or key[1] == "(covariates only)":
                continue
            assert np.allclose(X[:, :base.shape[1]], base), key

    def test_y_is_aligned_with_the_features_row_for_row(self, two_conditions):
        root, pheno, _ = two_conditions
        data, _, _ = B.assemble(root, {"movie": "cA", "rest": "cB"}, pheno,
                                bench_args(root))
        subs = sorted(set(pheno["sub"]))[:8]
        want = pheno.set_index("sub").loc[subs, "y"].to_numpy(float)
        for key, (_, y) in data.items():
            assert np.allclose(y, want), key

    def test_a_condition_with_no_table_at_all_names_both_commands(self, tmp_path):
        pheno = pd.DataFrame({"sub": ["A"], "y": [0], "Age": [30], "Sex": ["M"]})
        with pytest.raises(SystemExit) as e:
            B.assemble(tmp_path, {"movie": "nope"}, pheno, bench_args(tmp_path))
        msg = str(e.value)
        assert "fmri-decomp transitions" in msg
        assert "fmri-decomp static-fc" in msg

    def test_a_condition_with_no_fc_table_says_why_quality_is_needed(
            self, two_conditions):
        root, pheno, _ = two_conditions
        for p in (root / "static_fc").rglob("subjects.parquet"):
            if "cB" in str(p):
                p.unlink()
        with pytest.raises(SystemExit) as e:
            B.assemble(root, {"movie": "cA", "rest": "cB"}, pheno,
                       bench_args(root))
        assert "frac_good_frames" in str(e.value)
        assert "--covariates Age Sex" in str(e.value)


class TestOneScoringFunction:
    def test_the_cv_loop_is_imported_from_select_not_copied(self):
        """Property 3. Same folds, same metric, same estimators -- a second
        copy would drift and the two outputs would stop being comparable."""
        from fmri_decomposition import bstm_selection

        assert B.score_once is bstm_selection.score_once
        assert B.metric_name is bstm_selection.metric_name


class TestTaskFilter:
    def test_a_filter_narrows_before_the_duplicate_check(self):
        d = pd.DataFrame({"sub": ["A", "A", "B"], "task": ["m1", "m2", "m1"]})
        out = B.one_row_per_subject(d, "t", "c", tasks=["m1"])
        assert out["sub"].to_list() == ["A", "B"]

    def test_a_filter_matching_nothing_is_refused_where_it_happens(self):
        """Left to continue, this empties the frame and reappears later as
        "the phenotype join matched nothing", which blames the subject ids."""
        d = pd.DataFrame({"sub": ["A"], "task": ["m1"]})
        with pytest.raises(SystemExit) as e:
            B.one_row_per_subject(d, "t", "c", tasks=["nope"])
        assert "nope" in str(e.value) and "m1" in str(e.value)

    def test_no_filter_leaves_a_unique_table_alone(self):
        d = pd.DataFrame({"sub": ["A", "B"], "task": ["m", "m"]})
        assert len(B.one_row_per_subject(d, "t", "c", tasks=None)) == 2

    def test_the_shared_subject_set_is_surveyed_under_the_same_filter(
            self, two_conditions):
        """Surveying every row and filtering later would intersect onto
        subjects that no arm ends up carrying."""
        root, pheno, _ = two_conditions
        _, meta, _ = B.assemble(root, {"movie": "cA", "rest": "cB"}, pheno,
                                bench_args(root, tasks=["m", "rest"]))
        assert set(meta["n"]) == {8}

    def test_a_filter_excluding_a_condition_entirely_is_refused(
            self, two_conditions):
        root, pheno, _ = two_conditions
        with pytest.raises(SystemExit) as e:
            B.assemble(root, {"movie": "cA", "rest": "cB"}, pheno,
                       bench_args(root, tasks=["m"]))
        assert "--tasks ['m']" in str(e.value)
