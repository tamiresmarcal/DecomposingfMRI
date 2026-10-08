"""Stage 5c -- the three properties that make this a benchmark.

One subject set, one covariate block per condition, one scoring function. Each
has a test here, because each is the kind of thing that silently stops being
true: an arm quietly scored on more subjects than its rival, a covariate that
exists for one arm only, a second copy of the CV loop that drifts.

This stage computes no differences between arms -- the arms share their folds,
so a delta between two rows has no standard error any usual test supplies. The
deliverable is the one table, and `TestTheOneTable` pins that every arm and
condition reaches it as a row.
"""
import argparse

import numpy as np
import pandas as pd
import pytest

from fmri_decomposition import bstm_benchmark as B
from fmri_decomposition import static_fc as S


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


# --------------------------------------------------------------------------
# The one table IS the deliverable now, so it gets an end-to-end test rather
# than a unit test of a formatter: every arm and every condition has to arrive
# as a row, carrying enough to be read against the row above it.
# --------------------------------------------------------------------------
class TestTheOneTable:
    def run_it(self, two_conditions, tmp_path, **kw):
        root, pheno, subs = two_conditions
        ph = tmp_path / "pheno.csv"
        pd.DataFrame({"CCID": pheno["sub"],
                      "Age": pheno["Age"], "Sex": pheno["Sex"],
                      "hads": [["Normal", "Mild", "Moderate", "Severe"][int(v)]
                               for v in pheno["y"]]}).to_csv(ph, index=False)
        a = bench_args(root, target="hads", pheno=[f"{ph}:,"], id_col="CCID",
                       ordinal_levels=["Normal", "Mild", "Moderate", "Severe"],
                       arms=["cells", "occupancy", "fc"], n_jobs=1, show=20,
                       **kw)
        assert B.run(a) == 0
        out = root / "bstm_benchmark" / "target=hads"
        return pd.read_csv(out / "summary.csv"), out

    def test_every_arm_and_condition_is_a_row(self, two_conditions, tmp_path):
        d, _ = self.run_it(two_conditions, tmp_path)
        got = set(zip(d["condition"], d["arm"]))
        for cond in ("movie", "rest"):
            for arm in ("bstm:cells", "bstm:occupancy", "fc:edges",
                        "(covariates only)"):
                assert (cond, arm) in got, (cond, arm)

    def test_the_columns_needed_to_read_two_rows_against_each_other(
            self, two_conditions, tmp_path):
        d, _ = self.run_it(two_conditions, tmp_path)
        # mean to compare, std because a gap smaller than the seed spread is
        # not a result, n and n_features because an arm with more of either is
        # not comparable on mean alone.
        for c in ("model", "condition", "arm", "atlas", "window_s", "states",
                  "mean", "std", "min", "max", "count", "n", "n_features"):
            assert c in d.columns, c

    def test_it_is_sorted_best_first_within_a_model(self, two_conditions,
                                                    tmp_path):
        d, _ = self.run_it(two_conditions, tmp_path)
        for _, g in d.groupby("model"):
            assert g["mean"].is_monotonic_decreasing

    def test_no_delta_column_is_computed(self, two_conditions, tmp_path):
        """Deliberate. The arms share their folds, so a difference between two
        rows is dependent and has no standard error the usual tests supply; a
        `delta` column invites being read as one."""
        d, out = self.run_it(two_conditions, tmp_path)
        assert not [c for c in d.columns if "delta" in c or "contrast" in c]
        assert not (out / "contrasts.csv").exists()

    def test_the_arms_share_one_subject_count(self, two_conditions, tmp_path):
        d, _ = self.run_it(two_conditions, tmp_path)
        assert d["n"].nunique() == 1

    def test_design_and_manifest_travel_with_the_table(self, two_conditions,
                                                       tmp_path):
        d, out = self.run_it(two_conditions, tmp_path)
        note = (out / "DESIGN.md").read_text()
        assert "movie (cohort=cA)" in note and "rest (cohort=cB)" in note
        assert "NOT a test" in note
        assert (out / "scores.parquet").exists()
        assert (out / "figures" / "arms_by_condition.png").exists()
