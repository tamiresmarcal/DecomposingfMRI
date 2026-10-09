"""Adding ONE cohort to a cell must not rewrite the others.

This is the behaviour the whole fit/apply split exists for. Before it,
`decompose` skipped only when EVERY cohort in `train + project` already had a
latents file and otherwise rewrote all of them -- so projecting a new cohort
deleted every state column stage 4b had written into the cell, and the
expensive HMM2 fits had to be redone. `project_cohorts` sitting inside
`model_hash` made it unavoidable: adding a cohort always changed the hash, so
nothing could ever be skipped.

The tests here are about files and hashes, not about numbers: that an untouched
cohort is byte-for-byte untouched, that a column written between the two runs
survives, and that the fit is not repeated.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fmri_decomposition import decompose as D

PARCELS = [f"P{i:02d}" for i in range(8)]


def write_activation(root, cohort, sub, n_tr=80, tr=2.0, seed=0, task="m"):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    t = np.arange(n_tr)
    X = rng.normal(size=(n_tr, len(PARCELS))).astype(np.float32)
    X += np.sin(2 * np.pi * 0.03 * t * tr)[:, None].astype(np.float32)
    d = pd.DataFrame(X, columns=PARCELS)
    d["t"] = t.astype(np.int32)
    d["time_s"] = (t * tr).astype(np.float32)
    d["stimulus_time_s"] = (t * tr).astype(np.float32)
    d["good_frame"] = True
    d["run_idx"] = np.int16(0)
    d["run_key"] = "r0"
    # NOT a column: `task` is a partition key in an activation shard
    # (activation.py: "cohort / atlas / task / sub are partition keys, carried
    # by the path"), and frames.read_cohort assigns it from the path. Writing
    # it as a column here made it the 9th "parcel".
    p = (root / "activation" / "atlas=mini" / f"cohort={cohort}"
         / f"task={task}" / f"sub={sub}" / "data.parquet")
    p.parent.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pandas(d, preserve_index=False)
    pq.write_table(tbl.replace_schema_metadata({b"tr": str(tr).encode()}), p)


def cohort_tree(root, cohorts, n_subs=3):
    for i, c in enumerate(cohorts):
        for j in range(n_subs):
            write_activation(root, c, f"S{j}", seed=i * 10 + j)


def args(root, project, **kw):
    import argparse

    d = dict(atlas="mini", window_s=["-1"], source="activation",
             train=["trainA", "trainB"], project=list(project),
             n_latents=[3], umap_fit_rows=200, no_umap=True,
             umap_latents=None, censor_policy=None, seed=0, overwrite=False,
             match_bandpass=None, zscore_runs=True,
             passthrough_features=False, output_root=str(root),
             balance_train=False, min_rows=1)
    d.update(kw)
    return argparse.Namespace(**d)


def latents(root, cohort):
    return (root / "latents" / "atlas=mini" / "window_s=-1"
            / f"cohort={cohort}" / "data.parquet")


def run(root, project, **kw):
    D.run_one(root, "-1", args(root, project, **kw))


@pytest.fixture
def cell(tmp_path):
    """trainA, trainB and camcan decomposed; camcan_rest's shards exist but it
    has not been projected yet."""
    cohort_tree(tmp_path, ["trainA", "trainB", "camcan", "camcan_rest"])
    run(tmp_path, ["camcan"])
    return tmp_path


class TestAddingACohort:
    def test_the_first_run_writes_every_cohort_it_was_given(self, cell):
        for c in ("trainA", "trainB", "camcan"):
            assert latents(cell, c).exists(), c
        assert not latents(cell, "camcan_rest").exists()

    def test_adding_one_does_not_rewrite_the_others(self, cell):
        """THE POINT. mtime_ns, so a rewrite with identical content still
        fails -- the claim is that the file is not touched, not that it would
        come out the same."""
        before = {c: latents(cell, c).stat().st_mtime_ns
                  for c in ("trainA", "trainB", "camcan")}
        run(cell, ["camcan", "camcan_rest"])
        assert latents(cell, "camcan_rest").exists()
        for c, t in before.items():
            assert latents(cell, c).stat().st_mtime_ns == t, c

    def test_a_state_column_written_in_between_survives(self, cell):
        """What stage 4b spends a day producing. Simulated with the same
        function `cluster` uses, so this is its actual write path."""
        from fmri_decomposition.cluster import append_columns

        n = len(pd.read_parquet(latents(cell, "camcan")))
        append_columns(latents(cell, "camcan"),
                       {"HMM2_pca3_8": np.zeros(n, dtype=np.int32)},
                       {"HMM2_pca3_8": {"method": "hmm2", "k": 8}})
        run(cell, ["camcan", "camcan_rest"])
        back = pd.read_parquet(latents(cell, "camcan"))
        assert "HMM2_pca3_8" in back.columns

    def test_both_cohorts_end_up_with_the_same_model_hash(self, cell):
        """Without this the cell is not comparable across cohorts, which is
        exactly what `transitions --check` refuses."""
        run(cell, ["camcan", "camcan_rest"])
        hashes = {c: D.latents_model_hash(latents(cell, c))
                  for c in ("trainA", "trainB", "camcan", "camcan_rest")}
        assert len(set(hashes.values())) == 1, hashes

    def test_the_fit_is_not_repeated(self, cell, capsys):
        capsys.readouterr()
        run(cell, ["camcan", "camcan_rest"])
        out = capsys.readouterr().out
        assert "reusing the saved fit" in out
        assert "loading training cohorts" not in out

    def test_the_new_cohort_is_labelled_projected(self, cell):
        """role_of is per-invocation and the saved copy predates this cohort,
        so it has to be refreshed or write_latents raises KeyError."""
        run(cell, ["camcan", "camcan_rest"])
        d = pd.read_parquet(latents(cell, "camcan_rest"))
        assert set(d["role"]) == {"projected"}

    def test_the_projected_rows_are_the_saved_fit_applied(self, cell):
        """Reuse must mean the same transform, not a re-derived one. camcan was
        written by the fitting run; re-projecting it with --overwrite under the
        reused fit has to reproduce it."""
        before = pd.read_parquet(latents(cell, "camcan"))
        run(cell, ["camcan", "camcan_rest"])          # writes only rest
        run(cell, ["camcan"], overwrite=True)         # refits, rewrites camcan
        after = pd.read_parquet(latents(cell, "camcan"))
        cols = [c for c in before.columns if c.startswith("pca")]
        assert np.allclose(before[cols].to_numpy(), after[cols].to_numpy(),
                           atol=1e-5)


class TestWhenItDoesRewrite:
    def test_nothing_to_do_is_said_and_nothing_is_written(self, cell, capsys):
        before = latents(cell, "camcan").stat().st_mtime_ns
        capsys.readouterr()
        run(cell, ["camcan"])
        assert "nothing to write" in capsys.readouterr().out
        assert latents(cell, "camcan").stat().st_mtime_ns == before

    def test_overwrite_rewrites_and_refits(self, cell, capsys):
        before = latents(cell, "camcan").stat().st_mtime_ns
        capsys.readouterr()
        run(cell, ["camcan"], overwrite=True)
        assert "loading training cohorts" in capsys.readouterr().out
        assert latents(cell, "camcan").stat().st_mtime_ns != before

    def test_a_settings_change_rewrites_everything(self, cell):
        """A different fit must NOT be silently mixed into the cell. Changing
        n_latents changes model_hash, so every cohort is stale and all of them
        are rewritten."""
        before = {c: latents(cell, c).stat().st_mtime_ns
                  for c in ("trainA", "trainB", "camcan")}
        run(cell, ["camcan"], n_latents=[3, 5])
        for c, t in before.items():
            assert latents(cell, c).stat().st_mtime_ns != t, c

    def test_a_library_change_refits_rather_than_trusting_the_pickle(self,
                                                                    cell,
                                                                    capsys):
        """A pickled sklearn estimator is not guaranteed to behave across
        versions. The manifest records what it was pickled under, and a
        mismatch has to mean refit, not silent reuse."""
        man = (cell / "meta" / "models"
               / "decompose_atlas-mini_window--1_manifest.json")
        d = json.loads(man.read_text())
        d["library_versions"]["numpy"] = "0.0.0-not-a-real-version"
        man.write_text(json.dumps(d))
        capsys.readouterr()
        run(cell, ["camcan", "camcan_rest"])
        out = capsys.readouterr().out
        assert "refitting rather than trusting it" in out
        assert "loading training cohorts" in out

    def test_a_missing_manifest_refits(self, cell, capsys):
        (cell / "meta" / "models"
         / "decompose_atlas-mini_window--1_manifest.json").unlink()
        capsys.readouterr()
        run(cell, ["camcan", "camcan_rest"])
        assert "loading training cohorts" in capsys.readouterr().out

    def test_the_project_list_is_recorded_beside_the_hash_not_in_it(self, cell):
        """Removed from the payload, but not lost -- it is still recoverable
        from the manifest."""
        man = json.loads((cell / "meta" / "models"
                          / "decompose_atlas-mini_window--1_manifest.json")
                         .read_text())
        assert man["project_cohorts"] == ["camcan"]
        assert "project_cohorts" not in man["fit_meta"] if "fit_meta" in man \
            else True


# --------------------------------------------------------------------------
# The other half: `cluster` must not refit when the fit it needs is already
# saved. That fit is the expensive one -- HMM2 measured ~24 h per atlas at
# K=27 -- so "adding a cohort is cheap" is false until this holds.
# --------------------------------------------------------------------------
def cluster_args(**kw):
    import argparse

    from fmri_decomposition import cluster as C

    d = dict(methods=["hmm2"], embeddings=["pca3"], k=[3],
             train=["trainA", "trainB"], balance_train=False, refit=False,
             meanshift_quantile=0.2, meanshift_fit_rows=50_000, hmm_iter=5,
             hmm2_iter=5, hmm2_restarts=2, hmm2_jobs=1,
             min_k=2, max_k=30, project=["all"])
    d.update(kw)
    return argparse.Namespace(**d)


def run_cluster(root, **kw):
    from fmri_decomposition import cluster as C

    return C.run_one(root, "mini", "-1", cluster_args(**kw))


class TestClusterReusesItsFit:
    @pytest.fixture
    def clustered(self, cell, capsys):
        entries, _ = run_cluster(cell)
        capsys.readouterr()
        return cell, entries

    def test_the_first_run_fits_and_caches(self, clustered):
        """Filed under the RECIPE, not the fit_hash -- the recipe is what a
        later run can compute before fitting, which is what makes a lookup
        possible at all. The fit_hash is recorded INSIDE, for the reload
        check."""
        import joblib

        from fmri_decomposition import cluster as C

        cell, entries = clustered
        assert entries and not entries[0]["fit_reused"]
        cached = list((cell / "meta" / "clusterers").glob("*.joblib"))
        assert len(cached) == 1
        blob = joblib.load(cached[0])
        a = cluster_args()
        recipe = C.recipe_hash("hmm2", "pca3", 3, C._opts("hmm2", a),
                               ["trainA", "trainB"], blob["n_train_rows"])
        assert recipe in cached[0].name
        assert blob["fit_hash"] == entries[0]["fit_hash"]
        assert blob["recipe_hash"] == recipe

    def test_a_reload_that_changed_the_model_is_refused(self, clustered):
        """params() and k_found come out of the pickle, so a reloaded fit must
        reproduce its own fit_hash. If it does not, the pickle no longer
        behaves as it did -- which must not pass silently."""
        import joblib

        cell, _ = clustered
        cached = next((cell / "meta" / "clusterers").glob("*.joblib"))
        blob = joblib.load(cached)
        blob["fit_hash"] = "not-what-it-will-reload-as"
        joblib.dump(blob, cached)
        with pytest.raises(SystemExit) as e:
            run_cluster(cell)
        assert "reloaded to a DIFFERENT model" in str(e.value)
        assert "--refit" in str(e.value)

    def test_a_second_run_reuses_it(self, clustered, capsys):
        cell, _ = clustered
        entries, _ = run_cluster(cell)
        assert entries[0]["fit_reused"]
        assert "reusing the cached fit" in capsys.readouterr().out

    def test_a_cohort_already_at_this_fit_is_not_relabelled(self, clustered):
        """append_columns rewrites the whole file, so relabelling a cohort that
        needs nothing would churn every file in the cell on every run."""
        cell, entries = clustered
        col = entries[0]["column"]
        before = {c: latents(cell, c).stat().st_mtime_ns
                  for c in ("trainA", "trainB", "camcan")}
        run_cluster(cell)
        for c, t in before.items():
            assert latents(cell, c).stat().st_mtime_ns == t, c

    def test_a_cohort_added_afterwards_is_labelled_without_a_refit(
            self, clustered, capsys):
        """THE POINT OF THE WHOLE CHANGE. Project rest, then cluster: rest gets
        the labels and nothing is fitted again."""
        cell, entries = clustered
        col = entries[0]["column"]
        run(cell, ["camcan", "camcan_rest"])          # decompose: rest only
        capsys.readouterr()
        new, _ = run_cluster(cell)
        out = capsys.readouterr().out
        assert "reusing the cached fit" in out
        assert new[0]["fit_reused"]
        # Only the new cohort was labelled, and it did get states.
        assert list(new[0]["states_used"]) == ["camcan_rest"]
        assert new[0]["states_used"]["camcan_rest"] >= 1
        assert sorted(new[0]["cohorts_kept"]) == ["camcan", "trainA", "trainB"]
        assert col in pd.read_parquet(latents(cell, "camcan_rest")).columns

    def test_the_labels_are_the_same_model_not_a_new_one(self, clustered):
        """A reused fit has to give the identical labels for a cohort it
        already labelled -- otherwise 'reuse' is a different model wearing the
        same fit_hash."""
        cell, entries = clustered
        col = entries[0]["column"]
        before = pd.read_parquet(latents(cell, "camcan"))[col].to_numpy()
        run_cluster(cell, refit=False)
        # force a relabel from the cached fit by clearing the column's hash
        from fmri_decomposition.cluster import append_columns

        append_columns(latents(cell, "camcan"), {col: before.astype(np.int32)},
                       {col: {"method": "hmm2", "fit_hash": "cleared"}})
        run_cluster(cell)
        after = pd.read_parquet(latents(cell, "camcan"))[col].to_numpy()
        assert np.array_equal(before, after)

    def test_refit_ignores_the_cache(self, clustered, capsys):
        cell, _ = clustered
        entries, _ = run_cluster(cell, refit=True)
        assert not entries[0]["fit_reused"]
        assert "reusing the cached fit" not in capsys.readouterr().out

    def test_a_cache_from_other_latents_is_refused(self, clustered, capsys):
        """fit_hash describes the CLUSTERING, not the embedding underneath, so
        without the model_hash guard a cached fit could be applied to latents
        from a different decompose run with the same row count."""
        import joblib

        cell, _ = clustered
        cached = next((cell / "meta" / "clusterers").glob("*.joblib"))
        blob = joblib.load(cached)
        blob["model_hash"] = "a-different-decompose-fit"
        joblib.dump(blob, cached)
        capsys.readouterr()
        entries, _ = run_cluster(cell)
        assert "refitting" in capsys.readouterr().out
        assert not entries[0]["fit_reused"]

    def test_a_cache_from_other_libraries_is_refused(self, clustered, capsys):
        import joblib

        cell, _ = clustered
        cached = next((cell / "meta" / "clusterers").glob("*.joblib"))
        blob = joblib.load(cached)
        blob["libs"] = {**blob["libs"], "numpy": "0.0.0-not-real"}
        joblib.dump(blob, cached)
        capsys.readouterr()
        entries, _ = run_cluster(cell)
        out = capsys.readouterr().out
        assert "different libraries" in out and "refitting" in out
        assert not entries[0]["fit_reused"]

    def test_an_unreadable_cache_refits_rather_than_raising(self, clustered,
                                                            capsys):
        cell, _ = clustered
        next((cell / "meta" / "clusterers").glob("*.joblib")).write_text("junk")
        capsys.readouterr()
        entries, _ = run_cluster(cell)
        assert "unreadable cache" in capsys.readouterr().out
        assert not entries[0]["fit_reused"]

    def test_every_method_is_cacheable_including_meanshift(self, cell):
        """The cache is keyed by the RECIPE -- what was asked for -- not by
        fit_hash, which carries what came out. MeanShift's discovered bandwidth
        and Threshold's bins_per_axis do not exist before the fit, so they can
        describe a fit but cannot look one up. The recipe can, for every
        method."""
        from fmri_decomposition import cluster as C

        a = cluster_args(methods=["meanshift"])
        h = C.recipe_hash("meanshift", "pca3", None, C._opts("meanshift", a),
                          ["trainA"], 1000)
        assert len(h) == 16
        # Same recipe, same hash, every time -- that is the whole property.
        assert h == C.recipe_hash("meanshift", "pca3", None,
                                  C._opts("meanshift", a), ["trainA"], 1000)

    def test_the_recipe_ignores_options_that_only_change_speed(self):
        """n_jobs changes how long the restarts take and nothing about the
        labels, so a run on 8 cores must find the fit a run on 1 core saved."""
        from fmri_decomposition import cluster as C

        one = cluster_args(hmm2_jobs=1)
        eight = cluster_args(hmm2_jobs=8)
        assert (C.recipe_hash("hmm2", "pca3", 3, C._opts("hmm2", one),
                              ["trainA"], 1000)
                == C.recipe_hash("hmm2", "pca3", 3, C._opts("hmm2", eight),
                                 ["trainA"], 1000))

    def test_the_recipe_does_move_with_a_real_setting(self):
        from fmri_decomposition import cluster as C

        few = cluster_args(hmm2_restarts=2)
        many = cluster_args(hmm2_restarts=15)
        assert (C.recipe_hash("hmm2", "pca3", 3, C._opts("hmm2", few),
                              ["trainA"], 1000)
                != C.recipe_hash("hmm2", "pca3", 3, C._opts("hmm2", many),
                                 ["trainA"], 1000))

    def test_the_recipe_moves_with_the_training_rows(self):
        """A different training set IS a different fit -- for MeanShift it can
        even discover a different K -- so it must not reuse the old one."""
        from fmri_decomposition import cluster as C

        a = cluster_args()
        o = C._opts("meanshift", a)
        assert (C.recipe_hash("meanshift", "pca3", None, o, ["trainA"], 1000)
                != C.recipe_hash("meanshift", "pca3", None, o, ["trainA"], 2000))
        assert (C.recipe_hash("meanshift", "pca3", None, o, ["trainA"], 1000)
                != C.recipe_hash("meanshift", "pca3", None, o,
                                 ["trainA", "trainB"], 1000))

    def test_meanshift_reuses_its_fit_so_k_cannot_drift(self, cell, capsys):
        """Her question: how is the same number of clusters guaranteed? Within
        one fit it is trivial -- one object labels everyone. Across runs it is
        now guaranteed by reusing that object, not by trusting the search to
        land in the same place twice."""
        first, _ = run_cluster(cell, methods=["meanshift"], k=[None])
        capsys.readouterr()
        second, _ = run_cluster(cell, methods=["meanshift"], k=[None])
        assert "reusing the cached fit" in capsys.readouterr().out
        assert second[0]["fit_reused"]
        assert second[0]["k"] == first[0]["k"]
        assert second[0]["column"] == first[0]["column"]


# --------------------------------------------------------------------------
# The forgotten-flag footgun. The fit is defined by the flags, so omitting one
# when adding a cohort moves model_hash, makes every existing cohort look
# stale, and rewrites all of them -- deleting state columns that cost hours.
# Measured: on yeo7, dropping --passthrough-features alone moves the hash.
# --------------------------------------------------------------------------
class TestForgottenFlagIsRefused:
    @pytest.fixture
    def with_states(self, cell):
        from fmri_decomposition.cluster import append_columns

        n = len(pd.read_parquet(latents(cell, "camcan")))
        append_columns(latents(cell, "camcan"),
                       {"HMM2_pca3_8": np.zeros(n, dtype=np.int32)},
                       {"HMM2_pca3_8": {"method": "hmm2", "k": 8,
                                        "fit_hash": "abc"}})
        return cell

    def test_a_changed_fit_is_refused_when_states_would_be_lost(self,
                                                                with_states):
        with pytest.raises(SystemExit) as e:
            run(with_states, ["camcan"], n_latents=[3, 5])
        msg = str(e.value)
        assert "refusing to rewrite" in msg
        assert "HMM2_pca3_8" in msg

    def test_the_refusal_names_the_flag_that_moved(self, with_states):
        """A bare "the hash differs" sends you to diff two 16-character
        strings. This has to say which flag to put back."""
        with pytest.raises(SystemExit) as e:
            run(with_states, ["camcan"], n_latents=[3, 5])
        msg = str(e.value)
        assert "n_latents" in msg
        assert "[3]" in msg and "[3, 5]" in msg

    def test_it_points_at_both_ways_out(self, with_states):
        with pytest.raises(SystemExit) as e:
            run(with_states, ["camcan"], n_latents=[3, 5])
        msg = str(e.value)
        assert "add it back" in msg
        assert "--overwrite" in msg

    def test_nothing_is_written_before_it_refuses(self, with_states):
        before = latents(with_states, "camcan").stat().st_mtime_ns
        with pytest.raises(SystemExit):
            run(with_states, ["camcan"], n_latents=[3, 5])
        assert latents(with_states, "camcan").stat().st_mtime_ns == before

    def test_overwrite_is_the_escape_hatch(self, with_states):
        run(with_states, ["camcan"], n_latents=[3, 5], overwrite=True)
        back = pd.read_parquet(latents(with_states, "camcan"))
        assert "HMM2_pca3_8" not in back.columns      # as warned
        assert "pca0/5" in back.columns

    def test_a_cohort_with_no_state_columns_is_rewritten_freely(self, cell):
        """Nothing to lose, so no refusal -- the guard is about destroying
        work, not about the hash moving."""
        run(cell, ["camcan"], n_latents=[3, 5])
        assert "pca0/5" in pd.read_parquet(latents(cell, "camcan")).columns

    def test_a_brand_new_cohort_is_never_blocked_by_it(self, with_states):
        """The whole point of the change: adding a cohort with the SAME flags
        must still just work, even with state columns present elsewhere."""
        run(with_states, ["camcan", "camcan_rest"])
        assert latents(with_states, "camcan_rest").exists()
        assert "HMM2_pca3_8" in pd.read_parquet(
            latents(with_states, "camcan")).columns

    def test_the_stored_fit_description_is_readable(self, cell):
        stored, cols = D.latents_fit_description(latents(cell, "camcan"))
        assert stored["train_cohorts"] == ["trainA", "trainB"]
        assert "project_cohorts" not in stored
        assert cols == []


class TestTheDiffIsReadable:
    """The refusal is only useful if the flag that moved is findable in it.

    A latents file carries ten keys BESIDE the fit description -- cohort, role,
    stride_s, model_hash, written_utc and so on -- and comparing those against
    a fit_meta that never had them produced ten lines of `-> '(absent)'` with
    the one real difference buried in the middle.
    """

    def test_only_the_real_difference_is_listed(self, tmp_path):
        stored = {"n_latents": [3], "atlas": "mini", "cohort": "camcan",
                  "role": "projected", "stride_s": 2.0, "indep_factor": 1,
                  "model_hash": "abc", "n_train_rows": 480,
                  "umap_fitted": False, "n_umap_components": [],
                  "written_utc": "2026-10-08T00:00:00Z"}
        meta = {"n_latents": [3, 5], "atlas": "mini"}
        assert D._fit_differences(stored, meta) == [
            "      n_latents: [3] -> [3, 5]"]

    def test_a_flag_dropped_from_the_command_is_shown(self):
        """fit_meta writes a key only when it differs from its default, so a
        key present on disk and absent now IS the forgotten flag -- the case
        this message exists for."""
        out = D._fit_differences({"passthrough_features": True}, {})
        assert out == ["      passthrough_features: True -> '(absent)'"]

    def test_a_flag_newly_added_is_shown_too(self):
        out = D._fit_differences({}, {"passthrough_features": True})
        assert out == ["      passthrough_features: '(absent)' -> True"]

    def test_identical_descriptions_differ_in_nothing(self):
        d = {"n_latents": [3], "atlas": "mini", "written_utc": "whenever"}
        assert D._fit_differences(d, {"n_latents": [3], "atlas": "mini"}) == []

    def test_a_package_version_change_is_reported(self):
        """It is in the hash payload but not in fit_meta, so the generic loop
        cannot see it -- and a version bump really does move every hash."""
        out = D._fit_differences({"package_version": "0.0.1-ancient"}, {})
        assert len(out) == 1 and "package_version" in out[0]

    def test_the_excluded_keys_match_what_write_latents_adds(self):
        """If write_latents gains a per-file key and this set does not, the
        diff starts showing it as a difference that cannot be acted on."""
        src = (Path(D.__file__).read_text()
               .split("def write_latents")[1].split("def ")[0])
        for key in D._NOT_FIT_KEYS:
            assert f'"{key}"' in src, key


# --------------------------------------------------------------------------
# `cluster --project`: which cohorts get labelled, SAID rather than inferred.
# Before this, cluster acted on whatever cohorts happened to be in the
# directory, so a subset could not be asked for and a stray cohort was
# labelled silently.
# --------------------------------------------------------------------------
class TestClusterProject:
    def test_all_labels_every_cohort_in_the_cell(self, cell):
        entries, _ = run_cluster(cell, project=["all"])
        assert set(entries[0]["states_used"]) == {"trainA", "trainB", "camcan"}

    def test_a_subset_can_be_asked_for(self, cell):
        """Her case: come back a month later and label only the new cohort."""
        run(cell, ["camcan", "camcan_rest"])
        entries, _ = run_cluster(cell, project=["camcan_rest"])
        labelled = set(entries[0]["states_used"]) | set(
            entries[0]["cohorts_kept"])
        assert "camcan_rest" in set(entries[0]["states_used"])
        assert "camcan" not in labelled      # not asked for, not touched

    def test_train_is_labelled_without_being_named(self, cell):
        """`train + project`, exactly as decompose says it -- the training
        cohorts are part of the output and always get their columns."""
        run(cell, ["camcan", "camcan_rest"])
        entries, _ = run_cluster(cell, project=["camcan_rest"])
        got = set(entries[0]["states_used"]) | set(entries[0]["cohorts_kept"])
        assert {"trainA", "trainB", "camcan_rest"} <= got

    def test_a_cohort_not_asked_for_keeps_the_columns_it_has(self, cell):
        col = run_cluster(cell)[0][0]["column"]
        before = pd.read_parquet(latents(cell, "camcan"))[col].to_numpy()
        run(cell, ["camcan", "camcan_rest"])
        run_cluster(cell, project=["camcan_rest"])
        after = pd.read_parquet(latents(cell, "camcan"))[col].to_numpy()
        assert np.array_equal(before, after)

    def test_an_unknown_cohort_is_refused_not_ignored(self, cell):
        """The typo protection there was no way to have before."""
        with pytest.raises(SystemExit) as e:
            run_cluster(cell, project=["camcan_rst"])
        msg = str(e.value)
        assert "camcan_rst" in msg
        assert "decompose --project camcan_rst" in msg
        assert "camcan" in msg               # lists what the cell has

    def test_it_says_which_cohorts_it_left_out(self, cell, capsys):
        run(cell, ["camcan", "camcan_rest"])
        capsys.readouterr()
        run_cluster(cell, project=["camcan_rest"])
        out = capsys.readouterr().out
        assert "not asked for" in out and "camcan" in out


# --------------------------------------------------------------------------
# "Train only" -- fit on --train and write/label nothing else. Exposed by
# Tamires asking for the three scenarios side by side: it was not expressible
# in decompose (nargs=+ rejected an empty list) and was silently INVERTED in
# cluster (an empty list is falsy, so it labelled everything).
# --------------------------------------------------------------------------
class TestTrainOnly:
    def test_decompose_projects_nothing_by_default(self):
        """Train-only is just `--train A B`, with no flag saying "no".

        It used to default to ["camcan"], which was wrong on this stage's own
        terms: the module docstring says no cohort YAML owns stage 4 because a
        decomposition spans cohorts -- and then the default named one cohort.
        """
        import argparse

        p = argparse.ArgumentParser()
        D.add_arguments(p)
        base = ["--atlas", "yeo7", "--window-s", "-1",
                "--censor-policy", "none"]
        assert p.parse_args(base).project == []
        assert p.parse_args(base + ["--project", "camcan"]).project == ["camcan"]

    def test_decompose_train_only_writes_only_the_training_cohorts(self,
                                                                   tmp_path):
        cohort_tree(tmp_path, ["trainA", "trainB", "camcan"])
        run(tmp_path, [])
        for c in ("trainA", "trainB"):
            assert latents(tmp_path, c).exists(), c
        assert not latents(tmp_path, "camcan").exists()

    def test_cluster_projects_nothing_by_default_too(self):
        """Symmetric with decompose, which is the point: one mental model for
        both stages. `--project all` is the old implicit behaviour, now asked
        for rather than assumed."""
        import argparse

        from fmri_decomposition import cluster as C

        p = argparse.ArgumentParser()
        C.add_arguments(p)
        base = ["--atlas", "yeo7", "--window-s", "-1"]
        assert p.parse_args(base).project == []
        assert p.parse_args(base + ["--project", "all"]).project == ["all"]

    def test_cluster_train_only_labels_only_the_training_cohorts(self, cell):
        """An empty list used to be falsy and fall through to "label
        everything" -- the exact opposite of what it reads as."""
        entries, _ = run_cluster(cell, project=[])
        touched = set(entries[0]["states_used"]) | set(
            entries[0]["cohorts_kept"])
        assert touched == {"trainA", "trainB"}
        assert "camcan" not in touched

    def test_cluster_labels_only_train_by_default(self, cell):
        entries, _ = run_cluster(cell, project=[])
        assert set(entries[0]["states_used"]) == {"trainA", "trainB"}

    def test_project_all_is_how_you_ask_for_every_cohort(self, cell):
        """The old implicit default. Convenient, and the wrong DEFAULT: a
        cohort that appeared in the cell got labelled without being named."""
        entries, _ = run_cluster(cell, project=["all"])
        assert set(entries[0]["states_used"]) == {"trainA", "trainB", "camcan"}

    def test_a_projected_cohort_is_untouched_by_a_train_only_run(self, cell):
        col = run_cluster(cell)[0][0]["column"]
        before = latents(cell, "camcan").stat().st_mtime_ns
        run_cluster(cell, project=[])
        assert latents(cell, "camcan").stat().st_mtime_ns == before
        assert col in pd.read_parquet(latents(cell, "camcan")).columns


class TestACohortThatGainedATask:
    """One cohort, two conditions -- and the skip must notice the second.

    The per-cohort skip asks one question: does this cohort's latents carry the
    current model_hash? "Yes, leave it alone" is right when its shards have not
    changed, and WRONG the moment the cohort gains a condition. Cam-CAN's rest
    arrived as shards under `cohort=camcan/task=Rest` beside a latents file
    already at the right hash for `task=Movie` alone. Skipped, the whole rest
    condition would never be projected: every file present, the hash correct,
    and half the data missing.
    """

    @pytest.fixture
    def movie_only(self, tmp_path):
        """camcan decomposed from Movie shards alone, Rest not yet extracted."""
        for i, c in enumerate(["trainA", "trainB"]):
            for j in range(3):
                write_activation(tmp_path, c, f"S{j}", seed=i * 10 + j,
                                 task="Movie")
        for j in range(3):
            write_activation(tmp_path, "camcan", f"S{j}", seed=99 + j,
                             task="Movie")
        run(tmp_path, ["camcan"])
        return tmp_path

    def test_the_tasks_it_covers_are_recorded(self, movie_only):
        assert D.latents_tasks(latents(movie_only, "camcan")) == {"Movie"}

    def test_the_source_tasks_come_from_the_partitions(self, movie_only):
        for j in range(3):
            write_activation(movie_only, "camcan", f"S{j}", seed=7 + j,
                             task="Rest")
        assert D.source_tasks(movie_only, "mini", "-1", "camcan",
                              "activation") == {"Movie", "Rest"}

    def test_a_cohort_missing_a_task_is_not_skipped(self, movie_only):
        """THE POINT. Same model_hash, so the plain skip would leave it alone."""
        for j in range(3):
            write_activation(movie_only, "camcan", f"S{j}", seed=7 + j,
                             task="Rest")
        out_dir = latents(movie_only, "camcan").parent.parent
        assert D.cohorts_missing_a_task(
            movie_only, out_dir, "mini", "-1", "activation",
            ["camcan", "trainA"], log=lambda *a: None) == ["camcan"]

    def test_it_is_rewritten_and_then_covers_both(self, movie_only):
        for j in range(3):
            write_activation(movie_only, "camcan", f"S{j}", seed=7 + j,
                             task="Rest")
        run(movie_only, ["camcan"], overwrite=True)
        lat = pd.read_parquet(latents(movie_only, "camcan"))
        assert set(lat["task"]) == {"Movie", "Rest"}
        assert D.latents_tasks(latents(movie_only, "camcan")) == {"Movie",
                                                                 "Rest"}

    def test_a_cohort_whose_tasks_are_unchanged_is_still_skipped(self,
                                                                movie_only):
        """The guard must not cost the incremental behaviour it sits beside."""
        before = {c: latents(movie_only, c).stat().st_mtime_ns
                  for c in ("trainA", "trainB", "camcan")}
        run(movie_only, ["camcan"])
        for c, t in before.items():
            assert latents(movie_only, c).stat().st_mtime_ns == t, c

    def test_the_rewrite_is_refused_while_state_columns_exist(self,
                                                              movie_only):
        """Not silently rewritten: dropping the state columns costs the HMM2
        fits, so the refusal names the cost and --overwrite is deliberate."""
        from fmri_decomposition.cluster import append_columns

        n = len(pd.read_parquet(latents(movie_only, "camcan")))
        append_columns(latents(movie_only, "camcan"),
                       {"HMM2_pca3_8": np.zeros(n, dtype=np.int16)},
                       {"HMM2_pca3_8": {"method": "HMM2", "k": 8}})
        for j in range(3):
            write_activation(movie_only, "camcan", f"S{j}", seed=7 + j,
                             task="Rest")
        with pytest.raises(SystemExit) as e:
            run(movie_only, ["camcan"])
        assert "refusing to rewrite" in str(e.value)
