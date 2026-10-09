"""Stage 4's source dispatch, and the hash it must not disturb.

`model_hash` is the claim "these two files came from one fit". Adding a second
feature source had to leave every hash already stamped on disk alone, because a
changed hash turns valid latents into files that no longer match a re-run of the
command that produced them -- and nothing would report that, it would just look
like two incomparable fits.
"""
import argparse

import pytest

from fmri_decomposition import decompose as D


def parse(*argv):
    """Parse, supplying --censor-policy none unless the caller names one.

    The flag is REQUIRED on the real command line -- forgetting it silently
    produced an uncensored aperture once already -- so a test that wants the
    default behaviour has to say `none` out loud, exactly as a user does.
    """
    p = argparse.ArgumentParser()
    D.add_arguments(p)
    argv = list(argv)
    if "--censor-policy" not in argv:
        argv += ["--censor-policy", "none"]
    a = p.parse_args(argv)
    a.output_root = None                     # skips the censor-summary lookup
    return a


EDGES = ["A__B", "A__C", "B__C"]


class TestFitMetaIsHashStable:
    # Frozen on purpose. If this value changes, a dfc latents file written before
    # the change stops matching a re-run of the command that wrote it, and
    # nothing reports that -- it looks like two incomparable fits. So a change
    # here has to be deliberate: update the constant in the same commit, and say
    # in the message what moved.
    #
    # Changed TWICE, knowingly:
    #
    #   433fca34406aafde  the original, while `bins` was part of the payload
    #   8c7c88dca4e6b719  `bins` left, when quantile thresholding moved out of
    #                     this stage into `cluster`
    #   eacc79844d0ec658  `project_cohorts` left. It is a per-INVOCATION
    #                     quantity, not a property of the fit -- only --train
    #                     feeds the scaler, the PCA and the UMAP -- so two runs
    #                     differing only in who else was projected produce the
    #                     same model, and the hash claimed otherwise. While it
    #                     was hashed, adding a cohort to a cell ALWAYS changed
    #                     the hash, so no cohort could be skipped and every
    #                     addition rewrote and re-clustered the whole cell.
    DFC_HASH = "eacc79844d0ec658"

    def test_the_default_payload_is_unchanged(self):
        a = parse("--atlas", "yeo7", "--window-s", "30")
        assert D.model_hash(D.fit_meta(a, "30", EDGES), EDGES) == self.DFC_HASH

    def test_the_dfc_payload_carries_no_source_keys(self):
        a = parse("--atlas", "yeo7", "--window-s", "30")
        meta = D.fit_meta(a, "30", EDGES)
        for k in ("source", "match_bandpass", "zscore_runs"):
            assert k not in meta

    def test_the_project_list_is_not_part_of_the_fit_description(self):
        """The fit sees only --train. A hash that moved with --project made two
        identical fits look incomparable, and made every cohort addition a
        cell-wide rewrite."""
        a = parse("--atlas", "yeo7", "--window-s", "30")
        assert "project_cohorts" not in D.fit_meta(a, "30", EDGES)

    def test_two_project_lists_give_one_hash(self):
        one = parse("--atlas", "yeo7", "--window-s", "30",
                    "--project", "camcan")
        two = parse("--atlas", "yeo7", "--window-s", "30",
                    "--project", "camcan", "camcan_rest")
        assert (D.model_hash(D.fit_meta(one, "30", EDGES), EDGES)
                == D.model_hash(D.fit_meta(two, "30", EDGES), EDGES))

    def test_the_train_list_still_does_move_it(self):
        """The other half of the claim: --train IS the fit, so it must move the
        hash. Dropping project_cohorts must not have made the payload
        indifferent to who was fitted on."""
        one = parse("--atlas", "yeo7", "--window-s", "30",
                    "--train", "ds002837")
        two = parse("--atlas", "yeo7", "--window-s", "30",
                    "--train", "ds002837", "cneuromod")
        assert (D.model_hash(D.fit_meta(one, "30", EDGES), EDGES)
                != D.model_hash(D.fit_meta(two, "30", EDGES), EDGES))

    def test_the_payload_no_longer_describes_a_clustering(self):
        # `bins` configured the thresholding this stage used to do. It belongs to
        # stage 4b now, so it must not be in a hash that claims to describe THIS
        # fit -- a hash that over-claims is one that reports two identical fits
        # as different.
        a = parse("--atlas", "yeo7", "--window-s", "30")
        assert "bins" not in D.fit_meta(a, "30", EDGES)


class TestNoStatesHere:
    def test_bins_is_gone_from_the_command_line(self):
        p = argparse.ArgumentParser()
        D.add_arguments(p)
        with pytest.raises(SystemExit):
            p.parse_args(["--atlas", "yeo7", "--window-s", "30",
                          "--bins", "2", "3"])

    def test_three_components_are_still_required(self):
        # Every stage 4b clusterer works on a 3-D embedding, so a fit without
        # pca3 produces latents nothing can cluster -- and the failure has to
        # come at the fit, not later when `cluster` finds no pca3 columns.
        import numpy as np

        a = parse("--atlas", "yeo7", "--window-s", "30", "--pca-latents", "2",
                  "--no-umap")
        X = np.random.default_rng(0).normal(size=(60, 6)).astype(np.float32)
        feats = [f"A__{i}" for i in range(6)]
        with pytest.raises(SystemExit, match="include 3"):
            D.fit_models(X, feats, a, meta={})

    def test_latents_for_writes_no_state_column(self):
        import numpy as np
        import pandas as pd

        a = parse("--atlas", "yeo7", "--window-s", "30", "--pca-latents", "3",
                  "--no-umap")
        X = np.random.default_rng(0).normal(size=(60, 6)).astype(np.float32)
        feats = [f"A__{i}" for i in range(6)]
        models = D.fit_models(X, feats, a, meta={"model_hash": "x"})
        ident = pd.DataFrame({"cohort": "c", "task": "m", "sub": "01",
                              "window_id": np.arange(60)})
        out = D.latents_for(ident, X, models, "train")
        assert [c for c in out.columns if "Cluster" in c or "HMM" in c] == []
        assert "pca0/3" in out.columns

    def test_the_activation_payload_records_both_transforms(self):
        a = parse("--atlas", "yeo7", "--source", "activation")
        meta = D.fit_meta(a, "-1", ["P0", "P1"])
        assert meta["source"] == "activation"
        assert meta["match_bandpass"] == [0.01, 0.1]
        assert meta["zscore_runs"] is True

    def test_turning_a_transform_off_is_a_different_model(self):
        on = parse("--atlas", "yeo7", "--source", "activation")
        off = parse("--atlas", "yeo7", "--source", "activation",
                    "--no-zscore-runs", "--no-match-bandpass")
        f = ["P0", "P1"]
        assert (D.model_hash(D.fit_meta(on, "-1", f), f)
                != D.model_hash(D.fit_meta(off, "-1", f), f))

    def test_a_different_band_is_a_different_model(self):
        a = parse("--atlas", "yeo7", "--source", "activation")
        b = parse("--atlas", "yeo7", "--source", "activation",
                  "--match-bandpass", "0.01", "0.2")
        f = ["P0", "P1"]
        assert (D.model_hash(D.fit_meta(a, "-1", f), f)
                != D.model_hash(D.fit_meta(b, "-1", f), f))


class TestCensorPolicyIsNotForgettable:
    def test_omitting_it_is_an_error(self):
        # It used to default to None and print a warning. A warning in a
        # 30-line cluster log is not a guard: window_s=-1 was built uncensored
        # while every other aperture used `motion`, and nothing stopped it.
        p = argparse.ArgumentParser()
        D.add_arguments(p)
        with pytest.raises(SystemExit):
            p.parse_args(["--atlas", "yeo7", "--window-s", "30"])

    def test_none_is_how_you_opt_out(self, tmp_path):
        a = parse("--atlas", "yeo7", "--window-s", "30", "--dry-run",
                  "--censor-policy", "none")
        a.output_root = str(tmp_path)
        D.run(a)
        assert a.censor_policy is None       # normalised, so fit_meta is unchanged

    @pytest.mark.parametrize("spelling", ["none", "NONE", "  none  "])
    def test_opting_out_normalises_at_parse_time(self, spelling):
        # Not in run(): fit_meta is reachable without it, and the literal string
        # "none" in the payload is a DIFFERENT model_hash from None -- so an
        # uncensored fit would stop matching every uncensored file on disk for no
        # reason but spelling.
        a = parse("--atlas", "yeo7", "--window-s", "30",
                  "--censor-policy", spelling)
        assert a.censor_policy is None
        assert D.fit_meta(a, "30", EDGES)["censor_policy"] is None
        assert D.model_hash(D.fit_meta(a, "30", EDGES), EDGES) == \
            TestFitMetaIsHashStable.DFC_HASH

    def test_a_named_policy_is_a_different_model(self):
        a = parse("--atlas", "yeo7", "--window-s", "30")
        b = parse("--atlas", "yeo7", "--window-s", "30",
                  "--censor-policy", "motion")
        assert (D.model_hash(D.fit_meta(a, "30", EDGES), EDGES)
                != D.model_hash(D.fit_meta(b, "30", EDGES), EDGES))


class TestApertureGuards:
    def run(self, a, tmp_path):
        a.output_root = str(tmp_path)
        return D.run(a)

    def test_the_activation_source_names_its_own_aperture(self, tmp_path):
        a = parse("--atlas", "yeo7", "--source", "activation", "--dry-run")
        self.run(a, tmp_path)
        assert a.window_s == [D.ACTIVATION_WINDOW]

    def test_a_window_the_activation_source_cannot_honour_is_refused(self, tmp_path):
        # Silently overriding it would write the fit to a path nobody asked for.
        a = parse("--atlas", "yeo7", "--source", "activation",
                  "--window-s", "30")
        with pytest.raises(SystemExit, match="one aperture"):
            self.run(a, tmp_path)

    def test_minus_one_is_accepted_explicitly(self, tmp_path):
        a = parse("--atlas", "yeo7", "--source", "activation",
                  "--window-s", "-1", "--dry-run")
        self.run(a, tmp_path)
        assert a.window_s == ["-1"]

    def test_the_dfc_source_still_requires_a_window(self, tmp_path):
        a = parse("--atlas", "yeo7")
        with pytest.raises(SystemExit, match="--window-s is required"):
            self.run(a, tmp_path)


class TestSourceDispatch:
    def test_source_of_defaults_to_dfc_for_an_args_without_the_flag(self):
        assert D.source_of(argparse.Namespace()) == "dfc"

    def test_activation_shards_are_looked_for_under_activation(self, tmp_path):
        (tmp_path / "activation" / "atlas=yeo7" / "cohort=c" / "task=movie"
         / "sub=01").mkdir(parents=True)
        (tmp_path / "activation" / "atlas=yeo7" / "cohort=c" / "task=movie"
         / "sub=01" / "data.parquet").write_bytes(b"")
        found = D.source_shard_paths(tmp_path, "yeo7", "-1", "c", "activation")
        assert len(found) == 1

    def test_dfc_shards_are_not_found_by_the_activation_source(self, tmp_path):
        (tmp_path / "dfc" / "atlas=yeo7" / "window_s=30" / "cohort=c"
         / "task=movie" / "sub=01").mkdir(parents=True)
        (tmp_path / "dfc" / "atlas=yeo7" / "window_s=30" / "cohort=c"
         / "task=movie" / "sub=01" / "data.parquet").write_bytes(b"")
        assert D.source_shard_paths(tmp_path, "yeo7", "30", "c", "dfc")
        assert not D.source_shard_paths(tmp_path, "yeo7", "-1", "c",
                                        "activation")


class TestPassthroughFeatures:
    """`--passthrough-features` writes the input features beside the PCs.

    The point is interpretability: a state mean over `raw/AM .. raw/WM` reads as
    a network pattern, where the same mean over `pca0/14 .. pca13/14` does not.
    """

    def models(self, n_feat=6, n_latents=("3",), extra=()):
        """-> (args, PRISTINE X, feature names, models).

        The copy matters: `fit_models` scales with `StandardScaler(copy=False)`,
        so the matrix it is handed comes back already standardised. Production
        never notices, because `latents_for` is called per cohort on freshly
        read rows -- but a test that reuses the training matrix would be scaling
        twice and comparing the wrong thing.
        """
        import numpy as np

        a = parse("--atlas", "networks", "--source", "activation",
                  "--pca-latents", *n_latents, "--no-umap", *extra)
        X = np.random.default_rng(0).normal(size=(80, n_feat)).astype("float32")
        feats = [f"net{i}" for i in range(n_feat)]
        pristine = X.copy()
        models = D.fit_models(X, feats, a, meta={"model_hash": "x"})
        return a, pristine, feats, models

    def frame(self, X, models):
        """`latents_for` also scales in place (copy=False), so it gets a copy
        and the caller keeps its pristine X."""
        import numpy as np
        import pandas as pd

        ident = pd.DataFrame({"cohort": "c", "task": "m", "sub": "01",
                              "window_id": np.arange(len(X))})
        return D.latents_for(ident, X.copy(), models, "train")

    def test_off_by_default(self):
        _, X, _, models = self.models()
        out = self.frame(X, models)
        assert not [c for c in out.columns if c.startswith(D.RAW_PREFIX)]

    def test_on_it_writes_one_column_per_named_feature(self):
        _, X, feats, models = self.models(
            extra=("--passthrough-features",))
        out = self.frame(X, models)
        assert ([c for c in out.columns if c.startswith(D.RAW_PREFIX)]
                == [f"{D.RAW_PREFIX}{f}" for f in feats])

    def test_the_values_are_scaled_not_raw(self):
        """Z, not X. PCA is fitted on Z, so writing Z is what makes raw<N> and
        pca<N> differ by a rotation ONLY -- the whole equivalence argument."""
        import numpy as np

        _, X, feats, models = self.models(
            n_latents=("3", "6"), extra=("--passthrough-features",))
        expected = models["scaler"].transform(X.copy())
        out = self.frame(X, models)
        raw = out[[f"{D.RAW_PREFIX}{f}" for f in feats]].to_numpy(float)
        assert np.allclose(raw, expected, atol=1e-5)
        assert not np.allclose(raw, X, atol=1e-3)

    def test_a_full_rank_pca_is_a_rotation_of_those_columns(self):
        """The claim the raw arm rests on: at n_components == n_features the
        two embeddings are the same space, so an HMM on one is the same model."""
        import numpy as np

        _, X, feats, models = self.models(
            n_latents=("3", "6"), extra=("--passthrough-features",))
        out = self.frame(X, models)
        raw = out[[f"{D.RAW_PREFIX}{f}" for f in feats]].to_numpy(float)
        pcs = out[[f"pca{j}/6" for j in range(6)]].to_numpy(float)
        back = models["pca"][6].inverse_transform(pcs)
        assert np.abs(back - raw).max() < 1e-4

    def test_it_is_refused_above_the_feature_cap(self):
        from fmri_decomposition.io import PASSTHROUGH_MAX_FEATURES

        with pytest.raises(SystemExit) as e:
            self.models(n_feat=PASSTHROUGH_MAX_FEATURES + 1,
                        extra=("--passthrough-features",))
        msg = str(e.value)
        assert str(PASSTHROUGH_MAX_FEATURES) in msg
        assert "--source activation" in msg

    def test_asking_for_more_components_than_features_names_the_atlas_width(self):
        with pytest.raises(SystemExit) as e:
            self.models(n_feat=7, n_latents=("3", "14"))
        assert "only 7 feature" in str(e.value)
        assert "--n-latents 3 7" in str(e.value)


class TestNewFlagsDoNotMoveExistingHashes:
    """`fit_meta` extends its payload only where a value differs from the
    default -- the same rule `source` follows -- so adding these leaves every
    hash already stamped on disk alone."""

    def test_the_frozen_dfc_hash_still_holds(self):
        a = parse("--atlas", "yeo7", "--window-s", "30")
        assert (D.model_hash(D.fit_meta(a, "30", EDGES), EDGES)
                == TestFitMetaIsHashStable.DFC_HASH)

    def test_passthrough_absent_means_absent_from_the_payload(self):
        a = parse("--atlas", "yeo7", "--window-s", "30")
        assert "passthrough_features" not in D.fit_meta(a, "30", EDGES)

    def test_umap_latents_matching_n_latents_is_absent_from_the_payload(self):
        a = parse("--atlas", "yeo7", "--window-s", "30",
                  "--pca-latents", "2", "3", "--umap-latents", "2", "3")
        assert "umap_latents" not in D.fit_meta(a, "30", EDGES)

    def test_but_each_one_changes_the_hash_when_it_differs(self):
        base = parse("--atlas", "yeo7", "--window-s", "30")
        pt = parse("--atlas", "yeo7", "--window-s", "30",
                   "--passthrough-features")
        ul = parse("--atlas", "yeo7", "--window-s", "30",
                   "--pca-latents", "2", "3", "--umap-latents", "3")
        h = D.model_hash(D.fit_meta(base, "30", EDGES), EDGES)
        assert D.model_hash(D.fit_meta(pt, "30", EDGES), EDGES) != h
        assert D.model_hash(D.fit_meta(ul, "30", EDGES), EDGES) != h

    def test_umap_is_fitted_only_at_the_counts_asked_for(self):
        import numpy as np

        a = parse("--atlas", "yeo7", "--window-s", "30",
                  "--pca-latents", "3", "5", "--umap-latents", "3")
        assert D._umap_latents(a) == [3]
        b = parse("--atlas", "yeo7", "--window-s", "30", "--pca-latents", "3", "5")
        assert D._umap_latents(b) == [3, 5]
        assert isinstance(np.float32(1), np.floating)   # imports used above
