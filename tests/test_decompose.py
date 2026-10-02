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
    p = argparse.ArgumentParser()
    D.add_arguments(p)
    a = p.parse_args(list(argv))
    a.output_root = None                     # skips the censor-summary lookup
    return a


EDGES = ["A__B", "A__C", "B__C"]


class TestFitMetaIsHashStable:
    # Frozen on purpose, and verified against the commit BEFORE the activation
    # source existed rather than simply recorded from the code as it now stands.
    # If this value changes, every dfc latents file on disk stops matching a
    # re-run of the command that wrote it, and nothing would report that -- it
    # would look like two incomparable fits. So a change here has to be
    # deliberate: update the constant in the same commit and say why.
    DFC_HASH = "433fca34406aafde"

    def test_the_default_payload_is_unchanged(self):
        a = parse("--atlas", "yeo7", "--window-s", "30")
        assert D.model_hash(D.fit_meta(a, "30", EDGES), EDGES) == self.DFC_HASH

    def test_the_dfc_payload_carries_no_source_keys(self):
        a = parse("--atlas", "yeo7", "--window-s", "30")
        meta = D.fit_meta(a, "30", EDGES)
        for k in ("source", "match_bandpass", "zscore_runs"):
            assert k not in meta

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
