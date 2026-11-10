"""The three output halves, the version key, and what a version claims.

WHY THIS FILE EXISTS. `outputs/` used to be one flat level of tree names, and
the shape of it was a fact repeated in about forty places -- `root /
"transitions"` written inline in the package, in the notebooks' modules and in
tools/. Splitting it into extractions / modelling(version) exposed every one of
those as an assertion nobody had written down.

What is pinned here is the RULE, not the current spelling: a tree whose content
does not depend on which cohort plays which role lives outside every version,
and one whose content does lives inside one. Get that backwards and either a
version duplicates tens of GB of identical parcel timeseries, or two versions
overwrite each other's fits.
"""
import json
import os

import pytest

from fmri_decomposition import io


@pytest.fixture(autouse=True)
def _unpinned(monkeypatch):
    """No version pinned, unless a test pins one.

    `announce_version` writes the resolved version into the environment on
    purpose (see its own test), which makes it process-global -- so without
    this, one test that announces a version changes what every later test in
    the same process resolves. That is worth stating rather than relying on
    test order.
    """
    monkeypatch.delenv("FMRIDECOMP_MODELLING_VERSION", raising=False)


class TestWhichHalfEachTreeIsIn:
    """The split is by one question: does a role change change the bytes?"""

    def test_an_extraction_is_outside_every_version(self):
        for tree in io.EXTRACTION_TREES:
            p = io.tree_root("/o", tree).as_posix()
            assert p == f"/o/extractions/{tree}"
            assert "version=" not in p

    def test_a_fitted_tree_is_inside_a_version(self):
        for tree in io.UNSUPERVISED_TREES:
            assert io.tree_root("/o", tree).as_posix() == (
                f"/o/modelling/version=v0/unsupervised_model/{tree}")
        for tree in io.SUPERVISED_TREES:
            assert io.tree_root("/o", tree).as_posix() == (
                f"/o/modelling/version=v0/supervised_model/{tree}")

    def test_the_version_moves_only_the_fitted_trees(self):
        """The point of the split, as one assertion."""
        before = {t: io.tree_root("/o", t) for t in io.EXTRACTION_TREES}
        after = {t: io.tree_root("/o", t, "v7") for t in io.EXTRACTION_TREES}
        assert before == after
        assert io.tree_root("/o", "latents") != io.tree_root("/o", "latents", "v7")

    def test_an_unknown_tree_is_named_rather_than_guessed(self):
        with pytest.raises(SystemExit) as e:
            io.tree_root("/o", "trnasitions")
        assert "trnasitions" in str(e.value) and "transitions" in str(e.value)


class TestTheVersionIsAHiveKey:
    """Not the flat name `modelling_v0`, and that is the whole reason."""

    def test_every_version_is_one_glob(self, tmp_path):
        for v in ("v0", "v1", "v2"):
            io.tree_root(tmp_path, "latents", v).mkdir(parents=True)
        got = sorted(p.name for p in
                     (tmp_path / "modelling").glob("version=*"))
        assert got == ["version=v0", "version=v1", "version=v2"]

    def test_hive_key_reads_it_with_no_new_parsing(self, tmp_path):
        p = io.tree_root(tmp_path, "transitions", "v3") / "atlas=yeo7"
        assert io.hive_key(p, "version") == "v3"

    def test_a_value_that_cannot_be_a_directory_is_refused(self):
        for bad in ("", "  ", "a/b", "k=v"):
            with pytest.raises(SystemExit) as e:
                io.modelling_version(bad)
            assert "hive value" in str(e.value)


class TestWhereTheVersionComesFrom:
    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("FMRIDECOMP_MODELLING_VERSION", "v9")
        assert io.modelling_version("v1") == "v1"

    def test_then_the_environment(self, monkeypatch):
        monkeypatch.setenv("FMRIDECOMP_MODELLING_VERSION", "v9")
        assert io.modelling_version() == "v9"

    def test_then_v0(self, monkeypatch):
        monkeypatch.delenv("FMRIDECOMP_MODELLING_VERSION", raising=False)
        assert io.modelling_version() == io.DEFAULT_MODELLING_VERSION == "v0"

    def test_announcing_pins_it_for_the_rest_of_the_process(self, tmp_path,
                                                            monkeypatch):
        """One missed call site would otherwise read a different version.

        The path builders resolve the version per call, so the only alternative
        is threading `version=` through every internal function -- and a single
        miss then reads the wrong tree silently. Pinning it also reaches
        joblib's loky workers, which re-read the environment on import.
        """
        monkeypatch.delenv("FMRIDECOMP_MODELLING_VERSION", raising=False)
        io.announce_version(tmp_path, "v5", log=lambda *_: None)
        assert os.environ["FMRIDECOMP_MODELLING_VERSION"] == "v5"
        assert "version=v5" in io.latents_root(tmp_path, "yeo7", -1).as_posix()


class TestMetaFollowsItsData:
    """A fitted clusterer and a cohort's stage-2 manifest are not the same kind
    of fact, and they used to share one `outputs/meta/`."""

    def test_the_three_stages_land_in_three_places(self):
        assert io.meta_dir("/o", "extractions").as_posix() == "/o/extractions/meta"
        assert io.meta_dir("/o", "unsupervised").as_posix() == (
            "/o/modelling/version=v0/unsupervised_model/meta")
        assert io.meta_dir("/o", "supervised").as_posix() == (
            "/o/modelling/version=v0/supervised_model/meta")

    def test_every_meta_subtree_has_a_declared_stage(self):
        """So a caller holding a NAME never has to decide again."""
        assert set(io.META_STAGE.values()) == {"extractions", "unsupervised",
                                               "supervised"}
        for tree in io.SUPERVISED_TREES:
            assert io.META_STAGE[tree] == "supervised"

    def test_a_bad_stage_is_refused_and_points_at_the_map(self):
        with pytest.raises(SystemExit) as e:
            io.meta_dir("/o", "modelling")
        assert "META_STAGE" in str(e.value)


class TestAVersionRecordsWhatItIs:
    """`version=v0` is a directory name, and a name is not a checkable claim."""

    def test_the_first_run_claims_it(self, tmp_path):
        io.claim_version(tmp_path, "v0", unsupervised_train=["a", "b"])
        man = json.loads(io.version_manifest_path(tmp_path, "v0").read_text())
        assert man["unsupervised_train"] == ["a", "b"]
        assert man["version"] == "v0"

    def test_each_half_fills_its_own_and_keeps_the_other(self, tmp_path):
        io.claim_version(tmp_path, "v0", unsupervised_train=["a"])
        man = io.claim_version(tmp_path, "v0", supervised_train="camcan")
        assert man["unsupervised_train"] == ["a"]
        assert man["supervised_train"] == ["camcan"]

    def test_the_same_composition_again_is_fine(self, tmp_path):
        io.claim_version(tmp_path, "v0", supervised_train="camcan")
        assert io.claim_version(tmp_path, "v0",
                                supervised_train="camcan")["version"] == "v0"

    def test_a_different_composition_is_refused_with_both_printed(self, tmp_path):
        io.claim_version(tmp_path, "v0", supervised_train="camcan",
                         supervised_test="hcp7t_fix")
        with pytest.raises(SystemExit) as e:
            io.claim_version(tmp_path, "v0", supervised_train="hcp7t_fix")
        msg = str(e.value)
        assert "camcan" in msg and "hcp7t_fix" in msg
        assert "--modelling-version" in msg
        # and it did NOT overwrite the record it refused
        man = io.read_version_manifest(tmp_path, "v0")
        assert man["supervised_train"] == ["camcan"]

    def test_a_swap_of_roles_is_a_different_version_not_a_clash(self, tmp_path):
        """Her v2: Cam-CAN and HCP change places. A new version, and the state
        space it reuses is NAMED rather than copied."""
        io.claim_version(tmp_path, "v0", supervised_train="camcan",
                         supervised_test="hcp7t_fix")
        man = io.claim_version(tmp_path, "v2", supervised_train="hcp7t_fix",
                               supervised_test="camcan", unsupervised_from="v0")
        assert man["unsupervised_from"] == ["v0"]
        assert io.read_version_manifest(tmp_path, "v0")["supervised_train"] == ["camcan"]

    def test_a_knob_is_not_a_version(self, tmp_path):
        with pytest.raises(SystemExit) as e:
            io.claim_version(tmp_path, "v0", atlas="yeo7")
        assert "atlas" in str(e.value) and "knob INSIDE a version" in str(e.value)

    def test_verifying_without_writing_writes_nothing(self, tmp_path):
        io.claim_version(tmp_path, "v0", supervised_train="camcan", write=False)
        assert io.read_version_manifest(tmp_path, "v0") is None
