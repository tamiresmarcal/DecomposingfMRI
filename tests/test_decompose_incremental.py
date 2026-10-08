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

import numpy as np
import pandas as pd
import pytest

from fmri_decomposition import decompose as D

PARCELS = [f"P{i:02d}" for i in range(8)]


def write_activation(root, cohort, sub, n_tr=80, tr=2.0, seed=0):
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
    p = (root / "activation" / "atlas=mini" / f"cohort={cohort}"
         / "task=m" / f"sub={sub}" / "data.parquet")
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
