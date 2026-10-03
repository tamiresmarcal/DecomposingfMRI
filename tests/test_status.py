"""The read-only inventory, and the disagreements it exists to surface.

Each problem it reports is a state that LOOKS like success -- every file
present, every column there -- and is quietly not the analysis anyone intended.
Those are the ones no single stage's own --check can see, because each of them
spans two stages or two cohorts.
"""
import argparse
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fmri_decomposition import status as S


def args(**kw):
    d = dict(atlas=None, window_s=None, embeddings=["pca3"], min_k=2, max_k=64,
             output_root=None)
    d.update(kw)
    return argparse.Namespace(**d)


def latents(root, atlas="yeo7", w="30", cohorts=("a", "b"), states=(),
            emb=("pca0/3",), censor="motion", model_hash="h1", source="dfc"):
    for c in cohorts:
        d = pd.DataFrame({"cohort": c, "task": "m", "sub": "01",
                          "window_id": range(4)})
        for e in emb:
            d[e] = 0.0
        for st in states:
            d[st] = 0
        p = (root / "latents" / f"atlas={atlas}" / f"window_s={w}"
             / f"cohort={c}" / "data.parquet")
        p.parent.mkdir(parents=True, exist_ok=True)
        md = {b"censor_policy": json.dumps(censor).encode(),
              b"model_hash": json.dumps(model_hash).encode(),
              b"source": json.dumps(source).encode(),
              b"role": json.dumps("train").encode()}
        pq.write_table(pa.Table.from_pandas(d, preserve_index=False)
                         .replace_schema_metadata(md), p)


def trans(root, atlas="yeo7", w="30", state="HMM_pca3_8", cohorts=("a", "b")):
    for c in cohorts:
        d = pd.DataFrame({"task": "m", "sub": ["01", "02"],
                          "0->0": 1.0, "0->1": 0.0, "n_states": 8})
        p = (root / "transitions" / f"atlas={atlas}" / f"window_s={w}"
             / f"states={state}" / f"cohort={c}" / "subjects.parquet")
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(d, preserve_index=False), p)


class TestInventory:
    def test_counts_activation_shards_per_cohort(self, tmp_path):
        for c, n in (("a", 3), ("b", 5)):
            for i in range(n):
                p = (tmp_path / "activation" / "atlas=yeo7" / f"cohort={c}"
                     / "task=m" / f"sub={i:02d}" / "data.parquet")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"")
        got = S.activation(tmp_path).set_index("cohort")["shards"].to_dict()
        assert got == {"a": 3, "b": 5}

    def test_reads_latents_metadata_without_reading_rows(self, tmp_path):
        latents(tmp_path, states=["HMM_pca3_8"])
        d = S.latents(tmp_path)
        assert set(d["cohort"]) == {"a", "b"}
        assert d["censor"].unique().tolist() == ["motion"]
        assert d["n_states"].unique().tolist() == [1]

    def test_censor_says_subjects_only_when_there_is_no_window_gate(self, tmp_path):
        p = tmp_path / "censor" / "policy=motion" / "cohort=a" / "subjects.parquet"
        p.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pandas(
            pd.DataFrame({"keep": [True, True, False]}), preserve_index=False), p)
        d = S.censor(tmp_path)
        assert d["subjects_kept"].iloc[0] == 2 and d["of"].iloc[0] == 3
        assert d["window_gate"].iloc[0] == "subjects only"


class TestProblems:
    def test_cohorts_from_different_fits(self, tmp_path):
        # The one that silently destroys everything downstream: state 5 is only
        # the same state in two cohorts if one fit defined it.
        latents(tmp_path, cohorts=("a",), model_hash="h1", states=["HMM_pca3_8"])
        latents(tmp_path, cohorts=("b",), model_hash="h2", states=["HMM_pca3_8"])
        probs = S.problems(S.latents(tmp_path), args())
        assert any("DIFFERENT fits" in p for p in probs)

    def test_cohorts_disagreeing_on_the_censor_policy(self, tmp_path):
        latents(tmp_path, cohorts=("a",), censor="motion", states=["HMM_pca3_8"])
        latents(tmp_path, cohorts=("b",), censor=None, states=["HMM_pca3_8"])
        probs = S.problems(S.latents(tmp_path), args())
        assert any("disagree on the censor policy" in p for p in probs)

    def test_the_grid_spanning_two_policies(self, tmp_path):
        # Each aperture internally consistent, still not comparable.
        latents(tmp_path, w="30", censor="motion", states=["HMM_pca3_8"])
        latents(tmp_path, w="-1", censor=None, states=["HMM_pca3_8"])
        probs = S.problems(S.latents(tmp_path), args())
        assert any("SPANS 2 CENSOR POLICIES" in p for p in probs)

    def test_a_missing_embedding(self, tmp_path):
        latents(tmp_path, emb=("pca0/3",), states=["HMM_pca3_8"])
        probs = S.problems(S.latents(tmp_path), args(embeddings=["pca3", "umap3"]))
        assert any("umap3" in p for p in probs)

    def test_a_state_set_at_an_unusable_k(self, tmp_path):
        latents(tmp_path, states=["MeanShift_pca3_1", "HMM_pca3_8"])
        probs = S.problems(S.latents(tmp_path), args())
        assert any("MeanShift_pca3_1" in p for p in probs)

    def test_a_consistent_tree_reports_nothing(self, tmp_path):
        latents(tmp_path, states=["HMM_pca3_8"])
        assert S.problems(S.latents(tmp_path), args()) == []


class TestStale:
    def test_a_table_for_a_state_set_that_no_longer_exists(self, tmp_path):
        # Stage 3 re-run after stage 4: the columns are gone, the tables are not.
        latents(tmp_path, states=[])
        trans(tmp_path, state="HMM_pca3_8")
        out = S.stale(S.latents(tmp_path), S.transitions(tmp_path))
        assert len(out) == 1 and "HMM_pca3_8" in out[0]

    def test_a_k_free_method_changing_its_k_leaves_a_stale_table(self, tmp_path):
        # MeanShift_pca3_7 -> MeanShift_pca3_4 after a bandwidth change. The old
        # table keeps its own `states=` directory and looks current.
        latents(tmp_path, states=["MeanShift_pca3_4"])
        trans(tmp_path, state="MeanShift_pca3_7")
        trans(tmp_path, state="MeanShift_pca3_4")
        out = S.stale(S.latents(tmp_path), S.transitions(tmp_path))
        assert len(out) == 1
        assert "MeanShift_pca3_7" in out[0] and "MeanShift_pca3_4" not in out[0]

    def test_tables_matching_the_latents_are_not_stale(self, tmp_path):
        latents(tmp_path, states=["HMM_pca3_8"])
        trans(tmp_path, state="HMM_pca3_8")
        assert S.stale(S.latents(tmp_path), S.transitions(tmp_path)) == []

    def test_nothing_to_say_when_a_stage_has_not_run(self, tmp_path):
        latents(tmp_path, states=["HMM_pca3_8"])
        assert S.stale(S.latents(tmp_path), S.transitions(tmp_path)) == []
        assert S.stale(pd.DataFrame(), pd.DataFrame()) == []


class TestBehind:
    """The mirror of stale, and the more common state: stage 4b is cheap and gets
    re-run, stage 5a is a separate job and gets forgotten. Stage 6 then picks a
    winner among whatever has tables, without mentioning what it never saw."""

    def test_state_sets_with_no_transition_table(self, tmp_path):
        latents(tmp_path, states=["HMM_pca3_8", "MeanShift_pca3_4"])
        trans(tmp_path, state="HMM_pca3_8")
        out = S.behind(S.latents(tmp_path), S.transitions(tmp_path))
        assert len(out) == 1
        assert "MeanShift_pca3_4" in out[0] and "HMM_pca3_8" not in out[0]

    def test_no_transitions_at_all_lists_every_state_set(self, tmp_path):
        latents(tmp_path, states=["HMM_pca3_8", "MeanShift_pca3_4"])
        out = S.behind(S.latents(tmp_path), S.transitions(tmp_path))
        assert len(out) == 1 and "2 state set(s)" in out[0]

    def test_a_caught_up_cell_says_nothing(self, tmp_path):
        latents(tmp_path, states=["HMM_pca3_8"])
        trans(tmp_path, state="HMM_pca3_8")
        assert S.behind(S.latents(tmp_path), S.transitions(tmp_path)) == []


class TestShardGap:
    def test_fewer_dfc_shards_than_activation(self, tmp_path):
        act = pd.DataFrame([{"atlas": "yeo7", "cohort": "camcan", "shards": 648}])
        dfcs = pd.DataFrame([{"atlas": "yeo7", "window_s": "30",
                              "cohort": "camcan", "shards": 647}])
        out = S.shard_gap(act, dfcs)
        assert len(out) == 1 and "1 subject-task(s)" in out[0]

    def test_an_aperture_not_run_at_all_is_not_a_gap(self, tmp_path):
        # 0 means stage 3 was never run for it, which is a different thing and
        # already visible in the DFC table.
        act = pd.DataFrame([{"atlas": "yeo7", "cohort": "camcan", "shards": 648}])
        dfcs = pd.DataFrame([{"atlas": "yeo7", "window_s": "15",
                              "cohort": "camcan", "shards": 0}])
        assert S.shard_gap(act, dfcs) == []

    def test_matching_counts_say_nothing(self, tmp_path):
        act = pd.DataFrame([{"atlas": "yeo7", "cohort": "a", "shards": 86}])
        dfcs = pd.DataFrame([{"atlas": "yeo7", "window_s": "30",
                              "cohort": "a", "shards": 86}])
        assert S.shard_gap(act, dfcs) == []


class TestTransitionPeopleCount:
    def test_rows_and_people_are_reported_separately(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        # 6 rows, 2 people, 3 tasks each -- cneuromod's shape in miniature.
        d = pd.DataFrame({"task": ["e1", "e2", "e3"] * 2,
                          "sub": ["s1"] * 3 + ["s2"] * 3,
                          "0->0": 1.0, "n_states": 8})
        p = (tmp_path / "transitions" / "atlas=yeo7" / "window_s=30"
             / "states=HMM_pca3_8" / "cohort=cneuromod" / "subjects.parquet")
        p.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pandas(d, preserve_index=False), p)
        got = S.transitions(tmp_path).iloc[0]
        assert got["rows"] == 6 and got["subs"] == 2

    def test_shallow_skips_the_people_count(self, tmp_path):
        trans(tmp_path, state="HMM_pca3_8")
        assert S.transitions(tmp_path, deep=False)["subs"].isna().all()
