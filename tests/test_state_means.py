"""What a brain state looks like, in units that have names.

The whole argument for routing the raw arm THROUGH `decompose` at full rank --
rather than around it, duplicating the censor and role logic -- is that the
rotation is invertible and the fitted objects are on disk. These tests pin that:
a state mean recovered from `pca<N>` and the same state mean read off `raw<N>`
have to agree, or the shortcut was the right call after all.
"""
import json

import numpy as np
import pandas as pd
import pytest

from fmri_decomposition import state_means as S
from fmri_decomposition.io import RAW_PREFIX

FEATURES = ["AM", "CogAC", "Motor", "ToM", "VigAtt", "WM"]


def build(tmp_path, n_latents=(3, 6), passthrough=True, n_states=3, seed=0):
    """A complete cell: scaled latents, raw columns, a state column, and the
    saved models -- assembled the way `decompose` and `cluster` assemble it."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    n = 600
    centres = rng.normal(scale=4.0, size=(n_states, len(FEATURES)))
    labels = rng.integers(0, n_states, size=n)
    X = centres[labels] + rng.normal(size=(n, len(FEATURES)))

    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    models = {"scaler": scaler, "edges": list(FEATURES), "umap": {},
              "pca": {k: PCA(n_components=k, random_state=0).fit(Z)
                      for k in n_latents},
              "passthrough": passthrough}

    d = pd.DataFrame({"cohort": "a", "task": "m",
                      "sub": np.repeat([f"{i:02d}" for i in range(6)], n // 6),
                      "window_id": np.tile(np.arange(n // 6), 6)})
    for k in n_latents:
        arr = models["pca"][k].transform(Z)
        for j in range(k):
            d[f"pca{j}/{k}"] = arr[:, j].astype(np.float32)
    if passthrough:
        for j, f in enumerate(FEATURES):
            d[f"{RAW_PREFIX}{f}"] = Z[:, j].astype(np.float32)
    d["HMM2_pca6_3"] = labels.astype(np.int32)
    d["HMM2_raw6_3"] = labels.astype(np.int32)
    d["HMM2_pca3_3"] = labels.astype(np.int32)
    d["HMM2_umap3_3"] = labels.astype(np.int32)

    clusterers = {
        "HMM2_pca6_3": {"embedding": "pca6",
                        "embedding_columns": [f"pca{j}/6" for j in range(6)]},
        "HMM2_pca3_3": {"embedding": "pca3",
                        "embedding_columns": [f"pca{j}/3" for j in range(3)]},
        "HMM2_umap3_3": {"embedding": "umap3",
                         "embedding_columns": [f"umap{j}/3" for j in range(3)]},
        "HMM2_raw6_3": {"embedding": "raw6",
                        "embedding_columns": sorted(
                            f"{RAW_PREFIX}{f}" for f in FEATURES)},
        "HMM2_noprov_3": {"embedding": "pca3"},
    }
    table = pa.Table.from_pandas(d, preserve_index=False)
    table = table.replace_schema_metadata(
        {b"clusterers": json.dumps(clusterers).encode()})
    p = (tmp_path / "latents" / "atlas=networks" / "window_s=-1"
         / "cohort=a" / "data.parquet")
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, p)

    mdir = tmp_path / "meta" / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    import pickle
    (mdir / "decompose_atlas-networks_window--1.pkl").write_bytes(
        pickle.dumps(models))
    return centres, labels


class TestRoundTrip:
    def test_a_full_rank_pca_state_mean_comes_back_in_feature_units(self, tmp_path):
        centres, labels = build(tmp_path)
        out = S.means_in_feature_units(tmp_path, "networks", "-1", "HMM2_pca6_3")
        assert list(out.columns[2:]) == FEATURES
        # Each state's recovered mean is its generating centre, up to the noise
        # averaged over ~200 rows.
        for st in range(3):
            got = out.loc[st, FEATURES].to_numpy(float)
            assert np.abs(got - centres[st]).max() < 0.4, st

    def test_pca_and_raw_agree_on_the_same_states(self, tmp_path):
        """The equivalence the whole design rests on. If these disagreed, going
        through `decompose` would have been the wrong route."""
        build(tmp_path)
        a = S.means_in_feature_units(tmp_path, "networks", "-1", "HMM2_pca6_3")
        b = S.means_in_feature_units(tmp_path, "networks", "-1", "HMM2_raw6_3")
        assert np.allclose(a[FEATURES].to_numpy(float),
                           b[FEATURES].to_numpy(float), atol=1e-4)

    def test_full_rank_is_not_marked_lossy(self, tmp_path):
        build(tmp_path)
        out = S.means_in_feature_units(tmp_path, "networks", "-1", "HMM2_pca6_3")
        assert not out.attrs.get("lossy")

    def test_occupancy_is_reported_beside_the_means(self, tmp_path):
        build(tmp_path)
        out = S.means_in_feature_units(tmp_path, "networks", "-1", "HMM2_pca6_3")
        assert out["n_rows"].sum() == 600
        assert out["share"].sum() == pytest.approx(1.0)


class TestHonestyAboutWhatCannotBeRecovered:
    def test_a_reduced_pca_is_marked_lossy(self, tmp_path):
        """3 of 6 components inverted back to 6 parcels is the state's position
        in a 3-D subspace, not its mean over the parcels. Saying so is the
        difference between a figure and a misleading figure."""
        build(tmp_path)
        out = S.means_in_feature_units(tmp_path, "networks", "-1", "HMM2_pca3_3")
        assert out.attrs["lossy"]
        assert "3 of 6" in out.attrs["lossy"]

    def test_umap_is_refused_rather_than_approximated(self, tmp_path):
        build(tmp_path)
        with pytest.raises(SystemExit) as e:
            S.means_in_feature_units(tmp_path, "networks", "-1", "HMM2_umap3_3")
        assert "inverse_transform" in str(e.value)

    def test_a_column_with_no_provenance_names_the_fix(self, tmp_path):
        build(tmp_path)
        with pytest.raises(SystemExit) as e:
            S.means_in_feature_units(tmp_path, "networks", "-1", "HMM2_noprov_3")
        assert "embedding_columns" in str(e.value)
        assert "Re-run `cluster`" in str(e.value)

    def test_an_unknown_state_column_lists_what_is_there(self, tmp_path):
        build(tmp_path)
        with pytest.raises(SystemExit) as e:
            S.means_in_feature_units(tmp_path, "networks", "-1", "HMM9_pca3_3")
        assert "HMM2_pca6_3" in str(e.value)

    def test_missing_models_name_the_path_and_the_stage(self, tmp_path):
        build(tmp_path)
        for p in (tmp_path / "meta" / "models").iterdir():
            p.unlink()
        with pytest.raises(SystemExit) as e:
            S.means_in_feature_units(tmp_path, "networks", "-1", "HMM2_pca6_3")
        assert "decompose" in str(e.value)

    def test_an_absent_cohort_is_named(self, tmp_path):
        build(tmp_path)
        with pytest.raises(SystemExit) as e:
            S.means_in_feature_units(tmp_path, "networks", "-1", "HMM2_pca6_3",
                                     cohorts=["nope"])
        assert "nope" in str(e.value)
