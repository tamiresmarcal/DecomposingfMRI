"""Neurosynth profiles counted in the states' own parcels with NiMARE's
BrainMap decoder: a two-parcel toy grid, fake studies with known peaks.

The table must hold, per parcel, NiMARE's zForward for the studies with a
peak in that parcel, and painting it back with the same labels must give
each parcel's value exactly.
"""
import gzip

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

nimare = pytest.importorskip("nimare")
from nimare.decode.discrete import brainmap_decode  # noqa: E402

from fmri_decomposition.unsupervised_model.contextualization import refmaps as rm  # noqa: E402

N = 40


def _fake_neurosynth(d):
    nsd = d / "neurosynth"
    nsd.mkdir()
    pre = nsd / "data-neurosynth_version-7_"
    rng = np.random.default_rng(0)
    ids = np.arange(1, N + 1)
    vision = ids <= 20                          # half the studies are "vision"
    # vision studies mostly peak in parcel 1 (x=2), the rest in parcel 2 (x=12)
    in_p1 = np.where(vision, rng.random(N) < 0.8, rng.random(N) < 0.2)
    coords = pd.DataFrame({"id": ids, "x": np.where(in_p1, 2, 12), "y": 2, "z": 2})
    meta = pd.DataFrame({"id": ids, "space": "MNI"})
    for name, t in (("coordinates", coords), ("metadata", meta)):
        with gzip.open(f"{pre}{name}.tsv.gz", "wt") as f:
            t.to_csv(f, sep="\t", index=False)
    feats = sp.csr_matrix(np.c_[np.where(vision, 0.5, 0.0), np.where(~vision, 0.5, 0.0)])
    sp.save_npz(f"{pre}vocab-terms_source-abstract_type-tfidf_features.npz", feats)
    (nsd / "data-neurosynth_version-7_vocab-terms_vocabulary.txt").write_text("vision\naudition\n")
    return coords, feats


def test_nimare_zforward_per_parcel_and_paints_back(tmp_path, monkeypatch):
    coords, feats = _fake_neurosynth(tmp_path)
    monkeypatch.setitem(rm.TAXONOMIES, "toy", ("toy", {
        "Perception.Vision": ["vision"], "Perception.Audition": ["audition"]}))
    ref = nib.Nifti1Image(np.zeros((10, 4, 4), np.float32), np.diag([2.0, 2, 2, 1]))
    lab = np.zeros((10, 4, 4), int)
    lab[:3] = 1                      # x = 0..4 mm
    lab[5:8] = 2                     # x = 10..14 mm

    t = rm.neurosynth_behavioural_table(tmp_path, ref, "toy", radius_mm=2.0,
                                        labels=lab, key="toyatlas")
    assert list(t["parcel"]) == [1, 2] and "bn_index" not in t
    ann = pd.DataFrame({"id": np.arange(1, N + 1),
                        "Perception.Vision": feats[:, 0].toarray().ravel(),
                        "Perception.Audition": feats[:, 1].toarray().ravel()})
    for r, x in ((1, 2), (2, 12)):
        ids = sorted(coords.loc[coords["x"] == x, "id"])
        want = brainmap_decode(coords[["id"]], ann, ids,
                               features=["Perception.Vision", "Perception.Audition"],
                               correction=None)["zForward"]
        got = t.loc[t["parcel"] == r, want.index].to_numpy().ravel()
        assert np.allclose(got, np.nan_to_num(want.to_numpy(float)))
    # vision is enriched in parcel 1, not in parcel 2
    assert t.loc[0, "Perception.Vision"] > t.loc[1, "Perception.Vision"]
    assert (tmp_path / "neurosynth" / "toyatlas_nimare_toy.csv").exists()

    maps = rm.behavioural_maps(tmp_path, ref, t, labels=lab)
    v = maps.voxels[0].reshape(lab.shape)
    assert np.allclose(v[lab == 1], t.loc[0, maps.names[0]])
    assert np.allclose(v[lab == 2], t.loc[1, maps.names[0]])
    assert np.isnan(v[lab == 0]).all() and not maps.simulated


def test_lancaster_statistic_matches_the_plugin_formula(tmp_path, monkeypatch):
    """statistic='lancaster': foci counted strictly inside parcels, z as in the
    Mango Behavioral Analysis plugin, separate cache from the decoder's."""
    coords, feats = _fake_neurosynth(tmp_path)
    monkeypatch.setitem(rm.TAXONOMIES, "toy", ("toy", {
        "Perception.Vision": ["vision"], "Perception.Audition": ["audition"]}))
    ref = nib.Nifti1Image(np.zeros((10, 4, 4), np.float32), np.diag([2.0, 2, 2, 1]))
    lab = np.zeros((10, 4, 4), int)
    lab[:3] = 1
    lab[5:8] = 2
    t = rm.neurosynth_behavioural_table(tmp_path, ref, "toy", labels=lab, key="toyatlas",
                                        statistic="lancaster")
    assert (tmp_path / "neurosynth" / "toyatlas_lancaster_toy.csv").exists()
    vision = coords["id"] <= 20
    for r, x in ((1, 2), (2, 12)):
        count = (vision & (coords["x"] == x)).sum()
        p, pe, n = count / vision.sum(), 0.5, vision.sum()
        want = (p - pe) / np.sqrt((p * (1 - p) + pe * (1 - pe)) / n)
        assert np.isclose(t.loc[t.parcel == r, "Perception.Vision"].item(), want)
    assert t.loc[0, "Perception.Vision"] > 0 > t.loc[1, "Perception.Vision"]


def test_brainmap_db_columns_keeps_general_categories_before_children():
    cols = rm.brainmap_db_columns({"Action.Execution", "Action.Execution.Speech",
                                   "Perception.Vision", "Perception.Vision.Motion",
                                   "Emotion.Fear"})
    assert cols == ["Action.Execution", "Action.Execution.Speech", "Emotion.Fear",
                    "Perception.Vision", "Perception.Vision.Motion"]


def test_pure_taxonomy_is_one_own_name_per_category():
    tax = rm.TAXONOMIES["brainmap_pure"][1]
    assert all(len(v) == 1 for v in tax.values())
    assert set(tax) <= set(rm.BRAINMAP_TAXONOMY)
    for cat, (term,) in tax.items():
        leaf = cat.split(".")[-1].lower()
        assert term == leaf or term == f"{leaf} {cat.split('.')[-2].lower()}"


def test_lancaster_studies_counts_each_study_once(tmp_path, monkeypatch):
    coords, feats = _fake_neurosynth(tmp_path)
    # study 1 (vision) reports 9 extra peaks in parcel 1: by foci it weighs 10x
    pre = tmp_path / "neurosynth" / "data-neurosynth_version-7_coordinates.tsv.gz"
    extra = pd.DataFrame({"id": 1, "x": [2] * 9, "y": 2, "z": 2})
    with gzip.open(pre, "wt") as f:
        pd.concat([coords, extra]).to_csv(f, sep="\t", index=False)
    monkeypatch.setitem(rm.TAXONOMIES, "toy", ("toy", {"Perception.Vision": ["vision"]}))
    ref = nib.Nifti1Image(np.zeros((10, 4, 4), np.float32), np.diag([2.0, 2, 2, 1]))
    lab = np.zeros((10, 4, 4), int)
    lab[:3] = 1
    lab[5:8] = 2
    t = rm.neurosynth_behavioural_table(tmp_path, ref, "toy", labels=lab, key="toyatlas",
                                        statistic="lancaster_studies")
    vision = coords["id"] <= 20
    count = (vision & (coords["x"] == 2)).sum()           # studies, not foci
    p, pe, n = count / 20, 0.5, 20
    want = (p - pe) / np.sqrt((p * (1 - p) + pe * (1 - pe)) / n)
    assert np.isclose(t.loc[0, "Perception.Vision"], want)
    assert (tmp_path / "neurosynth" / "toyatlas_lancasterstudies_toy.csv").exists()


def test_other_bins_are_shown_as_general_categories():
    t = pd.DataFrame({"parcel": [1, 2], "Action.Execution.Other": [1.0, 2.0],
                      "Perception.Vision.Other": [0.0, 1.0], "Emotion.Fear": [3.0, 0.0]})
    lab = np.array([[[1, 2]]])
    maps = rm.behavioural_maps(None, None, t, labels=lab)
    assert maps.names == ["Action.Execution", "Perception.Vision", "Emotion.Fear"]


def test_lancaster_weighted_gives_each_study_one_vote(tmp_path, monkeypatch):
    coords, feats = _fake_neurosynth(tmp_path)
    # study 1 (vision) reports 9 extra peaks in parcel 2: weighted, it still
    # carries one vote, 1/10 in parcel 1 and 9/10 in parcel 2
    pre = tmp_path / "neurosynth" / "data-neurosynth_version-7_coordinates.tsv.gz"
    extra = pd.DataFrame({"id": 1, "x": [12] * 9, "y": 2, "z": 2})
    allc = pd.concat([coords, extra], ignore_index=True)
    with gzip.open(pre, "wt") as f:
        allc.to_csv(f, sep="\t", index=False)
    monkeypatch.setitem(rm.TAXONOMIES, "toy", ("toy", {"Perception.Vision": ["vision"]}))
    ref = nib.Nifti1Image(np.zeros((10, 4, 4), np.float32), np.diag([2.0, 2, 2, 1]))
    lab = np.zeros((10, 4, 4), int)
    lab[:3] = 1
    lab[5:8] = 2
    t = rm.neurosynth_behavioural_table(tmp_path, ref, "toy", labels=lab, key="toyatlas",
                                        statistic="lancaster_weighted")
    vis = allc[allc["id"] <= 20]
    w = 1.0 / vis["id"].map(vis["id"].value_counts())
    for r, x in ((1, 2), (2, 12)):
        p, pe, n = w[vis["x"] == x].sum() / 20, 0.5, 20
        want = (p - pe) / np.sqrt((p * (1 - p) + pe * (1 - pe)) / n)
        assert np.isclose(t.loc[t.parcel == r, "Perception.Vision"].item(), want)
    assert (tmp_path / "neurosynth" / "toyatlas_lancasterweighted_toy.csv").exists()


def test_reverse_z_judges_a_sparse_parcel_by_its_profile():
    # parcel 0 reports little (10) but all of it fear; parcel 1 reports a lot
    # (1000), 10% fear. Density says parcel 1; reverse inference says parcel 0.
    z = rm.reverse_z([10, 100], [10, 1000])
    assert z[0] > 0 > z[1]
    assert np.isnan(rm.reverse_z([0, 5], [0, 10])[0])


def test_brainmap_db_reverse_from_long_counts(tmp_path):
    d = tmp_path / "brainmap"
    d.mkdir()
    rows = []
    for parcel, (fear, vis) in {1: (9, 1), 2: (1, 9)}.items():
        for cat, v in (("Emotion.Fear", fear), ("Perception.Vision", vis)):
            rows.append(("normals", "count", cat, v, parcel))
    pd.DataFrame(rows, columns=["group", "measure", "category", "value", "parcel"]).to_csv(
        d / "toy_mango_long.csv", index=False)
    t = rm.brainmap_db_table(tmp_path, "toy", reverse=True).set_index("parcel")
    assert t.loc[1, "Emotion.Fear"] > 0 > t.loc[2, "Emotion.Fear"]
    assert t.loc[2, "Perception.Vision"] > 0


def test_lancaster_reverse_runs_and_separates_parcels(tmp_path, monkeypatch):
    _fake_neurosynth(tmp_path)
    monkeypatch.setitem(rm.TAXONOMIES, "toy", ("toy", {
        "Perception.Vision": ["vision"], "Perception.Audition": ["audition"]}))
    ref = nib.Nifti1Image(np.zeros((10, 4, 4), np.float32), np.diag([2.0, 2, 2, 1]))
    lab = np.zeros((10, 4, 4), int)
    lab[:3] = 1
    lab[5:8] = 2
    t = rm.neurosynth_behavioural_table(tmp_path, ref, "toy", labels=lab, key="toyatlas",
                                        statistic="lancaster_reverse")
    assert t.loc[0, "Perception.Vision"] > 0 > t.loc[1, "Perception.Vision"]
    assert t.loc[1, "Perception.Audition"] > 0


def test_nimare_zreverse_is_the_decoders_own_and_cached_beside_zforward(tmp_path, monkeypatch):
    coords, feats = _fake_neurosynth(tmp_path)
    monkeypatch.setitem(rm.TAXONOMIES, "toy", ("toy", {
        "Perception.Vision": ["vision"], "Perception.Audition": ["audition"]}))
    ref = nib.Nifti1Image(np.zeros((10, 4, 4), np.float32), np.diag([2.0, 2, 2, 1]))
    lab = np.zeros((10, 4, 4), int)
    lab[:3] = 1
    lab[5:8] = 2
    t = rm.neurosynth_behavioural_table(tmp_path, ref, "toy", radius_mm=2.0, labels=lab,
                                        key="toyatlas", statistic="zreverse")
    ann = pd.DataFrame({"id": np.arange(1, N + 1),
                        "Perception.Vision": feats[:, 0].toarray().ravel(),
                        "Perception.Audition": feats[:, 1].toarray().ravel()})
    ids = sorted(coords.loc[coords["x"] == 2, "id"])
    want = brainmap_decode(coords[["id"]], ann, ids,
                           features=["Perception.Vision", "Perception.Audition"],
                           correction=None)["zReverse"]
    assert np.allclose(t.loc[0, want.index].to_numpy(float), np.nan_to_num(want.to_numpy(float)))
    assert t.loc[0, "Perception.Vision"] > t.loc[1, "Perception.Vision"]
    nsd = tmp_path / "neurosynth"
    assert (nsd / "toyatlas_nimarereverse_toy.csv").exists()
    assert (nsd / "toyatlas_nimare_toy.csv").exists()
