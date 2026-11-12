"""Neurosynth profiles counted in the states' own parcels, not Brainnetome's.

A two-parcel toy grid and four fake studies with known peaks: the z of each
parcel must follow the binomial forward-inference formula, and painting the
table back with the same labels must give each parcel's value exactly.
"""
import gzip

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from fmri_decomposition.unsupervised_model.contextualization import refmaps as rm


def _fake_neurosynth(d):
    nsd = d / "neurosynth"
    nsd.mkdir()
    pre = nsd / "data-neurosynth_version-7_"
    # studies 1,2 peak in parcel 1 (x=2); 3 in parcel 2 (x=12); 4 far away
    coords = pd.DataFrame({"id": [1, 2, 3, 4], "x": [2, 2, 12, 60],
                           "y": [2, 2, 2, 60], "z": [2, 2, 2, 60]})
    meta = pd.DataFrame({"id": [1, 2, 3, 4], "space": ["MNI"] * 4})
    for name, t in (("coordinates", coords), ("metadata", meta)):
        with gzip.open(f"{pre}{name}.tsv.gz", "wt") as f:
            t.to_csv(f, sep="\t", index=False)
    # term "vision" loads on studies 1 and 3
    feats = sp.csr_matrix(np.array([[0.5], [0.0], [0.5], [0.0]]))
    sp.save_npz(f"{pre}vocab-terms_source-abstract_type-tfidf_features.npz", feats)
    (nsd / "data-neurosynth_version-7_vocab-terms_vocabulary.txt").write_text("vision\n")


def test_counts_in_given_parcels_and_paints_back(tmp_path, monkeypatch):
    _fake_neurosynth(tmp_path)
    monkeypatch.setitem(rm.TAXONOMIES, "toy", ("toy", {"Perception.Vision": ["vision"]}))
    ref = nib.Nifti1Image(np.zeros((10, 4, 4), np.float32), np.diag([2.0, 2, 2, 1]))
    lab = np.zeros((10, 4, 4), int)
    lab[:3] = 1                      # x = 0..4 mm
    lab[5:8] = 2                     # x = 10..14 mm

    t = rm.neurosynth_behavioural_table(tmp_path, ref, "toy", radius_mm=2.0,
                                        labels=lab, key="toyatlas")
    assert list(t["parcel"]) == [1, 2] and "bn_index" not in t
    act = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], float)
    p0, member = act.mean(0), np.array([1, 0, 1, 0], bool)
    n, k = member.sum(), act[member].sum(0)
    z = (k - n * p0) / np.sqrt(n * p0 * (1 - p0))
    assert np.allclose(t["Perception.Vision"], z)
    assert (tmp_path / "neurosynth" / "toyatlas_neurosynth_toy.csv").exists()

    maps = rm.behavioural_maps(tmp_path, ref, t, labels=lab)
    v = maps.voxels[0].reshape(lab.shape)
    assert np.allclose(v[lab == 1], z[0]) and np.allclose(v[lab == 2], z[1])
    assert np.isnan(v[lab == 0]).all() and not maps.simulated
