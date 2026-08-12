import numpy as np
import pandas as pd
import pytest

from fmri_decomposition.atlases.labels import SUB_DROP, plan_harvard_oxford_merge
from fmri_decomposition.atlases.registry import AtlasSpec, clean_name, get_atlas
from fmri_decomposition.atlases.spheres import coordinate_networks

nib = pytest.importorskip("nibabel")

# Abbreviated but structurally faithful nilearn label lists.
CORT = ["Background"] + [f"Left Region {i}" for i in range(1, 49)] \
                      + [f"Right Region {i}" for i in range(1, 49)]
SUB = [
    "Background",
    "Left Cerebral White Matter", "Left Cerebral Cortex", "Left Lateral Ventricle",
    "Left Thalamus", "Left Caudate", "Left Putamen", "Left Pallidum", "Brain-Stem",
    "Left Hippocampus", "Left Amygdala", "Left Accumbens",
    "Right Cerebral White Matter", "Right Cerebral Cortex", "Right Lateral Ventricle",
    "Right Thalamus", "Right Caudate", "Right Putamen", "Right Pallidum",
    "Right Hippocampus", "Right Amygdala", "Right Accumbens",
]


class TestHarvardOxfordMergePlan:
    def test_canonical_count_is_96_plus_15(self):
        plan = plan_harvard_oxford_merge(CORT, SUB)
        assert plan.n_cort == 96
        assert plan.n_sub == 15
        assert plan.n_parcels == 111

    def test_nuisance_compartments_are_dropped(self):
        plan = plan_harvard_oxford_merge(CORT, SUB)
        for bad in ("Left Cerebral Cortex", "Right Cerebral White Matter",
                    "Left Lateral Ventricle"):
            assert bad not in plan.names

    def test_legacy_definition_keeps_the_nuisance_nodes(self):
        """Why 111 and not 69: six of the 69 'regions' are nuisance compartments."""
        plan = plan_harvard_oxford_merge(CORT, SUB, drop=frozenset({"Background"}))
        assert "Left Cerebral Cortex" in plan.names
        assert plan.n_parcels == 96 + 21

    def test_ids_are_contiguous_and_start_at_one(self):
        plan = plan_harvard_oxford_merge(CORT, SUB)
        assert plan.final_ids == list(range(1, 112))
        assert min(plan.sub_remap.values()) == 97
        assert max(plan.sub_remap.values()) == 111

    def test_subcortical_ids_do_not_collide_with_cortical(self):
        plan = plan_harvard_oxford_merge(CORT, SUB)
        assert set(plan.cort_ids).isdisjoint(set(plan.sub_remap.values()))

    def test_drop_set_is_the_documented_one(self):
        assert "Brain-Stem" not in SUB_DROP
        assert "Left Cerebral Cortex" in SUB_DROP


def toy_label_img(shape=(8, 8, 6), n_parcels=4):
    data = np.zeros(shape, dtype=np.int16)
    bounds = np.linspace(0, shape[0], n_parcels + 1).astype(int)
    for p, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
        data[lo:hi] = p
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-8.0, -8.0, -6.0]
    return nib.Nifti1Image(data, affine)


class TestAtlasSpecContract:
    def test_label_table_columns_are_the_contract(self):
        img = toy_label_img()
        spec = AtlasSpec.from_label_image("toy", img, ["Left A", "Left B", "Right A", "Right B"])
        for col in ("index", "name", "hemi", "x", "y", "z", "network"):
            assert col in spec.labels.columns

    def test_hemi_is_inferred(self):
        spec = AtlasSpec.from_label_image(
            "toy", toy_label_img(), ["Left A", "Left B", "Right A", "Brain-Stem"])
        assert spec.labels["hemi"].tolist() == ["L", "L", "R", "B"]

    def test_centroids_are_in_world_coordinates(self):
        spec = AtlasSpec.from_label_image("toy", toy_label_img(),
                                          ["a", "b", "c", "d"])
        xs = spec.labels["x"].to_numpy()
        assert np.all(np.diff(xs) > 0), "slab centroids should increase along x"

    def test_duplicate_column_names_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate column"):
            AtlasSpec.from_label_image("toy", toy_label_img(),
                                       ["Same name", "Same, name", "c", "d"])

    def test_edge_count_and_names(self):
        spec = AtlasSpec.from_label_image("toy", toy_label_img(), ["a", "b", "c", "d"])
        assert spec.n_edges == 6
        assert spec.edge_names()[:3] == ["a__b", "a__c", "a__d"]
        assert len(spec.edge_names()) == spec.n_edges

    def test_write_labels_round_trips(self, tmp_path):
        spec = AtlasSpec.from_label_image("toy", toy_label_img(), ["a b", "c,d", "e", "f"])
        path = spec.write_labels(tmp_path / "atlas-toy_labels.csv")
        tbl = pd.read_csv(path)
        assert tbl["column"].tolist() == ["a_b", "cd", "e", "f"]

    def test_clean_name(self):
        assert clean_name("Heschl's Gyrus (includes H1 and H2)") == \
            "Heschls_Gyrus_includes_H1_and_H2"


class TestMembership:
    def test_rows_sum_to_one_for_non_empty_parcels(self):
        img = toy_label_img()
        spec = AtlasSpec.from_label_image("toy", img, ["a", "b", "c", "d"])
        M, counts = spec.membership(img)
        assert np.allclose(np.asarray(M.sum(axis=1)).ravel(), 1.0)
        assert counts.sum() == int((np.asarray(img.dataobj) > 0).sum())

    def test_masked_out_voxels_are_excluded(self):
        img = toy_label_img()
        spec = AtlasSpec.from_label_image("toy", img, ["a", "b", "c", "d"])
        mask = np.ones(img.shape, dtype=bool)
        mask[:2] = False                       # kills the first parcel entirely
        M, counts = spec.membership(img, mask)
        assert counts[0] == 0
        assert np.asarray(M[0].todense()).sum() == 0
        assert counts[1] > 0

    def test_empty_parcel_stays_in_the_table(self):
        """Never drop a column: 110 among 84 files with 111 fails at read time."""
        img = toy_label_img()
        spec = AtlasSpec.from_label_image("toy", img, ["a", "b", "c", "d"])
        mask = np.ones(img.shape, dtype=bool)
        mask[:2] = False
        M, counts = spec.membership(img, mask)
        assert M.shape[0] == spec.n_nodes == len(spec.columns) == 4


class TestSpheres:
    def test_aggregate_network_gives_one_node_per_network(self, tmp_path):
        csv = tmp_path / "coords.csv"
        pd.DataFrame({
            "network": ["A", "A", "B"], "node number": [1, 2, 1],
            "x": [-4, 4, 0], "y": [0, 0, 4], "z": [0, 0, 0], "name": ["na", "na", "nb"],
        }).to_csv(csv, index=False)
        spec = coordinate_networks(csv, aggregate="network", radius_mm=4)
        assert spec.n_nodes == 2
        assert spec.labels["name"].tolist() == ["A", "B"]
        assert spec.labels["n_seeds"].tolist() == [2, 1]
        assert spec.seed_group.tolist() == [0, 0, 1]

    def test_aggregate_node_keeps_every_coordinate(self, tmp_path):
        csv = tmp_path / "coords.csv"
        pd.DataFrame({"network": ["A", "A", "B"], "node number": [1, 2, 1],
                      "x": [-4, 4, 0], "y": [0, 0, 4], "z": [0, 0, 0]}).to_csv(csv, index=False)
        spec = coordinate_networks(csv, aggregate="node")
        assert spec.n_nodes == 3
        assert spec.labels["hemi"].tolist() == ["L", "R", "B"]

    def test_real_coordinate_file_gives_14_networks_and_254_nodes(self):
        import pathlib

        csv = pathlib.Path(__file__).parent / "data" / "mni_space_of_networks.csv"
        if not csv.exists():
            pytest.skip("real coordinate CSV not vendored into tests/data")
        net = coordinate_networks(csv, aggregate="network")
        node = coordinate_networks(csv, aggregate="node")
        assert net.n_nodes == 14 and net.n_edges == 91
        assert node.n_nodes == 254
        assert node.n_edges == 32_131

    def test_aggregate_must_be_valid(self, tmp_path):
        csv = tmp_path / "c.csv"
        pd.DataFrame({"network": ["A"], "x": [0], "y": [0], "z": [0]}).to_csv(csv, index=False)
        with pytest.raises(ValueError):
            coordinate_networks(csv, aggregate="parcel")

    def test_spheres_pick_up_voxels_within_the_radius(self, tmp_path):
        csv = tmp_path / "c.csv"
        pd.DataFrame({"network": ["A"], "node number": [1],
                      "x": [0.0], "y": [0.0], "z": [0.0]}).to_csv(csv, index=False)
        spec = coordinate_networks(csv, aggregate="node", radius_mm=4.0)
        img = toy_label_img()
        M, counts = spec.membership(img)
        assert counts[0] > 0


class TestRegistry:
    def test_unknown_atlas_names_are_rejected(self):
        with pytest.raises(KeyError):
            get_atlas("schaefer1000")

    def test_custom_label_image_is_first_class(self, tmp_path):
        img_path = tmp_path / "atlas.nii.gz"
        nib.save(toy_label_img(), img_path)
        csv = tmp_path / "labels.csv"
        pd.DataFrame({"index": [1, 2, 3, 4], "name": list("abcd"),
                      "network": ["n"] * 4}).to_csv(csv, index=False)
        spec = get_atlas("mine", labels_img=img_path, labels_csv=csv)
        assert spec.name == "mine" and spec.n_nodes == 4
