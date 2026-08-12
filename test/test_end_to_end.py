"""Fixture -> stage 2 -> stage 3, checking the properties the design promises."""

import numpy as np
import pandas as pd
import pytest

from fmri_decomposition.atlases.registry import get_atlas
from fmri_decomposition.cli import main
from fmri_decomposition.config import load_config
from fmri_decomposition.dfc import read_edges
from fmri_decomposition.fixtures import make_fixture

pytest.importorskip("nibabel")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("cohort")
    info = make_fixture(root, n_subs=4, n_tr=240)
    assert main(["validate", str(info["config"])]) == 0
    assert main(["extract", str(info["config"]), "--n-jobs", "1"]) == 0
    assert main(["dfc", str(info["config"]), "--n-jobs", "1"]) == 0
    return info


def test_activation_shards_exist_for_every_subject_and_atlas(built):
    cfg = load_config(built["config"])
    for atlas in cfg.atlases:
        files = sorted((cfg.output_root / "activation" / f"atlas={atlas}"
                        / "cohort=fixture").rglob("*.parquet"))
        assert len(files) == built["n_subs"], f"{atlas}: {files}"


def test_activation_contract_columns(built):
    cfg = load_config(built["config"])
    f = next((cfg.output_root / "activation").rglob("*.parquet"))
    df = pd.read_parquet(f)
    for col in ("t", "time_s", "stimulus_time_s", "good_frame", "run_idx", "run_key"):
        assert col in df.columns
    # read_shard restores the partition keys the path carries.
    from fmri_decomposition.io import read_shard

    restored = read_shard(f)
    assert restored["sub"].iloc[0] in {"01", "02", "03", "04", "05", "06"}
    assert restored["task"].iloc[0] == "testmovie"


def test_censored_frames_are_marked_and_dilated(built):
    cfg = load_config(built["config"])
    from fmri_decomposition.io import read_shard

    f = sorted((cfg.output_root / "activation" / "atlas=fixture_labels"
                / "cohort=fixture").rglob("*.parquet"))[0]
    df = read_shard(f)
    sub = str(df["sub"].iloc[0])
    censored = set(built["censored"][sub])
    bad = set(df.loc[~df["good_frame"], "t"].tolist())
    assert censored <= bad, "every censored TR must be flagged"
    assert len(bad) > len(censored), "the mask must be dilated by +/-1 TR"


def test_dfc_shards_exist_for_every_window_size(built):
    cfg = load_config(built["config"])
    for w in cfg.windows.sizes_s:
        wk = int(w)
        files = sorted((cfg.output_root / "dfc" / "atlas=fixture_labels"
                        / f"window_s={wk}" / "cohort=fixture").rglob("*.parquet"))
        assert len(files) == built["n_subs"], f"window {wk}: {files}"


def test_window_ids_align_across_subjects(built):
    """The structural promise of the stimulus grid."""
    cfg = load_config(built["config"])
    files = sorted((cfg.output_root / "dfc" / "atlas=fixture_labels" / "window_s=30"
                    / "cohort=fixture").rglob("*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    ref = frames[0].set_index("window_id")["stimulus_start_s"]
    for df in frames[1:]:
        other = df.set_index("window_id")["stimulus_start_s"]
        shared = ref.index.intersection(other.index)
        assert len(shared) > 10
        assert np.allclose(ref.loc[shared], other.loc[shared])


def test_window_count_matches_the_grid(built):
    cfg = load_config(built["config"])
    from fmri_decomposition.windows import make_stimulus_grid

    df = pd.read_parquet(sorted((cfg.output_root / "dfc" / "atlas=fixture_labels"
                                 / "window_s=30" / "cohort=fixture").rglob("*.parquet"))[0])
    expected = len(make_stimulus_grid(240.0, 30.0, cfg.windows.n_overlaps))
    assert len(df) == expected


def test_qc_columns_travel_with_the_edges(built):
    cfg = load_config(built["config"])
    df = pd.read_parquet(sorted((cfg.output_root / "dfc" / "atlas=fixture_labels"
                                 / "window_s=60" / "cohort=fixture").rglob("*.parquet"))[0])
    for col in ("n_tr_nominal", "n_tr_effective", "frac_good_frames",
                "crosses_run_boundary", "rank_deficient"):
        assert col in df.columns
    assert (df["n_tr_effective"] <= df["n_tr_available"]).all()
    assert (df["frac_good_frames"] <= 1.0).all()
    assert (df["n_tr_effective"] < df["n_tr_nominal"]).any(), "fixture censors frames"


def test_edges_recover_the_planted_network_structure(built):
    """Parcels 1-4 share latent A, 5-8 share latent B: within > between."""
    cfg = load_config(built["config"])
    atlas = get_atlas("fixture_labels", **cfg.atlas_params["fixture_labels"])
    f = sorted((cfg.output_root / "dfc" / "atlas=fixture_labels" / "window_s=120"
                / "cohort=fixture").rglob("*.parquet"))[0]
    edges = read_edges(f, atlas)
    names = atlas.edge_names()
    net = atlas.labels["network"].tolist()
    iu, ju = np.triu_indices(atlas.n_nodes, k=1)
    within = np.array([net[i] == net[j] for i, j in zip(iu, ju)])
    assert np.nanmean(edges[:, within]) > np.nanmean(edges[:, ~within]) + 0.2
    assert len(names) == edges.shape[1]


def test_labels_csv_is_written_next_to_the_data(built):
    cfg = load_config(built["config"])
    for atlas in cfg.atlases:
        p = cfg.output_root / "meta" / f"atlas-{atlas}_labels.csv"
        assert p.exists()
        tbl = pd.read_csv(p)
        assert {"index", "name", "hemi", "x", "y", "z", "network", "column"} <= set(tbl.columns)


def test_manifests_are_written_by_the_parent_only(built):
    cfg = load_config(built["config"])
    cm = cfg.output_root / "meta" / "cohorts" / "cohort=fixture"
    assert (cm / "manifest_activation.json").exists()
    assert (cm / "manifest_dfc.json").exists()
    assert not list(cfg.output_root.rglob("_metadata"))


def test_atlas_labels_stay_cohort_independent(built):
    """Label tables describe the atlas, not the cohort: one copy, shared."""
    cfg = load_config(built["config"])
    assert (cfg.output_root / "meta" / "atlas-fixture_labels_labels.csv").exists()
    assert not (cfg.output_root / "meta" / "cohorts" / "cohort=fixture"
                / "atlas-fixture_labels_labels.csv").exists()


def test_one_read_covers_every_cohort_at_one_atlas_and_window(built):
    """Atlas-first pays off here: a single dataset root, no path list."""
    from fmri_decomposition.io import dfc_root, open_dataset

    d = open_dataset(dfc_root(cfg_root(built), "fixture_labels", 30), stage="dfc")
    tbl = d.to_table(columns=["window_id", "cohort", "sub"])
    assert set(tbl.column("cohort").to_pylist()) == {"fixture"}
    subs = set(tbl.column("sub").to_pylist())
    assert len(subs) == built["n_subs"]
    assert "01" in subs, "leading zeros must survive: sub is a string, not an int"


def cfg_root(built):
    return load_config(built["config"]).output_root


def test_rerun_is_free(built):
    """skip_if_exists makes resume a no-op, which is what makes walltime kills cheap."""
    import json

    cfg = load_config(built["config"])
    assert main(["dfc", str(built["config"]), "--n-jobs", "1"]) == 0
    payload = json.loads((cfg.output_root / "meta" / "cohorts" / "cohort=fixture"
                          / "manifest_dfc.json").read_text())
    assert all(e["status"] == "skipped" for e in payload["entries"])


def test_masker_per_window_equivalence(built):
    """Slicing the parquet == re-fitting a masker per window, to float noise."""
    import nibabel as nib

    from fmri_decomposition.validate import equivalence_check

    cfg = load_config(built["config"])
    atlas = get_atlas("fixture_labels", **cfg.atlas_params["fixture_labels"])
    bold = next((cfg.derivatives_root).rglob("sub-*_task-testmovie_bold.nii.gz"))
    report = equivalence_check(nib.load(str(bold)), atlas,
                               windows=[(0, 30), (60, 120), (100, 130)])
    assert report["passes"].all(), report.to_string(index=False)


def test_sphere_atlas_produces_the_configured_number_of_networks(built):
    cfg = load_config(built["config"])
    df = pd.read_parquet(next((cfg.output_root / "activation" / "atlas=fixture_spheres"
                               / "cohort=fixture").rglob("*.parquet")))
    atlas = get_atlas("fixture_spheres", **cfg.atlas_params["fixture_spheres"])
    assert atlas.n_nodes == 2
    assert all(c in df.columns for c in atlas.columns)


def test_diagnose_writes_its_artifacts(built):
    cfg = load_config(built["config"])
    main(["diagnose", str(built["config"])])
    assert (cfg.output_root / "meta" / "cohorts" / "cohort=fixture"
            / "coverage.parquet").exists()
