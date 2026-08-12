"""A tiny synthetic cohort, so the pipeline can be exercised end to end
without touching real data, a cluster, or the network.

The signal is built so the outputs are checkable rather than merely
non-crashing: every subject shares a stimulus-locked latent component, so
inter-subject correlation is high and known, and censored frames are placed
at known indices with a large artifact so that failing to exclude them would
be visible in the edges.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE_TR = 1.0
FIXTURE_SHAPE = (12, 12, 10)
FIXTURE_N_TR = 240
FIXTURE_PARCELS = 8


def make_fixture(root: str | Path, n_subs: int = 4, n_tr: int = FIXTURE_N_TR,
                 tr: float = FIXTURE_TR, seed: int = 0,
                 censor_fraction: float = 0.05) -> dict:
    """Write a self-contained cohort under `root` and return the key paths."""
    import nibabel as nib

    rng = np.random.default_rng(seed)
    root = Path(root)
    deriv = root / "derivatives"
    deriv.mkdir(parents=True, exist_ok=True)

    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-12.0, -12.0, -10.0]

    # --- atlas: contiguous slabs along x, on the data grid -----------------
    labels = np.zeros(FIXTURE_SHAPE, dtype=np.int16)
    bounds = np.linspace(0, FIXTURE_SHAPE[0], FIXTURE_PARCELS + 1).astype(int)
    for p, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
        labels[lo:hi] = p
    atlas_path = root / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(labels, affine), atlas_path)

    atlas_csv = root / "atlas_labels.csv"
    pd.DataFrame({
        "index": np.arange(1, FIXTURE_PARCELS + 1),
        "name": [f"Left_Parcel{p}" if p <= FIXTURE_PARCELS // 2 else
                 f"Right_Parcel{p - FIXTURE_PARCELS // 2}"
                 for p in range(1, FIXTURE_PARCELS + 1)],
        "network": ["A"] * (FIXTURE_PARCELS // 2) + ["B"] * (FIXTURE_PARCELS // 2),
    }).to_csv(atlas_csv, index=False)

    # --- sphere coordinates, in this fixture's world space -----------------
    coords = []
    for p in range(FIXTURE_PARCELS):
        ijk = np.array([bounds[p] + 1, 6, 5, 1.0])
        xyz = (affine @ ijk)[:3]
        coords.append({"network": "A" if p < FIXTURE_PARCELS // 2 else "B",
                       "node number": p + 1, "x": xyz[0], "y": xyz[1], "z": xyz[2],
                       "name": f"net{'A' if p < FIXTURE_PARCELS // 2 else 'B'}"})
    coords_csv = root / "coords.csv"
    pd.DataFrame(coords).to_csv(coords_csv, index=False)

    # --- shared stimulus latents ------------------------------------------
    t = np.arange(n_tr) * tr
    latent_a = np.sin(2 * np.pi * t / 40.0)
    latent_b = np.sin(2 * np.pi * t / 63.0 + 0.7)

    mask = np.ones(FIXTURE_SHAPE, dtype=np.uint8)
    mask[0, 0, 0] = 0
    mask_path = root / "brain_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), mask_path)

    rows, censored_all = [], {}
    for s in range(1, n_subs + 1):
        sub = f"{s:02d}"
        func = deriv / f"sub-{sub}" / "func"
        func.mkdir(parents=True, exist_ok=True)

        data = np.zeros(FIXTURE_SHAPE + (n_tr,), dtype=np.float32)
        for p in range(1, FIXTURE_PARCELS + 1):
            latent = latent_a if p <= FIXTURE_PARCELS // 2 else latent_b
            sig = 100.0 + 8.0 * latent + rng.normal(0, 1.0, n_tr)
            data[labels == p] = sig[None, :] + rng.normal(
                0, 0.5, (int((labels == p).sum()), n_tr)
            ).astype(np.float32)

        n_cens = int(round(censor_fraction * n_tr))
        censored = np.sort(rng.choice(np.arange(5, n_tr - 5), size=n_cens, replace=False))
        data[..., censored] = 0.0          # AFNI zero-fills censored TRs
        censored_all[sub] = censored.tolist()

        bold = func / f"sub-{sub}_task-testmovie_bold.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), bold)

        keep = np.ones(n_tr)
        keep[censored] = 0
        np.savetxt(func / f"sub-{sub}_task-testmovie_censor.1D", keep, fmt="%d")

        rows.append({
            "participant_id": f"sub-{sub}", "sub": sub, "cohort": "fixture",
            "task": "testmovie", "excluded": False, "exclusion_reason": "",
            "end_movie": n_tr * tr, "group": "control",
        })

    # One recorded exclusion: absence must never be the way to exclude.
    rows.append({
        "participant_id": "sub-99", "sub": "99", "cohort": "fixture", "task": "testmovie",
        "excluded": True, "exclusion_reason": "corrupted file", "end_movie": n_tr * tr,
        "group": "control",
    })
    participants = root / "participants.csv"
    pd.DataFrame(rows).to_csv(participants, index=False)

    out_root = root / "outputs"
    config_path = root / "config.yaml"
    config_path.write_text(_CONFIG_TEMPLATE.format(
        deriv=deriv, out=out_root, participants=participants, tr=tr,
        duration=n_tr * tr, atlas_img=atlas_path, atlas_csv=atlas_csv,
        coords_csv=coords_csv,
    ))

    return {
        "root": root, "config": config_path, "derivatives": deriv, "output_root": out_root,
        "participants": participants, "atlas_img": atlas_path, "atlas_csv": atlas_csv,
        "coords_csv": coords_csv, "mask": mask_path, "censored": censored_all,
        "n_tr": n_tr, "tr": tr, "n_subs": n_subs,
    }


_CONFIG_TEMPLATE = """\
cohort: fixture
tr: {tr}
derivatives_root: {deriv}
output_root: {out}
space: MNI152NLin2009cAsym
participants: {participants}
smoothing_fwhm: null

atlases: [fixture_labels, fixture_spheres]
atlas_params:
  fixture_labels:
    labels_img: {atlas_img}
    labels_csv: {atlas_csv}
  fixture_spheres:
    csv_path: {coords_csv}
    radius_mm: 4.0
    aggregate: network

discovery:
  backend: glob
  bold_glob: "sub-*/func/sub-*_task-*_bold.nii.gz"
  include_tasks: [testmovie]

confounds:
  format: none
  strategy: none
  censor_glob: "sub-*_task-*_censor.1D"
  dilate_tr: 1

filtering:
  already_applied: true
  bandpass: [0.01, 1.0]

runs:
  mode: concat

trim:
  column: end_movie
  unit: seconds
  mode: end

stimulus:
  durations_s: {{testmovie: {duration}}}
  timing_source: identity
  unit_of_analysis: run

windows:
  sizes_s: [30, 60, 120]
  n_overlaps: 5
  drop_incomplete: true
"""
