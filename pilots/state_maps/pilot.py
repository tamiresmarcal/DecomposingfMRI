#!/usr/bin/env python3
"""Pilot: what each brain state looks like, and what it lines up with.

    python pilots/state_maps/pilot.py                       # simulated states
    python pilots/state_maps/pilot.py --state-means sm.csv  # real ones

One figure row per state: its mean activation on the cortical surface, and to
the right its correlation with three families of reference maps -- molecular
(19 PET receptor/transporter maps, neuromaps / Hansen 2022), behavioural
(BrainMap domains over Brainnetome subregions) and canonical (Yeo 2011's 7
networks). Stars mark correlations that survive a spin test, BH-corrected
within the family. A second figure is the same information for ONE state at
reading size, as ranked bars.

THE STATES ARE SIMULATED unless `--state-means` points at the CSV that
`fmri-decomp state-means --atlas harvardoxford ... --csv` writes. The
simulation is built to have a known answer: each state is a signed mix of
Yeo networks plus spatially smooth noise, sampled as a sticky Markov chain
of noisy frames and averaged back per label -- the same groupby mean
`state-means` does. If the Yeo panel does not recover the mix, the pipeline
is wrong, not the states.

The behavioural panel is SIMULATED until a real Brainnetome behavioural table
is given with `--behavioural` (see README.md for why and what format).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(HERE))
import refmaps as rm  # noqa: E402

# Diverging blue <-> red with a neutral grey midpoint; the dataviz reference
# palette's blue ramp and red pole.
DIVERGING = ["#104281", "#3987e5", "#cde2fb", "#f0efec", "#f8d3d2", "#e34948", "#8f1d1c"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SIMULATED_INK = "#b4560f"

FAMILY_TITLES = {
    "molecular": "Molecular  ·  PET receptors (neuromaps)",
    "behavioural": "Behavioural  ·  BrainMap × Brainnetome\n210 subregions",
    "behavioural_gyrus": "Behavioural  ·  BrainMap × Brainnetome\n40 gyri",
    "canonical": "Canonical  ·  Yeo 7 networks",
}
CARD_TITLES = {
    "molecular": "Molecular (PET)",
    "behavioural": "Behavioural · 210 subregions",
    "behavioural_gyrus": "Behavioural · 40 gyri",
    "canonical": "Canonical (Yeo 7)",
}

# Ground truth of the simulation: state -> {Yeo network: weight}.
SIM_STATES = {
    "Default": {"Default": 1.0, "Dorsal Attention": -0.6, "Somatomotor": -0.3},
    "Visual": {"Visual": 1.0, "Default": -0.4},
    "Sensorimotor": {"Somatomotor": 1.0, "Ventral Attention": 0.4, "Frontoparietal": -0.4},
    "Task-positive": {"Dorsal Attention": 0.8, "Frontoparietal": 0.8, "Default": -0.6},
    "Limbic": {"Limbic": 1.0, "Default": 0.3, "Visual": -0.4},
    "Salience": {"Ventral Attention": 1.0, "Somatomotor": 0.3, "Default": -0.5},
}


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("div", DIVERGING)


# ------------------------------------------------------------ simulation ---
def parcel_centroids(parc: rm.SurfaceParcellation, surf: rm.Surface) -> np.ndarray:
    xyz = {h: surf.pial[h][0].T for h in rm.HEMIS}
    return parc.parcellate(xyz).T                     # P, 3


def simulate_state_means(parc, surf, yeo_parcel, k_seed=0, n_frames=6000,
                         frame_noise=2.0, pattern_noise=0.45) -> pd.DataFrame:
    """Frames -> labels -> per-state mean, in `state-means`' CSV layout."""
    rng = np.random.default_rng(k_seed)
    P = len(parc.names)
    cen = parcel_centroids(parc, surf)
    d = np.linalg.norm(cen[:, None] - cen[None], axis=-1)
    smooth = np.exp(-(d / 25.0) ** 2)                 # 25 mm Gaussian kernel
    smooth /= smooth.sum(1, keepdims=True)

    patterns = []
    for mix in SIM_STATES.values():
        x = sum(w * yeo_parcel[rm.YEO7.index(n)] for n, w in mix.items())
        x = (x - x.mean()) / x.std()
        noise = smooth @ rng.normal(size=P)
        x = x + pattern_noise * noise / noise.std()
        patterns.append((x - x.mean()) / x.std())
    patterns = np.array(patterns)
    K = len(patterns)

    # Sticky chain: stay with p=.9, otherwise jump uniformly; uneven
    # occupancy from a Dirichlet over where the jumps land.
    land = rng.dirichlet(np.full(K, 3.0))
    labels = np.empty(n_frames, dtype=int)
    labels[0] = rng.choice(K, p=land)
    for t in range(1, n_frames):
        labels[t] = labels[t - 1] if rng.random() < 0.9 else rng.choice(K, p=land)
    frames = patterns[labels] + frame_noise * rng.normal(size=(n_frames, P))

    df = pd.DataFrame(frames, columns=parc.columns)
    df["state"] = labels
    g = df.groupby("state")
    out = g[parc.columns].mean()
    out.insert(0, "n_rows", g.size())
    out.insert(1, "share", out["n_rows"] / n_frames)
    out.attrs["names"] = list(SIM_STATES)
    return out


def read_state_means(path: Path, parc: rm.SurfaceParcellation) -> pd.DataFrame:
    """`state-means --csv` -> the cortical columns this surface has.

    Subcortical columns are reported and dropped: they have no vertices. A
    cortical column the CSV lacks is a hard error, since it means the states
    came from another atlas and the correlation would silently be over a
    different set of parcels.
    """
    t = pd.read_csv(path, index_col="state")
    missing = [c for c in parc.columns if c not in t.columns]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} Harvard-Oxford cortical columns "
                         f"missing, e.g. {missing[:3]}. Was it written with "
                         f"--atlas harvardoxford?")
    extra = [c for c in t.columns if c not in parc.columns and c not in ("n_rows", "share")]
    if extra:
        print(f"[state-means] ignoring {len(extra)} non-cortical columns: "
              f"{', '.join(extra[:6])}{' ...' if len(extra) > 6 else ''}")
    return t[["n_rows", "share"] + parc.columns]


# ---------------------------------------------------------------- figures ---
VIEWS = [("left", "lateral"), ("left", "medial"), ("right", "lateral"), ("right", "medial")]


def draw_brain(fig, cell, values, parc, fs, sulc, vlim):
    """Four views of one parcel map into a gridspec cell."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    from nilearn import plotting

    painted = parc.paint(values)
    sub = GridSpecFromSubplotSpec(1, 4, subplot_spec=cell, wspace=-0.12)
    for j, (hemi, view) in enumerate(VIEWS):
        ax = fig.add_subplot(sub[0, j], projection="3d")
        h = "L" if hemi == "left" else "R"
        plotting.plot_surf(
            fs["inflated"].parts[hemi], surf_map=painted[h],
            bg_map=sulc.data.parts[hemi], hemi=hemi, view=view, axes=ax,
            figure=fig, cmap=diverging_cmap(), vmin=-vlim, vmax=vlim,
            colorbar=False, bg_on_data=False, avg_method="median")
        ax.set_box_aspect(None, zoom=1.45)
        ax.set_axis_off()


def heat_panel(ax, r, p, q, names, simulated=False, header_colors=None):
    import matplotlib.patches as mpatches

    K, M = r.shape
    im = ax.imshow(r, cmap=diverging_cmap(), vmin=-1, vmax=1, aspect="auto",
                   interpolation="nearest")
    # 2px surface gaps between cells
    for x in np.arange(-0.5, M, 1.0):
        ax.axvline(x, color="white", lw=1.5)
    for y in np.arange(-0.5, K, 1.0):
        ax.axhline(y, color="white", lw=3.0)
    for i in range(K):
        for j in range(M):
            mark = ("★" if q[i, j] < 0.01 else "✱" if q[i, j] < 0.05
                    else "•" if p[i, j] < 0.05 else "")
            if mark:
                ax.text(j, i, mark, ha="center", va="center",
                        fontsize=11 if mark == "•" else 7.5,
                        color="white" if abs(r[i, j]) > 0.55 else INK)
    ax.set_xticks(range(M))
    ax.set_xticklabels([rm.short(n) for n in names], rotation=60, ha="left",
                       fontsize=7.5, color=INK2)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=12 if header_colors is not None else 2)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if header_colors is not None:
        for j, c in enumerate(header_colors):
            ax.add_patch(mpatches.Rectangle((j - 0.4, 1.006), 0.8, 0.012,
                                            transform=ax.get_xaxis_transform(),
                                            color=c, clip_on=False, lw=0))
    if simulated:
        ax.add_patch(mpatches.Rectangle((-0.5, -0.5), M, K, fill=False,
                                        hatch="////", ec=SIMULATED_INK,
                                        lw=0, alpha=0.35))
    return im


def overview_figure(state_df, names, parc, results, out: Path, simulated_states):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from nilearn import datasets

    fs = datasets.load_fsaverage("fsaverage5")
    sulc = datasets.load_fsaverage_data(mesh="fsaverage5", data_type="sulcal")
    X = state_df[parc.columns].to_numpy(float)
    Z = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    K = len(Z)

    fams = list(results)
    widths, cols = [0.9, 5.2], {}
    for f in fams:                      # a spacer, then one heat strip per family
        widths.append(0.3)
        cols[f] = len(widths)
        widths.append(len(results[f]["names"]) * 0.26)
    H = 1.25 * K + 3.1
    fig = plt.figure(figsize=(sum(widths) + 0.6, H), facecolor="white")
    gs = GridSpec(K, len(widths), figure=fig, width_ratios=widths, left=0.01,
                  right=0.99, top=1 - 2.0 / H, bottom=1.1 / H,
                  wspace=0.0, hspace=0.0)
    for i in range(K):
        lab = fig.add_subplot(gs[i, 0])
        lab.set_axis_off()
        lab.text(0.95, 0.62, f"S{i + 1}", ha="right", va="center", fontsize=12,
                 fontweight="bold", color=INK, transform=lab.transAxes)
        lab.text(0.95, 0.40, names[i], ha="right", va="center", fontsize=8,
                 color=INK2, transform=lab.transAxes)
        lab.text(0.95, 0.22, f"{state_df['share'].iloc[i]:.0%} of frames",
                 ha="right", va="center", fontsize=7, color=MUTED,
                 transform=lab.transAxes)
        draw_brain(fig, gs[i, 1], Z[i], parc, fs, sulc, vlim=2.5)

    for fam in fams:
        title = FAMILY_TITLES[fam]
        res = results[fam]
        ax = fig.add_subplot(gs[:, cols[fam]])
        im = heat_panel(ax, res["r"], res["p"], res["q"], res["names"],
                        simulated=res["simulated"],
                        header_colors=rm.YEO7_COLORS if fam == "canonical" else None)
        # Titles in figure coordinates so all three sit on one line,
        # whatever the length of their tick labels.
        sim = res["simulated"]
        fig.text(ax.get_position().x0, 1 - 0.75 / H,
                 title + ("  ·  SIMULATED" if sim else ""), ha="left", va="top",
                 fontsize=9.5, fontweight="bold", color=SIMULATED_INK if sim else INK)
    if any(results[f]["simulated"] for f in fams):
        fig.text(0.99, 0.004, "Behavioural panels: SIMULATED stand-in table on the real "
                 "Brainnetome parcellation (BrainMap profiles not loaded yet); the gyrus "
                 "level averages the same table over each gyrus's subregions.",
                 ha="right", va="bottom", fontsize=7, color=SIMULATED_INK)

    # Colour bars: r for the heat strips, z for the brains.
    cax = fig.add_axes([0.66, 0.62 / H, 0.16, 0.10 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("Pearson r across parcels · spin test: • p<.05 uncorrected, "
                 "✱ q<.05, ★ q<.01 (BH within family)",
                 fontsize=7, color=INK2)
    cb.ax.tick_params(labelsize=7, length=2, colors=INK2)
    cb.outline.set_visible(False)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    bax = fig.add_axes([0.12, 0.62 / H, 0.14, 0.10 / H])
    cb2 = fig.colorbar(ScalarMappable(Normalize(-2.5, 2.5), diverging_cmap()),
                       cax=bax, orientation="horizontal", ticks=[-2, 0, 2])
    cb2.set_label("state mean activation (z across cortical parcels)", fontsize=7, color=INK2)
    cb2.ax.tick_params(labelsize=7, length=2, colors=INK2)
    cb2.outline.set_visible(False)

    head = ("Brain states × reference maps — PILOT"
            + ("  (simulated states)" if simulated_states else ""))
    fig.text(0.01, 0.995, head, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.995 - 0.32 / H,
             "Harvard-Oxford cortical parcels (96) on fsaverage5 · lateral/medial, left then right",
             ha="left", va="top", fontsize=8, color=INK2)
    fig.savefig(out, dpi=200, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def state_card(i, state_df, names, parc, results, out: Path, top=8):
    """One state at reading size: four big views + ranked bars per family."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from nilearn import datasets

    fs = datasets.load_fsaverage("fsaverage5")
    sulc = datasets.load_fsaverage_data(mesh="fsaverage5", data_type="sulcal")
    X = state_df[parc.columns].to_numpy(float)
    z = (X[i] - X[i].mean()) / X[i].std()

    fams = list(results)
    fig = plt.figure(figsize=(6.0 + 2.35 * len(fams), 4.6), facecolor="white")
    gs = GridSpec(2, 1 + len(fams), figure=fig, width_ratios=[4.4] + [2.2] * len(fams),
                  height_ratios=[1, 1], left=0.01, right=0.985, top=0.80,
                  bottom=0.1, wspace=0.55, hspace=0.05)
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    from nilearn import plotting

    sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[:, 0], wspace=-0.1, hspace=-0.1)
    painted = parc.paint(z)
    for j, (hemi, view) in enumerate(VIEWS):
        ax = fig.add_subplot(sub[j % 2, j // 2], projection="3d")
        h = "L" if hemi == "left" else "R"
        plotting.plot_surf(fs["inflated"].parts[hemi], surf_map=painted[h],
                           bg_map=sulc.data.parts[hemi], hemi=hemi, view=view,
                           axes=ax, figure=fig, cmap=diverging_cmap(), vmin=-2.5,
                           vmax=2.5, colorbar=False, avg_method="median")
        ax.set_box_aspect(None, zoom=1.35)
        ax.set_axis_off()

    for c, fam in enumerate(fams, start=1):
        res = results[fam]
        r, q, nm = res["r"][i], res["q"][i], np.array(res["names"])
        order = np.argsort(-np.abs(r))[:top] if fam != "canonical" else np.argsort(-r)
        order = order[np.argsort(-r[order])]
        ax = fig.add_subplot(gs[:, c])
        y = np.arange(len(order))[::-1]
        if fam == "canonical":
            colors = [rm.YEO7_COLORS[k] for k in order]
        else:   # sign carries the colour; length carries the size
            colors = [DIVERGING[5] if v >= 0 else DIVERGING[1] for v in r[order]]
        ax.barh(y, r[order], color=colors, height=0.72, edgecolor="white", lw=1)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([rm.short(n) for n in nm[order]], fontsize=8, color=INK)
        for yy, k in zip(y, order):
            star = ("★" if q[k] < 0.01 else "✱" if q[k] < 0.05
                    else "•" if res["p"][i, k] < 0.05 else "")
            ax.text(r[k] + (0.03 if r[k] >= 0 else -0.03), yy,
                    f"{r[k]:+.2f}{(' ' + star) if star else ''}",
                    ha="left" if r[k] >= 0 else "right", va="center", fontsize=7,
                    color=INK2)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xticks([-1, 0, 1])
        ax.tick_params(labelsize=7, colors=INK2, length=2)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.set_xlabel("r (parcels)", fontsize=7.5, color=INK2)
        sim = res["simulated"]
        ax.set_title(CARD_TITLES[fam] + ("  SIMULATED" if sim else "")
                     + ("" if fam == "canonical" else f"\ntop {top} by |r|"),
                     fontsize=9, color=SIMULATED_INK if sim else INK, loc="left")
    fig.text(0.01, 0.975, f"S{i + 1} · {names[i]}", fontsize=14, fontweight="bold",
             color=INK, va="top")
    fig.text(0.01, 0.905, f"{state_df['share'].iloc[i]:.0%} of frames · mean activation, "
             "z across 96 cortical parcels · spin test: • p<.05 uncorrected, ✱ q<.05, ★ q<.01 "
             "(BH within family)",
             fontsize=8, color=INK2, va="top")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------- main ---
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=HERE / "data",
                   help="where fetch_reference_maps.sh put the maps")
    p.add_argument("--state-means", type=Path,
                   help="CSV from `fmri-decomp state-means --atlas harvardoxford --csv`")
    p.add_argument("--behavioural", type=Path,
                   help="real Brainnetome behavioural table (bn_index + one column per domain)")
    p.add_argument("--bn-levels", nargs="+", default=list(rm.BN_LEVELS),
                   choices=rm.BN_LEVELS,
                   help="Brainnetome level(s) for the behavioural panel: the 210 "
                        "subregions, the 40 gyri they group into, or both side by side")
    p.add_argument("--n-spins", type=int, default=1000)
    p.add_argument("--out", type=Path, default=HERE / "figures")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    surf = rm.load_surface(args.data)
    parc = rm.harvard_oxford_on_surface(args.data, surf)
    print(f"[atlas] Harvard-Oxford cortical: {len(parc.names)} parcels on fsaverage5")

    bt = rm.load_behavioural_table(args.behavioural) if args.behavioural else None
    refs = ([rm.receptor_maps(args.data, surf)]
            + [rm.behavioural_maps(args.data, surf, bt, level=lv) for lv in args.bn_levels]
            + [rm.yeo7_maps(args.data, surf)])
    yeo_parcel = parc.parcellate(refs[-1].vertex)

    if args.state_means:
        state_df = read_state_means(args.state_means, parc)
        names = [f"state {s}" for s in state_df.index]
    else:
        state_df = simulate_state_means(parc, surf, yeo_parcel, k_seed=args.seed)
        names = state_df.attrs["names"]
        state_df.to_csv(args.out / "simulated_state_means.csv")
    X = state_df[parc.columns].to_numpy(float)

    print(f"[spin] {args.n_spins} rotations")
    spins = rm.spin_indices(surf, args.n_spins, seed=args.seed)
    results, rows = {}, []
    for ref in refs:
        res = rm.spin_test(X, ref, parc, spins)
        res.update(names=ref.names, simulated=ref.simulated)
        results[ref.family] = res
        for i in range(len(X)):
            for j, n in enumerate(ref.names):
                rows.append({"state": i + 1, "state_name": names[i], "family": ref.family,
                             "map": n, "r": res["r"][i, j], "p_spin": res["p"][i, j],
                             "q_fdr": res["q"][i, j], "map_simulated": ref.simulated})
        print(f"[{ref.family}] {len(ref.names)} maps, "
              f"{(res['q'] < 0.05).sum()} of {res['q'].size} state×map pairs q<.05"
              + ("  (SIMULATED table)" if ref.simulated else ""))
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "state_map_correlations.csv", index=False)

    overview_figure(state_df, names, parc, results, args.out / "state_maps_overview.png",
                    simulated_states=not args.state_means)
    for i in range(len(X)):
        state_card(i, state_df, names, parc, results, args.out / f"state_card_S{i + 1}.png")
    (args.out / "provenance.json").write_text(json.dumps({
        "states": str(args.state_means) if args.state_means else "simulated",
        "simulation_truth": None if args.state_means else SIM_STATES,
        "families": {r.family: {"source": r.source, "maps": r.names,
                                "simulated": r.simulated} for r in refs},
        "atlas": "Harvard-Oxford cortical lateralised thr25 (96), fsaverage5",
        "n_spins": args.n_spins, "seed": args.seed,
    }, indent=2, ensure_ascii=False))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
