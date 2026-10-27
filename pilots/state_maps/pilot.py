#!/usr/bin/env python3
"""Pilot: what each brain state looks like, and what it lines up with.

    python pilots/state_maps/pilot.py                     # all four atlases, simulated
    python pilots/state_maps/pilot.py --atlases harvardoxford --state-means sm.csv

One figure per state atlas -- harvardoxford, networks_nodes, networks, yeo7,
the ones this pipeline extracts on. One row per state: its mean activation on
the cortex (lateral and medial, left then right) and on two subcortical
slices, then its correlation across parcels with four families of maps:

    molecular    19 PET receptor/transporter maps (neuromaps / Hansen 2022)
    behavioural  BrainMap categories over Brainnetome's 246 subregions, every
                 leaf of the taxonomy, grouped by domain
    Yeo 7        Yeo 2011's canonical networks
    networks 14  this repo's own coordinate networks

Plus a card per state at reading size, and every number in a CSV.

THE STATES ARE SIMULATED unless `--state-means` is the CSV `fmri-decomp
state-means --csv` writes. The simulation has a known answer: each state is a
signed mix of Yeo networks and Brainnetome's deep structures (amygdala,
hippocampus, basal ganglia, thalamus), plus spatially smooth noise, sampled as
a sticky Markov chain of noisy frames and averaged back per label -- the
groupby `state-means` does. If the Yeo panel does not recover the mix, the
pipeline is wrong, not the states.

The behavioural panel is SIMULATED until `--behavioural` gives the real
Brainnetome table (README.md: why, and the format).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refmaps as rm  # noqa: E402

ATLASES = ["harvardoxford", "networks_nodes", "networks", "yeo7"]

# Diverging blue <-> red with a neutral grey midpoint.
DIVERGING = ["#104281", "#3987e5", "#cde2fb", "#f0efec", "#f8d3d2", "#e34948", "#8f1d1c"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SIMULATED_INK = "#b4560f"

FAMILIES = ["molecular", "behavioural", "yeo7", "networks14"]
TITLES = {
    "molecular": "Molecular · PET receptors (neuromaps)",
    "behavioural": "Behavioural · BrainMap taxonomy × Brainnetome (246)",
    "yeo7": "Canonical · Yeo 7",
    "networks14": "Canonical · our 14 networks",
}
CARD_TITLES = {"molecular": "Molecular (PET)", "behavioural": "Behavioural (BrainMap)",
               "yeo7": "Yeo 7 networks", "networks14": "Our 14 networks"}
CARD_TOP = {"molecular": 8, "behavioural": 10, "yeo7": 7, "networks14": 14}

# Ground truth of the simulation: state -> {component: weight}. Components are
# Yeo networks and Brainnetome's deep gyri.
SIM_STATES = {
    "Default": {"Default": 1.0, "Dorsal Attention": -0.6, "Somatomotor": -0.3, "Hipp": 0.5},
    "Visual": {"Visual": 1.0, "Default": -0.4, "Tha": 0.3},
    "Sensorimotor": {"Somatomotor": 1.0, "Ventral Attention": 0.4, "Frontoparietal": -0.4,
                     "BG": 0.6, "Tha": 0.4},
    "Task-positive": {"Dorsal Attention": 0.8, "Frontoparietal": 0.8, "Default": -0.6, "BG": 0.3},
    "Limbic": {"Limbic": 1.0, "Default": 0.3, "Visual": -0.4, "Amyg": 0.9, "Hipp": 0.6},
    "Salience": {"Ventral Attention": 1.0, "Somatomotor": 0.3, "Default": -0.5,
                 "Tha": 0.5, "Amyg": 0.3},
}

NULL_LABEL = {"moran": "Moran spectral null", "burt": "variogram null (brainsmash)"}

SLICES = [("z", 2), ("y", -10)]          # basal ganglia + thalamus; amygdala/hippocampus


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("div", DIVERGING)


# ------------------------------------------------------------ simulation ---
def simulate_state_means(parc: rm.Parcellator, components: dict, seed: int,
                         n_frames: int = 6000, frame_noise: float = 2.0,
                         pattern_noise: float = 0.45) -> pd.DataFrame:
    """Frames -> labels -> per-state mean, in `state-means`' CSV layout."""
    rng = np.random.default_rng(seed)
    P = parc.n
    cen = parc.centroids
    d = np.linalg.norm(cen[:, None] - cen[None], axis=-1)
    smooth = np.exp(-(d / 25.0) ** 2)                 # 25 mm Gaussian kernel
    smooth /= smooth.sum(1, keepdims=True)

    patterns = []
    for mix in SIM_STATES.values():
        x = sum(w * components[c] for c, w in mix.items())
        x = (x - x.mean()) / (x.std() or 1.0)
        noise = smooth @ rng.normal(size=P)
        x = x + pattern_noise * noise / (noise.std() or 1.0)
        patterns.append((x - x.mean()) / x.std())
    patterns = np.array(patterns)
    K = len(patterns)

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
    return out


def read_state_means(path: Path, parc: rm.Parcellator) -> pd.DataFrame:
    """`state-means --csv` -> its parcel columns. A column the atlas has and
    the CSV lacks means the states came from another atlas: hard error."""
    t = pd.read_csv(path, index_col="state")
    missing = [c for c in parc.columns if c not in t.columns]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} of {parc.n} `{parc.spec.name}` columns "
                         f"missing, e.g. {missing[:3]}")
    return t[["n_rows", "share"] + parc.columns]


# ---------------------------------------------------------------- drawing ---
VIEWS = [("left", "lateral"), ("left", "medial"), ("right", "lateral"), ("right", "medial")]


class Brains:
    """Everything a state drawing needs, loaded once per atlas."""

    def __init__(self, parc: rm.Parcellator, surf: rm.Surface, deep: np.ndarray):
        from nilearn import datasets

        self.parc = parc
        self.deep = deep.reshape(parc.ref.shape[:3]) > 0   # slices show subcortex only
        self.fs = datasets.load_fsaverage("fsaverage5")
        self.sulc = datasets.load_fsaverage_data(mesh="fsaverage5", data_type="sulcal")
        self.template = datasets.load_mni152_template(resolution=1)
        self.vlab = rm.vertex_labels(parc, surf)

    def surface(self, fig, cells, values, vlim):
        from nilearn import plotting

        vals = np.concatenate([[np.nan], np.asarray(values, float)])
        for cell, (hemi, view) in zip(cells, VIEWS):
            ax = fig.add_subplot(cell, projection="3d")
            h = "L" if hemi == "left" else "R"
            plotting.plot_surf(
                self.fs["inflated"].parts[hemi], surf_map=vals[self.vlab[h]],
                bg_map=self.sulc.data.parts[hemi], hemi=hemi, view=view, axes=ax,
                figure=fig, cmap=diverging_cmap(), vmin=-vlim, vmax=vlim,
                colorbar=False, avg_method="median")
            ax.set_box_aspect(None, zoom=1.42)
            ax.set_axis_off()

    def slices(self, fig, cells, values, vlim):
        from nilearn import plotting

        img = self.parc.paint(values)
        img = img.__class__(np.where(self.deep, img.get_fdata(), 0).astype(np.float32), img.affine)
        for cell, (mode, cut) in zip(cells, SLICES):
            ax = fig.add_subplot(cell)
            plotting.plot_stat_map(
                img, bg_img=self.template, display_mode=mode, cut_coords=[cut],
                axes=ax, figure=fig, colorbar=False, cmap=diverging_cmap(),
                vmin=-vlim, vmax=vlim, symmetric_cbar=True, threshold=1e-3,
                annotate=False, draw_cross=False, black_bg=False, dim=-0.6,
                resampling_interpolation="nearest")
            ax.text(0.03, 0.03, f"{mode}={cut}", transform=ax.transAxes, fontsize=6,
                    color=MUTED, va="bottom")


def heat_panel(ax, res, header_colors=None):
    import matplotlib.patches as mpatches

    r, p, q, names = res["r"], res["p"], res["q"], res["names"]
    K, M = r.shape
    im = ax.imshow(r, cmap=diverging_cmap(), vmin=-1, vmax=1, aspect="auto",
                   interpolation="nearest")
    for x in np.arange(-0.5, M, 1.0):
        ax.axvline(x, color="white", lw=1.2)
    for y in np.arange(-0.5, K, 1.0):
        ax.axhline(y, color="white", lw=3.0)
    for i in range(K):
        for j in range(M):
            if not np.isfinite(p[i, j]):
                continue
            mark = ("★" if q[i, j] < 0.01 else "✱" if q[i, j] < 0.05
                    else "•" if p[i, j] < 0.05 else "")
            if mark:
                ax.text(j, i, mark, ha="center", va="center",
                        fontsize=10 if mark == "•" else 7,
                        color="white" if abs(r[i, j]) > 0.55 else INK)
    ax.set_xticks(range(M))
    ax.set_xticklabels([rm.short(n) for n in names], rotation=62, ha="left",
                       fontsize=7, color=INK2)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=10 if header_colors is not None else 2)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if header_colors is not None:
        for j, c in enumerate(header_colors):
            ax.add_patch(mpatches.Rectangle((j - 0.4, 1.006), 0.8, 0.012,
                                            transform=ax.get_xaxis_transform(),
                                            color=c, clip_on=False, lw=0))
    domains = [n.split(".")[0] for n in names]
    if any("." in n for n in names):
        # Brackets under the strip: one per BrainMap domain.
        tr = ax.get_xaxis_transform()
        start = 0
        for j in range(1, M + 1):
            if j == M or domains[j] != domains[start]:
                ax.plot([start - 0.35, j - 0.65], [-0.012, -0.012], transform=tr,
                        color=INK2, lw=1, clip_on=False)
                ax.text((start + j - 1) / 2, -0.03, domains[start], transform=tr,
                        ha="center", va="top", fontsize=7.5, color=INK, clip_on=False)
                start = j
    if res["simulated"]:
        ax.add_patch(mpatches.Rectangle((-0.5, -0.5), M, K, fill=False, hatch="////",
                                        ec=SIMULATED_INK, lw=0, alpha=0.3))
    return im


def overview_figure(atlas, src, state_df, names, results, brains, out: Path, simulated_states,
                    null="moran"):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    X = state_df[brains.parc.columns].to_numpy(float)
    Z = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    K = len(Z)

    cell_w = {"molecular": 0.25, "behavioural": 0.2, "yeo7": 0.26, "networks14": 0.26}
    widths, cols = [0.95, 5.0, 1.9], {}
    for f in FAMILIES:
        widths.append(0.35)
        cols[f] = len(widths)
        widths.append(len(results[f]["names"]) * cell_w[f])
    H = 1.3 * K + 3.4
    fig = plt.figure(figsize=(sum(widths) + 0.4, H), facecolor="white")
    gs = GridSpec(K, len(widths), figure=fig, width_ratios=widths, left=0.008,
                  right=0.995, top=1 - 2.1 / H, bottom=1.3 / H, wspace=0.0, hspace=0.0)
    for i in range(K):
        lab = fig.add_subplot(gs[i, 0])
        lab.set_axis_off()
        lab.text(0.95, 0.62, f"S{i + 1}", ha="right", va="center", fontsize=12,
                 fontweight="bold", color=INK, transform=lab.transAxes)
        lab.text(0.95, 0.40, names[i], ha="right", va="center", fontsize=8,
                 color=INK2, transform=lab.transAxes)
        lab.text(0.95, 0.22, f"{state_df['share'].iloc[i]:.0%} of frames", ha="right",
                 va="center", fontsize=7, color=MUTED, transform=lab.transAxes)
        sc = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[i, 1], wspace=-0.12)
        brains.surface(fig, [sc[0, j] for j in range(4)], Z[i], vlim=2.5)
        ss = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[i, 2], wspace=0.04)
        brains.slices(fig, [ss[0, j] for j in range(2)], Z[i], vlim=2.5)

    for fam in FAMILIES:
        res = results[fam]
        ax = fig.add_subplot(gs[:, cols[fam]])
        im = heat_panel(ax, res, header_colors=rm.YEO7_COLORS if fam == "yeo7" else None)
        sim = res["simulated"]
        fig.text(ax.get_position().x0, 1 - 0.78 / H,
                 TITLES[fam] + ("\nSIMULATED table" if sim else ""), ha="left", va="top",
                 fontsize=9.5, fontweight="bold", color=SIMULATED_INK if sim else INK)

    P = brains.parc.n
    no_null = not np.isfinite(results["molecular"]["p"]).any()
    cax = fig.add_axes([0.60, 0.42 / H, 0.12, 0.10 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label(f"Pearson r across {P} parcels" + (
        f" · no p-values: {P} parcels is a description, not a sample" if no_null else
        f" · {NULL_LABEL[null]}: • p<.05 uncorrected, ✱ q<.05, ★ q<.01 (BH within family)"),
        fontsize=7, color=INK2)
    bax = fig.add_axes([0.08, 0.42 / H, 0.12, 0.10 / H])
    cb2 = fig.colorbar(ScalarMappable(Normalize(-2.5, 2.5), diverging_cmap()), cax=bax,
                       orientation="horizontal", ticks=[-2, 0, 2])
    cb2.set_label("state mean activation (z across parcels)", fontsize=7, color=INK2)
    for c in (cb, cb2):
        c.ax.tick_params(labelsize=7, length=2, colors=INK2)
        c.outline.set_visible(False)

    head = (f"Brain states × reference maps — {atlas} ({P} parcels) — PILOT"
            + ("  (simulated states)" if simulated_states else ""))
    fig.text(0.008, 1 - 0.12 / H, head, ha="left", va="top", fontsize=13,
             fontweight="bold", color=INK)
    fig.text(0.008, 1 - 0.45 / H,
             f"atlas: {src} · cortex: fsaverage5, lateral/medial, left then right · "
             f"subcortex (Brainnetome amygdala, hippocampus, basal ganglia, thalamus): "
             f"MNI slices {', '.join(f'{m}={c}' for m, c in SLICES)}",
             ha="left", va="top", fontsize=8, color=INK2)
    if results["behavioural"]["simulated"]:
        fig.text(0.995, 0.06 / H, "Behavioural panel: SIMULATED stand-in table on the "
                 "real Brainnetome atlas (BrainMap profiles not loaded yet).",
                 ha="right", va="bottom", fontsize=7, color=SIMULATED_INK)
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)


def state_card(i, state_df, names, results, brains, out: Path, null="moran"):
    """One state at reading size: brain + ranked bars per family."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    X = state_df[brains.parc.columns].to_numpy(float)
    z = (X[i] - X[i].mean()) / X[i].std()

    fig = plt.figure(figsize=(17, 5.2), facecolor="white")
    gs = GridSpec(1, 5, figure=fig, width_ratios=[4.6, 2.1, 2.5, 2.0, 2.1], left=0.005,
                  right=0.99, top=0.80, bottom=0.1, wspace=0.62)
    left = GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[0, 0], wspace=-0.05,
                                   hspace=-0.05, height_ratios=[1, 1, 0.9])
    brains.surface(fig, [left[0, 0], left[1, 0], left[0, 1], left[1, 1]], z, vlim=2.5)
    brains.slices(fig, [left[2, 0], left[2, 1]], z, vlim=2.5)

    for c, fam in enumerate(FAMILIES, start=1):
        res = results[fam]
        r, p, q, nm = res["r"][i], res["p"][i], res["q"][i], np.array(res["names"])
        top = CARD_TOP[fam]
        order = np.argsort(-np.abs(np.nan_to_num(r)))[:top]
        order = order[np.argsort(-r[order])]
        ax = fig.add_subplot(gs[0, c])
        y = np.arange(len(order))[::-1]
        if fam == "yeo7":
            colors = [rm.YEO7_COLORS[k] for k in order]
        else:   # sign carries the colour, length the size
            colors = [DIVERGING[5] if v >= 0 else DIVERGING[1] for v in r[order]]
        ax.barh(y, r[order], color=colors, height=0.72, edgecolor="white", lw=1)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(y)
        lab = [n.replace(".", " · ") for n in nm[order]] if fam == "behavioural" else list(nm[order])
        ax.set_yticklabels(lab, fontsize=7.5, color=INK)
        for yy, k in zip(y, order):
            star = ""
            if np.isfinite(p[k]):
                star = "★" if q[k] < 0.01 else "✱" if q[k] < 0.05 else "•" if p[k] < 0.05 else ""
            ax.text(r[k] + (0.03 if r[k] >= 0 else -0.03), yy,
                    f"{r[k]:+.2f}{(' ' + star) if star else ''}",
                    ha="left" if r[k] >= 0 else "right", va="center", fontsize=7, color=INK2)
        ax.set_xlim(-1.1, 1.1)
        ax.set_xticks([-1, 0, 1])
        ax.tick_params(labelsize=7, colors=INK2, length=2)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.set_xlabel("r (parcels)", fontsize=7.5, color=INK2)
        sim = res["simulated"]
        sub = "" if top >= len(nm) else f"\ntop {top} by |r|"
        ax.set_title(CARD_TITLES[fam] + ("  SIMULATED" if sim else "") + sub,
                     fontsize=9, color=SIMULATED_INK if sim else INK, loc="left")
    P = brains.parc.n
    has_p = np.isfinite(results["molecular"]["p"]).any()
    fig.text(0.008, 0.975, f"S{i + 1} · {names[i]}   ({brains.parc.spec.name}, {P} parcels)",
             fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.008, 0.91, f"{state_df['share'].iloc[i]:.0%} of frames · mean activation, z "
             f"across parcels · " + (f"{NULL_LABEL[null]}: • p<.05 uncorrected, ✱ q<.05, ★ q<.01 "
                                     "(BH within family)" if has_p else
                                     f"no p-values ({P} parcels)"),
             fontsize=8, color=INK2, va="top")
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------- main ---
def run_atlas(atlas, args, ref, surf, refs_vox, subcort):
    spec, src = rm.state_atlas(atlas, args.data)
    parc = rm.Parcellator.build(spec, ref)
    out = args.out / atlas
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{atlas}] {parc.n} parcels ({src})")

    parcel_maps = {f: parc.parcellate(refs_vox[f].voxels) for f in FAMILIES}
    comps = {n: parcel_maps["yeo7"][k] for k, n in enumerate(rm.YEO7)}
    comps.update({g: parc.parcellate(v) for g, v in subcort.items()})

    if args.state_means:
        state_df = read_state_means(args.state_means, parc)
        names = [f"state {s}" for s in state_df.index]
    else:
        state_df = simulate_state_means(parc, comps, seed=args.seed)
        names = list(SIM_STATES)
        state_df.to_csv(out / "simulated_state_means.csv")
    X = state_df[parc.columns].to_numpy(float)

    results, rows = {}, []
    for k, fam in enumerate(FAMILIES):
        res = rm.null_test(X, parcel_maps[fam], parc.centroids, args.n_nulls,
                           seed=args.seed + 1000 * k, method=args.null)
        res.update(names=refs_vox[fam].names, simulated=refs_vox[fam].simulated)
        results[fam] = res
        for i in range(len(X)):
            for j, n in enumerate(res["names"]):
                rows.append({"atlas": atlas, "state": i + 1, "state_name": names[i],
                             "family": fam, "map": n, "r": res["r"][i, j],
                             "p_null": res["p"][i, j], "q_fdr": res["q"][i, j],
                             "map_simulated": res["simulated"]})
        sig = int(np.nansum(res["q"] < 0.05)) if np.isfinite(res["q"]).any() else "-"
        print(f"[{atlas}] {fam}: {len(res['names'])} maps, q<.05: {sig}")
    pd.DataFrame(rows).to_csv(out / "state_map_correlations.csv", index=False)

    brains = Brains(parc, surf, deep=sum(subcort.values()))
    overview_figure(atlas, src, state_df, names, results, brains,
                    out / "state_maps_overview.png", simulated_states=not args.state_means,
                    null=args.null)
    for i in range(len(X)):
        state_card(i, state_df, names, results, brains, out / f"state_card_S{i + 1}.png",
                   null=args.null)
    (out / "provenance.json").write_text(json.dumps({
        "atlas": atlas, "atlas_source": src, "n_parcels": parc.n,
        "states": str(args.state_means) if args.state_means else "simulated",
        "simulation_truth": None if args.state_means else SIM_STATES,
        "families": {f: {"source": refs_vox[f].source, "maps": refs_vox[f].names,
                         "simulated": refs_vox[f].simulated} for f in FAMILIES},
        "null": (f"{args.null}, {args.n_nulls} surrogates" if parc.n >= rm.MIN_PARCELS_FOR_NULL
                 else f"none ({parc.n} parcels)"),
        "seed": args.seed,
    }, indent=2, ensure_ascii=False))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=HERE / "data",
                   help="where fetch_reference_maps.sh put the maps")
    p.add_argument("--atlases", nargs="+", default=ATLASES, choices=ATLASES)
    p.add_argument("--state-means", type=Path,
                   help="CSV from `fmri-decomp state-means --csv` (one atlas only)")
    p.add_argument("--behavioural", type=Path,
                   help="real Brainnetome behavioural table (bn_index + one column per category)")
    p.add_argument("--n-nulls", type=int, default=10000,
                   help="surrogates per map. The smallest p is 1/(n+1), and BH over a "
                        "family of 114 (6 states x 19 PET maps) needs it well under "
                        "0.05/114: 1000 cannot reach q<.05 at all")
    p.add_argument("--null", choices=rm.NULLS, default="moran",
                   help="moran: Moran spectral randomisation (seconds); burt: brainsmash "
                        "variogram matching (hours for every family -- a cluster job)")
    p.add_argument("--out", type=Path, default=HERE / "figures")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)
    if args.state_means and len(args.atlases) != 1:
        p.error("--state-means is one atlas's states: pass exactly one --atlases")

    ref = rm.reference_grid(args.data)
    surf = rm.load_surface(args.data)
    bt = rm.load_behavioural_table(args.behavioural) if args.behavioural else None
    refs_vox = {
        "molecular": rm.receptor_maps(args.data, ref),
        "behavioural": rm.behavioural_maps(args.data, ref, bt, seed=args.seed),
        "yeo7": rm.yeo7_maps(args.data, ref),
        "networks14": rm.network14_maps(ref),
    }
    subcort = rm.subcortical_structures(args.data, ref)
    for atlas in args.atlases:
        run_atlas(atlas, args, ref, surf, refs_vox, subcort)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
