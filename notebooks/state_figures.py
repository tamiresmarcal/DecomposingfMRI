#!/usr/bin/env python3
"""Brain states x reference maps: the FIGURES, drawn from the tables
`state_correlations.py` wrote. Nothing here recomputes a number.

    python notebooks/state_figures.py notebooks/state_maps_results/simulated
    python notebooks/state_figures.py RUN_DIR --atlases harvardoxford \
        --bn-levels domain subcategory

Per atlas: one overview -- a row per state: cortex (lateral and medial, left
then right, fsaverage5) and two subcortical slices, then one heat strip per
reference family -- and one card per state with ranked bars. The families and
the BrainMap depth shown are chosen here; any subset of what the run computed.
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
from state_correlations import BN_LEVELS, FAMILIES  # noqa: E402

# Diverging blue <-> red with a neutral grey midpoint.
DIVERGING = ["#104281", "#3987e5", "#cde2fb", "#f0efec", "#f8d3d2", "#e34948", "#8f1d1c"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SIMULATED_INK = "#b4560f"

TITLES = {
    "molecular": "Molecular · PET receptors (neuromaps)",
    "behavioural_domain": "Behavioural · BrainMap domains",
    "behavioural": "Behavioural · BrainMap subcategories × Brainnetome (246)",
    "yeo7": "Canonical · Yeo 7",
    "networks14": "Canonical · our 14 networks",
}
CARD_TITLES = {"molecular": "Molecular (PET)", "behavioural_domain": "BrainMap domains",
               "behavioural": "BrainMap subcategories", "yeo7": "Yeo 7 networks",
               "networks14": "Our 14 networks"}
CARD_TOP = {"molecular": 8, "behavioural_domain": 5, "behavioural": 10, "yeo7": 7,
            "networks14": 14}
CELL_W = {"molecular": 0.25, "behavioural_domain": 0.3, "behavioural": 0.2, "yeo7": 0.26,
          "networks14": 0.26}
NULL_LABEL = {"moran": "Moran spectral null", "burt": "variogram null (brainsmash)"}

SLICES = [("z", 2), ("y", -10)]          # basal ganglia + thalamus; amygdala/hippocampus


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("div", DIVERGING)


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

    fams = list(results)
    widths, cols = [0.95, 5.0, 1.9], {}
    for f in fams:
        widths.append(0.35)
        cols[f] = len(widths)
        widths.append(len(results[f]["names"]) * CELL_W[f])
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

    for fam in fams:
        res = results[fam]
        ax = fig.add_subplot(gs[:, cols[fam]])
        im = heat_panel(ax, res, header_colors=rm.YEO7_COLORS if fam == "yeo7" else None)
        sim = res["simulated"]
        fig.text(ax.get_position().x0, 1 - 0.78 / H,
                 TITLES[fam] + ("\nSIMULATED table" if sim else ""), ha="left", va="top",
                 fontsize=9.5, fontweight="bold", color=SIMULATED_INK if sim else INK)

    P = brains.parc.n
    no_null = not any(np.isfinite(r["p"]).any() for r in results.values())
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
    if any(r["simulated"] for r in results.values()):
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

    fams = list(results)
    fig = plt.figure(figsize=(5.2 + 2.6 * len(fams), 5.2), facecolor="white")
    gs = GridSpec(1, 1 + len(fams), figure=fig, width_ratios=[4.6] + [2.2] * len(fams),
                  left=0.005, right=0.99, top=0.80, bottom=0.1, wspace=0.62)
    left = GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[0, 0], wspace=-0.05,
                                   hspace=-0.05, height_ratios=[1, 1, 0.9])
    brains.surface(fig, [left[0, 0], left[1, 0], left[0, 1], left[1, 1]], z, vlim=2.5)
    brains.slices(fig, [left[2, 0], left[2, 1]], z, vlim=2.5)

    for c, fam in enumerate(fams, start=1):
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
    has_p = any(np.isfinite(r["p"]).any() for r in results.values())
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
def load_results(atlas_dir: Path, families) -> tuple[pd.DataFrame, dict, dict]:
    """state_means.csv + correlations.csv -> (states, {family: r/p/q/names}, provenance)."""
    sm = pd.read_csv(atlas_dir / "state_means.csv", index_col="state")
    cor = pd.read_csv(atlas_dir / "correlations.csv")
    prov = json.loads((atlas_dir / "provenance.json").read_text())
    states = list(sm.index)
    results = {}
    for fam in families:
        c = cor[cor["family"] == fam]
        if c.empty:
            raise SystemExit(f"{atlas_dir}: no `{fam}` rows -- run state_correlations "
                             f"with that family first")
        names = list(dict.fromkeys(c["map"]))
        piv = {k: c.pivot(index="state", columns="map", values=v).loc[states, names].to_numpy(float)
               for k, v in (("r", "r"), ("p", "p_null"), ("q", "q_fdr"))}
        results[fam] = dict(piv, names=names, simulated=bool(c["map_simulated"].iloc[0]))
    return sm, results, prov


def draw(run_dir, atlases=None, families=None, bn_levels=("subcategory",), *,
         data: Path = HERE / "state_maps_data", cards: bool = True) -> list:
    """Draw every atlas under `run_dir`; return the figure paths.

    `families` picks the non-behavioural panels; `bn_levels` ("domain",
    "subcategory", or both) picks the BrainMap depth(s) shown.
    """
    run_dir = Path(run_dir)
    atlases = atlases or sorted(p.name for p in run_dir.iterdir() if (p / "correlations.csv").exists())
    families = [f for f in (families or ["molecular", "yeo7", "networks14"])
                if not f.startswith("behavioural")]
    chosen = families + [BN_LEVELS[b] for b in bn_levels]
    chosen = [f for f in FAMILIES if f in chosen]
    ref = rm.reference_grid(data)
    surf = rm.load_surface(data)
    deep = sum(rm.subcortical_structures(data, ref).values())
    paths = []
    for atlas in atlases:
        d = run_dir / atlas
        sm, results, prov = load_results(d, chosen)
        spec, _ = rm.state_atlas(atlas, data)
        parc = rm.Parcellator.build(spec, ref)
        brains = Brains(parc, surf, deep=deep)
        names = list(sm["name"]) if "name" in sm else [f"state {s}" for s in sm.index]
        null = prov.get("null_method", "moran")
        out = d / "state_maps_overview.png"
        overview_figure(atlas, prov["atlas_source"], sm, names, results, brains, out,
                        simulated_states=prov["states"] == "simulate", null=null)
        paths.append(out)
        if cards:
            for i in range(len(sm)):
                p = d / f"state_card_S{i + 1}.png"
                state_card(i, sm, names, results, brains, p, null=null)
                paths.append(p)
        print(f"[{atlas}] figures -> {d}")
    return paths


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="<out>/<states>/ from state_correlations.py")
    p.add_argument("--atlases", nargs="+", default=None)
    p.add_argument("--families", nargs="+", default=["molecular", "yeo7", "networks14"],
                   choices=["molecular", "yeo7", "networks14"])
    p.add_argument("--bn-levels", nargs="+", default=["subcategory"], choices=list(BN_LEVELS))
    p.add_argument("--no-cards", action="store_true")
    p.add_argument("--data", type=Path, default=HERE / "state_maps_data")
    a = p.parse_args(argv)
    draw(a.run_dir, a.atlases, a.families, a.bn_levels, data=a.data, cards=not a.no_cards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
