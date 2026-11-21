"""Figures of 11_normal_brain_cohorts.ipynb, drawn from unsupervised_model.cohort_tables. The state diagram and its
dynamics column are notebook 10's (figures.hads), with all their label rules."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from fmri_decomposition.figures import hads as hp
from fmri_decomposition.figures import state_names as sn
from fmri_decomposition.supervised_model import hads as ht

# one colour per cohort, in the order they are listed; a new cohort takes the next
PALETTE = ["#eda100", "#008300", "#6250d6", "#52514e", "#e34948", "#1baf7a", "#8c6d4f"]
DYN_UNIT = {"switch_rate": " (share of steps)", "mean_dwell_s": " (s)", "entropy_rate_bits": " (bits)",
            "dispersion": " (latent-space units)", "n_distinct_transitions": " (cells of K×K)",
            "self_transition_rate": " (share of steps)", "switches_per_min": " (per min)", "n_states_visited": " (states)"}


# how a cohort is named in the figures
COHORT_NAMES = {"ds002837": "NNDb", "cneuromod": "CNeuroMod", "camcan": "Cam-CAN", "hcp7t_fix": "HCP",
                "hcp7t": "HCP"}


def cname(c):
    return COHORT_NAMES.get(c, c)


def colours(cfg):
    return dict(zip(cfg.cohorts, PALETTE))

def save(fig, name, cfg):
    if cfg.save_dir:
        hp.Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        for ext in ("png", "svg"):
            fig.savefig(hp.Path(cfg.save_dir) / f"{name}.{ext}", dpi=200, bbox_inches="tight")


# ------------------------------------------------------------------ 1 rest | movie per cohort
def fig_diagrams(co, graphs, marks, dyn, fs=1.2, gap=2.0, away=4.0):
    """Rows = each cohort, then their mean; columns = rest | movie. Blue = movie differs from the same
    people's rest (task_change). The dynamics sit left of rest and right of movie."""
    cfg, K = co.cfg, co.cfg.K
    rows = [c for c in cfg.cohorts if (c, "rest") in graphs or (c, "movie") in graphs] + ["mean"]
    off = [(a, b) for a in range(K) for b in range(K) if a != b]
    base = set()
    for kind in ("rest", "movie"):
        if ("mean", kind) in graphs:
            base |= set(sorted(off, key=lambda e: -graphs["mean", kind][0][e])[:cfg.edge_top])
    hit, note = {}, {}
    for c in rows:
        t = marks[(marks["cohort"] == c) & marks["variable"].isin(co.cells)].set_index("variable").reindex(co.cells)
        hit[c] = t["hit"].fillna(False).to_numpy(bool).reshape(K, K)
        if (c, "rest") in graphs and (c, "movie") in graphs:
            nt = np.vectorize(ht.pp)(graphs[c, "movie"][0] - graphs[c, "rest"][0])
            nt[~hit[c]] = ""
        else:
            nt = np.full((K, K), "", dtype=object)
        note[c] = nt
    edges = {c: sorted(base | {e for e in off if hit[c][e]}) for c in rows}
    wmax = max(graphs[k][0][e] for k in graphs for e in edges[k[0]])
    omax = max(g[1].max() for g in graphs.values())
    fig, axes = plt.subplots(len(rows), 2, figsize=(12.5, 5.3 * len(rows)), squeeze=False)
    n = co.counts()
    for i, c in enumerate(rows):
        if c == "mean":
            k_rest = sum((x, "rest") in graphs for x in rows[:-1])
            k_movie = sum((x, "movie") in graphs for x in rows[:-1])
        who = f"{c} · {co.role(c).capitalize()}" if c != "mean" else None
        for j, kind in enumerate(("rest", "movie")):
            ax = axes[i, j]
            if (c, kind) not in graphs:
                ax.axis("off")
                ax.text(.5, .5, f"{c}\nno {kind} scan", ha="center", va="center", fontsize=11, color=hp.MUTED, transform=ax.transAxes)
                continue
            if kind == "rest":
                hp.diagram(ax, *graphs[c, kind], sorted(base), wmax, omax, fs=fs, room=fs)
            else:
                hp.diagram(ax, *graphs[c, kind], edges[c], wmax, omax, hit_T=hit[c], note_T=note[c], colour=hp.BLUE, fs=fs, room=fs)
            if c == "mean":
                k = k_rest if kind == "rest" else k_movie
                lead = f"Mean of the {k} cohort{'s' if k > 1 else ''} with {kind}"
            else:
                lead = f"{who} (n={int(n.loc[c, kind])})"
            ax.set_title(f"{lead} · {'Rest' if kind == 'rest' else 'Movie vs Rest'}", fontsize=11)
            ax.set_xlim(*((-2.6, 1.6) if j == 0 else (-1.6, 2.6)))
    fig.legend(handles=[Line2D([], [], color=hp.BLUE, lw=2.5,
                               label=f"blue = Movie differs from the same people's Rest (paired, FDR<{cfg.alpha:g} & "
                                     f"|log2FC|≥{cfg.fc_task:g}); label = change in pp"),
                        Line2D([], [], ls="none", label="beside: ⇄ switch rate · stopwatch = mean dwell · H entropy rate · "
                                                         f"✳ dispersion · # distinct transitions (blue: |log2FC|≥{cfg.fc_dyn:g})")],
               loc="lower center", fontsize=9.5)
    fig.suptitle(f"1  Normal brain · Rest | Movie per cohort and their mean · {ht.set_label(cfg.space)}",
                 x=.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, .6 / fig.get_figheight(), 1, 1 - .45 / fig.get_figheight()))
    fig.canvas.draw()
    for i, c in enumerate(rows):                                  # the dynamics, at the panel's real scale
        for j, kind in enumerate(("rest", "movie")):
            d = dyn[(dyn["cohort"] == c) & (dyn["task"] == kind)]
            if (c, kind) not in graphs or d.empty:
                continue
            ax = axes[i, j]
            ax.apply_aspect()
            b = ax.get_position()
            (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
            pt = max((x1 - x0) / (b.width * fig.get_figwidth() * 72), (y1 - y0) / (b.height * fig.get_figheight() * 72))
            hp.dynamics_column(ax, d, -1 if j == 0 else 1, hp.BLUE if kind == "movie" else hp.INK2, fs, 1.0, gap, pt, away)
    save(fig, "1_diagrams", cfg)
    plt.show()


# ------------------------------------------------------------------ 2 equivalence (TOST) and effect size (Cohen)
def _rows(ax, t, y0, x, lo, hi, scale, col, offsets, filled):
    for r in t.itertuples():
        y = y0 + offsets[r.cohort]
        ax.plot([scale * getattr(r, lo), scale * getattr(r, hi)], [y, y], color=col[r.cohort], lw=1.4)
        f = filled(r)
        ax.plot(scale * getattr(r, x), y, "o", ms=5, color=col[r.cohort], mfc=col[r.cohort] if f else "white", mew=1.4)

def _three_panels(co, title, draw_block, draw_dyn, legend, name, dyn_title, height=8):
    """Transitions (the strongest), occupancy, and each dynamics on its own axis."""
    cfg = co.cfg
    fig = plt.figure(figsize=(17, height))
    outer = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=.4)
    draw_block(fig.add_subplot(outer[0]), f"Transitions p(j | i) · the {cfg.top_transitions} strongest", co.strongest())
    draw_block(fig.add_subplot(outer[1]), "Occupancy (time in state)", co.occ)
    dyn_axes = [fig.add_subplot(x) for x in outer[2].subgridspec(len(co.dyn), 1, hspace=1.2)]
    for ax, v in zip(dyn_axes, co.dyn):
        draw_dyn(ax, v)
        ax.set_ylim(-.5, .5); ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=8)
    if dyn_axes:
        dyn_axes[0].text(.5, 2.3, dyn_title, transform=dyn_axes[0].transAxes, ha="center", fontsize=11)
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(f"{title} · {ht.set_label(cfg.space)}", x=.01, ha="left", fontsize=11)
    fig.subplots_adjust(left=.05, right=.95, top=1 - .9 / height, bottom=1.05 / height)
    save(fig, name, cfg)
    plt.show()

def _check(ax, ok, y=None):
    kw = dict(transform=ax.get_yaxis_transform(), x=1.02, y=y) if y is not None else dict(transform=ax.transAxes, x=1.03, y=0)
    ax.text(kw.pop("x"), kw.pop("y"), "✓" if ok else "✗", va="center", fontsize=11, color=hp.INK if ok else hp.RED, **kw)

def fig_tost(co, agree):
    """2a. Movie, in the variables' own units and fixed ranges: each cohort's mean (90% CI). Filled =
    equivalent to the others (TOST, 5%); * = differs from the others (Welch, BH-FDR < alpha)."""
    cfg, col = co.cfg, colours(co.cfg)
    labels = list(agree["cohort"].unique())
    offsets = dict(zip(labels, np.linspace(.28, -.28, len(labels)) if len(labels) > 1 else [0]))
    eqf = lambda r: r.equivalent

    def stars(ax, t, y0, scale):
        for r in t[t["differs"]].itertuples():
            ax.text(scale * r.mean_hi, y0 + offsets[r.cohort], " *", va="center", fontsize=11, color=col[r.cohort],
                    fontweight="bold")

    def block(ax, title, vs):
        for y, v in enumerate(vs[::-1]):
            t = agree[agree["variable"] == v]
            _rows(ax, t, y, "mean", "mean_lo", "mean_hi", 100, col, offsets, eqf)
            stars(ax, t, y, 100)
        ax.set_yticks(range(len(vs)), vs[::-1], fontsize=9)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(100))
        ax.set_xlabel("cohort mean, 90% CI", fontsize=9)
        ax.set_title(title, fontsize=11)

    def dyn(ax, v):
        t = agree[agree["variable"] == v]
        _rows(ax, t, 0, "mean", "mean_lo", "mean_hi", 1, col, offsets, eqf)
        stars(ax, t, 0, 1)
        lo_, hi_ = fixed_range(co, v)
        if np.isfinite(lo_):
            ax.set_xlim(lo_, hi_)
        ax.set_title(v + DYN_UNIT.get(v, ""), fontsize=9, loc="left", pad=3)

    legend = ([Line2D([], [], ls="none", marker="o", color=col[c], label=f"{c} ({co.role(c)})") for c in labels] +
              [Line2D([], [], ls="none", marker="o", mfc="white", mec=hp.INK2, label="hollow = not shown equivalent (TOST, 5%)"),
               Line2D([], [], ls="none", marker="$*$", ms=9, color=hp.INK2, label=f"* = differs from the others (Welch, FDR<{cfg.alpha:g})")])
    _three_panels(co, "2a  Movie: is every cohort equivalent to the others? (cohort means, in the variables' units)",
                  block, dyn, legend, "2a_tost", "Dynamics (each on its own axis)", height=6.2)

def fig_cohen(co, d):
    """2b. Movie: each cohort vs all the others as Cohen's d (95% CI); bands at 0.2 / 0.5 / 0.8."""
    col = colours(co.cfg)
    labels = list(d["cohort"].unique())
    offsets = dict(zip(labels, np.linspace(.3, -.3, len(labels)) if len(labels) > 1 else [0]))
    small = lambda r: abs(r.d) < .2
    lim = max(1.0, np.ceil(d[["lo", "hi"]].abs().max().max() * 2) / 2)

    def zones(ax, vertical=True):
        for w, a in ((.8, .08), (.5, .14), (.2, .3)):
            ax.axvspan(-w, w, color=hp.MUTED, alpha=a, lw=0)
        ax.axvline(0, color=hp.INK2, lw=.8)
        ax.set_xlim(-lim, lim)

    def block(ax, title, vs):
        zones(ax)
        for y, v in enumerate(vs[::-1]):
            t = d[d["variable"] == v]
            _rows(ax, t, y, "d", "lo", "hi", 1, col, offsets, small)
            _check(ax, (t["d"].abs() < .2).all(), y)
        ax.set_yticks(range(len(vs)), vs[::-1], fontsize=9)
        ax.set_xlabel("Cohen's d, cohort − all the others (95% CI)", fontsize=9)
        ax.set_title(title, fontsize=11)

    def dyn(ax, v):
        zones(ax)
        t = d[d["variable"] == v]
        _rows(ax, t, 0, "d", "lo", "hi", 1, col, offsets, small)
        ax.set_title(v, fontsize=9, loc="left", pad=3)
        _check(ax, (t["d"].abs() < .2).all())

    legend = ([Line2D([], [], ls="none", marker="o", color=col[c], label=f"{c} ({co.role(c)})") for c in labels] +
              [Line2D([], [], ls="none", marker="o", mfc="white", mec=hp.INK2, label="hollow = |d| ≥ 0.2"),
               Line2D([], [], color=hp.MUTED, lw=8, alpha=.5, label="bands: |d| < 0.2 small · 0.5 medium · 0.8 large"),
               Line2D([], [], ls="none", label="✓ = every cohort |d| < 0.2")])
    _three_panels(co, "2b  Movie: how far is each cohort from the others? (Cohen's d)",
                  block, dyn, legend, "2b_cohen", "Dynamics (same d axis)")


# ------------------------------------------------------------------ 3 train vs projected
def fig_train_vs_projected(co, means):
    """One row per projected cohort: every variable's mean against each training cohort's (a dot per
    training cohort); transitions and occupancy linear, the dynamics log-log."""
    cfg, col = co.cfg, colours(co.cfg)
    train = [c for c in means.index if means.loc[c, "role"] == "train"]
    proj = [c for c in means.index if means.loc[c, "role"] == "projected"]
    if not train or not proj:
        print("figure 3 needs at least one train and one projected cohort")
        return
    blocks = [("Transitions p(j | i)", co.cells, "linear"), ("Occupancy", co.occ, "linear"), ("Dynamics", co.dyn, "log")]
    fig, axes = plt.subplots(len(proj), 3, figsize=(17, 5.6 * len(proj)), squeeze=False)
    for i, pc in enumerate(proj):
        for ax, (title, vs, scale) in zip(axes[i], blocks):
            y = means.loc[pc, vs].astype(float)
            for c in train:
                ax.scatter(means.loc[c, vs].astype(float), y, s=24 if scale == "log" else 14, color=col[c], alpha=.85, lw=0)
            allv = means.loc[[*train, pc], vs].astype(float).stack()
            lo, hi = allv.min(), allv.max()
            if scale == "log":
                lo, hi = lo / 1.4, hi * 1.4
                ax.set_xscale("log"); ax.set_yscale("log")
                pts = sorted(((means.loc[train, v].astype(float).max(), y[v], v) for v in vs), key=lambda p: p[1])
                last = -np.inf                                   # names at least 1/14 of a decade apart
                for x, yy, v in pts:
                    ly = max(np.log10(yy), last + np.log10(hi / lo) / 14)
                    last = ly
                    ax.annotate(v, (x, yy), xytext=(x * 1.25, 10 ** ly), fontsize=8, color=hp.INK2, va="center",
                                arrowprops=dict(arrowstyle="-", lw=.4, color=hp.MUTED))
            else:
                pad = .06 * (hi - lo); lo, hi = max(lo - pad, 0), hi + pad
                for a in (ax.xaxis, ax.yaxis):
                    a.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=1))
            ax.plot([lo, hi], [lo, hi], color=hp.INK2, ls="--", lw=.8)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
            f = np.log if scale == "log" else (lambda z: z)
            r = {c: np.corrcoef(f(means.loc[c, vs].astype(float)), f(y))[0, 1] for c in train}
            text = f"r with {pc}: " + ", ".join(f"{c} {r[c]:.2f}" for c in train)
            if len(train) > 1:
                text += f" · train vs train: {np.corrcoef(f(means.loc[train[0], vs].astype(float)), f(means.loc[train[1], vs].astype(float)))[0, 1]:.2f}"
            ax.set_title(f"{title}\n{text}", fontsize=10)
            ax.set_xlabel("training cohort")
        axes[i, 0].set_ylabel(f"projected: {pc}", color=col[pc])
    fig.legend(handles=[Line2D([], [], ls="none", marker="o", color=col[c], label=f"{c} (train)") for c in train] +
                       [Line2D([], [], color=hp.INK2, ls="--", label="equal")], loc="lower center", ncol=len(train) + 1, fontsize=9)
    fig.suptitle(f"3  Does each projected cohort reproduce the training cohorts? Movie · one dot per variable and training "
                 f"cohort · {ht.set_label(cfg.space)}", x=.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, .5 / fig.get_figheight(), 1, 1 - .5 / fig.get_figheight()))
    save(fig, "3_train_vs_projected", cfg)
    plt.show()


# ------------------------------------------------------------------ 4 the matrices, TOST per cell
def _cax(fig, ax):
    """A thin colour-bar axis just right of `ax`, so no panel is shrunk to make room."""
    b = ax.get_position()
    return fig.add_axes([b.x1 + .006, b.y0, .007, b.height])

def _panels(n, height=10, per_row=None):
    """One fixed grid per panel: occupancy on top, the matrix, icons below; `per_row` panels a row
    (None = all in one row). Returns the figure and, per panel, dict(top, left, main, bottom)."""
    cols = min(n, per_row or n)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(7.4 * cols, height * rows))
    outer = fig.add_gridspec(rows, cols, wspace=.38, hspace=.3, left=.04, right=.95,
                             top=1 - .14 / rows, bottom=.03 / rows)
    out = []
    for k in range(n):
        g = outer[k // cols, k % cols].subgridspec(3, 2, width_ratios=[.001, 1], height_ratios=[.3, 1, .3],
                                                   wspace=.04, hspace=.12)
        main = fig.add_subplot(g[1, 1])
        out.append(dict(main=main, top=fig.add_subplot(g[0, 1], sharex=main), left=fig.add_subplot(g[1, 0], sharey=main),
                        bottom=fig.add_subplot(g[2, :])))
    return fig, out

def sig1(x):
    """One significant figure, without exponents: 0.12 -> 0.1, 16 -> 20, 0.57 -> 0.6."""
    v = float(f"{x:.1g}")
    return f"{v:.0f}" if v >= 1 else f"{v:g}"

def _bigger(fig, k=2):
    """Every text of the figure x k."""
    for t in fig.findobj(plt.matplotlib.text.Text):
        t.set_fontsize(t.get_fontsize() * k)

ICON = {"switch_rate": "⇄", "mean_dwell_s": "⏱", "entropy_rate_bits": "H", "dispersion": "✳", "n_distinct_transitions": "#"}
FMT = {"switch_rate": "{:.2f}", "mean_dwell_s": "{:.0f} s", "entropy_rate_bits": "{:.2f} bits", "dispersion": "{:.2f}",
       "n_distinct_transitions": "{:.0f}"}

def _group_colours(co, labels):
    """A cohort keeps its colour; any other group (a task, a film) takes the palette in order."""
    col = colours(co.cfg)
    return {g: col.get(g, PALETTE[k % len(PALETTE)]) for k, g in enumerate(labels)}

def _cell_text(ax, M, vmax, sd_matrix):
    K = len(M)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"±{100 * sd_matrix[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] > .6 * vmax else hp.INK)

def fig_matrices(co, groups, test="tost", sd=False, title=None, name="4_matrix", per_row=5):
    """4. One matrix per group (p(j | i), time in each state on top, the dynamics below), and with
    test="tost" or "cohen" a last panel testing every cell, occupancy and dynamics: each group vs all
    the others. test=None draws the groups only. groups = co.groups(...): {label: one row per subject}.
    sd = each cell's between-subject SD (pp), occupancy ± SD, dynamics ± SD. per_row = panels a row
    (the test panel counts as one)."""
    cfg, K = co.cfg, co.cfg.K
    labels = list(groups)
    col = _group_colours(co, labels)
    G = {g: co._graph(d.mean()) for g, d in groups.items()}
    fig, axes = _panels(len(labels) + (test is not None), per_row=per_row)
    vmax = max(G[g][0].max() for g in labels)
    for P, g in zip(axes, labels):
        P["main"].imshow(G[g][0], cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        role = f" ({co.role(g)})" if g in cfg.cohorts else ""
        h = groups[g]["scan_s"].sum() / 3600 if "scan_s" in groups[g] else np.nan
        P["main"].set_title(f"{g}{role} · n={len(groups[g])}" + (f" · {h:.1f} h" if np.isfinite(h) else ""),
                            fontsize=11, color=col[g])
        if sd:
            _cell_text(P["main"], G[g][0], vmax, groups[g][co.cells].std().to_numpy().reshape(K, K))
    im = axes[0]["main"].images[0]
    fig.colorbar(im, cax=_cax(fig, axes[len(labels) - 1]["main"])).set_label("p(j | i)")
    for P in axes:
        _state_ticks(P["main"], K, 8); P["main"].set_xlabel("to state j", fontsize=9)
        P["main"].set_ylabel("from state i", fontsize=9)
        P["left"].axis("off")

    # occupancy on top of every group
    ymax = max((G[g][1] + (groups[g][co.occ].std().to_numpy() if sd else 0)).max() for g in labels) * 1.15
    for P, g in zip(axes, labels):
        P["top"].bar(range(K), G[g][1], width=.8, color=col[g], alpha=.75,
                     yerr=groups[g][co.occ].std().to_numpy() if sd else None, error_kw=dict(lw=1, capsize=2, ecolor=hp.INK2))
        P["top"].set_ylim(0, ymax)
        P["top"].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
    axes[0]["top"].set_ylabel("time in state", fontsize=8)

    # the dynamics under every group
    def icons(P, texts, colours_):
        b = P["bottom"]
        b.set_xlim(-.5, len(co.dyn) - .5); b.set_ylim(0, 1); b.axis("off")
        for k, (v, text, c) in enumerate(zip(co.dyn, texts, colours_)):
            if ICON.get(v) == "⏱":
                b.plot(k, .62, marker=hp.STOPWATCH, ms=26, mfc="none", mec=c, mew=1.6)
            else:
                b.text(k, .62, ICON.get(v, v[:2]), ha="center", va="center", fontsize=13, color=c)
            b.text(k, .1, text, ha="center", va="center", fontsize=6, color=c, linespacing=1.1)

    for P, g in zip(axes, labels):
        texts = []
        for v in co.dyn:
            x = groups[g][v]
            f = FMT.get(v, "{:.2f}").split(" ")
            texts.append(f[0].format(x.mean()) + (" " + f[1] if len(f) > 1 else "") + (f"\n± {sig1(x.std())}" if sd else ""))
        icons(P, texts, [hp.INK2] * len(co.dyn))

    # the test panel
    what = ""
    if test is not None:
        P = axes[-1]
        ax = P["main"]
        variables = [*co.cells, *co.occ, *co.dyn]
        if test == "cohen":
            d = co.cohen(variables, groups)
            worst = d.assign(a=d["d"].abs()).groupby("variable")["a"].max()
            cmap = plt.matplotlib.colors.ListedColormap(["#f3f2ee", "#d8d6cf", "#f0a8a2", "#e34948"])
            norm = plt.matplotlib.colors.BoundaryNorm([0, .2, .5, .8, max(1, worst.max() + .01)], cmap.N)
            W = np.array([worst.get(c, np.nan) for c in co.cells]).reshape(K, K)
            ax.imshow(W, cmap=cmap, norm=norm, aspect="auto")
            for i in range(K):
                for j in range(K):
                    if np.isfinite(W[i, j]):
                        ax.text(j, i, f"{W[i, j]:.1f}", ha="center", va="center", fontsize=7, color=hp.INK)
            fig.colorbar(ax.images[0], cax=_cax(fig, ax), ticks=[0, .2, .5, .8]).set_label("largest |d|, group vs the others")
            ax.set_title(f"Cohen's d\n{int((W < .2).sum())}/{K * K} cells with every |d| < 0.2", fontsize=11)
            od = worst.reindex(co.occ)
            P["top"].bar(range(K), od.to_numpy(), width=.8, color=[cmap(norm(v)) for v in od])
            P["top"].set_ylim(0, max(1, np.nanmax(od) * 1.1))
            for w in (.2, .5, .8):
                P["top"].axhline(w, color=hp.MUTED, lw=.5, ls=":")
            P["top"].set_ylabel("largest |d|", fontsize=7)
            dd = worst.reindex(co.dyn)
            icons(P, [f"|d| {v:.2f}" for v in dd], [hp.RED if v >= .5 else hp.INK2 for v in dd])
            what = "Cohen's d per cell (< 0.2 small, 0.5 medium, 0.8 large)"
        elif test == "tost":
            t = co.agreement(variables, groups)
            far = t.assign(a=t[["lo", "hi"]].abs().max(axis=1)).groupby("variable")["a"].max()
            eq = t.groupby("variable")["equivalent"].all()
            n_eq = t.groupby("variable")["equivalent"].sum()
            W = np.array([100 * far.get(c, np.nan) for c in co.cells]).reshape(K, K)
            ax.imshow(W, cmap="Greys", vmin=0, vmax=2.5 * np.nanmax(W), aspect="auto")       # light: the marks stay readable
            for i in range(K):
                for j in range(K):
                    c = co.cells[i * K + j]
                    if c not in eq:
                        continue
                    if eq[c]:
                        ax.text(j, i, "✓", ha="center", va="center", fontsize=9, color=hp.BLUE)
                    else:
                        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=hp.RED, lw=2))
                        ax.text(j, i, f"{n_eq[c]}/{len(labels)}", ha="center", va="center", fontsize=7, color=hp.RED)
            fig.colorbar(ax.images[0], cax=_cax(fig, ax)).set_label("largest |group − others| (pp)")
            ax.set_title(f"TOST\n{int(eq.reindex(co.cells).sum())}/{K * K} cells equivalent in every group", fontsize=11)
            P["top"].bar(range(K), 100 * far.reindex(co.occ).to_numpy(), width=.8,
                         color=[hp.MUTED if e else hp.RED for e in eq.reindex(co.occ)])
            P["top"].set_ylabel("largest |diff| (pp)", fontsize=7)
            de = eq.reindex(co.dyn)
            icons(P, ["✓ equiv." if e else "✗" for e in de], [hp.INK2 if e else hp.RED for e in de])
            what = f"TOST per cell, ✓ = every group equivalent (± {cfg.sesoi_d:g} SD), red = k/N"
        else:
            raise SystemExit(f'test must be "tost", "cohen" or None, not {test!r}')
    for P in axes:                                               # titles above the occupancy bars
        P["top"].tick_params(labelbottom=False, labelsize=7)
        P["top"].spines["right"].set_visible(False); P["top"].spines["top"].set_visible(False)
        P["top"].set_title(P["main"].get_title(), fontsize=11, color=P["main"].title.get_color())
        P["main"].set_title("")
    head = title or "4  transitions per group"
    fig.suptitle(f"{head}" + (f" · {what}" if what else "") + f" · {ht.set_label(cfg.space)}", x=.01, ha="left", fontsize=11)
    _bigger(fig)
    save(fig, name + (f"_{test}" if test else "") + ("_sd" if sd else ""), cfg)
    plt.show()


def fig_matrix(co, graphs=None, agree=None, sd=False):
    """4a, the cohorts' movie with the TOST (kept for the notebook's earlier calls)."""
    fig_matrices(co, co.groups("cohort"), "tost", sd, "4a  Movie transitions per cohort", "4a_matrix")

def fig_matrix_cohen(co, graphs=None, d=None, sd=False):
    """4b, the cohorts' movie with Cohen's d (kept for the notebook's earlier calls)."""
    fig_matrices(co, co.groups("cohort"), "cohen", sd, "4b  Movie transitions per cohort", "4b_matrix")


# ------------------------------------------------------------------ 5 the latent space
def _hex(c):
    return plt.matplotlib.colors.to_hex(c)

def fig_latents(co, lat, by="participant", who=None, embedding="umap3", paths=False, same_axes=True, height=560):
    """5. One interactive 3D panel per cohort (plotly: drag to rotate, scroll to zoom), a dot per
    window. by = "participant" (a gradient, one colour per person), "task" (rest, each film),
    "cluster" (the state set the latents were loaded with) or "single": one person per cohort in
    colour, with their path, over everyone else in grey. who = {cohort: subject id, or n = the n-th
    subject in alphabetical order}, or one n for every cohort; a cohort left out gets its first.
    paths = join each person's windows in time order. Saved as an .html in cfg.save_dir."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise SystemExit("figure 5 needs plotly: pip install --user plotly")
    cfg, col = co.cfg, colours(co.cfg)
    labels = [c for c in cfg.cohorts if c in set(lat["cohort"])]
    A = {"atlas": cfg.space["atlas"], "window_s": cfg.space["window_s"], "states": cfg.space["states"], **lat.attrs}
    states = A["states"]
    if by == "cluster" and states not in lat:
        raise SystemExit(f"the latents carry no {states} column; latent_options(co) lists the state sets")
    pick = {}
    if by == "single":
        who = {c: who for c in labels} if isinstance(who, (int, np.integer)) else (who or {})
        for c in labels:
            subs = sorted(lat.loc[lat["cohort"] == c, "sub"].unique())
            w = who.get(c, 1)
            if isinstance(w, (int, np.integer)):                  # 1 = the first subject, alphabetically
                if not 1 <= w <= len(subs):
                    raise SystemExit(f"{c} has {len(subs)} subjects: who must be 1..{len(subs)}, not {w}")
                pick[c] = subs[w - 1]
            else:
                pick[c] = str(w).strip().upper()
                if pick[c] not in subs:
                    raise SystemExit(f"{pick[c]} is not in {c}; e.g. {subs[:3]}")
    titles = [f"{c} · {co.role(c)} · {lat.loc[lat['cohort'] == c, 'sub'].nunique()} participants"
              + (f"<br>{pick[c]} (#{sorted(lat.loc[lat['cohort'] == c, 'sub'].unique()).index(pick[c]) + 1})"
                 if by == "single" else "") for c in labels]
    fig = make_subplots(1, len(labels), specs=[[{"type": "scene"}] * len(labels)], subplot_titles=titles,
                        horizontal_spacing=.02)
    tasks = sorted(lat["task"].astype(str).unique(), key=lambda t: (t.lower() != "rest", t))
    tcol = {t: (hp.INK2 if t.lower() == "rest" else _hex(plt.cm.tab20(k % 20))) for k, t in enumerate(tasks)}
    seen = set()

    def dots(x, colour, name, j, size=2, opacity=.5, group=None, hover=None):
        show = group is not None and group not in seen
        if group is not None:
            seen.add(group)
        return go.Scatter3d(x=x["x"], y=x["y"], z=x["z"], mode="markers", name=name, legendgroup=group,
                            showlegend=show, marker=dict(size=size, color=colour, opacity=opacity),
                            text=hover, hoverinfo="text" if hover is not None else "skip")

    for j, c in enumerate(labels, start=1):
        g = lat[lat["cohort"] == c]
        hover = g["sub"] + " · " + g["task"].astype(str)
        if by == "participant":
            subs = sorted(g["sub"].unique())
            rank = g["sub"].map({s_: k for k, s_ in enumerate(subs)})
            fig.add_trace(go.Scatter3d(x=g["x"], y=g["y"], z=g["z"], mode="markers", showlegend=False, text=hover,
                                       hoverinfo="text", marker=dict(size=2, opacity=.5, color=rank, colorscale="Viridis",
                                       colorbar=dict(title="participant", x=j / len(labels) - .02, len=.6, thickness=10))),
                          1, j)
        elif by == "task":
            for t, h in g.groupby(g["task"].astype(str)):
                fig.add_trace(dots(h, tcol[t], t, j, group=t, hover=h["sub"] + " · " + t), 1, j)
        elif by == "cluster":
            for k, h in g.groupby(states):
                fig.add_trace(dots(h, _hex(plt.cm.tab10(int(k) % 10)), sn.label(int(k), "long"), j, group=f"s{int(k)}",
                                   hover=h["sub"] + f" · {sn.label(int(k))}"), 1, j)
        elif by == "single":
            mine = g["sub"] == pick[c]
            fig.add_trace(dots(g[~mine], hp.MUTED, "everyone else", j, opacity=.25, group="others"), 1, j)
            h = g[mine]
            fig.add_trace(go.Scatter3d(x=h["x"], y=h["y"], z=h["z"], mode="markers+lines", name=f"{c}: {pick[c]}",
                                       marker=dict(size=4, color=col[c]), line=dict(color=col[c], width=2),
                                       text=h["task"].astype(str), hoverinfo="text"), 1, j)
        if paths and by != "single":
            for _, h in g.groupby("sub"):
                fig.add_trace(go.Scatter3d(x=h["x"], y=h["y"], z=h["z"], mode="lines", showlegend=False, hoverinfo="skip",
                                           line=dict(color=hp.INK2, width=1), opacity=.25), 1, j)
    e = embedding[:-1]
    rng = {a: [lat[a].min(), lat[a].max()] for a in "xyz"}
    for j in range(1, len(labels) + 1):
        scene = dict(xaxis_title=f"{e} 1", yaxis_title=f"{e} 2", zaxis_title=f"{e} 3", aspectmode="cube")
        if same_axes:
            scene.update(xaxis_range=rng["x"], yaxis_range=rng["y"], zaxis_range=rng["z"])
        fig.update_layout(**{f"scene{'' if j == 1 else j}": scene})
    what = {"participant": "participant (gradient)", "task": "task", "cluster": f"state ({states})",
            "single": "one person per cohort vs everyone else"}[by]
    fig.update_layout(height=height, width=520 * len(labels), margin=dict(l=0, r=0, t=90, b=0),
                      title=f"5  Latent space ({embedding}) per cohort · one dot per window · colour = {what} · "
                            f"{A['atlas']} · {ht.window_name(A['window_s'])}", legend=dict(itemsizing="constant"))
    if cfg.save_dir:
        hp.Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        fig.write_html(hp.Path(cfg.save_dir) / f"5_latents_{embedding}_{by}.html", include_plotlyjs="cdn")
    return fig


def window_name(cfg):
    return ht.window_name(cfg.space["window_s"])


# ------------------------------------------------------------------ stability (manuscript figure 3)
ORANGE = "#eb6834"
NICE = {"transition_fit": "Transition structure (fit to reference)",
        "occupancy_fit": "Occupancy profile (fit to reference)",
        "dynamics_fit": "Dynamics profile (fit to reference)", "switch_rate": "Switch rate",
        "mean_dwell_s": "Mean dwell", "entropy_rate_bits": "Entropy rate", "dispersion": "Dispersion",
        "n_distinct_transitions": "Distinct transitions"}


def nice(v):
    if v in NICE:
        return NICE[v]
    if v.startswith("occ_"):
        return f"Occupancy {sn.label(v[4:])}"
    if "->" in v:
        a, b = v.split("->")
        return f"p({sn.label(a)} → {sn.label(b)})"
    return v



def fig_stability_plane(co, st, variables=None, x_cut=None, y_cut=0.4):
    """Panel B. x = movie vs rest shift (|d_av|), y = ICC(3,1) rest <-> movie (individuals keep their
    rank). Quadrants: stable trait / context-modulated trait / context-driven / unreliable."""
    t = st if variables is None else st.set_index("variable").loc[variables].reset_index()
    x_cut = co.cfg.sesoi_d if x_cut is None else x_cut
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.axvline(x_cut, color="#e5e4df", lw=1.5)
    ax.axhline(y_cut, color="#e5e4df", lw=1.5)
    x = t["task_d"].abs()
    c = [{"stable": hp.BLUE, "moderate": "#86b6ef"}.get(s, ORANGE) for s in t["tier"]]
    ax.errorbar(x, t["icc"], yerr=[t["icc"] - t["icc_lo"], t["icc_hi"] - t["icc"]], fmt="none", ecolor="#c9c8c2", lw=1, zorder=1)
    ax.scatter(x, t["icc"], s=70, c=c, edgecolor="white", linewidth=1.5, zorder=3)
    for xi, yi, v in zip(x, t["icc"], t["variable"]):
        ax.annotate(nice(v), (xi, yi), xytext=(6, 4), textcoords="offset points", fontsize=7.5, color=hp.INK)
    xmax = max(1.0, float(np.nanmax(x)) * 1.1)
    ax.set_xlim(0, xmax)
    ax.set_ylim(min(0, float(np.nanmin(t["icc_lo"]))), 1)
    kw = dict(fontsize=8, color=hp.INK2, style="italic", transform=ax.transAxes)
    ax.text(.02, .98, "stable trait", va="top", **kw)
    ax.text(.98, .98, "context-modulated trait", va="top", ha="right", **kw)
    ax.text(.98, .02, "context-driven", ha="right", **kw)
    ax.text(.02, .02, "unreliable", **kw)
    ax.set_xlabel("context shift: |d_av| movie vs rest (same people)", fontsize=9)
    ax.set_ylabel("individual consistency: ICC(3,1) rest ↔ movie", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("B  Stable in the mean, stable in the individual", fontsize=11, loc="left")
    save(fig, "stability_plane", co.cfg)
    return fig




# ------------------------------------------------------------------ fixed axes and one copy per figure
def fixed_range(co, v):
    """The same x range for a variable in every figure: probabilities (occupancy, transitions,
    the transition-pattern correlation) 0-1; the dynamics on their natural ranges -- switch rate
    0-1, mean dwell 0-60 s, entropy rate 0-log2(K) bits, distinct transitions 0-K^2. Dispersion is
    in the units of the state set's latent space (PCA and UMAP differ by an order of magnitude),
    so its range is 0 to the 99th percentile of the subjects' values, rounded up -- the same in
    every figure of a state set. `cfg.ranges` overrides any of these."""
    K = co.cfg.K
    if v.startswith("occ_") or "->" in v or v in ("transition_fit", "occupancy_fit", "dynamics_fit"):
        return (0.0, 1.0)
    if v in co.cfg.ranges:
        return tuple(co.cfg.ranges[v])
    top = {"entropy_rate_bits": np.log2(K), "n_distinct_transitions": K * K}
    if v in top:
        return (0.0, float(top[v]))
    vals = pd.concat([d[v] for d in co.data.values() if v in d]).dropna() if co.data else pd.Series(dtype=float)
    if not len(vals):
        return (np.nan, np.nan)
    hi = float(np.nanpercentile(vals, 99))
    step = 10 ** np.floor(np.log10(hi)) if hi > 0 else 1.0
    return (min(0.0, float(vals.min())), float(np.ceil(hi / step) * step))


def global_means(co, variables):
    """Every cohort and condition, each once: the reference the films are drawn against."""
    return pd.concat([d[variables].mean() for d in co.data.values()], axis=1).mean(axis=1)


def _show(fig):
    """Draw the figure ONCE in a notebook, then close it: an open figure plus a returned one is
    what made every figure appear twice."""
    plt.show()
    plt.close(fig)


# ------------------------------------------------------------------ variation between films (supplementary)
def _film_rows(fs):
    films = fs[["cohort", "film", "within"]].drop_duplicates(["cohort", "film"])
    return list(zip(films["cohort"], films["film"], films["within"]))


def fig_films_forest(co, fs, variables, rest=None, title="", name="films", per_row=None, labels=None):
    """One panel per variable, one row per film: the film's mean (95% CI). Filled = the film's
    viewers also saw other films of their cohort, hollow = different people per film. Dashed blue
    = `rest`, the global mean (every cohort and condition, each once; `global_means`). Every
    variable on its fixed range (`fixed_range`). Films grouped by cohort, no colour per cohort."""
    rows = _film_rows(fs)
    n = len(variables)
    per_row = per_row or n
    nrow = int(np.ceil(n / per_row))
    fig, axs = plt.subplots(nrow, per_row, figsize=(1.75 * per_row + 2.2, 0.33 * len(rows) * nrow + 1.8),
                            squeeze=False, gridspec_kw=dict(wspace=.12, hspace=.25))
    # each cohort gets a header line of its own above its tasks, so its name sits on no task's
    y, heads, yy, prev = [], [], 0.0, None
    for c, f, _ in rows:
        if c != prev:
            heads.append((c, yy))
            yy -= 1
            prev = c
        y.append(yy)
        yy -= 1
    y = np.array(y)
    pos = {(c, f): yi for yi, (c, f, _) in zip(y, rows)}
    for i, v in enumerate(variables):
        ax = axs[i // per_row, i % per_row]
        t = fs[fs["variable"] == v]
        for r in t.itertuples():
            yi = pos[r.cohort, r.film]
            ax.plot([r.lo, r.hi], [yi, yi], color=hp.INK2, lw=1.1)
            ax.plot(r.mean, yi, "o", ms=4.5, color=hp.INK2, mfc=hp.INK2 if r.within else "white", mew=1.1)
        lo, hi = fixed_range(co, v)
        if np.isfinite(lo):
            ax.set_xlim(lo, hi)
        if rest is not None and v in rest and np.isfinite(rest[v]):
            ax.axvline(rest[v], color=hp.BLUE, lw=1, ls="--")
        ax.set_title((labels or {}).get(v, nice(v)), fontsize=8.5)
        ax.set_yticks(y, [f for _, f, _ in rows] if i % per_row == 0 else [], fontsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.xaxis.grid(True, color="#e5e4df")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=7)
        for k, (c, yh) in enumerate(heads):                        # cohort separators and names
            if k:
                ax.axhline(yh + .5, color="#c9c8c2", lw=.8)
            if i % per_row == 0:
                ax.text(-0.02, yh, cname(c), transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=8, color=hp.INK, fontweight="bold")
        ax.set_ylim(y.min() - .7, .7)
    for j in range(n, nrow * per_row):
        axs[j // per_row, j % per_row].set_visible(False)
    fig.suptitle(title, fontsize=11, x=.02, ha="left")
    fig.text(.02, .005, "Task mean (95% CI). Filled = the same people did several tasks of their cohort; "
             "hollow = different people per task. Dashed blue = global mean (every cohort and condition).",
             fontsize=7.5, color=hp.INK2)
    save(fig, name, co.cfg)
    _show(fig)
    return fig


def fig_films_matrices(co, fs, per_row=6):
    """One p(j | i) per film and the films' mean, on one 0-1 colour scale. Title = cohort · film."""
    K = co.cfg.K
    rows = _film_rows(fs)
    cells = co.cells
    M = {(c, f): fs[(fs.cohort == c) & (fs.film == f)].set_index("variable").loc[cells] for c, f, _ in rows}
    n = len(rows) + 1
    nrow = int(np.ceil(n / per_row))
    fig, axs = plt.subplots(nrow, per_row, figsize=(1.9 * per_row, 2.1 * nrow + .6), squeeze=False)
    # a row nobody in a task left (e.g. a state CNeuroMod never visits) is NaN, not 0: the mean
    # is over the tasks that have it, and a missing row is drawn hatched grey, not white (= 0)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)          # a row no task has: NaN
        mean = np.nanmean([m["mean"].to_numpy(float) for m in M.values()], axis=0).reshape(K, K)
    panels = [("mean of the tasks", mean)] + [(f"{cname(c)} · {f}", m["mean"].to_numpy(float).reshape(K, K))
                                              for (c, f), m in M.items()]
    for k, (title, T) in enumerate(panels):
        ax = axs[k // per_row, k % per_row]
        ax.set_facecolor("#ecebe6")
        im = ax.imshow(np.ma.masked_invalid(T), cmap="Blues", vmin=0, vmax=1)
        for r_ in np.flatnonzero(np.isnan(T).all(1)):
            ax.add_patch(plt.Rectangle((-.5, r_ - .5), K, 1, fc="#ecebe6", ec="#c9c8c2", hatch="////", lw=0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=7, color=hp.INK2)
    for j in range(n, nrow * per_row):
        axs[j // per_row, j % per_row].set_visible(False)
    fig.suptitle("Transitions p(j | i), task by task — rows: from state i, columns: to state j; "
                 "hatched: state not left in that task", fontsize=11,
                 x=.02, ha="left")
    fig.colorbar(im, ax=axs.ravel().tolist(), fraction=.015, pad=.01, label="p(j | i)").outline.set_visible(False)
    save(fig, "films_transitions", co.cfg)
    _show(fig)
    return fig


# ------------------------------------------------------------------ manuscript figure 3, as one figure
TIER_COL = {"stable": hp.BLUE, "moderate": "#86b6ef", "context-dependent": ORANGE}


def _replication_panel(ax, pairs):
    t = pairs.copy()
    if "kind" not in t:
        t["kind"] = "between cohorts"
    t["_o"] = t["kind"].map({"vs mean": 0, "between cohorts": 0, "rest vs movie": 1})
    t = t.sort_values(["_o", "r_norm"], ascending=[True, False])
    y = np.arange(len(t))[::-1]
    ax.axvline(1, color=hp.INK2, lw=1, ls="--")
    ax.text(1, 1.01, "noise ceiling", ha="center", va="bottom", fontsize=7.5, color=hp.INK2,
            transform=ax.get_xaxis_transform())
    for yi, r in zip(y, t.itertuples()):
        c = ORANGE if r.kind == "rest vs movie" else hp.BLUE
        if hasattr(r, "lo") and np.isfinite(getattr(r, "lo", np.nan)):
            ax.plot([r.lo, r.hi], [yi, yi], color=c, lw=1.3)
        ax.plot(r.r_norm, yi, "o", color=c, ms=6)
    nm = lambda x: cname(x.split()[0])
    lab = [f"{nm(r.a)}: rest – movie" if r.kind == "rest vs movie" else
           f"{nm(r.a)} – mean of the others" if r.kind == "vs mean" else
           f"{nm(r.a)} – {nm(r.b)}" for r in t.itertuples()]
    ax.set_yticks(y, lab, fontsize=8)
    lo = min(.5, float(np.nanmin(t.get("lo", t["r_norm"]))) - .05)
    ax.set_xlim(lo, 1.08)
    ax.set_xlabel("transition-structure similarity / noise ceiling", fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#e5e4df")
    ax.set_axisbelow(True)


P_COLS = {"difference": ("task_p_fdr", "p\ndifference"), "tost": ("task_p_tost_fdr", "p\nequivalence"),
          "tier": ("task_p_tier", "P(tier)")}
TIER_STARS = (.90, .95, .99)                              # P(tier) for *, **, ***
TIER_SOLID = .95                                          # a filled marker: P(tier) at least this


def stars(p):
    return "" if not np.isfinite(p) else "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def tier_stars(p):
    return "" if not np.isfinite(p) else "*" * int(sum(p >= c for c in TIER_STARS))


def fmt_tier(p):
    return "" if not np.isfinite(p) else f"{p:.2f}"


def fmt_p(p):
    if not np.isfinite(p):
        return ""
    return "<.001" if p < .001 else f"{p:.3f}".lstrip("0") if p < .01 else f"{p:.2f}".lstrip("0") if p < .995 else "1"


def _p_columns(ax, t, y, pvalue, tests, x0=1.07, headers=False):
    """Confidence right of the panel, one column per test, exact or as stars.
    tier: P(tier), the share of bootstrap |d| in the tier of the observed |d| (stars: * >= .90,
          ** >= .95, *** >= .99);
    difference: paired t-test movie vs rest, BH-FDR (* <.05, ** <.01, *** <.001);
    tost: equivalence at +-sesoi_d SD, BH-FDR (a small p = within the bounds, i.e. stable)."""
    for k, test in enumerate(tests):
        col, head = P_COLS[test]
        if col not in t:
            continue
        x = x0 + .22 * k
        if headers:                                          # off: explained in the legend
            ax.text(x, len(t) - .3, head, transform=ax.get_yaxis_transform(), ha="center",
                    va="bottom", fontsize=8, color=hp.INK2)
        fmt = {("tier", "stars"): tier_stars, ("tier", "exact"): fmt_tier}.get(
            (test, pvalue), stars if pvalue == "stars" else fmt_p)
        for yi, p in zip(y, t[col].to_numpy(float)):
            ax.text(x, yi, fmt(p), transform=ax.get_yaxis_transform(),
                    ha="center", va="center", fontsize=8 if pvalue == "exact" else 10, color=hp.INK)


def _tests(test):
    return {"both": ["difference", "tost"], "all": ["tier", "difference", "tost"]}.get(
        test, [test] if isinstance(test, str) else list(test))


def family(v):
    if v == "transition_fit" or "->" in v:
        return "transitions"
    if v == "occupancy_fit" or v.startswith("occ_"):
        return "occupancy"
    return "dynamics"


def _mean_column(ax, t, y, x=1.1):
    """Beside each row, the values behind the effect: is a large effect a small change? With
    task_variation, the lowest and highest task mean; with stability, the mean rest -> movie."""
    rng_ = "task_mean_lo" in t
    a_col, b_col = ("task_mean_lo", "task_mean_hi") if rng_ else ("task_mean_rest", "task_mean_movie")
    ax.text(x, len(t) - .3, "task means\nmin – max" if rng_ else "mean\nrest → movie",
            transform=ax.get_yaxis_transform(), ha="center", va="bottom", fontsize=8, color=hp.INK2)
    arrow = " – " if rng_ else " → "
    f = lambda a: "" if not np.isfinite(a) else f"{a:.2f}".lstrip("0") if abs(a) < 1 else f"{a:.3g}"
    for yi, a, b in zip(y, t.get(a_col, pd.Series(np.nan, index=t.index)),
                        t.get(b_col, pd.Series(np.nan, index=t.index))):
        if np.isfinite(a) and np.isfinite(b):
            ax.text(x, yi, f"{f(a)}{arrow}{f(b)}", transform=ax.get_yaxis_transform(), ha="center",
                    va="center", fontsize=7, color=hp.INK)


def _tier_panel(ax, st, variables, pvalue=None, test="tier", separators=False, sort=True, means=False,
                headers=False):
    """pvalue: None, "exact" or "stars"; test: "tier" (P(tier)), "difference", "tost", "both"
    (difference + tost), "all", or a list (see _p_columns). A thin line separates the families
    (transitions, occupancy, dynamic metrics). A row with too few people scanned in both
    conditions to estimate (NaN) says so, with its n."""
    omega = st.attrs.get("scale") == "omega2"             # task_variation: share of variance, not |d|
    vpc = st.attrs.get("measure") == "vpc"
    cuts = (.01, .06) if omega else (.2, .5)
    t = st.set_index("variable").loc[[v for v in variables if v in set(st["variable"])]].reset_index()
    if sort:   # families in the order given; inside each, the profile r first, then by |d| (top = smallest)
        fam_rank = {f: k for k, f in enumerate(dict.fromkeys(family(v) for v in t["variable"]))}
        # transitions also grouped by the state they leave: all i -> . together, i in order
        key = pd.DataFrame({"f": [fam_rank[family(v)] for v in t["variable"]],
                            "fit": [0 if v.endswith("_fit") else 1 for v in t["variable"]],
                            "src": [int(v.split("->")[0]) if "->" in v else -1 for v in t["variable"]],
                            "d": t["task_d"].abs().fillna(np.inf)})
        t = t.loc[key.sort_values(["f", "fit", "src", "d"], kind="stable").index].reset_index(drop=True)
    y = np.arange(len(t))[::-1]
    ax.axvspan(0, cuts[0], color="#e8f0fb")
    ax.axvspan(cuts[0], cuts[1], color="#f3f6fb")
    d = t["task_d"].clip(lower=0) if omega else t["task_d"].abs()
    has = t["task_lo"].notna().to_numpy() & t["task_hi"].notna().to_numpy()   # per row: one NaN must not
    if has.any():                                                             # hide every other CI
        if omega:
            lo, hi = t["task_lo"].clip(lower=0), t["task_hi"].clip(lower=0)
        else:
            lo = np.minimum(t["task_lo"].abs(), t["task_hi"].abs())
            hi = np.maximum(t["task_lo"].abs(), t["task_hi"].abs())
            lo[np.sign(t["task_lo"]) != np.sign(t["task_hi"])] = 0       # the CI crosses 0
        ax.hlines(y[has], lo[has], hi[has], color="#9b9a95", lw=1)
    cols = [TIER_COL.get(x, "#c9c8c2") for x in t["tier"]]
    ax.scatter(d, y, c=cols, s=45, zorder=3, edgecolor="white")
    for yi, dv, n in zip(y, d, t.get("n_paired", pd.Series(np.nan, index=t.index))):
        if not np.isfinite(dv):
            ax.text(cuts[0] / 10, yi, ("too few values to estimate" if omega else
                                       f"too few people with both conditions (n = {int(n) if np.isfinite(n) else 0})"),
                    va="center", fontsize=6.5, color=hp.MUTED)
    src = [v.split("->")[0] if "->" in v else None for v in t["variable"]]
    for k in range(1, len(src)):                                   # a faint line between source states
        if src[k] is not None and src[k - 1] is not None and src[k] != src[k - 1]:
            ax.axhline(y[k] + .5, color="#e2e1dc", lw=.6)
    fam = [family(v) for v in t["variable"]] if separators else []
    for k in range(1, len(fam)):
        if fam[k] != fam[k - 1]:
            ax.axhline(y[k] + .5, color="#c9c8c2", lw=.8)
    ax.set_yticks(y, [nice(v) for v in t["variable"]], fontsize=8)
    top = len(t) - .3
    xmax = max(cuts[1] * 2, float(np.nanmax(np.r_[d.to_numpy(float), 0])) * 1.1)
    for x, lab in (() if not headers else ((cuts[0] / 2, "stable"), ((cuts[0] + cuts[1]) / 2, "moderate"),
                   (max(cuts[1] * 1.5, (cuts[1] + xmax) / 2), "task-dependent" if omega else "context-dependent"))):
        ax.text(x, top, lab, ha="center", fontsize=10, color=TIER_COL["context-dependent" if lab == "task-dependent" else lab],
                gid="scaled")                                                       # fixed size: fits its band
    ax.set_xlim(0, xmax)
    ax.set_ylim(-.7, len(t) + .2)
    ax.set_xlabel(("share of variance due to task (95% CI)" if vpc else "variance explained by task, ω² (95% CI)")
                  if omega else
                  "|d| movie vs rest, same people (95% CI)", fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if means:
        _mean_column(ax, t, y)
    if pvalue:
        _p_columns(ax, t, y, pvalue, _tests(test), x0=1.36 if means else 1.07, headers=headers)


def fig_figure3(co, st, tier_variables=None, font=1.45, diagram_fs=2.0, occupancy="each", name="figure3",
                pvalue=None, test="tier", panel="task_person", st_transitions=None, guides=None,
                transitions="fit"):
    """Manuscript Figure 3 as one figure, rest and movie together (each person's mean of the
    conditions they have; `groups(tasks="all")`).
    A  every cohort: occupancy, p(j | i) and the dynamic metrics (mean, 95% CI);
    B  the reference transition graph: the mean of the cohorts in A, each cohort once;
    C  which properties depend on context (`stability` tiers: movie vs rest, same people).
    font scales every text; diagram_fs the diagram's own (its labels are placed for that size).
    occupancy="each": C lists every state's occupancy; "fit": one row, the occupancy profile's
    correlation with the reference (`occupancy_fit`; st must include it).
    panel: bottom right, "task_person" (VPC task x VPC person, needs combine="mixed"; st_transitions
    adds every transition as small points) or "tiers" (task share with tiers and stars).
    guides: dashed lines on both axes of the task x person panel, e.g. True = (.01, .06) (Cohen's
    small / medium shares of variance) or a tuple of your own; None = none.
    pvalue / test: the p-value column(s) right of C (`_tier_panel`): None, "exact" or "stars";
    "tier" (P(tier), default), "difference" (paired t), "tost" (equivalence), "both", "all".
    transitions: "structure" | "median" | "average" | "both", how the VPC panel summarises the transitions
    (one model of the whole structure, or the median of the individual p(j | i); _task_person_panel).
    """
    from matplotlib.gridspec import GridSpec

    cfg, K = co.cfg, co.cfg.K
    col = colours(cfg)
    G = co.groups(by="cohort", tasks="all")
    cohorts = list(G)
    graphs = {c: co._graph(d.mean(numeric_only=True)) for c, d in G.items()}
    T_ref = np.mean([g[0] for g in graphs.values()], axis=0)
    o_ref = np.mean([g[1] for g in graphs.values()], axis=0)
    off = [(a, b) for a in range(K) for b in range(K) if a != b]
    edges = sorted(sorted(off, key=lambda e: -T_ref[e])[:cfg.edge_top])
    agree = co.agreement(co.dyn, groups=G)

    fig = plt.figure(figsize=(17, 15))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, .95], hspace=.12)
    n = len(cohorts)
    gA = gs[0].subgridspec(2, n + 1, height_ratios=[.35, 1], width_ratios=[1] * n + [1.25], hspace=.12, wspace=.35)
    occ_axes, mat_axes, dyn_axes = [], [], []
    for i, c in enumerate(cohorts):
        T, o = graphs[c]
        axo = fig.add_subplot(gA[0, i])
        occ_axes.append(axo)
        oc = G[c][co.occ]                                    # 95% CI of the mean over the cohort's people
        h = stats.t.ppf(.975, np.maximum(oc.count() - 1, 1)) * oc.std(ddof=1) / np.sqrt(oc.count())
        axo.bar(range(K), o, color=col[c], width=.7, yerr=h.to_numpy(float),
                error_kw=dict(ecolor=hp.INK2, elinewidth=1, capsize=0))
        axo.set_xticks([])
        axo.set_ylim(0, 1)
        axo.set_title(f"{cname(c)}{' *' if co.role(c) == 'train' else ''}", fontsize=10)
        for sp in ("top", "right"):
            axo.spines[sp].set_visible(False)
        axo.tick_params(labelsize=7)
        if i == 0:
            axo.set_ylabel("occupancy", fontsize=8)
        axm = fig.add_subplot(gA[1, i])
        mat_axes.append(axm)
        axm.imshow(T, cmap="Blues", vmin=0, vmax=1)
        _state_ticks(axm, K, 7, mode="number")       # Figure 3: numbers; the diagram names the states
        axm.set_xlabel("to state j", fontsize=8)
        if i == 0:
            axm.set_ylabel("from state i", fontsize=8)
    # the dynamics column spans occupancy + matrices: a spacer at the bottom stands for the matrices'
    # x labels, so its last axis ends where the matrices do
    gD = gA[:, n].subgridspec(len(co.dyn) + 1, 1, height_ratios=[1] * len(co.dyn) + [.35], hspace=1.3)
    for r_, v in enumerate(co.dyn):
        axd = fig.add_subplot(gD[r_])
        dyn_axes.append(axd)
        for j, c in enumerate(cohorts):                       # mean and 95% CI over the cohort's people
            x = G[c][v].dropna()
            h = stats.t.ppf(.975, max(len(x) - 1, 1)) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
            yy = -j * .3
            axd.plot([x.mean() - h, x.mean() + h], [yy, yy], color=col[c], lw=1.3)
            axd.plot(x.mean(), yy, "o", ms=4, color=col[c], mfc=col[c])
        axd.set_yticks([])
        axd.set_ylim(-.3 * len(cohorts), .3)
        lo_, hi_ = fixed_range(co, v)
        if np.isfinite(lo_):
            axd.set_xlim(lo_, hi_)
        for sp in ("top", "right", "left"):
            axd.spines[sp].set_visible(False)
        axd.tick_params(labelsize=7)
        axd.set_title(nice(v), fontsize=8, pad=2)

    # bottom: the task x person panel on the LEFT and wide; the diagram on the right in a narrower slot
    # (its equal-aspect box would leave the rest of a wide slot empty), both the same height
    tp = panel == "task_person" and "var_person" in st
    gB = gs[1].subgridspec(1, 2, width_ratios=[1.55, 1] if tp else [1, 1], wspace=.12 if tp else .3)
    ax = fig.add_subplot(gB[1] if tp else gB[0])
    hp.diagram(ax, T_ref, o_ref, edges, 1.0, 1.0, fs=diagram_fs)
    co.diagram_p_min = min(T_ref[e] for e in edges)       # for the legend: the thinnest arrow drawn
    ax = fig.add_subplot(gB[0] if tp else gB[1])
    occ = co.occ if occupancy == "each" else ["occupancy_fit"]
    if tp:                                       # VPC task x VPC person (mixed model)
        ax.set_box_aspect(3.5 / 3.2 / 1.55)      # as tall as the diagram, 1.55 times as wide
        _task_person_panel(ax, co, st, st_transitions, tier_variables or ["transition_fit", *co.dyn, *occ],
                           fs=9 / font, guides=guides, transitions=transitions, grow=font)   # scaled back up below
    else:                                        # "tiers": the share due to task, with tiers and stars
        _tier_panel(ax, st, tier_variables or ["transition_fit", *co.dyn, *occ], pvalue, test)
    for t in fig.findobj(plt.matplotlib.text.Text):              # every text larger, but the diagram's:
        if t.get_gid() != "scaled":                              # its labels were placed at diagram_fs
            t.set_fontsize(t.get_fontsize() * font)
    # occupancy: as tall as its matrix, its top on the top of the dynamics column (the cohorts' titles
    # on the line of the first dynamic metric's title); the space left goes between the two
    fig.canvas.draw()
    top = dyn_axes[0].get_position().y1 if dyn_axes else None
    for axo, axm in zip(occ_axes, mat_axes):
        mp = axm.get_position()
        if top is not None:
            h = min(mp.height, top - mp.y1 - .015)          # never into the matrix
            if h > .02:
                axo.set_position([mp.x0, top - h, mp.width, h])
    fig.suptitle(f"Figure 3 · {ht.set_label(cfg.space)}", x=.02, ha="left", fontsize=12)
    save(fig, name, cfg)
    _show(fig)
    return fig


def fig_replication(co, pairs, name="Sd_replication"):
    """Supplementary: the transition structure replicates (`replication`)."""
    fig, ax = plt.subplots(figsize=(8, .45 * len(pairs) + 1.6))
    _replication_panel(ax, pairs)
    ax.set_title("The transition structure replicates", fontsize=11, loc="left")
    _bigger(fig, 1.3)
    save(fig, name, co.cfg)
    _show(fig)
    return fig


def fig_aggregation_check(co, n_boot=2000, min_occ=0.01, name="check_aggregation", pvalue=None, test="tier",
                          means=False, measure="task", combine="random", st=None):
    """Does one number per family lose information? Movie vs rest |d_av| (same people, 95% CI); transitions
    on the left, dynamic metrics over occupancy on the right: the family's profile r with the reference on top, then every member of the family --
    transitions p(j | i) (rows of states with mean occupancy >= min_occ), occupancy of each state,
    and each dynamic metric. Same tiers as Figure 3C."""
    occ = pd.concat([v[co.occ].mean() for v in co.data.values()], axis=1).mean(axis=1).to_numpy()
    keep = [c for c in co.cells if occ[int(c.split("->")[0])] >= min_occ]
    fams = [("Transitions", "transition_fit", keep), ("Dynamic metrics", "dynamics_fit", list(co.dyn)),
            ("Occupancy", "occupancy_fit", list(co.occ))]
    vs = [v for _, f, m in fams for v in [f, *m]]
    if st is None:                               # st given (e.g. the cached mixed models): no refit
        st = (co.task_variation(vs, n_boot=n_boot, combine=combine) if measure == "task"
              else co.stability(vs, n_boot=n_boot))
    # measure: "task" = omega^2 of task, every task of every cohort; "rest_movie" = |d_av| (old)
    from matplotlib.gridspec import GridSpec
    h = [len(m) + 1 for _, _, m in fams]
    fig = plt.figure(figsize=(16 if means else 14, max(6, .2 * h[0] + 2)))
    gs = GridSpec(2, 2, figure=fig, height_ratios=h[1:], wspace=(1.1 if pvalue else .9) + (.5 if means else 0), hspace=.35,
                  left=.2, right=.86)
    axs = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
    for ax, (title, f, members) in zip(axs, fams):            # transitions left; metrics over occupancy right
        _tier_panel(ax, st, [f, *members], pvalue, test, means=means)
        ax.axhline(len(members) - .5, color="#c9c8c2", lw=.8)       # the profile above, members below
        ax.set_title(title, fontsize=12, loc="left")
        ax.tick_params(axis="y", labelsize=7 if len(members) > 20 else 9)
    axs[1].set_xlabel("")                                      # one x label in the right column, at the bottom
    what = "ω² of task, every task of every cohort" if measure == "task" else "|d_av| movie vs rest"
    fig.suptitle(f"Aggregation check · {ht.set_label(co.cfg.space)} · {what}", x=.02, ha="left", fontsize=12)
    save(fig, name, co.cfg)
    _show(fig)
    return fig, st


# ------------------------------------------------------------------ variance partition (mixed model)
PART_COL = {"cohort": "#9b9a95", "task": ORANGE, "person": hp.BLUE, "residual": "#dcdbd5"}


def _ordered(co, st, variables):
    """Figure 3's order: transitions, dynamic metrics, occupancy; inside, profile first, then by VPC."""
    t = st.set_index("variable").loc[[v for v in variables if v in set(st["variable"])]].reset_index()
    fam_rank = {f: k for k, f in enumerate(dict.fromkeys(family(v) for v in t["variable"]))}
    key = pd.DataFrame({"f": [fam_rank[family(v)] for v in t["variable"]],
                        "fit": [0 if v.endswith("_fit") else 1 for v in t["variable"]],
                        "src": [int(v.split("->")[0]) if "->" in v else -1 for v in t["variable"]],
                        "d": t["task_d"].fillna(np.inf)})
    return t.loc[key.sort_values(["f", "fit", "src", "d"], kind="stable").index].reset_index(drop=True)


def _partition_panel(ax, t, cohort, fs=10):
    parts = [k for k in PART_COL if cohort or k != "cohort"]
    y = np.arange(len(t))[::-1]
    for yi, r in zip(y, t.itertuples()):
        left = 0.0
        within = sum(getattr(r, f"var_{k}") for k in parts)
        for k in parts:
            w = getattr(r, f"share_{k}") if cohort else getattr(r, f"var_{k}") / within
            if np.isfinite(w):
                ax.barh(yi, w, left=left, color=PART_COL[k], height=.72, edgecolor="white", lw=.5)
                left += w
        if np.isfinite(r.task_d):
            ax.text(1.02, yi, f"{r.task_d:.3f}", va="center", fontsize=fs - 1, transform=ax.get_yaxis_transform())
    ax.text(1.02, len(t) - .35, "VPC task\n(within cohorts)", va="bottom", fontsize=8, color=hp.INK2,
            transform=ax.get_yaxis_transform())
    vs = list(t["variable"])
    for k in range(1, len(vs)):                                  # white gaps: families, and source states
        src = [v.split("->")[0] if "->" in v else None for v in (vs[k], vs[k - 1])]
        if family(vs[k]) != family(vs[k - 1]) or vs[k - 1].endswith("_fit") or (src[0] and src[0] != src[1]):
            ax.axhline(y[k] + .5, color="white", lw=3)
    ax.set_yticks(y, [nice(v) for v in vs], fontsize=fs)
    ax.set_xlim(0, 1)
    ax.set_ylim(-.6, len(t) - .4)
    ax.set_xlabel("share of the total variance" if cohort else "share of the variance within cohorts", fontsize=10)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return parts


def fig_variance_partition(co, st, variables=None, name="variance_partition", cohort=True, st_transitions=None,
                           min_occ=0.01):
    """One stacked bar per property: the share of its TOTAL variance due to cohort, task, person
    and residual, from `task_variation(combine="mixed")` (shares from one mixed model). Beside each
    bar, the VPC of the task within cohorts (Figure 3, bottom right). cohort=False: three colours,
    the variance WITHIN cohorts (task, person, residual) -- each part over their sum, so the orange
    part is that VPC itself. st_transitions: the same for every transition p(j | i) (rows of states
    with mean occupancy >= min_occ; grouped by the state left, then by VPC), drawn on the left."""
    from matplotlib.patches import Patch
    if "share_task" not in st:
        raise SystemExit('fig_variance_partition needs task_variation(..., combine="mixed")')
    variables = variables or ["transition_fit", *co.dyn, *co.occ]
    t = _ordered(co, st, variables)
    if st_transitions is not None:
        occ = pd.concat([v[co.occ].mean() for v in co._ref_tables()], axis=1).mean(axis=1).to_numpy()
        cells = ["transition_fit", *[c for c in co.cells if occ[int(c.split("->")[0])] >= min_occ]]
        tt = _ordered(co, st_transitions, cells)
        h = max(.42 * len(t), .2 * len(tt)) + 2
        fig, (axl, ax) = plt.subplots(1, 2, figsize=(17, h), gridspec_kw=dict(wspace=.55))
        _partition_panel(axl, tt, cohort, fs=7)
        axl.set_title("Transitions", loc="left", fontsize=11)
        ax.set_title("Transition structure, dynamic metrics and occupancy", loc="left", fontsize=11)
    else:
        h = .42 * len(t) + 1.8
        fig, ax = plt.subplots(figsize=(10, h))
    parts = _partition_panel(ax, t, cohort)
    fig.subplots_adjust(bottom=1.3 / h, top=1 - .8 / h, left=.12 if st_transitions is not None else .25, right=.93)
    fig.legend(handles=[Patch(color=PART_COL[k], label=k) for k in parts], loc="lower center",
               ncol=len(parts), frameon=False, fontsize=9)
    fig.suptitle(f"Variance partition · {ht.set_label(co.cfg.space)}", x=.02, ha="left", fontsize=12)
    save(fig, name, co.cfg)
    _show(fig)
    return fig


def variance_table(co, st, variables=None, digits=3, st_transitions=None, min_occ=0.01):
    """The partition as a table: one row per property (Figure 3's order) and, with st_transitions,
    one per transition p(j | i) (rows of states with mean occupancy >= min_occ; grouped by the state
    left). Variance components on z-scored values; VPC of task and of person within cohorts (VPC
    task with its 95% CI); Nakagawa's marginal / conditional R^2."""
    variables = variables or ["transition_fit", *co.dyn, *co.occ]
    parts = [("property", _ordered(co, st, variables))]
    if st_transitions is not None:
        occ = pd.concat([v[co.occ].mean() for v in co._ref_tables()], axis=1).mean(axis=1).to_numpy()
        cells = [c for c in co.cells if occ[int(c.split("->")[0])] >= min_occ]
        parts.append(("transition", _ordered(co, st_transitions, cells)))
    rows = []
    for kind, t in parts:
        rows.append(pd.DataFrame({
            "family": [family(v) if kind == "property" else "transition p(j | i)" for v in t["variable"]],
            "property": [nice(v) for v in t["variable"]],
            "σ² cohort": t["var_cohort"], "σ² task": t["var_task"], "σ² person": t["var_person"],
            "σ² residual": t["var_residual"],
            "VPC task": t["task_d"], "VPC 95% CI": [f"{a_:.{digits}f}–{b_:.{digits}f}" if np.isfinite(a_) else ""
                                                     for a_, b_ in zip(t["task_lo"], t["task_hi"])],
            "VPC person": _vpc_within(t, "person"),
            "R² marginal": t["r2_marginal"], "R² conditional": t["r2_conditional"]}))
    return pd.concat(rows, ignore_index=True).round(digits)


FAMILY_COL = {"transitions": "#1f4e9c", "dynamics": "#2a9d6f", "occupancy": ORANGE}


def _vpc_within(t, k):
    tot = t["var_task"] + t["var_person"] + t["var_residual"]
    return t[f"var_{k}"] / tot


TRANSITION_LIGHT = "#9cc3f5"


def _task_person_panel(ax, co, st, st_transitions=None, variables=None, min_occ=0.01, fs=9, guides=None,
                       transitions="fit", grow=1.0):
    """transitions: how the transitions are summarised -- "structure" (one model of the whole
    structure, transition_fit), "median" / "average" (the median / mean VPC task and VPC person
    of the individual p(j | i) drawn in light blue) or "both" (structure + median). Occupancy points
    are labelled by the state's name (state_names; S<k> without names), the colour says occupancy. Occupancies are labelled by state number here (S3), whatever
    state_names says: the panel is about variance shares, the diagram beside it names the states."""
    from matplotlib.lines import Line2D
    fig = ax.figure
    transitions = {"structure": "fit", "mean": "average"}.get(transitions, transitions)
    if transitions not in ("fit", "median", "average", "both"):
        raise SystemExit(f'transitions must be "structure", "median", "average" or "both", got {transitions!r}')
    agg = np.nanmean if transitions == "average" else np.nanmedian       # "both": structure + median
    summary = "Average Transition" if transitions == "average" else "Median Transition"
    variables = variables or ["transition_fit", *co.dyn, *co.occ]
    if transitions in ("median", "average") and st_transitions is not None:
        variables = [v for v in variables if v != "transition_fit"]
    t = st.set_index("variable").loc[[v for v in variables if v in set(st["variable"])]].reset_index()
    tc = None
    if st_transitions is not None:
        occ = pd.concat([v[co.occ].mean() for v in co._ref_tables()], axis=1).mean(axis=1).to_numpy()
        cells = [c for c in co.cells if occ[int(c.split("->")[0])] >= min_occ]
        tc = st_transitions.set_index("variable").loc[[c for c in cells if c in set(st_transitions["variable"])]]
        ax.scatter(_vpc_within(tc, "task"), _vpc_within(tc, "person"), s=18, color=TRANSITION_LIGHT,
                   edgecolor="none", zorder=2)
    x, y = _vpc_within(t, "task").to_numpy(float), _vpc_within(t, "person").to_numpy(float)
    names = list(t["variable"])
    if transitions != "fit" and tc is not None and len(tc):   # the light points' median / mean
        x = np.append(x, agg(_vpc_within(tc, "task").to_numpy(float)))
        y = np.append(y, agg(_vpc_within(tc, "person").to_numpy(float)))
        names.append("transitions_mean")
    names = pd.Series(names)
    ok = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[ok], y[ok], s=75, c=[FAMILY_COL[family("transition_fit" if v == "transitions_mean" else v)]
                                      for v in names[ok]], edgecolor="white", zorder=3)
    allx = np.r_[x, _vpc_within(tc, "task").to_numpy(float)] if tc is not None else x
    ally = np.r_[y, _vpc_within(tc, "person").to_numpy(float)] if tc is not None else y
    xmax = max(.1, np.nanmax(allx) * 1.08)                    # every point inside, the light ones too
    ymax = max(.2, np.nanmax(ally) * 1.1)
    ax.set_xlim(-.01 * xmax, xmax)
    ax.set_ylim(0, ymax)
    if guides:
        for g in ((.01, .06) if guides is True else guides):
            ax.axvline(g, color="#9b9a95", lw=.8, ls="--", zorder=1)
            ax.axhline(g, color="#9b9a95", lw=.8, ls="--", zorder=1)
    fig.canvas.draw()
    bb = ax.get_window_extent()
    # labels: for each point, the first candidate offset whose box overlaps no placed label nor point
    placed, pts = [], ax.transData.transform(np.c_[x[ok], y[ok]])
    renderer = fig.canvas.get_renderer()
    dot = 5 * fig.dpi / 72                                 # a point's radius, in pixels
    # the default place is right beside the point, no line; a label that cannot go there tries the
    # nearest free place (8 directions, growing distance) and gets a thin line back to its point.
    # Boxes are MEASURED at the size the text ends up at (grow: fig_figure3 enlarges every text after)
    cands = [(6, 0)] + [(r_ * cx, r_ * cy) for r_ in (6, 10, 15, 21, 28, 36, 46, 58, 72)
                        for cx, cy in ((1, .8), (1, -.8), (-1, 0), (-1, .8), (-1, -.8), (0, 1), (0, -1), (1, 0))][1:]
    for (xi, yi), (px, py), v in zip(np.c_[x[ok], y[ok]], pts, names[ok]):
        lab = (summary if v == "transitions_mean" else
               "Transition Structure" if v == "transition_fit" else
               sn.label(v[4:], "name") if str(v).startswith("occ_") else      # the state's name (colour = occupancy)
               nice(v).replace(" (fit to reference)", ""))
        t = ax.annotate(lab, (xi, yi), xytext=(0, 0), textcoords="offset points", fontsize=fs * grow, va="center")
        choice = cands[0]
        for dx, dy in cands:
            t.set_position((dx, dy))
            t.set_ha("left" if dx > 0 else "right" if dx < 0 else "center")
            b = t.get_window_extent(renderer)
            box = (b.x0, b.y0, b.x1, b.y1)
            clash = any(not (box[2] < q[0] or box[0] > q[2] or box[3] < q[1] or box[1] > q[3]) for q in placed)
            clash |= any(box[0] - dot < qx < box[2] + dot and box[1] - dot < qy < box[3] + dot for qx, qy in pts)
            clash |= box[0] < bb.x0 + 4 or box[1] < bb.y0 + 4 or box[2] > bb.x1 or box[3] > bb.y1   # off the axes
            if not clash:
                choice = (dx, dy)
                break
        t.remove()
        dx, dy = choice
        t = ax.annotate(lab, (xi, yi), xytext=(dx, dy), textcoords="offset points", fontsize=fs * grow, color=hp.INK,
                        ha="left" if dx > 0 else "right" if dx < 0 else "center", va="center",
                        arrowprops=(dict(arrowstyle="-", color="#9b9a95", lw=.7, shrinkA=0, shrinkB=4)
                                    if choice != cands[0] else None))
        b = t.get_window_extent(renderer)
        placed.append((b.x0, b.y0, b.x1, b.y1))
        t.set_fontsize(fs)                                   # back to the size the caller will scale
    ax.set_xlabel("Task Variance Partition Coefficient", fontsize=10)
    ax.set_ylabel("Person Variance Partition Coefficient", fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    struct = {"fit": "Transition Structure", "median": "Median Transition", "average": "Average Transition",
              "both": "Transition Structure / Median Transition"}[transitions]
    handles = [Line2D([], [], ls="none", marker="o", ms=8, color=c, label=struct if k == "transitions" else k.title())
               for k, c in FAMILY_COL.items()]
    if st_transitions is not None:
        handles.insert(1, Line2D([], [], ls="none", marker="o", ms=5, color=TRANSITION_LIGHT, label="Transitions p(j | i)"))
    ax.legend(handles=handles, frameon=False, loc="upper right")


def fig_task_person(co, st, st_transitions=None, variables=None, name="task_vs_person", min_occ=0.01, guides=None,
                    transitions="fit", title=None):
    """One point per property: x = VPC of the task, y = VPC of the person, both shares of the
    variance within cohorts (`task_variation(combine="mixed")`). Colour = family. Upper left:
    individual and task-invariant; lower right: task-driven and not individual. st_transitions
    adds every transition p(j | i) as small unlabelled points (rows of states with mean occupancy
    >= min_occ). Labels are placed off the points with a leader line, avoiding each other."""
    if "var_person" not in st:
        raise SystemExit('fig_task_person needs task_variation(..., combine="mixed")')
    fig, ax = plt.subplots(figsize=(10, 7))
    _task_person_panel(ax, co, st, st_transitions, variables, min_occ, guides=guides, transitions=transitions)
    fig.suptitle(title or f"Task vs person · {ht.set_label(co.cfg.space)}", x=.02, ha="left", fontsize=12)
    save(fig, name, co.cfg)
    _show(fig)
    return fig


# ------------------------------------------------------------------ figure 1: the states in time
def _state_ticks(ax, K, fs=7, mode=None):
    """A K x K transition matrix's axes: S0..S(K-1), or the states' names (state_names) when set;
    mode="number" keeps the numbers whatever state_names says."""
    named = sn.active() and mode != "number"
    ax.set_xticks(range(K), [sn.label(k, mode) for k in range(K)], fontsize=fs,
                  rotation=60 if named else 0, ha="right" if named else "center")
    ax.set_yticks(range(K), [sn.label(k, mode) for k in range(K)], fontsize=fs)


def state_colours(K):
    return [plt.cm.tab10(k % 10) for k in range(K)]


def fig_state_timeline(co, tl, title="", name="state_timeline", max_people=80, seed=0):
    """Left: one row per viewer, one column per window of the same film, colour = state (at most
    max_people rows, drawn at random); the strip on top is the most common state at each moment.
    Right: the share of viewers in each state at each moment of the film."""
    from matplotlib.colors import ListedColormap
    K = co.cfg.K
    cols = state_colours(K)
    R = tl.pivot_table(index="sub", columns="stimulus_start_s", values="state", aggfunc="first")
    share = np.array([(R == k).sum(0) / R.notna().sum(0) for k in range(K)])
    rng = np.random.default_rng(seed)
    show = R.loc[rng.permutation(R.index)[:max_people]] if len(R) > max_people else R
    mode = np.nanargmax(share, 0)
    t = R.columns.to_numpy(float)
    fig = plt.figure(figsize=(16, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[.06, 1], width_ratios=[1.5, 1], hspace=.04, wspace=.12)
    cmap = ListedColormap(cols)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow([mode], aspect="auto", cmap=cmap, vmin=-.5, vmax=K - .5, extent=(t[0], t[-1], 0, 1))
    ax0.set_axis_off()
    ax0.set_title("most common state at each moment", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax.imshow(np.ma.masked_invalid(show.to_numpy(float)), aspect="auto", cmap=cmap, vmin=-.5, vmax=K - .5,
              interpolation="nearest", extent=(t[0], t[-1], len(show), 0))
    ax.set_yticks([])
    ax.set_ylabel(f"viewers ({len(show)} of {len(R)})", fontsize=9)
    ax.set_xlabel("film time (s, window start)", fontsize=9)
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.stackplot(t, share, colors=cols, linewidth=0)
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("film time (s)", fontsize=9)
    ax2.set_ylabel("share of viewers in each state", fontsize=9)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    fig.legend(handles=[Patch(color=cols[k], label=sn.label(k, "long")) for k in range(K)], loc="upper center",
               ncol=K if not sn.active() else min(K, 4),
               frameon=False, fontsize=9, bbox_to_anchor=(.5, -.02))
    fig.suptitle(title, x=.02, ha="left", fontsize=12)
    save(fig, name, co.cfg)
    _show(fig)
    return fig


def fig_state_detail(co, F, M, tl, state, n_regions=18, n_traces=3, title="", name=None, min_share=None):
    """One state in detail. Right: mean activation of the regions (rows, clustered) in every state
    (columns), the n_regions most active in `state` and the n_regions least; the chosen state's
    column framed. Left: the n_traces regions most active in `state`, their activation over the
    film (mean over viewers, band = SD), on the share of viewers in `state` at each moment (grey
    fill, right axis)."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    K = co.cfg.K
    cols = state_colours(K)
    order = M.loc[state].sort_values(ascending=False)
    regions = list(dict.fromkeys([*order.index[:n_regions // 2 + n_regions % 2], *order.index[-(n_regions // 2):]]))
    H = M[regions].T                                              # regions x states
    Z = linkage(H.to_numpy(), "average") if len(H) > 2 else None   # regions: hierarchical clustering
    if Z is not None:
        H = H.iloc[leaves_list(Z)]
    top = list(order.index[:n_traces])
    g = F.groupby("t")[top]
    mean, sd = g.mean(), g.std()
    share = tl.assign(on=tl["state"] == state).groupby("stimulus_start_s")["on"].mean()
    fig = plt.figure(figsize=(17, max(6, .32 * len(H) + 2)))
    gs = fig.add_gridspec(len(top), 2, width_ratios=[1.6, 1], wspace=.55, hspace=.25)
    tr_cols = [plt.cm.Set2(i) for i in range(len(top))]
    for i, r in enumerate(top):
        ax = fig.add_subplot(gs[i, 0])
        axs = ax.twinx()
        axs.fill_between(share.index, share.to_numpy(), step="post", color="#d9d8d3", lw=0)
        axs.set_ylim(0, 1)
        axs.set_ylabel(f"share in S{state}" if i == len(top) // 2 else "", fontsize=8, color="#8a8984")
        axs.tick_params(labelsize=7, colors="#8a8984")
        ax.set_zorder(axs.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.plot(mean.index, mean[r], color=tr_cols[i], lw=1.2)
        ax.fill_between(mean.index, mean[r] - sd[r], mean[r] + sd[r], color=tr_cols[i], alpha=.25, lw=0)
        ax.set_title(r, fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
        ax.set_xlim(mean.index.min(), mean.index.max())
        if i < len(top) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("film time (s)", fontsize=9)
        ax.set_ylabel("activation (z)", fontsize=8)
    ax = fig.add_subplot(gs[:, 1])
    v = np.nanmax(np.abs(H.to_numpy()))
    im = ax.imshow(H.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks(range(H.shape[1]), [sn.label(k) for k in H.columns],
                  rotation=0 if not sn.active() else 60, ha="center" if not sn.active() else "left")
    for k, tl_ in zip(H.columns, ax.get_xticklabels()):
        tl_.set_color(cols[int(k)])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(H)), H.index, fontsize=8)
    ax.yaxis.tick_right()
    ax.add_patch(plt.Rectangle((list(H.columns).index(state) - .5, -.5), 1, len(H), fill=False, ec="k", lw=1.6))
    fig.colorbar(im, ax=ax, location="bottom", fraction=.03, pad=.02, label="mean activation (z)")
    if Z is not None:                                             # the regions' dendrogram, beside the rows
        from scipy.cluster.hierarchy import dendrogram
        pos = ax.get_position()                                   # after the colorbar took its share
        dax = fig.add_axes([pos.x0 - .07, pos.y0, .065, pos.height])
        dendrogram(Z, orientation="left", ax=dax, no_labels=True, color_threshold=0,
                   above_threshold_color=hp.INK2, link_color_func=lambda _: hp.INK2)
        dax.set_ylim(10 * len(H), 0)                              # leaf i at 5 + 10 i = row i, top down
        dax.set_axis_off()
    fig.suptitle(title or f"State {sn.label(state)} in detail", x=.02, ha="left", fontsize=12)
    save(fig, name or f"state_detail_S{state}", co.cfg)
    _show(fig)
    return fig


# ------------------------------------------------------------------ latent-space diagnostics (notebook 08)
VARIANT_NAMES = {"official": "official fit", "official_z": "official fit, z-scored per person after",
                 "base": "z-score per edge, UMAP on 30k windows (as the official)",
                 "all_rows": "UMAP on every training window", "no_scaler": "no z-score before the UMAP",
                 "person_scaled": "edges z-scored per person before the UMAP",
                 "nndb_only": "fitted on NNDb alone (as the old version)"}


def fig_embeddings(co, E, panel="variant", colour="cohort", panels=None, titles=None, max_points=15000,
                   seed=0, height=480, ncols=3, name=None, title=None, size=2, lines=()):
    """One interactive 3D panel per value of `panel` (plotly), a dot per window, at most `max_points`
    per panel (at random). colour: "cohort", "role", "kind" (rest | movie), "task" (each film),
    "state", "sub" (one gradient over people) or "time" (the window's place in its film, 0 = start,
    1 = end). Each panel keeps its own axes. size: marker size, a number or {panel value: size};
    lines: panel values drawn as a path in time order (stimulus_start_s), e.g. a group trajectory."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise SystemExit("needs plotly: pip install --user plotly")
    panels = list(panels or dict.fromkeys(E[panel]))
    titles = titles or {}
    rows = -(-len(panels) // ncols)
    ncol = min(ncols, len(panels))
    fig = make_subplots(rows, ncol, specs=[[{"type": "scene"}] * ncol] * rows, horizontal_spacing=.01,
                        vertical_spacing=.06, subplot_titles=[titles.get(p, VARIANT_NAMES.get(p, str(p))) for p in panels])
    col = colours(co.cfg)
    tasks = sorted(E["task"].astype(str).unique(), key=lambda t: (t.lower() != "rest", t)) if "task" in E else []
    tcol = {t: (hp.INK2 if t.lower() == "rest" else _hex(plt.cm.tab20(k % 20))) for k, t in enumerate(tasks)}
    seen = set()
    for j, p in enumerate(panels):
        r, c = j // ncol + 1, j % ncol + 1
        g = E[E[panel] == p]
        sz = size.get(p, 2) if isinstance(size, dict) else size
        if len(g) > max_points:
            g = g.sample(max_points, random_state=seed)
        hover = g["cohort"].map(cname) + " · " + g["sub"].astype(str) + " · " + g["task"].astype(str)

        def add(h, colour_, label):
            show = label not in seen
            seen.add(label)
            fig.add_trace(go.Scatter3d(x=h["x"], y=h["y"], z=h["z"], mode="markers", name=label, legendgroup=label,
                                       showlegend=show, marker=dict(size=sz, color=colour_, opacity=.5),
                                       text=hover.loc[h.index], hoverinfo="text"), r, c)
        if colour == "time" and "stimulus_start_s" not in g:
            raise SystemExit("colour='time' needs stimulus_start_s: load_latents(..., extra=('stimulus_start_s',))")
        if colour == "cohort":
            for k, h in g.groupby("cohort", sort=False):
                add(h, col.get(k, hp.INK2), cname(k))
        elif colour == "role":
            for k, h in g.groupby("role", sort=False):
                add(h, PALETTE[0] if k == "train" else PALETTE[2], "training cohorts" if k == "train" else "projected cohorts")
        elif colour == "kind":
            for k, h in g.groupby("kind", sort=False):
                add(h, hp.INK2 if k == "rest" else ORANGE, k)
        elif colour == "task":
            for k, h in g.groupby(g["task"].astype(str), sort=False):
                add(h, tcol[k], k)
        elif colour == "state":
            for k, h in g.groupby("state", sort=True):
                if int(k) >= 0:
                    add(h, _hex(plt.cm.tab10(int(k) % 10)), sn.label(int(k)))
        elif colour in ("sub", "time"):
            if colour == "sub":
                key = g["cohort"] + ":" + g["sub"].astype(str)
                v = key.map({s_: k for k, s_ in enumerate(sorted(key.unique()))})
                scale, label = "Viridis", "person"
            else:
                t = g["stimulus_start_s"].astype(float)
                v = t / t.groupby([g["cohort"], g["task"]]).transform("max").replace(0, np.nan)
                scale, label = "Plasma", "time in film (0 = start, 1 = end)"
            if p in lines:
                o = g["stimulus_start_s"].astype(float).argsort().to_numpy()
                g, v, hover = g.iloc[o], v.iloc[o], hover.iloc[o]
            fig.add_trace(go.Scatter3d(x=g["x"], y=g["y"], z=g["z"], mode="markers+lines" if p in lines else "markers",
                                       showlegend=False, text=hover, line=dict(color="rgba(80,80,80,.35)", width=2),
                                       hoverinfo="text", marker=dict(size=sz, opacity=.5 if p not in lines else .9,
                                                                     color=v, colorscale=scale,
                                       colorbar=dict(title=label, len=.5, thickness=10) if j == 0 else None,
                                       showscale=j == 0)), r, c)
        elif colour in g and pd.api.types.is_numeric_dtype(g[colour]):
            has = g[colour].notna()
            if (~has).any():
                add(g[~has], hp.MUTED, "no score")
            h = g[has]
            fig.add_trace(go.Scatter3d(x=h["x"], y=h["y"], z=h["z"], mode="markers", showlegend=False,
                                       text=hover.loc[h.index] + " · " + colour + " " + h[colour].round(1).astype(str),
                                       hoverinfo="text", marker=dict(size=sz, opacity=.6, color=h[colour], colorscale="Viridis",
                                       cmin=E[colour].min(), cmax=E[colour].max(), showscale=j == 0,
                                       colorbar=dict(title=colour, len=.5, thickness=10) if j == 0 else None)), r, c)
        else:
            raise SystemExit(f"colour={colour!r}: cohort, role, kind, task, state, sub, time or a numeric column")
        fig.update_layout(**{f"scene{'' if j == 0 else j + 1}": dict(aspectmode="cube", xaxis_title="1",
                                                                       yaxis_title="2", zaxis_title="3")})
    fig.update_layout(height=height * rows, width=460 * ncol, margin=dict(l=0, r=0, t=70, b=0),
                      legend=dict(itemsizing="constant"),
                      title=title or f"one dot per window · colour = {colour}")
    if co.cfg.save_dir and name:
        hp.Path(co.cfg.save_dir).mkdir(parents=True, exist_ok=True)
        fig.write_html(hp.Path(co.cfg.save_dir) / f"{name}.html", include_plotlyjs="cdn")
    return fig


def fig_window_stats(co, ws, ps, name="window_stats", fs=11):
    """Does the number of volumes in a window change the correlations it yields? Left: each person's
    mean |r| over windows and edges against the volumes per window. Right: each edge's variance over
    a person's windows (mean over edges) against the same, with the variance a correlation would
    have from sampling alone if the connectivity never changed, (1 - r^2)^2 / (n - 1) at the median
    r. A cohort above its neighbours on the right is noisier because its windows are shorter in
    volumes, not because its connectivity moves more. One dot per person x task."""
    col = colours(co.cfg)
    pm = (ws.groupby(["cohort", "task", "sub"], as_index=False)
          .agg(mean_abs_r=("mean_abs_r", "mean"), n_tr=("n_tr_effective", "median")))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for c in [x for x in co.cfg.cohorts if x in set(ps["cohort"])]:
        a, b = pm[pm["cohort"] == c], ps[ps["cohort"] == c]
        kw = dict(s=9, alpha=.45, color=col[c], lw=0)
        ax[0].scatter(a["n_tr"], a["mean_abs_r"], label=f"{cname(c)} ({a['n_tr'].median():.0f} volumes)", **kw)
        ax[1].scatter(b["n_tr"], b["edge_var"], **kw)
    n = np.linspace(max(4, ps["n_tr"].min() * .9), ps["n_tr"].max() * 1.05, 100)
    r = float(np.median(ps["mean_r"]))
    ax[1].plot(n, (1 - r ** 2) ** 2 / (n - 1), color=hp.INK2, ls="--", lw=1.2,
               label=f"sampling alone, r = {r:.2f}")
    ax[0].set(xlabel="volumes per window", ylabel="mean |r| (edges, windows)")
    ax[1].set(xlabel="volumes per window", ylabel="variance of an edge over a person's windows")
    ax[0].legend(fontsize=fs - 2, frameon=False, markerscale=2)
    ax[1].legend(fontsize=fs - 2, frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, name, co.cfg)
    return fig


def fig_mapper(co, graphs, colour="state", titles=None, name=None, fs=11, panel_size=4.6, ncols=None, title=None):
    """Mapper shape graphs side by side, {title: (rows, nodes, edges, positions)} from
    ct.mapper_graph + ct.graph_layout. A dot per node (area = how many windows it holds), a line
    per shared window. colour: the node's most common "cohort", "state", "task" or "kind", or its
    mean "time" in the film (0 = start, 1 = end)."""
    col = colours(co.cfg)
    nc = min(ncols or len(graphs), len(graphs))
    nr = -(-len(graphs) // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(panel_size * nc, panel_size * nr), squeeze=False)
    for ax in axes.flat[len(graphs):]:
        ax.set_axis_off()
    handles = {}
    for ax, (ptitle, (E, nodes, edges, P)) in zip(axes.flat, graphs.items()):
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(P[edges], colors=hp.MUTED, lw=.5, alpha=.6, zorder=1))
        size = np.array([len(n) for n in nodes], float)
        area = 6 + 120 * np.sqrt(size / size.max())
        if colour == "time":
            t = E["stimulus_start_s"].astype(float)
            t = (t / t.groupby([E["cohort"], E["task"]]).transform("max").replace(0, np.nan)).to_numpy()
            v = [np.nanmean(t[n]) for n in nodes]
            sc = ax.scatter(P[:, 0], P[:, 1], s=area, c=v, cmap="plasma", vmin=0, vmax=1, lw=.3,
                            edgecolor="white", zorder=2)
        else:
            key = E[colour].astype(str).to_numpy() if colour != "state" else E["state"].astype(int).to_numpy()
            dom = [pd.Series(key[n]).mode().iloc[0] for n in nodes]
            if colour == "cohort":
                cmap = {k: col.get(k, hp.INK2) for k in set(dom)}
                label = cname
            elif colour == "state":
                cmap = {k: (hp.MUTED if int(k) < 0 else _hex(plt.cm.tab10(int(k) % 10))) for k in set(dom)}
                label = lambda k: sn.label(k) if int(k) >= 0 else "no state"
            elif colour == "kind":
                cmap = {k: hp.INK2 if k == "rest" else ORANGE for k in set(dom)}
                label = str
            else:
                ks = sorted(set(E[colour].astype(str)))
                cmap = {k: _hex(plt.cm.tab20(ks.index(k) % 20)) for k in set(dom)}
                label = str
            ax.scatter(P[:, 0], P[:, 1], s=area, c=[cmap[k] for k in dom], lw=.3, edgecolor="white", zorder=2)
            for k in sorted(set(dom), key=str):
                handles[label(k)] = Line2D([], [], marker="o", ls="", color=cmap[k], ms=6)
        m = E.attrs.get("mapper", {})
        ax.set_title(f"{ptitle}\n{len(nodes)} nodes · {len(edges)} edges · {len(E):,} points", fontsize=fs)
        ax.set_axis_off()
    if colour == "time":
        fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=.6, label="time in film (0 = start, 1 = end)")
    elif handles:
        fig.legend(handles.values(), handles.keys(), loc="center right", frameon=False, fontsize=fs - 1,
                   title=colour)
    if title:
        fig.suptitle(title, fontsize=fs + 1, y=1 + .12 / nr)
    if name:
        save(fig, name, co.cfg)
    return fig


def fig_band_check(co, B, name="band_check", fs=11):
    """Mean |r| (left) and edge variance over windows (right) per cohort, the band as extracted
    (hollow) and filtered 0.01-0.1 Hz (filled); a dot per person x task, the line joins the medians."""
    col = colours(co.cfg)
    cohorts = [c for c in co.cfg.cohorts if c in set(B["cohort"])]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, v, lab in ((ax[0], "mean_abs_r", "mean |r| (edges, windows)"),
                      (ax[1], "edge_var", "variance of an edge over a person's windows")):
        for i, c in enumerate(cohorts):
            med = []
            for k, (b, fill) in enumerate((("as extracted", False), ("0.01-0.1 Hz", True))):
                y = B.loc[(B["cohort"] == c) & (B["band"] == b), v]
                x = i + (k - .5) * .3 + np.random.default_rng(0).uniform(-.05, .05, len(y))
                a.scatter(x, y, s=14, facecolor=col[c] if fill else "none", edgecolor=col[c], lw=.8, alpha=.7)
                med.append((i + (k - .5) * .3, y.median()))
            a.plot(*zip(*med), color=col[c], lw=1.5)
        a.set_xticks(range(len(cohorts)), [cname(c) for c in cohorts])
        a.set_ylabel(lab)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].legend([Line2D([], [], marker="o", ls="", mfc="none", mec=hp.INK2),
                  Line2D([], [], marker="o", ls="", color=hp.INK2)],
                 ["band as extracted", "filtered 0.01–0.1 Hz"], frameon=False, fontsize=fs - 2)
    fig.tight_layout()
    save(fig, name, co.cfg)
    return fig


def _bh(p):
    """Benjamini-Hochberg q-values; NaN stays NaN."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        v = p[ok]
        o = np.argsort(v)
        r = v[o] * len(v) / np.arange(1, len(v) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        q_ok = np.empty_like(v)
        q_ok[o] = np.minimum(r, 1)
        q[ok] = q_ok
    return q


def guide_table(co, st, st_transitions=None, guides=(.01, .06), min_occ=0.01, inside=True, sort_by="task",
                family_of="all"):
    """The task x person panel as a table: every point with its VPC task and VPC person, each with
    its 95% CI and P(tier) (the probability that the VPC lies in its own band of the guides, from
    the delta-method interval of the mixed model; the person's needs mixed models fitted after it
    was added -- older caches have no person interval), and the band each falls in. p-values:
    "p task < .06" (one-sided Wald on the logit, H0: VPC >= .06), "p task > .01" (H0: VPC <= .01),
    "p LRT task" (H0: no task component; likelihood ratio, mixed models fitted after it was added),
    the same for person, each with its BH-FDR q over the rows shown.
    inside=True keeps the points with BOTH shares under the upper guide; False = every point.
    sort_by: "task" or "person", the P(tier) the rows are ordered by (most certain first).
    family_of: "all", "occupancy", "dynamics" or "transitions" (structure and p(j | i)).
    Transitions: those of rows of states occupied >= min_occ, as drawn."""
    lo_g, hi_g = (.01, .06) if guides is True or guides is None else guides
    parts = [st.assign(_fam=[family(v) for v in st["variable"]])]
    if st_transitions is not None:
        occ = pd.concat([v[co.occ].mean() for v in co._ref_tables()], axis=1).mean(axis=1).to_numpy()
        cells = [c for c in co.cells if occ[int(c.split("->")[0])] >= min_occ]
        tc = st_transitions[st_transitions["variable"].isin(cells)]
        parts.append(tc.assign(_fam="transitions"))
    t = pd.concat(parts, ignore_index=True).drop_duplicates("variable")
    for c in ("task_lo", "task_hi", "task_p_tier", "person_lo", "person_hi", "person_p_tier",
              "task_p_lrt", "person_p_lrt"):
        if c not in t:
            t[c] = np.nan
    band = lambda x: np.where(np.isnan(x), "", np.where(x < lo_g, f"< {lo_g:g}", np.where(x < hi_g, f"{lo_g:g}-{hi_g:g}", f">= {hi_g:g}")))
    ci = lambda a, b_: [f"[{u:.3f}, {w:.3f}]" if np.isfinite(u) else "" for u, w in zip(a, b_)]

    def name(v):
        v = str(v)
        if v.startswith("occ_"):
            return f"Occupancy · {sn.label(v[4:], 'both')}"
        if "->" in v:
            a, b_ = v.split("->")
            return f"{sn.label(a, 'both')} → {sn.label(b_, 'both')}"
        return nice(v).replace(" (fit to reference)", "")
    vt, vp = _vpc_within(t, "task").to_numpy(float), _vpc_within(t, "person").to_numpy(float)
    out = pd.DataFrame({"variable": t["variable"].to_numpy(), "element": [name(v) for v in t["variable"]],
                        "family": t["_fam"],
                        "VPC task": vt, "task 95% CI": ci(t["task_lo"], t["task_hi"]),
                        "P(tier) task": t["task_p_tier"].to_numpy(float), "task band": band(vt),
                        "VPC person": vp, "person 95% CI": ci(t["person_lo"], t["person_hi"]),
                        "P(tier) person": t["person_p_tier"].to_numpy(float), "person band": band(vp)})
    # p-values. Wald, one-sided, on the logit scale of the delta-method interval the model gave
    # (se = (logit hi - logit lo) / 3.92): "VPC < upper guide" (H0: VPC >= .06, the share is small)
    # and "VPC > lower guide" (H0: VPC <= .01, it is not negligible). LRT: "the component exists"
    # (H0: its variance is 0), from the mixed model itself. q = BH-FDR over the rows of the table.
    from scipy.stats import norm
    lgt = lambda x: np.log(x / (1 - x))
    for k_, vv in (("task", vt), ("person", vp)):
        lo_, hi_ = t[f"{k_}_lo"].to_numpy(float), t[f"{k_}_hi"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            se = (lgt(hi_) - lgt(lo_)) / (2 * 1.959964)
            z_hi = (lgt(vv) - lgt(hi_g)) / se
            z_lo = (lgt(vv) - lgt(lo_g)) / se
        ok_ = np.isfinite(se) & (se > 0)
        out[f"p {k_} < {hi_g:g}"] = np.where(ok_, norm.cdf(z_hi), np.nan)
        out[f"p {k_} > {lo_g:g}"] = np.where(ok_, norm.sf(z_lo), np.nan)
        out[f"p LRT {k_}"] = t[f"{k_}_p_lrt"].to_numpy(float)
    for c in [c for c in out if c.startswith("p ")]:
        out[c.replace("p ", "q ", 1)] = _bh(out[c].to_numpy(float))
    if family_of != "all":
        if family_of not in ("occupancy", "dynamics", "transitions"):
            raise SystemExit(f'family_of must be "all", "occupancy", "dynamics" or "transitions", got {family_of!r}')
        out = out[out["family"] == family_of]
    if inside:
        out = out[(out["VPC task"] < hi_g) & (out["VPC person"] < hi_g)]
    if sort_by not in ("task", "person"):
        raise SystemExit(f'sort_by must be "task" or "person", got {sort_by!r}')
    if out[f"P(tier) {sort_by}"].isna().all():
        print(f"no P(tier) {sort_by} in these mixed models (fitted before it was added): rerun "
              f"slurm/mixed_model.sbatch with --overwrite")
    for k_ in ("task", "person"):                              # the interval as numbers, for the bar plot
        out[f"{k_}_lo"], out[f"{k_}_hi"] = t[f"{k_}_lo"].to_numpy(float), t[f"{k_}_hi"].to_numpy(float)
    return out.sort_values(f"P(tier) {sort_by}", ascending=False).reset_index(drop=True).round(3)


def fig_guide_bars(co, tab, order=("task", "person"), dot="below", use_q=True, guides=(.01, .06), name="vpc_bars",
                   fs=8, labels="both", max_elements=None, ci=False, tests="both"):
    """guide_table drawn: one row per element (y), two horizontal bars under each other -- VPC task
    (dark) and VPC person (light) -- on the VPC axis (x), with the guides as dashed lines.
    Marks right of a bar (or of its interval): * / ** = the component exists (likelihood ratio,
    H0: no task / person variance; q or p < .05 / .01); a dot = the share is significantly on the
    stated side of a guide: dot="below" -> under the upper guide (H0: VPC >= .06, small / stable),
    "above" -> over the lower guide (H0: VPC <= .01, not negligible). use_q: FDR q-values (else p).
    order: ("task", "person") sorts by VPC task then VPC person; ("person", "task") the other way.
    labels: "both" (S3 · name), "number" (S3) or "name" for the states in the element names.
    max_elements: draw only the first N rows of that order. ci: the 95% interval of each VPC as a
    thin line, where the mixed model gave one. tests: which marks to draw -- "both", "lrt" (the
    asterisks), "guide" (the dots) or "none"."""
    from matplotlib.lines import Line2D
    lo_g, hi_g = (.01, .06) if guides is True or guides is None else guides
    if dot not in ("below", "above"):
        raise SystemExit('dot must be "below" or "above"')
    if labels not in ("both", "number", "name"):
        raise SystemExit('labels must be "both", "number" or "name"')
    if tests not in ("both", "lrt", "guide", "none"):
        raise SystemExit('tests must be "both", "lrt", "guide" or "none"')
    show_lrt, show_guide = tests in ("both", "lrt"), tests in ("both", "guide")
    t = tab.sort_values([f"VPC {order[0]}", f"VPC {order[1]}"], ascending=True).reset_index(drop=True)
    if max_elements:
        t = t.head(int(max_elements))
    n = len(t)

    def label_of(v):
        v = str(v)
        if v.startswith("occ_"):
            return f"Occupancy · {sn.label(v[4:], labels)}"
        if "->" in v:
            a, b_ = v.split("->")
            return f"{sn.label(a, labels)} → {sn.label(b_, labels)}"
        return nice(v).replace(" (fit to reference)", "")
    names = [label_of(v) for v in t["variable"]] if "variable" in t else list(t["element"])
    col = {"task": "#1f4e9c", "person": "#9cc3f5"}
    fig, ax = plt.subplots(figsize=(7.5, .9 + .2 * n))
    pre = "q" if use_q else "p"
    for i, r in t.iterrows():
        for j, k in enumerate(order):                       # the first of `order` on top of its pair
            y = i + (-.19 if j == 0 else .19)
            v = r[f"VPC {k}"]
            if not np.isfinite(v):
                continue
            ax.barh(y, v, height=.36, color=col[k], lw=0)
            end = v
            lo_, hi_ = r.get(f"{k}_lo", np.nan), r.get(f"{k}_hi", np.nan)
            if ci and np.isfinite(lo_) and np.isfinite(hi_) and hi_ > lo_:
                ax.plot([lo_, hi_], [y, y], color=hp.INK2, lw=.8, zorder=3)
                end = max(v, hi_)
            marks = ""
            pl = r.get(f"{pre} LRT {k}", np.nan)
            if show_lrt and np.isfinite(pl) and pl < .05:
                marks += "**" if pl < .01 else "*"
            pg = r.get(f"{pre} {k} < {hi_g:g}" if dot == "below" else f"{pre} {k} > {lo_g:g}", np.nan)
            if show_guide and np.isfinite(pg) and pg < .05:
                marks += " •" if marks else "•"
            if marks:                                        # just right of the bar / interval, in points
                ax.annotate(marks, (end, y), xytext=(2, 0), textcoords="offset points", va="center",
                            ha="left", fontsize=fs, color=hp.INK)
    for g in (lo_g, hi_g):
        ax.axvline(g, color="#9b9a95", lw=.8, ls="--", zorder=0)
    ax.set_yticks(range(n), names, fontsize=fs)
    ax.set_ylim(n - .5, -.5)                                 # first row on top
    ax.set_xlabel("Variance Partition Coefficient", fontsize=fs + 1)
    cols = ["VPC task", "VPC person"] + (["task_hi", "person_hi"] if ci and "task_hi" in t else [])
    ax.set_xlim(0, max(.1, np.nanmax(t[cols].to_numpy(float)) * 1.12))
    ax.tick_params(axis="x", labelsize=fs)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    side = f"VPC < {hi_g:g}" if dot == "below" else f"VPC > {lo_g:g}"
    hand = [Patch(color=col[order[0]], label=order[0].capitalize()),
            Patch(color=col[order[1]], label=order[1].capitalize())]
    if show_lrt:
        hand.append(Line2D([], [], ls="none", marker="$*$", ms=7, color=hp.INK,
                           label=f"component exists ({pre} < .05; ** < .01; likelihood ratio)"))
    if show_guide:
        hand.append(Line2D([], [], ls="none", marker="o", ms=3, color=hp.INK, label=f"{side} ({pre} < .05, one-sided)"))
    if ci:
        hand.append(Line2D([], [], color=hp.INK2, lw=.8, label="95% CI"))
    ax.legend(handles=hand, frameon=False, fontsize=fs, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2,
              borderaxespad=0)
    fig.tight_layout()
    save(fig, name, co.cfg)
    return fig
