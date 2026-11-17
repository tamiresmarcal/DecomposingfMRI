"""Figures of 11_normal_brain_cohorts.ipynb, drawn from unsupervised_model.cohort_tables. The state diagram and its
dynamics column are notebook 10's (figures.hads), with all their label rules."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.lines import Line2D

from fmri_decomposition.figures import hads as hp
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
        P["main"].set_xticks(range(K)); P["main"].set_yticks(range(K)); P["main"].set_xlabel("to state j", fontsize=9)
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
                fig.add_trace(dots(h, _hex(plt.cm.tab10(int(k) % 10)), f"state {int(k)}", j, group=f"s{int(k)}",
                                   hover=h["sub"] + f" · state {int(k)}"), 1, j)
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
        return f"Occupancy S{v[4:]}"
    if "->" in v:
        return f"p(S{v.replace('->', ' → S')})"
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
    mean = np.mean([m["mean"].to_numpy(float) for m in M.values()], axis=0).reshape(K, K)
    panels = [("mean of the tasks", mean)] + [(f"{cname(c)} · {f}", m["mean"].to_numpy(float).reshape(K, K))
                                              for (c, f), m in M.items()]
    for k, (title, T) in enumerate(panels):
        ax = axs[k // per_row, k % per_row]
        im = ax.imshow(T, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=7, color=hp.INK2)
    for j in range(n, nrow * per_row):
        axs[j // per_row, j % per_row].set_visible(False)
    fig.suptitle("Transitions p(j | i), film by film — rows: from state i, columns: to state j", fontsize=11,
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


def _p_columns(ax, t, y, pvalue, tests):
    """Confidence right of the panel, one column per test, exact or as stars.
    tier: P(tier), the share of bootstrap |d| in the tier of the observed |d| (stars: * >= .90,
          ** >= .95, *** >= .99);
    difference: paired t-test movie vs rest, BH-FDR (* <.05, ** <.01, *** <.001);
    tost: equivalence at +-sesoi_d SD, BH-FDR (a small p = within the bounds, i.e. stable)."""
    for k, test in enumerate(tests):
        col, head = P_COLS[test]
        if col not in t:
            continue
        x = 1.07 + .22 * k
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


def _tier_panel(ax, st, variables, pvalue=None, test="tier", separators=False, sort=True):
    """pvalue: None, "exact" or "stars"; test: "tier" (P(tier)), "difference", "tost", "both"
    (difference + tost), "all", or a list (see _p_columns). A thin line separates the families
    (transitions, occupancy, dynamic metrics). A row with too few people scanned in both
    conditions to estimate (NaN) says so, with its n."""
    t = st.set_index("variable").loc[[v for v in variables if v in set(st["variable"])]].reset_index()
    if sort:   # families in the order given; inside each, the profile r first, then by |d| (top = smallest)
        fam_rank = {f: k for k, f in enumerate(dict.fromkeys(family(v) for v in t["variable"]))}
        key = pd.DataFrame({"f": [fam_rank[family(v)] for v in t["variable"]],
                            "fit": [0 if v.endswith("_fit") else 1 for v in t["variable"]],
                            "d": t["task_d"].abs().fillna(np.inf)})
        t = t.loc[key.sort_values(["f", "fit", "d"], kind="stable").index].reset_index(drop=True)
    y = np.arange(len(t))[::-1]
    ax.axvspan(0, .2, color="#e8f0fb")
    ax.axvspan(.2, .5, color="#f3f6fb")
    d = t["task_d"].abs()
    has = t["task_lo"].notna().to_numpy() & t["task_hi"].notna().to_numpy()   # per row: one NaN must not
    if has.any():                                                             # hide every other CI
        lo = np.minimum(t["task_lo"].abs(), t["task_hi"].abs())
        hi = np.maximum(t["task_lo"].abs(), t["task_hi"].abs())
        lo[np.sign(t["task_lo"]) != np.sign(t["task_hi"])] = 0       # the CI crosses 0
        ax.hlines(y[has], lo[has], hi[has], color="#9b9a95", lw=1)
    cols = [TIER_COL.get(x, "#c9c8c2") for x in t["tier"]]
    ax.scatter(d, y, c=cols, s=45, zorder=3, edgecolor="white")
    for yi, dv, n in zip(y, d, t.get("n_paired", pd.Series(np.nan, index=t.index))):
        if not np.isfinite(dv):
            ax.text(.02, yi, f"too few people with both conditions (n = {int(n) if np.isfinite(n) else 0})",
                    va="center", fontsize=6.5, color=hp.MUTED)
    fam = [family(v) for v in t["variable"]] if separators else []
    for k in range(1, len(fam)):
        if fam[k] != fam[k - 1]:
            ax.axhline(y[k] + .5, color="#c9c8c2", lw=.8)
    ax.set_yticks(y, [nice(v) for v in t["variable"]], fontsize=8)
    top = len(t) - .3
    for x, lab in ((.1, "stable"), (.35, "moderate"), (max(.75, d.max() * .7), "context-dependent")):
        ax.text(x, top, lab, ha="center", fontsize=10, color=TIER_COL[lab], gid="scaled")   # fixed size: fits its band
    ax.set_xlim(0, max(1.0, float(d.max()) * 1.1))
    ax.set_ylim(-.7, len(t) + .2)
    ax.set_xlabel("|d| movie vs rest, same people (95% CI)", fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if pvalue:
        _p_columns(ax, t, y, pvalue, _tests(test))


def fig_figure3(co, st, tier_variables=None, font=1.45, diagram_fs=2.0, occupancy="each", name="figure3",
                pvalue=None, test="tier"):
    """Manuscript Figure 3 as one figure, rest and movie together (each person's mean of the
    conditions they have; `groups(tasks="all")`).
    A  every cohort: occupancy, p(j | i) and the dynamic metrics (mean, 90% CI);
    B  the reference transition graph: the mean of the cohorts in A, each cohort once;
    C  which properties depend on context (`stability` tiers: movie vs rest, same people).
    font scales every text; diagram_fs the diagram's own (its labels are placed for that size).
    occupancy="each": C lists every state's occupancy; "fit": one row, the occupancy profile's
    correlation with the reference (`occupancy_fit`; st must include it).
    pvalue / test: the p-value column(s) right of C (`_tier_panel`): None, "exact" or "stars";
    "tier" (P(tier), default), "difference" (paired t), "tost" (equivalence), "both", "all"."""
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
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.15], hspace=.12)
    n = len(cohorts)
    gA = gs[0].subgridspec(2, n + 1, height_ratios=[.35, 1], width_ratios=[1] * n + [1.25], hspace=.25, wspace=.35)
    for i, c in enumerate(cohorts):
        T, o = graphs[c]
        axo = fig.add_subplot(gA[0, i])
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
        axm.imshow(T, cmap="Blues", vmin=0, vmax=1)
        axm.set_xticks(range(K))
        axm.set_yticks(range(K))
        axm.tick_params(labelsize=7)
        axm.set_xlabel("to state j", fontsize=8)
        if i == 0:
            axm.set_ylabel("from state i", fontsize=8)
    gD = gA[:, n].subgridspec(len(co.dyn), 1, hspace=1.1)
    for r_, v in enumerate(co.dyn):
        axd = fig.add_subplot(gD[r_])
        t = agree[agree["variable"] == v]
        for j, rr in enumerate(t.itertuples()):
            yy = -j * .3
            axd.plot([rr.mean_lo, rr.mean_hi], [yy, yy], color=col[rr.cohort], lw=1.3)
            axd.plot(rr.mean, yy, "o", ms=4, color=col[rr.cohort],
                     mfc=col[rr.cohort])
        axd.set_yticks([])
        axd.set_ylim(-.3 * len(cohorts), .3)
        lo_, hi_ = fixed_range(co, v)
        if np.isfinite(lo_):
            axd.set_xlim(lo_, hi_)
        for sp in ("top", "right", "left"):
            axd.spines[sp].set_visible(False)
        axd.tick_params(labelsize=7)
        axd.set_title(nice(v), fontsize=8, pad=2)

    gB = gs[1].subgridspec(1, 2, width_ratios=[1.25, 1], wspace=.35)
    ax = fig.add_subplot(gB[0])
    hp.diagram(ax, T_ref, o_ref, edges, 1.0, 1.0, fs=diagram_fs)
    co.diagram_p_min = min(T_ref[e] for e in edges)       # for the legend: the thinnest arrow drawn
    ax = fig.add_subplot(gB[1])
    occ = co.occ if occupancy == "each" else ["occupancy_fit"]
    _tier_panel(ax, st, tier_variables or ["transition_fit", *co.dyn, *occ], pvalue, test)
    for t in fig.findobj(plt.matplotlib.text.Text):              # every text larger, but the diagram's:
        if t.get_gid() != "scaled":                              # its labels were placed at diagram_fs
            t.set_fontsize(t.get_fontsize() * font)
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


def fig_aggregation_check(co, n_boot=2000, min_occ=0.01, name="check_aggregation", pvalue=None, test="tier"):
    """Does one number per family lose information? Movie vs rest |d_av| (same people, 95% CI); transitions
    on the left, dynamic metrics over occupancy on the right: the family's profile r with the reference on top, then every member of the family --
    transitions p(j | i) (rows of states with mean occupancy >= min_occ), occupancy of each state,
    and each dynamic metric. Same tiers as Figure 3C."""
    occ = pd.concat([v[co.occ].mean() for v in co.data.values()], axis=1).mean(axis=1).to_numpy()
    keep = [c for c in co.cells if occ[int(c.split("->")[0])] >= min_occ]
    fams = [("Transitions", "transition_fit", keep), ("Dynamic metrics", "dynamics_fit", list(co.dyn)),
            ("Occupancy", "occupancy_fit", list(co.occ))]
    st = co.stability([v for _, f, m in fams for v in [f, *m]], n_boot=n_boot)
    from matplotlib.gridspec import GridSpec
    h = [len(m) + 1 for _, _, m in fams]
    fig = plt.figure(figsize=(14, max(6, .2 * h[0] + 2)))
    gs = GridSpec(2, 2, figure=fig, height_ratios=h[1:], wspace=1.1 if pvalue else .9, hspace=.35,
                  left=.2, right=.95)
    axs = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
    for ax, (title, f, members) in zip(axs, fams):            # transitions left; metrics over occupancy right
        _tier_panel(ax, st, [f, *members], pvalue, test)
        ax.axhline(len(members) - .5, color="#c9c8c2", lw=.8)       # the profile above, members below
        ax.set_title(title, fontsize=12, loc="left")
        ax.tick_params(axis="y", labelsize=7 if len(members) > 20 else 9)
    axs[1].set_xlabel("")                                      # one x label in the right column, at the bottom
    fig.suptitle(f"Aggregation check · {ht.set_label(co.cfg.space)} · |d_av| movie vs rest", x=.02, ha="left", fontsize=12)
    save(fig, name, co.cfg)
    _show(fig)
    return fig, st
