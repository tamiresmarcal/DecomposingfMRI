"""Figures of 11_normal_brain_cohorts.ipynb, drawn from cohort_tables. The state diagram and its
dynamics column are notebook 10's (hads_plots), with all their label rules."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import hads_plots as hp
import hads_tables as ht

# one colour per cohort, in the order they are listed; a new cohort takes the next
PALETTE = ["#eda100", "#008300", "#6250d6", "#52514e", "#e34948", "#1baf7a", "#8c6d4f"]
DYN_UNIT = {"switch_rate": " (share of steps)", "mean_dwell_s": " (s)", "entropy_rate_bits": " (bits)",
            "dispersion": " (latent-space units)", "n_distinct_transitions": " (cells of K×K)",
            "self_transition_rate": " (share of steps)", "switches_per_min": " (per min)", "n_states_visited": " (states)"}


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
        top = {"entropy_rate_bits": np.log2(cfg.K), "n_distinct_transitions": cfg.K ** 2}   # the most K states allow
        if v in top:
            ax.set_xlim(0, top[v])
        elif v in cfg.ranges:
            ax.set_xlim(*cfg.ranges[v])
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

def _matrices(co, graphs):
    """The panels, and every cohort's movie p(j | i) drawn in its own; the comparison's main is
    returned as axes[-1]["main"] for the caller."""
    cfg, col = co.cfg, colours(co.cfg)
    labels = [c for c in cfg.cohorts if (c, "movie") in graphs]
    fig, axes = _panels(len(labels) + 1)
    vmax = max(graphs[c, "movie"][0].max() for c in labels)
    for P, c in zip(axes, labels):
        im = P["main"].imshow(graphs[c, "movie"][0], cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        P["main"].set_title(f"{c} ({co.role(c)})", fontsize=11, color=col[c])
    fig.colorbar(im, cax=_cax(fig, axes[-2]["main"])).set_label("p(j | i)")
    for P in axes:
        P["main"].set_xticks(range(cfg.K)); P["main"].set_yticks(range(cfg.K)); P["main"].set_xlabel("to state j", fontsize=9)
    return fig, axes, labels

def _panels(n, height=7.4):
    """One fixed grid per panel: occupancy on top, row entropy on the left, the matrix, icons below.
    Returns the figure and, per panel, dict(top, left, main, bottom)."""
    fig = plt.figure(figsize=(5.4 * n, height))
    outer = fig.add_gridspec(1, n, wspace=.38, left=.04, right=.95, top=.86, bottom=.03)
    out = []
    for k in range(n):
        g = outer[k].subgridspec(3, 2, width_ratios=[.001, 1], height_ratios=[.3, 1, .3], wspace=.04, hspace=.12)
        main = fig.add_subplot(g[1, 1])
        out.append(dict(main=main, top=fig.add_subplot(g[0, 1], sharex=main), left=fig.add_subplot(g[1, 0], sharey=main),
                        bottom=fig.add_subplot(g[2, :])))
    return fig, out

ICON = {"switch_rate": "⇄", "mean_dwell_s": "⏱", "entropy_rate_bits": "H", "dispersion": "✳", "n_distinct_transitions": "#"}
FMT = {"switch_rate": "{:.2f}", "mean_dwell_s": "{:.0f} s", "entropy_rate_bits": "{:.2f} bits", "dispersion": "{:.2f}",
       "n_distinct_transitions": "{:.0f}"}

def _margins(co, axes, labels, compare_left, compare_icons):
    """The dynamics as icons + values under every matrix. compare_left = (values per state, colours, label); compare_icons =
    (texts per dynamics, colours) for the comparison matrix."""
    col, K = colours(co.cfg), co.cfg.K
    mv = co.movie()
    for P, c in zip(axes, [*labels, None]):
        ax, left, bottom = P["main"], P["left"], P["bottom"]
        left.axis("off")                                          # no side panel: the dynamics are below
        ax.set_ylabel("from state i", fontsize=9)
        bottom.set_xlim(-.5, len(co.dyn) - .5); bottom.set_ylim(0, 1); bottom.axis("off")
        for k, v in enumerate(co.dyn):
            if c is not None:
                text, colour = FMT.get(v, "{:.2f}").format(mv.loc[mv["cohort"] == c, v].mean()), hp.INK2
            else:
                text, colour = compare_icons[0][k], compare_icons[1][k]
            if ICON.get(v) == "⏱":
                bottom.plot(k, .62, marker=hp.STOPWATCH, ms=14, mfc="none", mec=colour, mew=1.3)
            else:
                bottom.text(k, .62, ICON.get(v, v[:2]), ha="center", va="center", fontsize=13, color=colour)
            bottom.text(k, .18, text, ha="center", va="center", fontsize=7.5, color=colour)

def _occ_bars(co, graphs, axes, labels, compare):
    """A bar chart of time in each state above every matrix, aligned with its columns (state j).
    compare = (values per state, colours per state, label) for the bars above the comparison matrix."""
    col, K = colours(co.cfg), co.cfg.K
    ymax = max(graphs[c, "movie"][1].max() for c in labels) * 1.15
    tops = []
    for P, c in zip(axes, labels):
        top, ax = P["top"], P["main"]
        top.bar(range(K), graphs[c, "movie"][1], width=.8, color=col[c], alpha=.75)
        top.set_ylim(0, ymax)
        top.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
        tops.append(top)
    values, colours_, label = compare
    top = axes[-1]["top"]
    top.bar(range(K), values, width=.8, color=colours_)
    if label == "largest |d|":                                    # d on its own scale: 0.8 = large
        top.set_ylim(0, max(1, np.nanmax(values) * 1.1))
        for w in (.2, .5, .8):
            top.axhline(w, color=hp.MUTED, lw=.5, ls=":")
    top.set_ylabel(label, fontsize=7)
    tops.append(top)
    tops[0].set_ylabel("time in state", fontsize=8)
    for t, P in zip(tops, axes):
        ax = P["main"]
        t.tick_params(labelbottom=False, labelsize=7)
        t.spines["right"].set_visible(False); t.spines["top"].set_visible(False)
        t.set_title(ax.get_title(), fontsize=11, color=ax.title.get_color())
        ax.set_title("")
    return tops

def fig_matrix_cohen(co, graphs, d):
    """4b. Movie p(j | i) per cohort, then per cell the largest |Cohen's d| of a cohort vs the others."""
    cfg, K = co.cfg, co.cfg.K
    labels = [c for c in cfg.cohorts if (c, "movie") in graphs]
    fig, axes, labels = _matrices(co, graphs)
    t = d[d["variable"].isin(co.cells)]
    worst = t.assign(a=t["d"].abs()).groupby("variable")["a"].max()
    W = np.array([worst.get(c, np.nan) for c in co.cells]).reshape(K, K)
    ax = axes[-1]["main"]
    cmap = plt.matplotlib.colors.ListedColormap(["#f3f2ee", "#d8d6cf", "#f0a8a2", "#e34948"])
    norm = plt.matplotlib.colors.BoundaryNorm([0, .2, .5, .8, max(1, np.nanmax(W) + .01)], cmap.N)
    im2 = ax.imshow(W, cmap=cmap, norm=norm, aspect="auto")
    for i in range(K):
        for j in range(K):
            if np.isfinite(W[i, j]):
                ax.text(j, i, f"{W[i, j]:.1f}", ha="center", va="center", fontsize=7, color=hp.INK)
    cb = fig.colorbar(im2, cax=_cax(fig, ax), ticks=[0, .2, .5, .8])
    cb.set_label("largest |d|, cohort vs the others")
    od = d[d["variable"].isin(co.occ)].assign(a=lambda x: x["d"].abs()).groupby("variable")["a"].max().reindex(co.occ)
    ax.set_title(f"Cohen's d per cell\n{int((W < .2).sum())}/{K * K} with every |d| < 0.2", fontsize=11)
    _occ_bars(co, graphs, axes, labels, (od.to_numpy(), [cmap(norm(v)) for v in od], "largest |d|"))
    if not d["variable"].isin(co.row_h).any():                    # the row entropies, if the table lacks them
        d = pd.concat([d, co.cohen(co.row_h)], ignore_index=True)
    hd = d[d["variable"].isin(co.row_h)].assign(a=lambda x: x["d"].abs()).groupby("variable")["a"].max().reindex(co.row_h)
    dd = d[d["variable"].isin(co.dyn)].assign(a=lambda x: x["d"].abs()).groupby("variable")["a"].max().reindex(co.dyn)
    _margins(co, axes, labels, (hd.to_numpy(), [cmap(norm(v)) for v in hd], "largest |d|"),
             ([f"|d| {v:.2f}" for v in dd], [hp.RED if v >= .5 else hp.INK2 for v in dd]))
    fig.suptitle(f"4b  Movie transitions per cohort · per cell the cohort furthest from the others, Cohen's d "
                 f"(< 0.2 small, 0.5 medium, 0.8 large) · {ht.set_label(cfg.space)}", x=.01, ha="left", fontsize=11)
    save(fig, "4b_matrix_cohen", cfg)
    plt.show()

def fig_matrix(co, graphs, agree):
    """4a. Movie p(j | i) per cohort, then a TOST per cell: ✓ = every cohort equivalent, red = k/N."""
    cfg, col, K = co.cfg, colours(co.cfg), co.cfg.K
    labels = [c for c in cfg.cohorts if (c, "movie") in graphs]
    t = agree[agree["variable"].isin(co.cells)]
    worst = t.assign(a=t[["lo", "hi"]].abs().max(axis=1)).groupby("variable")["a"].max()
    eq = t.groupby("variable")["equivalent"].all()
    n_eq = t.groupby("variable")["equivalent"].sum()
    fig, axes, labels = _matrices(co, graphs)
    ax = axes[-1]["main"]
    W = np.array([100 * worst.get(c, np.nan) for c in co.cells]).reshape(K, K)
    im2 = ax.imshow(W, cmap="Greys", vmin=0, vmax=2.5 * np.nanmax(W), aspect="auto")         # light: the marks stay readable
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
    fig.colorbar(im2, cax=_cax(fig, ax)).set_label("largest |cohort − others| (pp)")
    ax.set_title(f"TOST per cell\n{int(eq.sum())}/{len(eq)} equivalent in every cohort", fontsize=11)
    o = agree[agree["variable"].isin(co.occ)]
    ow = o.assign(a=o[["lo", "hi"]].abs().max(axis=1)).groupby("variable")["a"].max().reindex(co.occ)
    oe = o.groupby("variable")["equivalent"].all().reindex(co.occ)
    _occ_bars(co, graphs, axes, labels, (100 * ow.to_numpy(), [hp.MUTED if e else hp.RED for e in oe], "largest |diff| (pp)"))
    extra = co.agreement([*co.row_h])
    hw = extra.assign(a=extra[["lo", "hi"]].abs().max(axis=1)).groupby("variable")["a"].max().reindex(co.row_h)
    he = extra.groupby("variable")["equivalent"].all().reindex(co.row_h)
    de = agree[agree["variable"].isin(co.dyn)].groupby("variable")["equivalent"].all().reindex(co.dyn)
    _margins(co, axes, labels, (hw.to_numpy(), [hp.MUTED if e else hp.RED for e in he], "largest |diff| (bits)"),
             (["✓ equiv." if e else "✗" for e in de], [hp.INK2 if e else hp.RED for e in de]))
    fig.suptitle(f"4a  Movie transitions per cohort · ✓ = every cohort equivalent to the others (bounds ± {cfg.sesoi_d:g} SD); "
                 f"red = k/N cohorts equivalent · {ht.set_label(cfg.space)}", x=.01, ha="left", fontsize=11)
    save(fig, "4a_matrix_tost", cfg)
    plt.show()
