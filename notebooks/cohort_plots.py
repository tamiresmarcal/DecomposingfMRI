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
DYN_UNIT = {"mean_dwell_s": " (s)", "entropy_rate_bits": " (bits)"}


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
        who = f"Mean of the {len(rows) - 1} cohorts" if c == "mean" else f"{c} · {co.role(c).capitalize()}"
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
            size = "" if c == "mean" else f" (n={int(n.loc[c, kind])})"
            ax.set_title(f"{who} · {'Rest' if kind == 'rest' else 'Movie vs Rest'}{size}", fontsize=11)
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


# ------------------------------------------------------------------ 2 equivalence, TOST
def _tost_rows(ax, t, y0, scale, col, offsets):
    for r in t.itertuples():
        y = y0 + offsets[r.cohort]
        ax.plot([scale * r.lo, scale * r.hi], [y, y], color=col[r.cohort], lw=1.4)
        ax.plot(scale * r.diff, y, "o", ms=5, color=col[r.cohort], mfc=col[r.cohort] if r.equivalent else "white", mew=1.4)

def fig_agreement(co, agree):
    """Movie: each cohort minus all the others (90% CI) for the strongest transitions, every occupancy
    and every dynamics; grey = the TOST bounds; filled = equivalent; ✓ = every cohort equivalent."""
    cfg, col = co.cfg, colours(co.cfg)
    labels = list(agree["cohort"].unique())
    offsets = dict(zip(labels, np.linspace(.3, -.3, len(labels)) if len(labels) > 1 else [0]))
    fig = plt.figure(figsize=(17, 8))
    outer = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=.4)
    panels = [(fig.add_subplot(outer[0]), f"Transitions p(j | i) · the {cfg.top_transitions} strongest", co.strongest()),
              (fig.add_subplot(outer[1]), "Occupancy (time in state)", co.occ)]
    for ax, title, vs in panels:
        for y, v in enumerate(vs[::-1]):
            t = agree[agree["variable"] == v]
            b = 100 * t["bound"].iloc[0]
            ax.plot([-b, b], [y, y], color=hp.MUTED, lw=10, alpha=.35, solid_capstyle="butt")
            _tost_rows(ax, t, y, 100, col, offsets)
            ok = t["equivalent"].all()
            ax.text(1.02, y, "✓" if ok else "✗", transform=ax.get_yaxis_transform(), va="center", fontsize=11,
                    color=hp.INK if ok else hp.RED)
        ax.axvline(0, color=hp.INK2, lw=.8)
        ax.set_yticks(range(len(vs)), vs[::-1], fontsize=9)
        ax.set_xlabel("cohort − all the others (pp), 90% CI", fontsize=9)
        ax.set_title(title, fontsize=11)
    dyn_axes = [fig.add_subplot(x) for x in outer[2].subgridspec(len(co.dyn), 1, hspace=1.2)]
    for ax, v in zip(dyn_axes, co.dyn):
        t = agree[agree["variable"] == v]
        b = t["bound"].iloc[0]
        ax.axvspan(-b, b, color=hp.MUTED, alpha=.35, lw=0)
        _tost_rows(ax, t, 0, 1, col, offsets)
        ax.axvline(0, color=hp.INK2, lw=.8)
        ax.set_ylim(-.5, .5); ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_title(v + DYN_UNIT.get(v, "") + "  ·  cohort − others", fontsize=9, loc="left", pad=3)
        ok = t["equivalent"].all()
        ax.text(1.03, 0, "✓" if ok else "✗", transform=ax.transAxes, va="center", fontsize=11, color=hp.INK if ok else hp.RED)
        ax.tick_params(labelsize=8)
    if dyn_axes:
        dyn_axes[0].text(.5, 2.3, "Dynamics (each on its own axis)", transform=dyn_axes[0].transAxes, ha="center", fontsize=11)
    fig.legend(handles=[Line2D([], [], ls="none", marker="o", color=col[c], label=f"{c} ({co.role(c)})") for c in labels] +
                       [Line2D([], [], ls="none", marker="o", mfc="white", mec=hp.INK2, label="hollow = not shown equivalent"),
                        Line2D([], [], color=hp.MUTED, lw=8, alpha=.5, label=f"equivalence bounds ± {cfg.sesoi_d:g} pooled SD"),
                        Line2D([], [], ls="none", label="✓ = every cohort equivalent (TOST, 5%)")],
               loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(f"2  Movie: is every cohort equivalent to the others? TOST: equivalent when the 90% CI lies inside "
                 f"the bounds · {ht.set_label(cfg.space)}", x=.01, ha="left", fontsize=11)
    fig.subplots_adjust(left=.05, right=.95, top=.87, bottom=.13)
    save(fig, "2_agreement_tost", cfg)
    plt.show()


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
                for v in vs:
                    ax.annotate(v, (means.loc[train, v].astype(float).mean(), y[v]), xytext=(8, -10),
                                textcoords="offset points", fontsize=8, color=hp.INK2)
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
def fig_matrix(co, graphs, agree):
    """Movie p(j | i) per cohort, then a TOST per cell: ✓ = every cohort equivalent, red = k/N."""
    cfg, col, K = co.cfg, colours(co.cfg), co.cfg.K
    labels = [c for c in cfg.cohorts if (c, "movie") in graphs]
    t = agree[agree["variable"].isin(co.cells)]
    worst = t.assign(a=t[["lo", "hi"]].abs().max(axis=1)).groupby("variable")["a"].max()
    eq = t.groupby("variable")["equivalent"].all()
    n_eq = t.groupby("variable")["equivalent"].sum()
    fig, axes = plt.subplots(1, len(labels) + 1, figsize=(4.3 * (len(labels) + 1), 5))
    vmax = max(graphs[c, "movie"][0].max() for c in labels)
    for ax, c in zip(axes, labels):
        im = ax.imshow(graphs[c, "movie"][0], cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(f"{c} ({co.role(c)})", fontsize=11, color=col[c])
    fig.colorbar(im, ax=axes[:-1], shrink=.8, pad=.01).set_label("p(j | i)")
    ax = axes[-1]
    W = np.array([100 * worst.get(c, np.nan) for c in co.cells]).reshape(K, K)
    im2 = ax.imshow(W, cmap="Greys", vmin=0, vmax=2.5 * np.nanmax(W))         # light: the marks stay readable
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
    fig.colorbar(im2, ax=ax, shrink=.8, pad=.04).set_label("largest |cohort − others| (pp)")
    ax.set_title(f"TOST per cell\n{int(eq.sum())}/{len(eq)} equivalent in every cohort", fontsize=11)
    for a in axes:
        a.set_xticks(range(K)); a.set_yticks(range(K)); a.set_xlabel("to state j"); a.set_ylabel("from state i")
    fig.suptitle(f"4  Movie transitions per cohort · ✓ = every cohort equivalent to the others (bounds ± {cfg.sesoi_d:g} SD); "
                 f"red = k/N cohorts equivalent · {ht.set_label(cfg.space)}", x=.01, ha="left", fontsize=11)
    save(fig, "4_matrix_tost", cfg)
    plt.show()
