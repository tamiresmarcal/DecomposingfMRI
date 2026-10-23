"""Figures of 10_hads_report.ipynb, drawn from the tables of hads_tables.

Every function takes a table and the Settings (`cfg`) and draws one figure; nothing here
computes a statistic. The state diagram's rules (sizes, label placement) are in `diagram`.
"""
import dataclasses
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Patch

INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#a3a29d", "#fcfcfb"
RED, BLUE, GREY, NODE = "#e34948", "#2a78d6", "#a3a29d", "#ecebe7"
BLOCK_COLOUR = {"covariates": "#a3a29d", "cells": "#2a78d6", "occupancy": "#eb6834", "dynamics": "#1baf7a"}
BLOCK_MARKER = {"cells": "o", "occupancy": "s", "dynamics": "^"}
TREE_COLOUR = {"bstm": "#2a78d6", "fcm": "#eb6834"}
TASK_COLOURS, TASK_MARKERS = ["#1baf7a", "#6250d6"], ["o", "D"]
# 8 validated hues, then two neutrals for slots 9-10; the marker shape is the second encoding
SET_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
               "#008300", "#6250d6", "#e34948", "#52514e", "#8c6d4f"]
SET_MARKERS = ["o", "s", "^", "D", "v"]
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",     # sequential blue, light -> dark,
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]     # for the ordered axes
ORDERED = {"window": "window_s", "K": "K"}
UP, DOWN = "#e34948", "#2a78d6"
VALUE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("value", [DOWN, "#c3c2bd", UP])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlesize": 11,
    "axes.titlecolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 9, "legend.frameon": False})


def save(fig, name, cfg):
    if cfg.save_dir:
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        for ext in ("png", "svg"):
            fig.savefig(Path(cfg.save_dir) / f"{name}.{ext}", dpi=200, bbox_inches="tight")

def rule(v, cfg, alpha=None, fc_line=None, xlim=None):
    """The pass rule at figure time: p_fdr is already BH, so a new alpha or fc_line (0 = off) only
    re-marks the points, with no recomputing. Returns (v with a new `hit`, cfg with the new values)."""
    new = {k: x for k, x in dict(alpha=alpha, fc_line=fc_line, xlim=xlim).items() if x is not None}
    if not new:
        return v, cfg
    cfg = dataclasses.replace(cfg, **new)
    return v.assign(hit=(v["p_fdr"] < cfg.alpha) & (v["x"].abs() >= (cfg.fc_line or 0))), cfg

def _panels(cfg):
    return [(t, lab) for t in cfg.tasks for lab in cfg.targets]


# ------------------------------------------------------------------------ volcanos ---
def label_points(ax, v, name, cfg, rows=None):
    """Name every passing point (or `rows`; or the label_top smallest p if none) in a column right
    of the panel, top to bottom in the points' order, with a thin line to each: names never overlap."""
    t = v[v["hit"]].nsmallest(cfg.label_max, "p") if rows is None else rows
    t = (t if len(t) else v.nsmallest(cfg.label_top, "p")).copy()
    if t.empty:
        return
    t["y"] = -np.log10(t["p"])
    lo, hi = ax.get_ylim()
    t["f"] = ((t["y"] - lo) / (hi - lo)).clip(0, 1)             # height as a fraction of the panel
    t = t.sort_values("f", ascending=False)
    gap = min(.035, .98 / len(t))
    pos, cur = [], 1.0
    for f in t["f"]:                                           # as close to the point as the one above allows
        cur = min(f, cur - gap) if pos else min(f, 1.0)
        pos.append(cur)
    floor = 0.0                                                # second pass, bottom-up: nothing below 0
    for i in range(len(pos) - 1, -1, -1):
        pos[i] = max(pos[i], floor)
        floor = pos[i] + gap
    for r, yl in zip(t.itertuples(), pos):
        ax.annotate(name(r), (r.x, r.y), xytext=(1.02, yl), textcoords="axes fraction",
                    fontsize=6.5, color=INK2, va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=.4, color=MUTED, shrinkA=0, shrinkB=2, relpos=(0, .5)))

def _cuts(ax, v, cfg):
    """p = 0.05 (dotted), the FDR cut (dashed), the effect-size cut (dashed verticals), a fixed x range."""
    ax.axhline(-np.log10(.05), color=MUTED, ls=":", lw=.8)
    fdr = v["p_fdr"] < cfg.alpha
    if fdr.any():                                              # the FDR cut itself, whatever the effect size
        ax.axhline(-np.log10(v.loc[fdr, "p"].max()), color=INK2, ls="--", lw=.8)
    ax.axvline(0, color=MUTED, lw=.6)
    for side in (-1, 1) if cfg.fc_line else ():
        ax.axvline(side * cfg.fc_line, color=INK2, ls="--", lw=.8)
    lim = cfg.xlim or 1.08 * max(v["x"].abs().max(), 1.1 * (cfg.fc_line or 0))
    ax.set_xlim(-lim, lim)
    out = int((v["x"].abs() > lim).sum())
    if out:                                                    # beyond a fixed xlim: say so, do not hide it
        ax.text(.99, .01, f"{out} beyond ±{lim:g}", transform=ax.transAxes, ha="right", fontsize=7, color=INK2)

def volcano_panel(ax, v, cfg):
    """One set's variables: colour = variable type, outlined = passes. Named: every dynamics
    variable, whatever its p, then the most significant passing ones up to label_max."""
    y = -np.log10(v["p"])
    for block, m in BLOCK_MARKER.items():
        h, c = v["block"] == block, BLOCK_COLOUR[block]
        ax.scatter(v["x"][h & ~v["hit"]], y[h & ~v["hit"]], s=16, marker=m, color=c, alpha=.35, lw=0)
        ax.scatter(v["x"][h & v["hit"]], y[h & v["hit"]], s=40, marker=m, color=c, edgecolor=INK, lw=.6)
    _cuts(ax, v, cfg)
    dyn = v[v["block"] == "dynamics"]
    named = pd.concat([dyn, v[v["hit"] & (v["block"] != "dynamics")].nsmallest(max(cfg.label_max - len(dyn), 0), "p")])
    label_points(ax, v, lambda t: t.variable, cfg, rows=named)
    n = f"n={int(v['n_group'].iloc[0])} vs {int(v['n_normal'].iloc[0])}" if "n_group" in v else f"n={int(v['n'].median())}"
    ax.set_title(f"{v['task'].iloc[0]} · {v['phenotype'].iloc[0]}   ({int(v['hit'].sum())} / {len(v)} {cfg.pass_text}, {n})",
                 fontsize=10)

def fig_volcanos(v, cfg, title, name, xlabel, alpha=None, fc_line=None, xlim=None):
    """One panel per (task, phenotype) of v: rows = tasks, columns = phenotypes.
    alpha / fc_line / xlim change the rule or the x range for this figure only."""
    v, cfg = rule(v, cfg, alpha, fc_line, xlim)
    rows, cols = list(dict.fromkeys(v["task"])), list(dict.fromkeys(v["phenotype"]))
    fig, axes = plt.subplots(len(rows), len(cols), squeeze=False, figsize=(8.5 * len(cols), 5.6 * len(rows) + .6))
    for (i, t), (j, lab) in ((a, b) for a in enumerate(rows) for b in enumerate(cols)):
        volcano_panel(axes[i, j], v[(v["task"] == t) & (v["phenotype"] == lab)], cfg)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$-\log_{10}$ p")
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    fig.legend(handles=[Line2D([], [], ls="none", marker=m, ms=6, color=BLOCK_COLOUR[b], label=b)
                        for b, m in BLOCK_MARKER.items()] +
                       [Line2D([], [], ls="none", marker="o", ms=7, mfc=INK2, mec=INK, label=f"outlined = {cfg.pass_text}")],
               loc="lower center", ncol=4, fontsize=7.5)
    fig.suptitle(title, x=.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .97), w_pad=8)
    save(fig, name, cfg)
    plt.show()

def palette(v, by):
    """{level: colour}: sequential for window and K (they have an order), categorical otherwise."""
    if by == "block":
        return {b: c for b, c in BLOCK_COLOUR.items() if b in set(v["block"])}
    if by in ORDERED:
        lev = v.groupby(by)[ORDERED[by]].first().sort_values().index.tolist()
        idx = np.linspace(0, len(RAMP) - 1, len(lev)).round().astype(int)
        return dict(zip(lev, [RAMP[i] for i in idx]))
    lev = v[by].value_counts().index.tolist()
    return {l: SET_COLOURS[k % len(SET_COLOURS)] for k, l in enumerate(lev)}

def multiverse_panel(ax, v, by, colours, cfg):
    v = v.sample(frac=1, random_state=0)                       # no group always drawn on top
    y, sig, c = -np.log10(v["p"]), v["hit"], v[by].map(colours)
    ax.scatter(v["x"][~sig], y[~sig], s=4, c=c[~sig], alpha=.25, lw=0, rasterized=True)
    ax.scatter(v["x"][sig], y[sig], s=12, c=c[sig], alpha=.95, lw=0, rasterized=True)
    _cuts(ax, v, cfg)
    label_points(ax, v, lambda t: f"{t.variable} · {t.set}", cfg)
    ax.set_title(f"{v['task'].iloc[0]} · {v['phenotype'].iloc[0]}   ({int(sig.sum()):,} / {len(v):,} {cfg.pass_text})",
                 fontsize=10)

def _multiverse_legend(fig, colours, cfg, h, ncol=8):
    """Under the title, right-aligned (h = figure height in inches)."""
    fig.legend(handles=[Line2D([], [], ls="none", marker="o", ms=6, color=c, label=str(l)) for l, c in colours.items()] +
                       [Line2D([], [], ls="none", label=f"solid = {cfg.pass_text} (FDR over the panel)")],
               loc="upper right", ncol=min(len(colours) + 1, ncol), fontsize=7, bbox_to_anchor=(1, 1 - .3 / h))

def fig_multiverse(multi, cfg, title, name, xlabel, by="block", alpha=None, fc_line=None, xlim=None):
    """One panel per (task, phenotype) in multi, coloured by `by`: "block" (cells, occupancy,
    dynamics), "window", "atlas", "method" or "K"; a list draws one figure per colouring.
    alpha / fc_line / xlim change the rule or the x range for this figure only."""
    multi, cfg = rule(multi, cfg, alpha, fc_line, xlim)
    rows, cols = list(dict.fromkeys(multi["task"])), list(dict.fromkeys(multi["phenotype"]))
    for by in [by] if isinstance(by, str) else by:
        colours = palette(multi, by)
        h = 5.2 * len(rows) + .9
        fig, axes = plt.subplots(len(rows), len(cols), squeeze=False, figsize=(9.5 * len(cols), h))
        for (i, t), (j, lab) in ((a, b) for a in enumerate(rows) for b in enumerate(cols)):
            multiverse_panel(axes[i, j], multi[(multi["task"] == t) & (multi["phenotype"] == lab)], by, colours, cfg)
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$-\log_{10}$ p")
        for ax in axes[-1]:
            ax.set_xlabel(xlabel)
        _multiverse_legend(fig, colours, cfg, h)
        fig.suptitle(f"{title} · colour = {by}", x=.01, y=1 - .08 / h, va="top", ha="left", fontsize=10, color=INK)
        fig.tight_layout(h_pad=2.5, rect=(0, 0, 1, 1 - .75 / h))
        save(fig, f"{name}_{by}", cfg)
        plt.show()


# --------------------------------------------------------------------- performance ---
def draw_band(ax, mean, sd, x=None):
    """One model's Age + Sex baseline: dashed mean, ± sd band (full width, or over x = (lo, hi))."""
    lo, hi = x or ax.get_xlim()
    ax.fill_between([lo, hi], mean - sd, mean + sd, color=MUTED, alpha=.18, lw=0)
    ax.plot([lo, hi], [mean] * 2, color=INK2, ls="--", lw=.9)
    ax.axhline(0, color=MUTED, lw=.6)

def fig_rankings(rows, cfg, above_both):
    """B1: every row of the four rankings, one block of panels per model over its own baseline.
    above_both(d) -> the rows above their baseline at every task (Study.above_both)."""
    tasks, models = list(cfg.tasks), list(dict.fromkeys(sorted(rows["model"])))
    runs = [(t, tree) for t in tasks for tree in ("bstm", "fcm")]
    origin = dict(zip(tasks, ["o", "D", "s", "^"]))            # which task a winner came from
    both = SET_COLOURS[2:]                                     # blue and orange belong to the winners

    def winners(d, tree):
        t = d[d["tree"] == tree]
        return {task: g.loc[g["mean"].idxmax()] for task, g in t.groupby("task", sort=False)}

    def plot_runs(ax, d):
        rng = np.random.default_rng(0)
        keep = above_both(d)
        shown = keep[:cfg.highlight_max]
        style = {e: (both[k % len(both)], SET_MARKERS[k // len(both) % len(SET_MARKERS)]) for k, e in enumerate(shown)}
        offset = dict(zip(shown, np.linspace(-.3, .3, len(shown)) if len(shown) > 1 else [0]))
        win = {tree: winners(d, tree) for tree in ("bstm", "fcm")}
        ax.set_xlim(-.6, len(runs) - .4)
        b = d.iloc[0]
        draw_band(ax, b["base_mean"], b["base_sd"])
        for i, (task, tree) in enumerate(runs):
            g, c = d[(d["task"] == task) & (d["tree"] == tree)], TREE_COLOUR[tree]
            ax.scatter(i + rng.uniform(-.14, .14, len(g)), g["mean"], s=10, color=c, alpha=.2, lw=0)
            for r in g[g["entity"].isin(style)].itertuples():  # same x offset in both tasks
                cc, m = style[r.entity]
                ax.errorbar(i + offset[r.entity], r.mean, yerr=r.std, fmt=m, ms=6, lw=1.2,
                            color=cc, mec=SURFACE, mew=.8, zorder=3)
            for o, w in win[tree].items():                     # filled where it won, hollow carried over
                r = g[g["entity"] == w["entity"]]
                if r.empty:
                    continue
                own = o == task
                ax.errorbar(i - .38 if own else i + .38, r["mean"].iloc[0], yerr=r["std"].iloc[0],
                            fmt=origin[o], ms=8, lw=1.5, color=c, zorder=4,
                            mfc=c if own else SURFACE, mec=SURFACE if own else c, mew=1.5)
        ax.set_xticks(range(len(runs)), [f"{tree}\n{task}" for task, tree in runs])
        for k in range(1, len(tasks)):
            ax.axvline(2 * k - .5, color=MUTED, lw=.6)
        handles = [Line2D([], [], ls="none", marker=origin[o], ms=7, color=TREE_COLOUR[tree],
                          label=f"{tree} best at {o}:  {w['label']} · {w['arm']}")
                   for tree in ("bstm", "fcm") for o, w in win[tree].items()]
        handles += [Line2D([], [], ls="none", marker=style[e][1], ms=6, color=style[e][0], label=e) for e in shown]
        if len(keep) > len(shown):
            handles.append(Line2D([], [], ls="none", label=f"… and {len(keep) - len(shown)} more (table below)"))
        handles.append(Patch(color=MUTED, alpha=.35, label=f"Age + Sex baseline:  {b['base_mean']:.3f} ± "
                                                           f"{b['base_sd']:.3f}  (n={int(b['base_n'])})"))
        return handles, (f"large = best at each task: filled where it won, hollow carried over\n"
                         f"coloured = {len(keep)} row(s) above the baseline at {' AND '.join(tasks)} (best {len(shown)} shown)")

    n_lines = cfg.highlight_max + 2 * len(tasks) + 3
    fig, axes = plt.subplots(2 * len(models), len(cfg.targets), squeeze=False,
                             figsize=(6.5 * len(cfg.targets), len(models) * (4 + .16 * n_lines)),
                             gridspec_kw={"height_ratios": [4, .16 * n_lines] * len(models)})
    for k, model in enumerate(models):
        for j, lab in enumerate(cfg.targets):
            ax, leg = axes[2 * k, j], axes[2 * k + 1, j]
            if (k, j) != (0, 0):
                ax.sharey(axes[0, 0])
            handles, title = plot_runs(ax, rows[(rows["phenotype"] == lab) & (rows["model"] == model)])
            ax.set_title(f"{model} · {lab}")
            leg.axis("off")
            leg.legend(handles=handles, loc="upper center", fontsize=7, title=title, title_fontsize=7)
        axes[2 * k, 0].set_ylabel("CV score (Spearman)")
    fig.suptitle("B1  every row · dashed = that model's Age + Sex baseline, band ± sd · coloured = above it at "
                 f"{' and '.join(tasks)}, ± sd over seeds", fontsize=8, color=INK2, y=.995)
    fig.tight_layout()
    save(fig, "B1_rankings", cfg)
    plt.show()

def fig_factors(rows, cfg):
    """B2: one figure per axis of the grid (cfg.factors), rest and movie side by side per group,
    one row of panels per model over its own baseline (the model axis: one panel, a band each)."""
    tasks = list(cfg.tasks)
    colour, marker = dict(zip(tasks, TASK_COLOURS)), dict(zip(tasks, TASK_MARKERS))
    numeric = {"window": "window_s", "K_": "K"}                # ordered by size; the rest by median score
    pool = rows[rows["tree"].isin(cfg.factor_trees)]
    models = list(dict.fromkeys(sorted(pool["model"])))
    base = pool.groupby(["phenotype", "model"])[["base_mean", "base_sd"]].first()

    def levels(d, col):
        if col in numeric:
            return d.groupby(col)[numeric[col]].first().sort_values().index.tolist()
        return d.groupby(col)["mean"].median().sort_values(ascending=False).index.tolist()

    def compare(ax, d, col, lab, ms, lev):
        d = d.dropna(subset=[col])
        rng = np.random.default_rng(0)
        ax.set_xlim(-.6, len(lev) - .4)
        if col == "model":                                     # each model over its own band
            for i, v in enumerate(lev):
                draw_band(ax, *base.loc[(lab, v)], (i - .45, i + .45))
        else:
            draw_band(ax, *base.loc[(lab, ms[0])])
        for j, task in enumerate(tasks):
            dx, c = (j - (len(tasks) - 1) / 2) * .3, colour[task]
            for i, v in enumerate(lev):
                s = d.loc[(d[col] == v) & (d["task"] == task), "mean"]
                if s.empty:
                    continue
                ax.scatter(i + dx + rng.uniform(-.06, .06, len(s)), s, s=8, color=c, alpha=.25, lw=0)
                q1, q2, q3 = np.percentile(s, [25, 50, 75])
                ax.plot([i + dx] * 2, [q1, q3], color=c, lw=3, solid_capstyle="round", zorder=3)
                ax.plot(i + dx, q2, marker[task], ms=7, color=c, mec=SURFACE, mew=1, zorder=4)
        long = max(map(len, map(str, lev))) > 8
        ax.set_xticks(range(len(lev)), lev, rotation=30 if long else 0, ha="right" if long else "center")

    for k, (col, name) in enumerate(cfg.factors.items(), start=1):
        blocks = [None] if col == "model" else models          # None = every model in one panel
        fig, axes = plt.subplots(len(blocks), len(cfg.targets), squeeze=False, sharey=True,
                                 figsize=(6.5 * len(cfg.targets), 3.6 * len(blocks) + .4))
        lev = levels(pool.dropna(subset=[col]), col)
        for r, model in enumerate(blocks):
            for ax, lab in zip(axes[r], cfg.targets):
                d = pool[(pool["phenotype"] == lab) & ((pool["model"] == model) if model else True)]
                compare(ax, d, col, lab, [model] if model else models, lev)
                ax.set_title(f"{model} · {lab}" if model else lab)
            axes[r, 0].set_ylabel("CV score (Spearman)")
        handles = [Line2D([], [], ls="none", marker=marker[t], ms=7, color=colour[t],
                          label=f"{t}: median, bar = middle half") for t in tasks]
        fig.legend(handles=handles + [Patch(color=MUTED, alpha=.35, label="that model's Age + Sex baseline ± sd")],
                   loc="upper right", ncol=3, fontsize=7, bbox_to_anchor=(1, 1.01))
        fig.suptitle(f"B2.{k}  {name}  ·  {', '.join(cfg.factor_trees)}", x=.01, ha="left", fontsize=10, color=INK, y=1.0)
        fig.tight_layout()
        save(fig, f"B2_{k}_{col}", cfg)
        plt.show()

def fig_performance(perf, cfg, title, legend, model):
    """E1: one model, one panel per phenotype; x = bstm | fcm, colour = task, a line per task
    from bstm to fcm; mean ± sd over seeds, over that model's Age + Sex baseline.
    legend = the label of each tree (what was selected)."""
    tasks, trees = list(cfg.tasks), ["bstm", "fcm"]
    colour, marker = dict(zip(tasks, TASK_COLOURS)), dict(zip(tasks, TASK_MARKERS))
    fig, axes = plt.subplots(1, len(cfg.targets), squeeze=False, sharey=True, figsize=(4.5 * len(cfg.targets), 4.9))
    for ax, lab in zip(axes[0], cfg.targets):
        p = perf[perf["phenotype"] == lab]
        ax.set_xlim(-.4, 1.4)
        if len(p):
            draw_band(ax, p["base_mean"].iloc[0], p["base_sd"].iloc[0])
        ends = []                                              # (y, x, task): the line's name at its end
        for k, task in enumerate(tasks):
            h = p[p["task"] == task].set_index("tree").reindex(trees)
            x = np.arange(len(trees)) + (k - (len(tasks) - 1) / 2) * .06
            ax.plot(x, h["mean"], color=colour[task], lw=1.4, alpha=.7)
            ax.errorbar(x, h["mean"], yerr=h["std"], fmt=marker[task], ls="none", ms=8, lw=1.6, color=colour[task],
                        mec=SURFACE, mew=1, zorder=3)
            last = h["mean"].last_valid_index()
            if last is not None:
                ends.append([h.loc[last, "mean"], x[trees.index(last)] + .1, task])
        ax.autoscale_view()
        gap = .05 * np.diff(ax.get_ylim())[0]                  # names never closer than this
        ends.sort()
        for e0, e1 in zip(ends, ends[1:]):
            e1[0] = max(e1[0], e0[0] + gap)
        for y, x, task in ends:
            ax.text(x, y, task, color=colour[task], va="center", fontsize=8)
        ax.set_xticks(range(len(trees)), ["bstm\n(transitions)", "fcm\n(static FC)"])
        ax.set_title(lab)
    axes[0, 0].set_ylabel("CV score (Spearman), mean ± sd over seeds")
    fig.legend(handles=[Line2D([], [], ls="none", marker=marker[t], ms=7, color=colour[t], label=t) for t in tasks] +
                       [Line2D([], [], ls="none", label=f"{tree} = {legend[tree]}") for tree in trees] +
                       [Patch(color=MUTED, alpha=.35, label=f"{model} on Age + Sex only (the baseline) ± sd")],
               loc="lower center", ncol=1, fontsize=7.5)
    fig.suptitle(title, x=.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, .2, 1, .95))
    save(fig, "E1_performance", cfg)
    plt.show()


# ---------------------------------------------------------------------- SHAP -----
def fig_shares(contrib, cfg, title):
    """C: one bar per model, grouped by model type, best CV first; segments = share of its mean |SHAP|."""
    order = ["covariates", "cells", "occupancy", "dynamics"]

    def plot(ax, c):
        share = c.pivot_table(index="label", columns="block", values="share", aggfunc="sum").reindex(columns=order).fillna(0)
        first = c.groupby("label")[["model", "cv"]].first().sort_values(["model", "cv"], ascending=[True, False])
        share = share.loc[first.index[::-1]]                   # barh draws bottom-up
        left = np.zeros(len(share))
        for b in order:
            ax.barh(range(len(share)), share[b], left=left, height=.75, color=BLOCK_COLOUR[b], edgecolor=SURFACE, lw=1)
            left += share[b].to_numpy()
        cv = c.groupby("label")["cv"].first()
        ax.set_yticks(range(len(share)), [f"{l}   ρ={cv[l]:.2f}" for l in share.index], fontsize=6.5)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        bottom_up = first["model"].to_numpy()[::-1]
        for k in np.nonzero(bottom_up[1:] != bottom_up[:-1])[0]:          # between two models
            ax.axhline(k + .5, color=INK, lw=.8)
        for m in dict.fromkeys(bottom_up):                                 # the model's name, once
            ax.text(1.01, np.nonzero(bottom_up == m)[0].mean(), m, transform=ax.get_yaxis_transform(),
                    rotation=-90, va="center", fontsize=8, color=INK2)

    n = contrib.groupby(["task", "phenotype"])["label"].nunique().max()
    h = len(cfg.tasks) * (.9 + .19 * n)
    fig, axes = plt.subplots(len(cfg.tasks), len(cfg.targets), squeeze=False, figsize=(7.5 * len(cfg.targets), h))
    for (i, task), (j, lab) in ((a, b) for a in enumerate(cfg.tasks) for b in enumerate(cfg.targets)):
        plot(axes[i, j], contrib[(contrib["task"] == task) & (contrib["phenotype"] == lab)])
        axes[i, j].set_title(f"{task} · {lab}", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("share of the model's mean |SHAP| (out-of-fold)")
    fig.legend(handles=[Patch(color=BLOCK_COLOUR[b], label=b) for b in order], loc="upper right",
               ncol=4, fontsize=8, bbox_to_anchor=(1, 1.0))
    fig.suptitle(title, x=.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - .3 / h))
    save(fig, "C_shares", cfg)
    plt.show()

def fig_predicted(deep, cfg, title):
    """E2.1: observed HADS category vs out-of-fold prediction; bar = middle half, marker = median."""
    fig, axes = plt.subplots(len(cfg.tasks), len(cfg.targets), squeeze=False, sharey=True,
                             figsize=(5.5 * len(cfg.targets), 3.4 * len(cfg.tasks)))
    rng = np.random.default_rng(0)
    for (i, task), (j, lab) in ((a, b) for a in enumerate(cfg.tasks) for b in enumerate(cfg.targets)):
        ax, d = axes[i, j], deep[task, lab]
        levels = [k for k in range(len(cfg.ordinal_levels)) if (d["y"] == k).any()]
        for x, k in enumerate(levels):
            p = d["pred"][d["y"] == k]
            ax.scatter(x + rng.uniform(-.18, .18, len(p)), p, s=7, color=INK2, alpha=.3, lw=0)
            q1, q2, q3 = np.percentile(p, [25, 50, 75])
            ax.plot([x, x], [q1, q3], color=INK, lw=3, solid_capstyle="round")
            ax.plot(x, q2, "o", ms=7, color=INK, mec=SURFACE, mew=1)
        ax.set_xticks(range(len(levels)), [f"{cfg.ordinal_levels[k]}\n(n={int((d['y'] == k).sum())})" for k in levels])
        same = "" if np.isnan(d["rho_selection"]) else ("  ✓ = select-bstm" if abs(d["rho"] - d["rho_selection"]) < 1e-6
                                                         else f"  (select-bstm: {d['rho_selection']:.2f})")
        ax.set_title(f"{task} · {lab}   (out-of-fold ρ = {d['rho']:.2f}{same})", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("predicted HADS level")
    fig.suptitle(title, x=.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .96))
    save(fig, "E2_1_predicted", cfg)
    plt.show()

def fig_stability(deep, cfg, title):
    """E2.2: each panel's top_vars variables (average over seeds): one dot per seed, bar = average.
    Right: the direction (↑ higher values push the prediction up, ↓ down, ↕ the seeds disagree)
    and how many seeds rank it in their own top_vars."""
    fig, axes = plt.subplots(len(cfg.tasks), len(cfg.targets), squeeze=False,
                             figsize=(6.8 * len(cfg.targets), (1 + .32 * cfg.top_vars) * len(cfg.tasks)))
    for (i, task), (j, lab) in ((a, b) for a in enumerate(cfg.tasks) for b in enumerate(cfg.targets)):
        ax, d = axes[i, j], deep[task, lab]
        M, sign = d["by_seed"], np.sign(d["sign_by_seed"])
        top = np.argsort(M.mean(0))[::-1][:cfg.top_vars][::-1]
        in_top = (np.argsort(np.argsort(-M, axis=1), axis=1) < cfg.top_vars).sum(0)
        for row, v in enumerate(top):
            ax.barh(row, M[:, v].mean(), height=.6, color=BLOCK_COLOUR[d["blocks"][v]], alpha=.45, lw=0)
            ax.scatter(M[:, v], np.full(len(M), row), s=12, color=INK, zorder=3, lw=0)
            arrow = "↑" if (sign[:, v] > 0).all() else "↓" if (sign[:, v] < 0).all() else "↕"
            ax.text(1.01, row, f"{arrow} {in_top[v]}/{len(M)}", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=7, color=INK if in_top[v] == len(M) else MUTED)
        ax.set_yticks(range(len(top)), [d["names"][v] for v in top], fontsize=7.5)
        ax.set_title(f"{task} · {lab}", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("mean |SHAP| (out-of-fold), one dot per fold seed")
    fig.legend(handles=[Patch(color=BLOCK_COLOUR[b], alpha=.45, label=b) for b in BLOCK_COLOUR] +
                       [Line2D([], [], ls="none", label=f"↑/↓ = direction in every seed (↕ = they disagree); "
                                                        f"k/{len(cfg.seeds_stab)} = seeds ranking it top {cfg.top_vars}")],
               loc="upper right", ncol=5, fontsize=7.5, bbox_to_anchor=(1, 1.0))
    fig.suptitle(title, x=.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .96))
    save(fig, "E2_2_stability", cfg)
    plt.show()


# ------------------------------------------------------------------ the state graph ---
def pct(v):
    """A share as %, never rounded to a misleading 0: a rarely visited state still has transitions."""
    return "0%" if v == 0 else "<0.1%" if v < .001 else f"{100 * v:.1f}%" if v < .01 else f"{100 * v:.0f}%"

def away(direction):
    """ha / va that make a label grow AWAY from its anchor along `direction` (never back over it)."""
    x, y = direction / np.linalg.norm(direction)
    return ("left" if x > .35 else "right" if x < -.35 else "center",
            "bottom" if y > .35 else "top" if y < -.35 else "center")

def box(xy, text, fs=1.0):
    """A label's box in data units, (x, y, half width, half height), for ~6.5 pt text (x fs) on a ~4.6 in panel."""
    return (xy[0], xy[1], (.5 * len(text) * .042 + .02) * fs, .05 * fs)

def clear_of(b, centres, radii, boxes):
    """True if label box b touches no state's circle and no label already placed."""
    x, y, hw, hh = b
    dx = np.maximum(np.abs(centres[:, 0] - x) - hw, 0)
    dy = np.maximum(np.abs(centres[:, 1] - y) - hh, 0)
    return (bool((dx ** 2 + dy ** 2 > (radii + .015) ** 2).all())
            and all(abs(x - bx) > hw + bw or abs(y - by) > hh + bh for bx, by, bw, bh in boxes))

def diagram(ax, T, occ, edges, wmax, omax, compare=False, hit_T=None, hit_occ=None, note_T=None, note_occ=None,
            colour=RED, fs=1.0):
    """A state diagram on a circle. Circle area = time in state, loop = p(stay), arrow width =
    p(j | i). With `compare`, everything not highlighted is grey. hit_T / hit_occ are drawn in
    `colour`, with note_T / note_occ (pp) written beside them; an empty note is not written."""
    K = len(occ)
    hit_T = np.zeros((K, K), bool) if hit_T is None else hit_T
    hit_occ = np.zeros(K, bool) if hit_occ is None else hit_occ
    ang = np.pi / 2 - 2 * np.pi * np.arange(K) / K
    pos = np.column_stack([np.cos(ang), np.sin(ang)])
    radius = lambda o: .05 + .17 * np.sqrt(o / omax)          # area ~ time in state
    base = GREY if compare else INK2
    radii = np.array([radius(o) for o in occ])
    nodes = [ax.add_patch(Circle(pos[k], radius(occ[k]), zorder=4, lw=1.2 if hit_occ[k] else .7,
                                 fc=matplotlib.colors.to_rgba(colour, .3) if hit_occ[k] else NODE,
                                 ec=colour if hit_occ[k] else INK2)) for k in range(K)]
    placed = []                                               # label boxes, so none lands on another
    for k in range(K):                                        # loops and the states' labels first:
        u, r = pos[k] / np.linalg.norm(pos[k]), radius(occ[k])   # their places are fixed
        th = np.degrees(np.arctan2(u[1], u[0]))
        ax.add_patch(Arc(pos[k] + u * (r + .045), .11, .11, theta1=th - 140, theta2=th + 140, zorder=2,
                         lw=.3 + 3 * T[k, k], color=colour if hit_T[k, k] else base))
        occ_note = note_occ[k] if hit_occ[k] and note_occ is not None and note_occ[k] else None
        ax.text(*pos[k], str(k), ha="center", va="center", fontsize=7.5 * fs, color=INK, zorder=6)
        at, lab = pos[k] * (1 + (r + .36) / np.linalg.norm(pos[k])), pct(occ[k]) + (f" ({occ_note})" if occ_note else "")
        ax.text(*at, lab, ha="center", va="center", fontsize=6.5 * fs, color=colour if occ_note else INK2, zorder=6)
        placed.append(box(at, lab, fs))                           # time in state, beside it
        if hit_T[k, k] and note_T is not None and note_T[k, k]:   # a loop's change: from its outer
            side = np.array([u[1], -u[0]])                    # corner, sideways, a little over it
            d = (u + side) / np.linalg.norm(u + side)
            ha, va = away(d)
            corner = pos[k] + u * (r + .045) + d * .055
            ax.annotate(note_T[k, k], corner, xytext=-1.5 * d, textcoords="offset points", ha=ha, va=va,
                        fontsize=6.2 * fs, color=colour, zorder=7,
                        bbox=dict(boxstyle="round,pad=.12", fc=SURFACE, ec="none", alpha=.8))
            _, _, hw, hh = box(corner, note_T[k, k], fs)
            placed.append(box(corner + np.array([{"left": hw, "right": -hw}.get(ha, 0),
                                                 {"bottom": hh, "top": -hh}.get(va, 0)]), note_T[k, k], fs))
    for i, j in edges:                                        # transitions
        hit = hit_T[i, j]
        ax.add_patch(FancyArrowPatch(pos[i], pos[j], connectionstyle="arc3,rad=.18", arrowstyle="-|>",
                                     patchA=nodes[i], patchB=nodes[j], shrinkA=2, shrinkB=3,
                                     mutation_scale=6 + 6 * T[i, j] / wmax,   # colour never changes the width
                                     lw=.3 + 4.5 * T[i, j] / wmax,
                                     color=colour if hit else base, alpha=1 if hit else .7, zorder=3 if hit else 2))
        if hit and note_T is not None and note_T[i, j]:       # beside the middle of the curve
            # centred just outside the curve's middle (the curve itself bulges .09 out): beside the
            # line on a long arrow, over it only where a short one leaves no room. If that spot is
            # on a circle or another label, slide away from the curve, outward first, to the nearest
            # free one.
            (x1, y1), (x2, y2) = pos[i], pos[j]
            perp = np.array([y2 - y1, x1 - x2])
            curve, n = (pos[i] + pos[j]) / 2 + .09 * perp, perp / np.linalg.norm(perp)
            candidates = [(pos[i] + pos[j]) / 2 + .17 * perp] + [
                curve + side * t * n for t in (.08, .12, .16, .2, .25, .3, .36, .42, .5) for side in (1, -1)]
            mid = next((c for c in candidates if clear_of(box(c, note_T[i, j], fs), pos, radii, placed)), candidates[0])
            ax.text(*mid, note_T[i, j], ha="center", va="center", fontsize=6.5 * fs, color=colour, zorder=5,
                    bbox=dict(boxstyle="round,pad=.15", fc=SURFACE, ec="none", alpha=.85))
            placed.append(box(mid, note_T[i, j], fs))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.75, 1.75)                                  # room for the top / bottom labels
    ax.set_aspect("equal")
    ax.axis("off")

# the dynamics beside the diagrams: symbol, name, format
DYN_SHOW = {"switch_rate": ("⇄", "switch rate", "{:.2f}"), "mean_dwell_s": ("τ", "mean dwell", "{:.0f} s"),
            "entropy_rate_bits": ("H", "entropy rate", "{:.2f} bits"), "dispersion": ("◌", "dispersion", "{:.2f}"),
            "n_distinct_transitions": ("#", "distinct transitions", "{:.0f}"),
            "self_transition_rate": ("↺", "self-transition rate", "{:.2f}"),
            "switches_per_min": ("⇄/min", "switches per min", "{:.1f}"),
            "n_states_visited": ("K", "states visited", "{:.1f}")}

def dynamics_column(ax, dyn, side, colour, fs=1.0, icon=1.0):
    """The panel's dynamics as a column of symbol + value, outside the circle on `side` (-1 left,
    +1 right). A passing one is coloured, with its change underneath."""
    x, ys = 1.95 * side, np.linspace(1.35, -1.25, len(dyn))         # the full height of the panel
    for y, r in zip(ys, dyn.itertuples()):
        sym, _, f = DYN_SHOW[r.variable]
        c = colour if r.hit else INK2
        ax.text(x, y + .07 * icon * fs, sym, ha="center", va="center", fontsize=10.5 * fs * icon, color=c)
        ax.text(x, y - .08 * fs - .02 * icon, f.format(r.value), ha="center", va="center", fontsize=6.3 * fs, color=c)
        if r.hit:
            ax.text(x, y - .2 * fs - .02 * icon, f"{100 * r.change:+.0f}%", ha="center", va="center", fontsize=6.3 * fs, color=c)

def fig_graph(pooled, marks, cfg, refs, colours, title, meaning, name, dyn=None, fs=1.0, icon=1.0):
    """Rows = Normal, then each phenotype; columns = tasks. pooled[(task, group)] = (p(j | i), occ, n).
    refs[(task, group)] = the panel it is compared with (its items in marks are highlighted in
    colours[(task, group)]); a panel without a ref is drawn as it is. dyn (Study.dynamics_marks) adds
    each panel's dynamics as a column of symbols: left of the first task, right of the others. Every panel draws the Normal
    group's edge_top strongest arrows of every task, plus its own highlighted ones, on ONE width scale."""
    tasks, groups = list(cfg.tasks), ["Normal", *cfg.targets]
    K = len(pooled[tasks[0], "Normal"][1])
    off = [(a, b) for a in range(K) for b in range(K) if a != b]
    base = set().union(*(sorted(off, key=lambda e: -pooled[t, "Normal"][0][e])[:cfg.edge_top] for t in tasks))
    panels = {}
    for t in tasks:
        for g in groups:
            hT, hO = np.zeros((K, K), bool), np.zeros(K, bool)
            nT, nO = np.full((K, K), "", dtype=object), [""] * K
            if (t, g) in refs:
                for r in marks[(marks["task"] == t) & (marks["group"] == g) & marks["hit"]].itertuples():
                    if r.block == "cells":
                        a, b = map(int, r.variable.split("->"))
                        hT[a, b], nT[a, b] = True, r.note
                    else:
                        k = int(r.variable.split("_")[1])
                        hO[k], nO[k] = True, r.note
            panels[t, g] = (hT, hO, nT, nO, sorted(base | {e for e in off if hT[e]}))
    wmax = max(pooled[k][0][e] for k, p in panels.items() for e in p[4])
    omax = max(pooled[k][1].max() for k in panels)
    w = 4.6 + (.9 if dyn is not None else 0)
    fig, axes = plt.subplots(len(groups), len(tasks), squeeze=False, figsize=(w * len(tasks), 4.7 * len(groups) + .6))
    for (i, g), (j, t) in ((a, b) for a in enumerate(groups) for b in enumerate(tasks)):
        T, occ, n = pooled[t, g]
        hT, hO, nT, nO, edges = panels[t, g]
        who = "Normal on both scales" if g == "Normal" else f"{g}, {cfg.case_from}+"
        ref = refs.get((t, g))
        if dyn is not None:
            side = -1 if j == 0 else 1
            dynamics_column(axes[i, j], dyn[(dyn["task"] == t) & (dyn["group"] == g)], side,
                            colours.get((t, g), INK2) if ref is not None else INK2, fs, icon)
        if ref is None:
            diagram(axes[i, j], T, occ, edges, wmax, omax, fs=fs)
            axes[i, j].set_title(f"{t} · {who} (n={n})", fontsize=9.5 * fs)
            if dyn is not None:
                axes[i, j].set_xlim(*((-2.25, 1.6) if j == 0 else (-1.6, 2.25)))
            continue
        # against another group, the rest goes grey; against the same group in the other task, it
        # stays dark: the Normal diagram is still itself, only its changes are marked
        diagram(axes[i, j], T, occ, edges, wmax, omax, compare=ref[1] != g, hit_T=hT, hit_occ=hO, note_T=nT,
                note_occ=nO, colour=colours[t, g], fs=fs)
        vs = ref[0] if ref[1] == g else ref[1]
        name_of = {RED: "red", BLUE: "blue"}.get(colours[t, g], "highlighted")
        axes[i, j].set_title(f"{t} · {who} (n={n})\nvs {vs}  ·  {int(hT.sum() + hO.sum())} in {name_of}", fontsize=9.5 * fs)
        if dyn is not None:
            axes[i, j].set_xlim(*((-2.25, 1.6) if j == 0 else (-1.6, 2.25)))
    fig.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=10, mfc=NODE, mec=INK2, label="circle area = time in state (%)"),
        Line2D([], [], color=INK2, lw=2.5, label="arrow width = p(j | i): share of moves out of i that go to j"),
        Line2D([], [], ls="none", marker=r"$\circlearrowleft$", ms=12, color=INK2, label="loop = p(stay): thicker = held longer"),
        *[Line2D([], [], color=c, lw=2.5, label=m) for c, m in meaning],
        *([Line2D([], [], ls="none", label="beside: " + "   ".join(f"{DYN_SHOW[v][0]} {DYN_SHOW[v][1]}"
                                                               for v in dict.fromkeys(dyn["variable"])))]
          if dyn is not None else [])],
        loc="lower center", ncol=2, fontsize=7.5 * fs, bbox_to_anchor=(.5, 0))
    fig.suptitle(title, x=.01, ha="left", fontsize=10 * fs)
    fig.tight_layout(rect=(0, .07, 1, .97))
    save(fig, name, cfg)
    plt.show()
