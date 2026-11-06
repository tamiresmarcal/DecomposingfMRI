"""Figures of 12_symptom_distributions.ipynb: symptom severity in Cam-CAN vs HCP.

The bands and the id matching are NOT here -- they are in
`tools/symptom_distribution.py`, whose `load()` this module consumes. Nothing
here computes a cutoff.

THE DESIGN DECISION THAT SHAPES EVERY FIGURE
--------------------------------------------
HADS runs 0-21 per subscale; the ASR's DSM-oriented raw scores run over a much
shorter range, and their published cutoffs exist only on the T-score. So:

  - NO figure puts the two cohorts' raw scores on a shared x axis. Two measures
    of different scale get small multiples, each with its own axis -- never one
    axis they both have to pretend to fit, and never two y-scales on one plot.
  - NO figure draws the HCP's cutoffs on a raw histogram. Those cutoffs are
    defined on T, and a rule at "T=65" has no x position on a raw axis. The
    HADS rules DO appear on its raw histograms, because HADS cutoffs are raw.
  - The band-composition figure carries the warning in its own subtitle, because
    that is the figure someone will screenshot without the notebook around it.

`fig_quantiles` is the one honest side-by-side: a Q-Q plot pairs the two scales
at matched percentiles, so it needs neither a shared axis nor an assumed maximum.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Same tokens as notebooks/hads_plots.py -- one project, one look.
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#a3a29d", "#fcfcfb"
# The two cohorts are an IDENTITY, so categorical, assigned in fixed order and
# never cycled. Validated as a pair: worst adjacent dE 24.7 (protan), 33.6
# (normal vision), both above the 8 / 15 floors, contrast >= 3:1 on this surface.
COHORT_COLOUR = {"camcan": "#2a78d6", "hcp": "#eb6834"}
COHORT_LABEL = {"camcan": "Cam-CAN (HADS)", "hcp": "HCP (ASR)"}
# Severity is ORDERED, so sequential: one hue, light -> dark. Five steps of the
# blue ramp, lightest = Zero. Not the categorical hues -- rank is not identity.
SEVERITY_RAMP = ["#dbe9fa", "#9ec6f2", "#5598e7", "#2a78d6", "#184f95"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlesize": 11,
    "axes.titlecolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 9, "legend.frameon": False})


def save(fig, name, save_dir=None):
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        for ext in ("png", "svg"):
            fig.savefig(Path(save_dir) / f"{name}.{ext}", dpi=200, bbox_inches="tight")


# ------------------------------------------------------------------ tables ---
def band_table(d, constructs=("anxiety", "depression")):
    """Rows of (cohort, construct, band, n, pct) with Zero split out of the lowest band.

    Returned as a list of dicts rather than a DataFrame so this module stays
    importable without pandas; the notebook wraps it in one.
    """
    rows = []
    for coh, blk in d.items():
        for con in constructs:
            sc = blk["scales"].get(con)
            if not sc or not sc["band"]:
                continue
            band, order = dict(sc["band"]), list(sc["band_order"])
            total = sum(band.values())
            zero_ok = sc["zero_n"] is not None and sc["summary"].get("n") == total
            if zero_ok:
                lowest = order[0]
                band = {"Zero": sc["zero_n"],
                        f"{lowest} (>0)": band.get(lowest, 0) - sc["zero_n"],
                        **{k: v for k, v in band.items() if k != lowest}}
                order = ["Zero", f"{lowest} (>0)"] + order[1:]
            for name in order:
                n = band.get(name, 0)
                rows.append({"cohort": coh, "construct": con, "band": name,
                             "n": n, "pct": 100 * n / total if total else 0.0,
                             "total": total})
    return rows


def summary_table(d, constructs=("anxiety", "depression")):
    """Scale-free comparison rows: percentiles, floor fraction, spread."""
    rows = []
    for coh, blk in d.items():
        for con in constructs:
            sc = blk["scales"].get(con)
            if not sc or not sc["summary"]:
                continue
            s = sc["summary"]
            rows.append({"cohort": coh, "construct": con, "column": sc["raw_col"],
                         "n": s["n"], "median": s["med"], "p75": s["p75"],
                         "p90": s["p90"], "p95": s["p95"], "max": s["max"],
                         "sd": round(s["sd"], 2),
                         "pct_at_floor": round(100 * s["floor_frac"], 1),
                         "scale_max": s["scale_max"],
                         "sd_over_max": (round(s["sd_over_max"], 3)
                                         if s["sd_over_max"] else None)})
    return rows


# -------------------------------------------------------------- histograms ---
def fig_histograms(d, constructs=("anxiety", "depression"), save_dir=None):
    """Small multiples of the RAW score -- rows = construct, cols = cohort, own x axis each.

    One series per panel, so no legend box: each panel's title names it. The
    HADS panels carry their cutoff rules; the ASR panels deliberately do not,
    because the ASR's cutoffs live on T and have no position on this axis.
    """
    cohorts = [c for c in ("camcan", "hcp") if c in d]
    fig, axes = plt.subplots(len(constructs), len(cohorts),
                             figsize=(4.6 * len(cohorts), 2.7 * len(constructs)),
                             squeeze=False)
    for i, con in enumerate(constructs):
        for j, coh in enumerate(cohorts):
            ax = axes[i][j]
            sc = d[coh]["scales"].get(con)
            if not sc or not sc["raw"]:
                ax.set_axis_off()
                ax.text(.5, .5, f"no {con} data", ha="center", va="center",
                        color=MUTED, transform=ax.transAxes)
                continue
            v = np.asarray(sc["raw"], float)
            lo, hi = int(v.min()), int(v.max())
            edges = np.arange(lo - .5, hi + 1.5, 1.0)
            ax.hist(v, bins=edges, color=COHORT_COLOUR[coh], lw=0, rwidth=.86)
            # Cutoff rules ONLY where the cutoffs are defined on this quantity.
            if sc["banded_on"] in ("raw", "category") and sc["cutoffs"]:
                for k, c in enumerate(sc["cutoffs"]):
                    if lo <= c <= hi + 1:
                        ax.axvline(c - .5, color=INK2, ls="--", lw=.8)
                        ax.text(c - .4, ax.get_ylim()[1] * .80,
                                sc["band_order"][k + 1], fontsize=7, color=INK2,
                                ha="left", va="top")
            n0 = sc["zero_n"]
            ax.text(.98, .96, f"n={len(v)}\nzero: {n0} ({100*n0/len(v):.0f}%)",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color=INK2, linespacing=1.4)
            ax.set_title(f"{COHORT_LABEL[coh]} · {con}", loc="left")
            ax.set_xlabel(sc["raw_col"] or con)
            if j == 0:
                ax.set_ylabel("subjects")
            ax.set_xticks(range(lo, hi + 1, max(1, (hi - lo) // 10 or 1)))
    fig.suptitle("Raw symptom scores, each on its own scale",
                 x=.01, ha="left", y=1.0, va="bottom", fontsize=10.5, color=INK)
    fig.text(.01, -.02,
             "Separate axes on purpose: HADS runs 0-21, the ASR's raw scores do not. "
             "Dashed rules are HADS cutoffs; the ASR's exist only on T, so none are drawn there.",
             fontsize=7.5, color=INK2, ha="left")
    fig.tight_layout()
    save(fig, "F1_histograms", save_dir)
    return fig


# ------------------------------------------------------------------- bands ---
def fig_bands(d, constructs=("anxiety", "depression"), save_dir=None):
    """Band composition as segmented bars: one bar per cohort x construct.

    Colour is the band's RANK WITHIN ITS OWN BAR, not a global name order. That
    matters because the two instruments have different band schemes -- HADS has
    five segments here (Zero, Normal>0, Mild, Moderate, Severe), the ASR four
    (Zero, Normal>0, Borderline, Clinical) -- so a shared name->colour legend
    would either run off the end of the ramp (an earlier draft painted
    Borderline and Clinical the same blue) or imply that "Mild" and
    "Borderline" are the same thing. They are not.

    So there is no name legend. EVERY segment is directly labelled with its own
    band name, n and %, which carries identity in text rather than in colour;
    the ramp key says only what the ramp means. Segments too narrow to hold a
    label get it above the bar on a leader.
    """
    rows = band_table(d, constructs)
    if not rows:
        raise SystemExit("band_table returned nothing -- check load()")
    bars = []
    for coh in ("camcan", "hcp"):
        for con in constructs:
            seg = [r for r in rows if r["cohort"] == coh and r["construct"] == con]
            if seg:
                bars.append((f"{COHORT_LABEL[coh]}\n{con}", seg))

    fig, ax = plt.subplots(figsize=(9.4, 1.15 * len(bars) + 1.4))
    for y, (label, seg) in enumerate(bars):
        left = 0.0
        n_seg = len(seg)
        for rank, r in enumerate(seg):
            # rank within THIS bar, spread across the ramp so the darkest step
            # is always the most severe band present, whatever the scheme
            idx = round(rank * (len(SEVERITY_RAMP) - 1) / max(1, n_seg - 1))
            col = SEVERITY_RAMP[idx]
            w = r["pct"]
            if w <= 0:
                continue
            ax.barh(y, w, left=left, height=.52, color=col, lw=1.4,
                    edgecolor=SURFACE)          # the 2px surface gap between fills
            pc = f"{w:.1f}%" if 0 < w < 1 else f"{w:.0f}%"
            txt = f"{r['band']}\n{r['n']}  ·  {pc}"
            if w >= 11:
                ax.text(left + w / 2, y, txt, ha="center", va="center",
                        fontsize=7.2, color=SURFACE if idx >= 3 else INK,
                        linespacing=1.3)
            else:
                # Too narrow to hold text. The leader goes DOWN, into the gap
                # below this bar -- never up, where an earlier draft put it
                # straight through the bar above. Two heights so neighbouring
                # slivers stagger instead of overprinting each other.
                h = .36 if rank % 2 else .58
                ax.plot([left + w / 2] * 2, [y + .26, y + h], color=MUTED, lw=.6,
                        clip_on=False, zorder=1)
                ax.text(left + w / 2, y + h + .03, txt.replace("\n", "  "),
                        ha="center", va="top", fontsize=6.8, color=INK2,
                        clip_on=False)
            left += w
        ax.text(101.5, y, f"n={seg[0]['total']}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b[0] for b in bars], fontsize=8.5, color=INK2)
    ax.invert_yaxis()
    ax.set_ylim(len(bars) - .15, -.55)
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of subjects", labelpad=6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Severity bands, by cohort and construct", loc="left", pad=14)

    # The ramp key: what the colour ORDER means, since the names differ per bar.
    kx, ky, kw = .012, -.055 - .055 * (4 - len(bars)) / 4, .022
    for i, c in enumerate(SEVERITY_RAMP):
        fig.patches.append(plt.Rectangle((kx + i * kw, ky), kw, .028, facecolor=c,
                                         edgecolor=SURFACE, lw=1.2,
                                         transform=fig.transFigure, clip_on=False))
    fig.text(kx + len(SEVERITY_RAMP) * kw + .012, ky + .014,
             "less severe  \u2192  more severe   (rank within each bar; the band names differ "
             "between the two instruments, so they are on the segments, not in a legend)",
             fontsize=7.3, color=INK2, va="center")
    fig.text(kx, ky - .075,
             "THE PERCENTAGES ARE NOT COMPARABLE BETWEEN COHORTS. HADS 8+ is a sensitive "
             "screening line; ASR T>=65 is the 93rd percentile of a reference sample by "
             "construction.\nA gap between the two is mostly a gap between two conventions. "
             "For the comparison, read F3 and the percentile table.",
             fontsize=7.5, color=INK2, ha="left", linespacing=1.5)
    fig.tight_layout()
    save(fig, "F2_bands", save_dir)
    return fig


# --------------------------------------------------------------- quantiles ---
def fig_quantiles(d, constructs=("anxiety", "depression"), save_dir=None):
    """Q-Q: the two scales paired at matched percentiles. The scale-free comparison.

    Needs no shared axis and no assumed maximum -- each axis is one cohort's own
    scale, and a point is "the score the pth percentile of each cohort reaches".
    A flat run along either axis is that cohort's floor.
    """
    if "camcan" not in d or "hcp" not in d:
        raise SystemExit("fig_quantiles needs both cohorts")
    ps = np.arange(1, 100)
    fig, axes = plt.subplots(1, len(constructs), figsize=(4.1 * len(constructs), 3.6),
                             squeeze=False)
    for k, con in enumerate(constructs):
        ax = axes[0][k]
        a = d["camcan"]["scales"].get(con)
        b = d["hcp"]["scales"].get(con)
        if not (a and b and a["raw"] and b["raw"]):
            ax.set_axis_off()
            continue
        x = np.percentile(a["raw"], ps)
        y = np.percentile(b["raw"], ps)
        # Neutral ink, NOT a cohort colour: this mark is the relationship
        # between the two cohorts, and painting it in one of their identity
        # hues would read as "this line is HCP", which it is not.
        ax.plot(x, y, color=INK2, lw=2, solid_capstyle="round")
        for p, m in ((50, "o"), (90, "s"), (95, "D")):
            i = p - 1
            ax.plot(x[i], y[i], m, ms=8, color=INK2,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            ax.annotate(f"p{p}", (x[i], y[i]), textcoords="offset points",
                        xytext=(7, -3), fontsize=7.5, color=INK2)
        ax.set_title(con, loc="left")
        ax.set_xlabel(f"Cam-CAN  {a['raw_col']}")
        ax.set_ylabel(f"HCP  {b['raw_col']}")
        fa, fb = a["summary"]["floor_frac"], b["summary"]["floor_frac"]
        ax.text(.98, .04, f"at floor:  Cam-CAN {100*fa:.0f}%   HCP {100*fb:.0f}%",
                transform=ax.transAxes, ha="right", fontsize=7.5, color=INK2)
    fig.suptitle("Q-Q: the same percentile in each cohort",
                 x=.01, ha="left", y=1.0, va="bottom", fontsize=10.5, color=INK)
    fig.text(.01, -.04,
             "One point per percentile: its x is the score Cam-CAN reaches there, its y what HCP "
             "reaches. No shared axis and no assumed maximum,\nso this is the comparison the band "
             "percentages cannot make. A flat run means that cohort's floor.",
             fontsize=7.5, color=INK2, ha="left", linespacing=1.5)
    fig.tight_layout()
    save(fig, "F3_quantiles", save_dir)
    return fig


# ---------------------------------------------- the whole HCP instrument ---
# Four families, assigned in fixed order and never cycled. Validated: worst
# adjacent dE 9.1 (protan), 22.9 (normal vision). Two of the four fall below
# 3:1 against this surface, which obligates visible labels -- every row here
# is named, and the notebook ships the table beside the figure.
FAMILY_COLOUR = {"DSM-oriented": "#2a78d6", "syndrome": "#eb6834",
                 "broad-band": "#1baf7a", "SSAGA": "#eda100"}


def hcp_scale_table(scales):
    """The rows of `symptom_distribution.hcp_scales`, minus the raw values."""
    return [{k: v for k, v in r.items() if k != "values"} for r in scales]


def fig_hcp_scales(scales, save_dir=None):
    """Every HCP scale ranked by how usable it is as a regression target.

    The quantity on the axis is the FLOOR FRACTION: the share of subjects
    sitting at the scale's minimum. It is the thing that decides power. A
    Spearman over a column where most subjects tie at the floor has far less
    power than its n suggests, and more subjects do not fix it -- which is the
    whole reason this cohort's n was raised in the first place.

    Resolution (the count of distinct values observed) is printed rather than
    plotted: it is a different measure on a different scale, and giving it a
    second x-axis on the same plot is the one chart mistake worth naming.
    Shorter bar + more distinct values = better target.

    Binary columns sit at the bottom of their family and are marked, because
    their "floor fraction" is just a base rate and a Spearman on them is a
    point-biserial. Reading them against the counts would be comparing two
    different quantities on one axis.
    """
    rows, labels, colours, heads = [], [], [], []
    y = 0
    for fam in FAMILY_COLOUR:
        got = [r for r in scales if r["family"] == fam]
        if not got:
            continue
        heads.append((y, fam))
        y += 1
        for r in got:
            rows.append((y, r))
            labels.append((y, r))
            colours.append(FAMILY_COLOUR[fam])
            y += 1
        y += .4
    if not rows:
        raise SystemExit("no HCP scales found -- check the phenotype columns")

    fig, ax = plt.subplots(figsize=(9.6, .30 * y + 1.6))
    for (yy, r), col in zip(rows, colours):
        v = r["pct_at_floor"]
        ax.plot([0, v], [yy, yy], color=col, lw=2, solid_capstyle="round",
                alpha=.55 if r["binary"] else 1.0)
        ax.plot(v, yy, "o", ms=8, color=col, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=3)
        tag = f"{v:.0f}%"
        ax.text(v + 1.4, yy, tag, va="center", fontsize=7.4, color=INK2)
        if r["binary"]:
            # Spell the positive rate out. The bar is the share at the
            # MINIMUM, so on a 0/1 flag a long bar means the condition is
            # RARE -- the opposite of what a long bar looks like it means.
            # A reader took agoraphobia's 93% for 93% prevalence; it is 7%.
            note = (f"BINARY — {r['pct_above_floor']:.0f}% positive; "
                    f"the bar is the {r['pct_at_floor']:.0f}% who scored "
                    f"{r['floor_value']:.0f}")
        else:
            note = f"{r['n_distinct']} values"
        if r["pairs_with"]:
            note += f"   ·   pairs with {r['pairs_with']}"
        ax.text(101, yy, note, va="center", fontsize=7, color=INK2)

    for yy, fam in heads:
        ax.text(-1, yy, fam, va="center", ha="right", fontsize=8.5,
                color=FAMILY_COLOUR[fam], fontweight="bold")
    ax.set_yticks([yy for yy, _ in labels])
    # The COLUMN NAME, not only the readable label: the column is what goes in
    # --target, and a reader who wants "the anxiety/depression mix" has no way
    # to get from the words "anxious/depressed" to ASR_Anxd_Raw otherwise.
    ax.set_yticklabels([f"{r['label']}   {r['column']}" for _, r in labels],
                       fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of subjects AT THE SCALE MINIMUM   (lower = more usable "
                  "as a regression target)", labelpad=6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=MUTED, alpha=.25, lw=.6)
    ax.set_axisbelow(True)
    ax.set_title("HCP: every symptom scale, by how much variance it leaves to "
                 "predict", loc="left", pad=12)
    fig.text(.01, -.03 - .004 * len(rows) / 4,
             "A LONG BAR MEANS THE SCALE IS MOSTLY EMPTY, NOT THAT THE SYMPTOM IS COMMON. On a "
             "0/1 flag it reads backwards: agoraphobia at 93% means 93% scored ZERO, so 7% have "
             "it.\nThe positive rate is written out for every binary row.   ·   Raw scores, "
             "never T: a T-score has age and sex normed out of it in the reference sample, which "
             "Cam-CAN's HADS never had.\nThe distinct-value count is printed, not plotted — it "
             "is a different measure on a different scale, and a second x-axis for it would be a "
             "dual-axis chart.",
             fontsize=7.5, color=INK2, ha="left", linespacing=1.5)
    fig.tight_layout()
    save(fig, "F4_hcp_scales", save_dir)
    return fig


def fig_hcp_grid(scales, ncol=4, save_dir=None):
    """A small histogram of every HCP scale, each on its own axis.

    Small multiples rather than one shared axis, for the same reason F1 uses
    them: these scales have different item counts and different ranges, and
    one axis they all had to fit would compress the short ones to nothing.
    """
    scales = [r for r in scales if r["values"]]
    nrow = -(-len(scales) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 1.75 * nrow),
                             squeeze=False)
    for i, r in enumerate(scales):
        ax = axes[i // ncol][i % ncol]
        v = np.asarray(r["values"], float)
        lo, hi = int(v.min()), int(v.max())
        counts, _, _ = ax.hist(v, bins=np.arange(lo - .5, hi + 1.5), lw=0,
                               color=FAMILY_COLOUR[r["family"]], rwidth=.86)
        # Headroom for the floor annotation. Without it a two-bar panel (the
        # binary SSAGA flags) fills its axes to the top and the text lands on
        # the bar.
        ax.set_ylim(0, counts.max() * 1.30)
        ax.set_title(f"{r['label']}", loc="left", fontsize=8.5)
        tag = (f"{r['pct_above_floor']:.0f}% positive" if r["binary"]
               else f"floor {r['pct_at_floor']:.0f}%")
        ax.text(.98, .93, tag, ha="right", va="top", transform=ax.transAxes,
                fontsize=7, color=INK2)
        ax.tick_params(labelsize=7)
        ax.set_xlabel(r["column"], fontsize=6.8, color=MUTED)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    for j in range(len(scales), nrow * ncol):
        axes[j // ncol][j % ncol].set_axis_off()
    handles = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLOUR.items()
               if any(r["family"] == f for r in scales)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, bbox_to_anchor=(.5, -.015))
    fig.suptitle("HCP: the shape of every scale, each on its own axis",
                 x=.01, ha="left", y=1.0, va="bottom", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, .03, 1, 1))
    save(fig, "F5_hcp_grid", save_dir)
    return fig
