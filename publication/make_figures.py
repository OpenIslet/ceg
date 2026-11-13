#!/usr/bin/env python3
"""Vector figures for the CEG V3.2 publication PDF.

Run from this directory:
    PYTHONDONTWRITEBYTECODE=1 python3 -B make_figures.py

Figures 1 and 2 are schematics transcribed node-for-node from the two Mermaid
blocks of CEG-JOURNAL-MANUSCRIPT-V3.2.md (sections 3.1 and 4.1).  Figure 3 is
plotted from ../benchmark_results.json and ../real_text_results.json only; no
count is typed into this file.  Each figure is written to ./figures/ as PDF
(embedded in the article), SVG, and a PNG preview.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "figures"

BASE, MID, SMALL = 8.0, 7.0, 6.0
INK = "#1a1a1a"
GREY = "#6b6b6b"
BLUE = "#0072B2"
AMBER = "#E69F00"
BLUE_FILL = "#dbeaf5"
AMBER_FILL = "#fbeccb"
GREY_FILL = "#e9e9e9"
MONO = "DejaVu Sans Mono"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": BASE,
        "axes.titlesize": BASE,
        "axes.labelsize": BASE,
        "xtick.labelsize": SMALL,
        "ytick.labelsize": SMALL,
        "legend.fontsize": MID,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "svg.fonttype": "path",
        "svg.hashsalt": "ceg-publication",
        "axes.linewidth": 0.6,
        "figure.dpi": 200,
    }
)


def check_text_inside(fig, stem: str) -> None:
    """Fail if any text falls outside the figure or two texts overlap."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [(t.get_text(), t.get_window_extent(r)) for t in fig.findobj(matplotlib.text.Text)
             if t.get_text().strip() and t.get_visible()]
    fb = fig.bbox
    outside = [s for s, b in boxes if b.x0 < fb.x0 - 0.5 or b.x1 > fb.x1 + 0.5 or b.y0 < fb.y0 - 0.5 or b.y1 > fb.y1 + 0.5]
    overlap = [(a, c) for i, (a, ba) in enumerate(boxes) for c, bc in boxes[i + 1:]
               if ba.overlaps(bc) and min(ba.x1, bc.x1) - max(ba.x0, bc.x0) > 1 and min(ba.y1, bc.y1) - max(ba.y0, bc.y0) > 1]
    if outside or overlap:
        raise SystemExit(f"{stem}: text outside figure {outside!r}; overlapping text {overlap!r}")


def save(fig, stem: str) -> None:
    OUT.mkdir(exist_ok=True)
    check_text_inside(fig, stem)
    fig.savefig(OUT / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / f"{stem}.svg", metadata={"Date": None})
    fig.savefig(OUT / f"{stem}.png", dpi=300)  # raster preview for the visual audit only
    plt.close(fig)


def blank_axes(fig, w: float, h: float):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return ax


def box(ax, cx, cy, w, h, face, edge, ls="-", lw=0.9):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.05",
        facecolor=face, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=3,
    )
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, style="-|>", color=INK, lw=0.9, ls="-", z=2):
    a = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=7, color=color,
        linewidth=lw, linestyle=ls, shrinkA=0, shrinkB=0, zorder=z,
    )
    ax.add_patch(a)
    return a


def path_arrow(ax, pts):
    """Polyline with an arrow head on the last segment."""
    xs, ys = zip(*pts[:-1])
    ax.plot(xs, ys, color=INK, lw=0.9, zorder=2, solid_joinstyle="round", solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1])


# --------------------------------------------------------------------------
# Figure 1 - Hasse diagram of the model_system axis (section 3.1)
# --------------------------------------------------------------------------
def figure1() -> None:
    W, H = 6.5, 3.75
    fig = plt.figure(figsize=(W, H))
    ax = blank_axes(fig, W, H)

    styles = {
        "DIRECT_MATCH": (BLUE_FILL, BLUE),
        "BOUNDED_BRIDGE": (AMBER_FILL, AMBER),
        "INADMISSIBLE_TRANSFER": (GREY_FILL, GREY),
    }
    # element -> (x, y, descriptor from the Mermaid block, transfer class toward
    # clinical_t1d_islet_graft on this axis as returned by
    # ContextLatticeSpec.evaluate_transfer in ceg_reference.py v3.2; the classes are
    # checked against the live lattice by t1d/check_t1d_vignette.py --check)
    nodes = {
        "clinical_t1d_islet_graft": (2.67, 3.20, "deployment target", "DIRECT_MATCH"),
        "humanized_mouse_allograft": (1.44, 2.26, "human immune cells in vivo", "BOUNDED_BRIDGE"),
        "human_sc_or_donor_islet_invitro": (3.90, 2.26, "human islet target cell", "BOUNDED_BRIDGE"),
        "primary_islet_or_sc_floor": (2.67, 1.32, "bridge floor", "BOUNDED_BRIDGE"),
        "immortalized_beta_line": (2.67, 0.38, "below floor", "INADMISSIBLE_TRANSFER"),
        "adult_population_t2d_genetics": (5.32, 0.95, "no common upper bound", "INADMISSIBLE_TRANSFER"),
    }
    bw, bh = 2.14, 0.50
    edges = [
        ("immortalized_beta_line", "primary_islet_or_sc_floor"),
        ("primary_islet_or_sc_floor", "humanized_mouse_allograft"),
        ("primary_islet_or_sc_floor", "human_sc_or_donor_islet_invitro"),
        ("humanized_mouse_allograft", "clinical_t1d_islet_graft"),
        ("human_sc_or_donor_islet_invitro", "clinical_t1d_islet_graft"),
    ]
    for lo, hi in edges:
        x0, y0 = nodes[lo][0], nodes[lo][1] + bh / 2
        x1, y1 = nodes[hi][0], nodes[hi][1] - bh / 2
        ax.plot([x0, x1], [y0, y1], color=INK, lw=0.9, zorder=1, solid_capstyle="round")

    for name, (x, y, desc, cls) in nodes.items():
        face, edge = styles[cls]
        box(ax, x, y, bw, bh, face, edge)
        ax.text(x, y + 0.085, name, ha="center", va="center", family="DejaVu Sans Mono",
                fontsize=SMALL, color=INK, zorder=4)
        ax.text(x, y - 0.115, desc, ha="center", va="center", fontsize=SMALL, color=INK, zorder=4)

    ax.text(2.67, 2.26, r"$\parallel$", ha="center", va="center", fontsize=BASE + 2, color=INK)
    ax.text(2.67, 1.86, "incomparable", ha="center", va="center", fontsize=SMALL, color=GREY)

    yb = 0.85
    ax.plot([1.48, 3.86], [yb, yb], color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=0)
    ax.text(1.42, yb, "floor\nboundary", ha="right", va="center", fontsize=SMALL, color=GREY,
            linespacing=1.2)

    arrow(ax, (0.08, 0.38), (0.08, 3.20), color=GREY, lw=0.8)
    ax.text(0.21, 1.79, r"$x \preceq y$: $y$ covers $x$", ha="center", va="center",
            rotation=90, fontsize=SMALL, color=GREY)

    kx, ky = 4.48, 3.55
    ax.text(kx, ky, "Transfer class toward\nclinical_t1d_islet_graft", ha="left", va="top",
            fontsize=MID, color=INK, linespacing=1.25)
    for i, cls in enumerate(["DIRECT_MATCH", "BOUNDED_BRIDGE", "INADMISSIBLE_TRANSFER"]):
        face, edge = styles[cls]
        yy = ky - 0.50 - 0.21 * i
        box(ax, kx + 0.11, yy, 0.22, 0.13, face, edge, lw=0.8)
        ax.text(kx + 0.30, yy, cls, ha="left", va="center", family="DejaVu Sans Mono",
                fontsize=SMALL, color=INK)
    save(fig, "fig1_context_poset_hasse")


# --------------------------------------------------------------------------
# Figure 2 - two-regime workflow and the commit verifier (section 4.1).
# Every box and arrow below corresponds to one node or edge of the Mermaid
# block; the node identifiers are given in the comments.
# --------------------------------------------------------------------------
def figure2() -> None:
    W, H = 6.5, 4.78
    fig = plt.figure(figsize=(W, H))
    ax = blank_axes(fig, W, H)

    def frame(x0, y0, x1, y1, face, edge, lw, ls="-"):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=face, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=0))

    def node(cx, cy, w, h, text, mono=False):
        box(ax, cx, cy, w, h, "white", INK)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=SMALL, zorder=4, linespacing=1.22,
                family=MONO if mono else "DejaVu Sans")

    xc = (3.91, 4.91, 5.91)
    bw = 0.86
    y1, h1 = 3.84, 0.52
    y2, h2 = 2.78, 0.88
    y3, h3 = 1.62, 0.88

    # ---- subgraph R1 --------------------------------------------------------
    frame(0.08, 0.12, 2.46, 2.74, "#f4f4f4", GREY, 0.9, ls=(0, (4, 2.5)))
    ax.text(0.18, 2.58, "Regime 1: in-context exploration", ha="left", va="center",
            fontsize=MID, fontweight="bold", color=INK)
    ax.text(0.18, 2.43, "& hypothesis discovery", ha="left", va="center",
            fontsize=MID, fontweight="bold", color=INK)
    ax.text(0.18, 2.20, "(primary mode; unconstrained;\ndesign intent, not evaluated)", ha="left", va="center",
            fontsize=SMALL, color=INK, linespacing=1.2)
    hyp = (1.27, y3)
    ctx = (1.27, 0.52)
    node(*hyp, 2.10, 0.50, "Free-form exploration and\nfluid hypothesis loop")                      # HYP
    node(*ctx, 2.10, 0.60, "In-context multi-study graph:\nEvidenceAtom, ContextSignature,\n"
                           "$T^{+}$ and $T^{-}$")                                                   # CTX
    arrow(ax, (1.07, ctx[1] + 0.30), (1.07, hyp[1] - 0.25))                                        # CTX --> HYP
    arrow(ax, (1.47, hyp[1] - 0.25), (1.47, ctx[1] + 0.30))                                        # HYP --> CTX

    # ---- CAND ---------------------------------------------------------------
    cand = (1.27, (y1 + y2) / 2)
    node(*cand, 2.20, 0.60, "SynthesisClaimRecord +\nTypedProposition +\nReversiblePatchEvent", mono=True)
    arrow(ax, (2.20, hyp[1] + 0.25), (2.20, cand[1] - 0.30))                                       # HYP --> CAND
    ax.text(2.13, (2.74 + cand[1] - 0.30) / 2, "candidate\ncommits", ha="right", va="center", fontsize=SMALL,
            linespacing=1.15)

    # ---- subgraph R2 --------------------------------------------------------
    frame(3.40, 0.12, 6.42, 4.66, "white", INK, 1.0)
    ax.text(3.50, 4.49, "Regime 2: deterministic commit gate", ha="left", va="center",
            fontsize=MID, fontweight="bold", color=INK)
    ax.text(3.50, 4.33, "(guardrail; two-layer $I_1$ to $I_6$ verifier)", ha="left", va="center",
            fontsize=SMALL, color=INK)
    node(xc[0], y1, bw, h1, "Raw corpus $D_0$\n(as extracted)")                                    # RAW
    node(xc[1], y1, bw, h1, "Settled corpus\n$D_m$ (after\ncorrections)")                          # CUR
    node(xc[2], y1, bw, h1, "Typed atoms &\nposet $\\mathcal{G}$ (findings\n& context map)")       # G
    arrow(ax, (xc[0] + bw / 2, y1), (xc[1] - bw / 2, y1))                                          # RAW --> CUR
    arrow(ax, (xc[1] + bw / 2, y1), (xc[2] - bw / 2, y1))                                          # CUR --> G
    ax.text((xc[0] + xc[1]) / 2, y1 - h1 / 2 - 0.05, "$I_2$ forward/\ninverse replay", ha="center", va="top",
            fontsize=SMALL, linespacing=1.15)
    ax.text((xc[1] + xc[2]) / 2, y1 - h1 / 2 - 0.05, "$I_1$ bidirectional\nbinding", ha="center", va="top",
            fontsize=SMALL, linespacing=1.15)
    node(xc[0], y2, bw, h2, "$I_3^{\\mathrm{tok}}$:\nUnicode screen +\nSI units\n(numbers, units\n& scripts)")  # TOK
    node(xc[1], y2, bw, h2, "$I_3^{\\mathrm{clause}}$:\nper-clause scope,\nbounds & arms\n(lint)")       # CLS
    node(xc[2], y2, bw, h2, "$I_3^{\\mathrm{typed}}$:\n7-coordinate\nhomomorphism\n(exact 7-field\nmatch)")  # TYP
    arrow(ax, (xc[0] + bw / 2, y2), (xc[1] - bw / 2, y2))                                          # TOK --> CLS
    arrow(ax, (xc[1] + bw / 2, y2), (xc[2] - bw / 2, y2))                                          # CLS --> TYP
    arrow(ax, (xc[2], y1 - h1 / 2), (xc[2], y2 + h2 / 2))                                          # G --> TYP
    node(xc[2], y3, bw, h3, "$I_4$ poset\ntransfer &\nfalsifier gate\n(context-gap\ncheck)")       # POS
    node(xc[1], y3, bw, h3, "$I_5$ digitization\ngate & terminal\ndecision (primary-\nsource check)")  # PRM
    arrow(ax, (xc[2], y2 - h2 / 2), (xc[2], y3 + h3 / 2))                                          # TYP --> POS
    arrow(ax, (xc[2] - bw / 2, y3), (xc[1] + bw / 2, y3))                                          # POS --> PRM
    lock = (4.41, 0.52)
    lw_, lh = 1.86, 0.54
    node(*lock, lw_, lh, "Layer 2: step, attestation &\nsection digest locks\n"
                         "($I_6$ non-prose lock, ledger digest)")                                   # LOCK
    arrow(ax, (xc[1], y3 - h3 / 2), (xc[1], lock[1] + lh / 2))                                     # PRM --> LOCK

    # ---- edges between the regimes -----------------------------------------
    xr = cand[0] + 1.10
    path_arrow(ax, [(xr, cand[1] + 0.10), (2.62, cand[1] + 0.10), (2.62, y1), (xc[0] - bw / 2, y1)])  # CAND --> RAW
    ax.text(2.68, y1 + 0.06, "curation edits", ha="left", va="bottom", fontsize=SMALL)
    path_arrow(ax, [(xr, cand[1] - 0.10), (2.62, cand[1] - 0.10), (2.62, y2), (xc[0] - bw / 2, y2)])  # CAND --> TOK
    ax.text(2.68, y2 + 0.06, "$I_6$ anchors &\ndisjointness", ha="left", va="bottom", fontsize=SMALL,
            linespacing=1.15)
    arrow(ax, (xc[1] - bw / 2, y3), (hyp[0] + 1.05, y3))                                           # PRM --> HYP
    ax.text(xc[0], y3 + 0.07, "rejected:\nviolation codes\nreturned to\ncontext", ha="center", va="bottom",
            fontsize=SMALL, linespacing=1.15)
    arrow(ax, (lock[0] - lw_ / 2, lock[1]), (ctx[0] + 1.05, lock[1]))                              # LOCK --> CTX
    ax.text(2.93, lock[1] + 0.07, "accepted:\npersisted to the\nshared graph", ha="center", va="bottom",
            fontsize=SMALL, linespacing=1.15)
    save(fig, "fig2_two_regime_workflow")


# --------------------------------------------------------------------------
# Figure 3 - empirical summary (section 6.2), plotted from the JSON outputs
# --------------------------------------------------------------------------
REGIMES = [
    ("V0_citation_presence", "V0 citation presence"),
    ("V1_study_pool_numeric", "V1 study-wide bag of numbers"),
    ("V2_prose_pooled_permitted", "V2 lint, pooled permitted"),
    ("V2s_prose_pooled_rejected", "V2s lint, pooled rejected"),
    ("V3_full_ceg", "V3 full CEG"),
]
SUITE_LABEL = {"core": "Core", "stress": "Stress", "residual": "Residual",
               "redteam": "Red-team", "gauntlet": "Gauntlet"}
FAMILY_LABEL = {
    "B_P1": "B-P1\ndirect\npromotion",
    "B_P2": "B-P2\nbounded\nbridge",
    "B_P3": "B-P3\nsafety\nscoping",
    "B_P4": "B-P4\ncompar.\nsyntax",
    "D1": "D1\ncross-trial\npooling",
    "D2": "D2\narm\nswap",
    "D3": "D3\nsubgroup\nas overall",
    "D4": "D4\nunit or\nrounding",
    "D5": "D5\npreclinical\nas clinical",
    "D6": "D6\ntheft or\nnegation",
}
CODE_COLOURS = ["#E69F00", "#D55E00", "#CC79A7", "#009E73", "#F0E442"]
REFUSED_GREY = "#b9b9b9"
CMAP = LinearSegmentedColormap.from_list("ceg_blues", ["#eef2f7", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])


def heat(fig, x0, ytop, cw, rh, cells, col_labels, row_labels):
    """Count heat map. cells[r][c] = (numerator, denominator), or None where a
    suite has no cases of that kind. Units: inches."""
    W, H = fig.get_size_inches()
    nr, nc = len(cells), len(cells[0])
    ax = fig.add_axes([x0 / W, (ytop - nr * rh) / H, nc * cw / W, nr * rh / H])
    for r in range(nr):
        for c in range(nc):
            v = cells[r][c]
            if v is None:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor="white", edgecolor="#c8c8c8",
                                           linewidth=0.5, hatch="////"))
                ax.text(c + 0.5, r + 0.5, "\u2013", ha="center", va="center", fontsize=SMALL, color=GREY,
                        bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
                continue
            num, den = v
            frac = num / den
            ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=CMAP(frac), edgecolor="white", linewidth=1.2))
            ax.text(c + 0.5, r + 0.5, f"{num}/{den}", ha="center", va="center", fontsize=SMALL,
                    color="white" if frac > 0.55 else INK)
    ax.set_xlim(0, nc)
    ax.set_ylim(nr, 0)
    ax.set_xticks([c + 0.5 for c in range(nc)])
    ax.set_xticklabels(col_labels, fontsize=SMALL, linespacing=1.1)
    ax.xaxis.tick_top()
    ax.set_yticks([r + 0.5 for r in range(nr)])
    ax.set_yticklabels(row_labels, fontsize=SMALL)
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def panel_title(fig, x, y, letter, text):
    W, H = fig.get_size_inches()
    fig.text(x / W, y / H, letter, ha="left", va="baseline", fontsize=BASE + 1, fontweight="bold", color=INK)
    fig.text((x + 0.19) / W, y / H, text, ha="left", va="baseline", fontsize=BASE, color=INK)


def figure3() -> None:
    bench = json.loads((ROOT / "benchmark_results.json").read_text(encoding="utf-8"))
    real = json.loads((ROOT / "real_text_results.json").read_text(encoding="utf-8"))
    summ = bench["overall_summary"]
    ta = real["track_a_verbatim_prose"]
    tb = real["track_b_grounded_synthesis"]
    rows = [label for _, label in REGIMES]
    none = [""] * len(rows)
    suites = list(SUITE_LABEL)
    if set(suites) != set(bench["suites"]):
        raise SystemExit("suite list differs from benchmark_results.json")

    # panels a, b: the counts of Table 2
    a_cells = [[(summ[s][k]["adversarial_rejected"], summ[s][k]["adversarial_total"]) for s in suites]
               for k, _ in REGIMES]
    b_cells = [[(summ[s][k]["valid_accepted"], summ[s][k]["valid_total"]) if bench["suites"][s]["valid_cases"] else None
                for s in suites] for k, _ in REGIMES]
    suite_labels = [SUITE_LABEL[s] for s in suites]

    # panel c: the counts of Table 9b
    def fam_key(name):
        parts = name.split("_")
        return "_".join(parts[:2]) if parts[0] == "B" else parts[0]

    valid_f = [(f, d) for f, d in tb["by_family"].items() if not d["is_adversarial"]]
    adv_f = [(f, d) for f, d in tb["by_family"].items() if d["is_adversarial"]]
    cv_cells = [[(d["by_verifier"][k]["accepted"], d["n"]) for _, d in valid_f] for k, _ in REGIMES]
    ca_cells = [[(d["by_verifier"][k]["rejected"], d["n"]) for _, d in adv_f] for k, _ in REGIMES]

    # panel d: Track A source-anchored verdicts; V3 refusals by primary code
    anchored = ta["source_anchored_summary"]
    codes = dict(ta["v3_primary_outcome_codes"])
    v3_accepted = codes.pop("accepted")
    code_order = sorted(codes, key=lambda c: (-codes[c], c))
    n_sent = ta["total_verbatim_sentences"]
    if v3_accepted != anchored["V3_full_ceg"]["accepted"] or sum(codes.values()) != anchored["V3_full_ceg"]["refused"]:
        raise SystemExit("Track A outcome codes do not reconcile with the V3 source-anchored summary")
    if len(code_order) > len(CODE_COLOURS):
        raise SystemExit("more Track A outcome codes than colours")

    W, H = 6.5, 6.12
    fig = plt.figure(figsize=(W, H))
    cw, rh = 0.46, 0.27
    xa = 1.46
    xb = xa + 5 * cw + 0.18
    top1, top2 = 5.62, 3.20
    heat(fig, xa, top1, cw, rh, a_cells, suite_labels, rows)
    heat(fig, xb, top1, cw, rh, b_cells, suite_labels, none)
    panel_title(fig, xa, top1 + 0.34, "a", "Adversarial cases rejected")
    panel_title(fig, xb, top1 + 0.34, "b", "Valid cases accepted")
    fig.text(xa / W, (top1 - 5 * rh - 0.17) / H,
             "Cell shading in a\u2013c: share of cases; darker = higher = better. Hatched: suite has no valid cases.",
             ha="left", va="baseline", fontsize=SMALL, color=GREY)

    xc2 = xa + len(valid_f) * cw + 0.14
    heat(fig, xa, top2, cw, rh, cv_cells, [FAMILY_LABEL[fam_key(f)] for f, _ in valid_f], rows)
    heat(fig, xc2, top2, cw, rh, ca_cells, [FAMILY_LABEL[fam_key(f)] for f, _ in adv_f], none)
    fig.text((xa + len(valid_f) * cw / 2) / W, (top2 + 0.44) / H, "valid records accepted", ha="center",
             va="baseline", fontsize=SMALL, color=INK, fontstyle="italic")
    fig.text((xc2 + len(adv_f) * cw / 2) / W, (top2 + 0.44) / H, "adversarial records rejected", ha="center",
             va="baseline", fontsize=SMALL, color=INK, fontstyle="italic")
    panel_title(fig, xa, top2 + 0.64, "c", "Real-text Track B records, by family")

    # panel d
    bar_w, bar_top, bar_rh = 2.60, 1.36, 0.20
    axd = fig.add_axes([xa / W, (bar_top - 5 * bar_rh) / H, bar_w / W, 5 * bar_rh / H])
    for i, (k, _) in enumerate(REGIMES):
        acc, ref = anchored[k]["accepted"], anchored[k]["refused"]
        if acc + ref != n_sent:
            raise SystemExit(f"Track A {k}: accepted + refused != total sentences")
        axd.barh(i, acc, color=BLUE, height=0.66, zorder=2)
        if acc:
            axd.text(acc / 2, i, str(acc), ha="center", va="center", fontsize=SMALL, color="white", zorder=3)
        if k == "V3_full_ceg":
            left = acc
            for c, colour in zip(code_order, CODE_COLOURS):
                axd.barh(i, codes[c], left=left, color=colour, height=0.66, zorder=2, edgecolor="white", linewidth=0.4)
                left += codes[c]
        elif ref:
            axd.barh(i, ref, left=acc, color=REFUSED_GREY, height=0.66, zorder=2)
            axd.text(acc + ref / 2, i, str(ref), ha="center", va="center", fontsize=SMALL, color=INK, zorder=3)
    axd.set_ylim(4.5, -0.5)
    axd.set_xlim(0, n_sent)
    axd.set_yticks(range(5))
    axd.set_yticklabels(rows, fontsize=SMALL)
    axd.set_xticks([0, n_sent])
    axd.set_xlabel("Source-anchored verbatim sentences", fontsize=MID, labelpad=1)
    axd.tick_params(axis="y", length=0, pad=2)
    axd.tick_params(axis="x", length=2.5, width=0.6)
    for side in ("top", "right", "left"):
        axd.spines[side].set_visible(False)
    panel_title(fig, xa, bar_top + 0.14, "d", "Real-text Track A verdicts")

    # key for panel d
    kx, ky, dy = xa + bar_w + 0.20, bar_top - 0.08, 0.148
    entries = [(BLUE, "accepted", False), (REFUSED_GREY, "refused", False)]
    for j, (colour, label, _) in enumerate(entries):
        fig.patches.append(plt.Rectangle(((kx) / W, (ky - j * dy - 0.04) / H), 0.11 / W, 0.08 / H,
                                         facecolor=colour, edgecolor="none", transform=fig.transFigure))
        fig.text((kx + 0.17) / W, (ky - j * dy) / H, label, ha="left", va="center", fontsize=SMALL, color=INK)
    y0 = ky - len(entries) * dy - 0.03
    fig.text(kx / W, y0 / H, "V3 refusals by primary violation code", ha="left", va="center",
             fontsize=SMALL, color=INK, fontstyle="italic")
    for j, (c, colour) in enumerate(zip(code_order, CODE_COLOURS)):
        yy = y0 - (j + 1) * dy
        fig.patches.append(plt.Rectangle((kx / W, (yy - 0.04) / H), 0.11 / W, 0.08 / H,
                                         facecolor=colour, edgecolor="none", transform=fig.transFigure))
        fig.text((kx + 0.17) / W, yy / H, f"{codes[c]}", ha="left", va="center", fontsize=SMALL, color=INK)
        fig.text((kx + 0.36) / W, yy / H, c, ha="left", va="center", fontsize=SMALL - 0.5, color=INK, family=MONO)
    save(fig, "fig3_empirical_summary")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    print("figures written:", ", ".join(sorted(p.name for p in OUT.iterdir())))
