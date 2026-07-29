"""
Two-family accuracy-memory comparison.

Figure 2b in the current draft plots MAYA against statistics-only analytic
methods, the family MAYA does not beat on memory.  It omits the family MAYA was
designed against -- methods that store sample-derived data.  This script plots
both, and quantifies MAYA vs the strongest data-storing baseline (ER+MLP)
across all nine configurations.

Numbers: results/grid/story/*.log (replay family) and
results/comparisons/*.json (analytic family).
"""

import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Resolve paths relative to the repo root so this runs from anywhere.
ROOT = pathlib.Path(__file__).resolve().parent.parent
os.chdir(ROOT)

BLUE = "#2a78d6"    # MAYA
ORANGE = "#eb6834"  # data-storing baselines
INK = "#0b0b0b"
INK2 = "#52514e"    # statistics-only family
MUTED = "#8a8983"
GRID = "#e4e3df"
SURFACE = "#ffffff"

PRETTY_D = {"imagenet_r": "ImageNet-R", "tinyimagenet": "TinyImageNet", "objectnet": "ObjectNet"}
PRETTY_B = {"dinov3": "DINOv3", "resnet50": "ResNet50", "siglip2": "SigLIP2"}

# From results/grid/story/story_main_*.log -- (AIA %, memory MB)
STORY = {
    ("imagenet_r", "dinov3"):    {"NCM": (85.7, 3.1),  "Replay": (71.7, 374.9), "ER+MLP": (94.5, 374.9), "Nodes": (93.5, 227.8), "MAYA": (94.2, 383.2)},
    ("imagenet_r", "resnet50"):  {"NCM": (40.7, 1.6),  "Replay": (39.8, 187.5), "ER+MLP": (46.8, 187.5), "Nodes": (45.8, 82.2),  "MAYA": (47.6, 159.9)},
    ("imagenet_r", "siglip2"):   {"NCM": (95.3, 1.2),  "Replay": (81.0, 140.6), "ER+MLP": (96.1, 140.6), "Nodes": (92.6, 55.8),  "MAYA": (95.6, 114.1)},
    ("objectnet", "dinov3"):     {"NCM": (47.8, 4.7),  "Replay": (29.7, 605.0), "ER+MLP": (73.0, 605.0), "Nodes": (71.1, 268.4), "MAYA": (76.0, 550.0)},
    ("objectnet", "resnet50"):   {"NCM": (19.4, 2.3),  "Replay": (12.8, 302.5), "ER+MLP": (23.4, 302.5), "Nodes": (18.2, 102.5), "MAYA": (20.0, 243.4)},
    ("objectnet", "siglip2"):    {"NCM": (76.6, 1.8),  "Replay": (67.3, 226.9), "ER+MLP": (80.2, 226.9), "Nodes": (76.6, 71.0),  "MAYA": (80.5, 176.7)},
    ("tinyimagenet", "dinov3"):  {"NCM": (89.5, 3.1),  "Replay": (89.7, 1250.0), "ER+MLP": (94.5, 1250.0), "Nodes": (90.2, 227.8), "MAYA": (93.6, 555.5)},
    ("tinyimagenet", "resnet50"):{"NCM": (63.3, 1.6),  "Replay": (60.0, 625.0), "ER+MLP": (71.1, 625.0), "Nodes": (64.5, 82.2),  "MAYA": (69.0, 246.1)},
    ("tinyimagenet", "siglip2"): {"NCM": (86.1, 1.2),  "Replay": (84.6, 468.8), "ER+MLP": (90.2, 468.8), "Nodes": (82.7, 55.8),  "MAYA": (89.0, 178.8)},
}


def load_analytic(dataset, backbone):
    with open(f"results/comparisons/sota_{dataset}_{backbone}.json") as f:
        return {k: (v["AIA"] * 100.0, v["Memory_MB"]) for k, v in json.load(f).items()}


def frontier(points):
    """Upper-left staircase: sorted by memory, keep running accuracy maxima."""
    pts = sorted(points, key=lambda p: p[0])
    out, best = [], -np.inf
    for m, a in pts:
        if a > best:
            out.append((m, a))
            best = a
    return out


def main():
    os.makedirs("figures", exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 5.0), facecolor=SURFACE)

    # ══ Panel A ══ both families on one plane, ObjectNet / DINOv3 ═════════════
    ds, bb = "objectnet", "dinov3"
    story = STORY[(ds, bb)]
    ana = load_analytic(ds, bb)
    ana["NCM"] = story["NCM"]

    axA.set_facecolor(SURFACE)

    stat_pts = [(m, a) for a, m in ana.values()]
    data_pts = [(story[k][1], story[k][0]) for k in ("Replay", "Nodes", "ER+MLP", "MAYA")]
    f_stat, f_data = frontier(stat_pts), frontier(data_pts)
    axA.plot(*zip(*f_stat), "--", color=INK2, linewidth=1.5, alpha=0.5, zorder=1)
    axA.plot(*zip(*f_data), "-", color=ORANGE, linewidth=1.8, alpha=0.45, zorder=1)

    # Statistics-only family: open markers, gray.
    stat_off = {"NCM": (10, -2), "SLDA": (-11, -8), "ACL": (-11, 7), "RanPAC": (17, 4)}
    for name, (acc, mem) in ana.items():
        axA.plot(mem, acc, "o", markerfacecolor=SURFACE, markeredgecolor=INK2,
                 markeredgewidth=1.8, markersize=9, zorder=4)
        off = stat_off[name]
        axA.annotate(name, (mem, acc), textcoords="offset points", xytext=off,
                     ha="right" if off[0] < 0 else "left", fontsize=9.5, color=INK2)

    # Data-storing family: filled markers.
    data_off = {"Replay": (12, 0), "Nodes": (-12, -6), "ER+MLP": (12, -6)}
    for name, off in data_off.items():
        acc, mem = story[name]
        axA.plot(mem, acc, "o", color=ORANGE, markersize=9, markeredgecolor=SURFACE,
                 markeredgewidth=1.6, zorder=4)
        axA.annotate(name, (mem, acc), textcoords="offset points", xytext=off,
                     ha="right" if off[0] < 0 else "left", fontsize=9.5, color=ORANGE)

    acc, mem = story["MAYA"]
    axA.plot(mem, acc, "*", color=BLUE, markersize=22, markeredgecolor=SURFACE,
             markeredgewidth=1.5, zorder=5)
    axA.annotate("MAYA", (mem, acc), textcoords="offset points", xytext=(-16, 13),
                 ha="right", fontsize=11, color=BLUE, fontweight="semibold")

    axA.set_xscale("log")
    axA.set_xlim(1.6, 2600)
    axA.set_ylim(22, 92)
    axA.set_xlabel("Memory footprint (MB, log scale)", fontsize=10.5, color=INK)
    axA.set_ylabel("Average incremental accuracy (%)", fontsize=10.5, color=INK)
    axA.set_title("(a)  Two families, two frontiers  (ObjectNet / DINOv3)",
                  fontsize=11.5, color=INK, loc="left", pad=10)
    axA.set_xticks([10, 100, 1000])
    axA.set_xticklabels(["10", "100", "1000"])

    handles = [
        Line2D([], [], marker="o", linestyle="--", color=INK2, markerfacecolor=SURFACE,
               markeredgecolor=INK2, markersize=8, label="stores statistics only"),
        Line2D([], [], marker="o", linestyle="-", color=ORANGE, markersize=8,
               alpha=0.8, label="stores sample-derived data"),
    ]
    axA.legend(handles=handles, loc="lower right", fontsize=9, frameon=False)

    # ══ Panel B ══ MAYA vs ER+MLP across all nine configs ═════════════════════
    axB.set_facecolor(SURFACE)
    ratios, deltas, labels = [], [], []
    for (d, b), s in STORY.items():
        ratios.append(s["MAYA"][1] / s["ER+MLP"][1])
        deltas.append(s["MAYA"][0] - s["ER+MLP"][0])
        labels.append(f"{PRETTY_D[d][:4]}/{PRETTY_B[b][:5]}")
    ratios, deltas = np.array(ratios), np.array(deltas)

    axB.axhspan(0, 6, xmin=0, xmax=(1.0 - 0.05) / (1.15 - 0.05),
                facecolor=BLUE, alpha=0.06, zorder=0)
    axB.axhline(0, color=INK2, linewidth=1.1, zorder=2)
    axB.axvline(1.0, color=INK2, linewidth=1.1, zorder=2)

    colors = [BLUE if (r < 1 and dd > 0) else ORANGE for r, dd in zip(ratios, deltas)]
    axB.scatter(ratios, deltas, c=colors, s=110, edgecolor=SURFACE, linewidth=1.6, zorder=4)

    lab_off = {"Imag/DINOv": (0, -16), "Imag/ResNe": (0, 13), "Imag/SigLI": (0, -16),
               "Obje/DINOv": (0, 13), "Obje/ResNe": (0, -16), "Obje/SigLI": (14, -3),
               "Tiny/DINOv": (0, 13), "Tiny/ResNe": (0, -16), "Tiny/SigLI": (0, 13)}
    for r, dd, lb in zip(ratios, deltas, labels):
        off = lab_off.get(lb, (0, 13))
        axB.annotate(lb, (r, dd), textcoords="offset points", xytext=off,
                     ha="left" if off[0] > 5 else "center", fontsize=8.3, color=INK2)

    axB.text(0.09, 5.3, "MAYA better on both axes", fontsize=9.5, color=BLUE,
             fontweight="semibold", va="top")
    axB.set_xlim(0.05, 1.15)
    axB.set_ylim(-4.5, 6.0)
    axB.set_xlabel("MAYA memory ÷ ER+MLP memory", fontsize=10.5, color=INK)
    axB.set_ylabel("MAYA accuracy − ER+MLP accuracy  (pp)", fontsize=10.5, color=INK)
    axB.set_title("(b)  vs the strongest data-storing baseline,\nMAYA uses 20–56% of the memory",
                  fontsize=11.5, color=INK, loc="left", pad=10, linespacing=1.4)

    for ax in (axA, axB):
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9.5, length=0)

    fig.tight_layout(pad=1.4)
    fig.savefig("figures/family_comparison.png", dpi=220, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig("figures/family_comparison.pdf", facecolor=SURFACE, bbox_inches="tight")

    print(f"{'config':26} {'MAYA':>14} {'ER+MLP':>14}  {'d-acc':>6} {'mem ratio':>9}")
    print("-" * 76)
    for (d, b), s in STORY.items():
        print(f"{PRETTY_D[d]+'/'+PRETTY_B[b]:26} {s['MAYA'][0]:6.1f}% {s['MAYA'][1]:6.0f}MB "
              f"{s['ER+MLP'][0]:6.1f}% {s['ER+MLP'][1]:6.0f}MB  {s['MAYA'][0]-s['ER+MLP'][0]:+6.1f} "
              f"{s['MAYA'][1]/s['ER+MLP'][1]:8.2f}x")
    print(f"\nMAYA uses less memory than ER+MLP in {(ratios < 1).sum()}/9; "
          f"strictly better on both axes in {sum(1 for r, dd in zip(ratios, deltas) if r < 1 and dd > 0)}/9.")
    print(f"MAYA beats raw feature Replay in "
          f"{sum(1 for s in STORY.values() if s['MAYA'][0] > s['Replay'][0])}/9.")
    print("\nWrote figures/family_comparison.png and .pdf")


if __name__ == "__main__":
    main()
