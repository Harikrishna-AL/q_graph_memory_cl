"""
Accuracy-memory Pareto audit for MAYA.

Answers "is MAYA just buying accuracy with memory?" by (a) showing MAYA's
footprint is a tunable knob (the K-sweep curve) rather than a fixed cost, and
(b) auditing, across all nine configurations, whether MAYA sits on the
accuracy-memory Pareto frontier against the statistics-only analytic methods.

Reads results/parsed/all_results.json and results/comparisons/sota_*.json.
Nothing is hardcoded -- rerun after any new experiment and the figure follows.
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

# ── Palette (dataviz reference instance, light surface) ──────────────────────
BLUE = "#2a78d6"   # categorical slot 1 -> MAYA
RED = "#e34948"    # diverging warm pole -> below frontier
INK = "#0b0b0b"
INK2 = "#52514e"   # baselines
MUTED = "#8a8983"
GRID = "#e4e3df"
SURFACE = "#ffffff"

DATASETS = ["imagenet_r", "tinyimagenet", "objectnet"]
BACKBONES = ["dinov3", "resnet50", "siglip2"]
PRETTY_D = {"imagenet_r": "ImageNet-R", "tinyimagenet": "TinyImageNet", "objectnet": "ObjectNet"}
PRETTY_B = {"dinov3": "DINOv3", "resnet50": "ResNet50", "siglip2": "SigLIP2"}
FOCUS = ("objectnet", "dinov3")


def load():
    with open("results/parsed/all_results.json") as f:
        parsed = json.load(f)
    sota = {}
    for ds in DATASETS:
        for bb in BACKBONES:
            with open(f"results/comparisons/sota_{ds}_{bb}.json") as f:
                sota[f"{ds}/{bb}"] = {k: (v["AIA"] * 100.0, v["Memory_MB"])
                                      for k, v in json.load(f).items()}
    return parsed, sota


def baselines_with_ncm(parsed, sota, key):
    """Statistics-only family: SLDA / ACL / RanPAC, plus NCM as the floor."""
    b = dict(sota[key])
    s = parsed[key]["story"]["1"]
    b["NCM"] = (s["aia"], s["mem_mb"])
    return b


def frontier_audit(parsed, sota):
    """Best statistics-only baseline reachable at or below MAYA's memory."""
    rows = []
    for ds in DATASETS:
        for bb in BACKBONES:
            key = f"{ds}/{bb}"
            maya = parsed[key]["story"]["5"]
            base = baselines_with_ncm(parsed, sota, key)
            cheaper = {k: v for k, v in base.items() if v[1] <= maya["mem_mb"]}
            name, (acc, mem) = max(cheaper.items(), key=lambda kv: kv[1][0])
            rows.append(dict(ds=ds, bb=bb, maya_acc=maya["aia"], maya_mem=maya["mem_mb"],
                             rival=name, rival_acc=acc, rival_mem=mem,
                             delta=maya["aia"] - acc))
    return rows


def main():
    parsed, sota = load()
    rows = frontier_audit(parsed, sota)

    print(f"{'config':28} {'MAYA':>16}  {'best <= MAYA mem':>26}  {'delta':>7}")
    print("-" * 84)
    for r in sorted(rows, key=lambda r: r["delta"]):
        cfg = f"{PRETTY_D[r['ds']]}/{PRETTY_B[r['bb']]}"
        print(f"{cfg:28} {r['maya_acc']:6.1f}% {r['maya_mem']:6.0f}MB  "
              f"{r['rival']:>8} {r['rival_acc']:6.1f}% {r['rival_mem']:6.0f}MB  {r['delta']:+6.1f}")
    n_win = sum(1 for r in rows if r["delta"] > 0)
    print(f"\nMAYA on the frontier in {n_win}/9 configurations.")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.9), facecolor=SURFACE)

    # ══ Panel A ══ focus configuration on the accuracy-memory plane ═══════
    fkey = f"{FOCUS[0]}/{FOCUS[1]}"
    sweep = parsed[fkey]["ablation"]["k_sweep"]
    ks = [d["k"] for d in sweep]
    k_acc = [d["aia"] for d in sweep]
    k_mem = [d["mem_mb"] for d in sweep]
    base = baselines_with_ncm(parsed, sota, fkey)

    axA.set_facecolor(SURFACE)
    best_i = int(np.argmax(k_acc))
    axA.add_patch(plt.Rectangle((1.0, k_acc[best_i]), k_mem[best_i] - 1.0, 100 - k_acc[best_i],
                                facecolor=RED, alpha=0.055, zorder=0, linewidth=0))
    axA.text(2.6, 98.5, "dominates MAYA\n(less memory, higher accuracy)",
             fontsize=8.5, color=RED, va="top", ha="left", alpha=0.9, linespacing=1.35)

    axA.plot(k_mem, k_acc, "-", color=BLUE, linewidth=2.0, zorder=3, solid_capstyle="round")
    axA.plot(k_mem, k_acc, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE,
             markeredgewidth=2, zorder=4)
    k_off = {1: (12, 0), 16: (-6, -17), 64: (-2, -17), 128: (13, -2)}
    for k, m, a in zip(ks, k_mem, k_acc):
        off = k_off.get(k, (0, -17))
        axA.annotate(f"K={k}", (m, a), textcoords="offset points", xytext=off,
                     ha="left" if off[0] > 5 else "center",
                     va="center" if off[0] > 5 else "top",
                     fontsize=9, color=BLUE, fontweight="semibold")
    axA.text(700, 52, "MAYA\n(sweep node\nbudget $K$)", ha="center", va="center",
             fontsize=10.5, color=BLUE, fontweight="semibold", linespacing=1.35)

    marks = {"NCM": ("s", (11, -1)), "SLDA": ("^", (-12, -7)), "ACL": ("D", (-12, 7)),
             "RanPAC": ("v", (8, 13))}
    for name, (mk, off) in marks.items():
        acc, mem = base[name]
        axA.plot(mem, acc, mk, color=INK2, markersize=8, markeredgecolor=SURFACE,
                 markeredgewidth=1.6, zorder=4)
        axA.annotate(f"{name}  {acc:.1f}%", (mem, acc), textcoords="offset points",
                     xytext=off, ha="right" if off[0] < 0 else "left",
                     fontsize=9.5, color=INK2)

    axA.set_xscale("log")
    axA.set_xlim(2.2, 1500)
    axA.set_ylim(25, 100)
    axA.set_xlabel("Memory footprint (MB, log scale)", fontsize=10.5, color=INK)
    axA.set_ylabel("Average incremental accuracy (%)", fontsize=10.5, color=INK)
    axA.set_title(f"(a)  {PRETTY_D[FOCUS[0]]} / {PRETTY_B[FOCUS[1]]}: MAYA's memory is a knob,"
                  "\nbut ACL sits above its curve",
                  fontsize=11.5, color=INK, loc="left", pad=10, linespacing=1.4)
    axA.set_xticks([10, 100, 1000])
    axA.set_xticklabels(["10", "100", "1000"])

    # ══ Panel B ══ frontier audit across all nine configurations ══════════
    axB.set_facecolor(SURFACE)
    rows_sorted = sorted(rows, key=lambda r: r["delta"])
    y = np.arange(len(rows_sorted))
    deltas = [r["delta"] for r in rows_sorted]
    colors = [BLUE if d > 0 else RED for d in deltas]

    axB.barh(y, deltas, color=colors, height=0.62, zorder=3)
    axB.axvline(0, color=INK2, linewidth=1.2, zorder=4)
    for yi, r in zip(y, rows_sorted):
        axB.text(0.45, yi, f"{r['delta']:+.1f} pp   vs {r['rival']} at {r['rival_mem']:.0f} MB",
                 va="center", ha="left", fontsize=8.8, color=INK2)

    axB.set_yticks(y)
    axB.set_yticklabels([f"{PRETTY_D[r['ds']]} / {PRETTY_B[r['bb']]}" for r in rows_sorted],
                        fontsize=9.5, color=INK)
    axB.set_xlim(-6.2, 9.0)
    axB.set_xticks([-6, -4, -2, 0])
    axB.set_xlabel("MAYA accuracy − best baseline at ≤ MAYA's memory  (pp)",
                   fontsize=10.5, color=INK)
    axB.set_title(f"(b)  MAYA is off the accuracy-memory frontier\n"
                  f"in {9 - n_win} of 9 configurations",
                  fontsize=11.5, color=INK, loc="left", pad=10, linespacing=1.4)

    handles = [Line2D([], [], marker="s", linestyle="", color=BLUE, markersize=8,
                      label="MAYA on the frontier"),
               Line2D([], [], marker="s", linestyle="", color=RED, markersize=8,
                      label="a cheaper baseline is more accurate")]
    axB.legend(handles=handles, loc="upper left", fontsize=9, frameon=False,
               bbox_to_anchor=(0.005, 0.46), handletextpad=0.5)

    for ax in (axA, axB):
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9.5, length=0)
    axB.grid(axis="y", visible=False)

    fig.tight_layout(pad=1.4)
    pathlib.Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/pareto_audit.png", dpi=220, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig("figures/pareto_audit.pdf", facecolor=SURFACE, bbox_inches="tight")
    print("\nWrote figures/pareto_audit.{png,pdf}")


if __name__ == "__main__":
    main()
