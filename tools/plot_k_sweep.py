"""
Figure 3 -- node-budget (K) ablation, read from results/parsed/.

Replaces tools/legacy/plot_k_sweep.py, which had the April numbers inlined and
kept reproducing them after the June re-run.

Panel (a) plots ObjectNet/DINOv3 against two reference lines -- Standard NCM and
the global-projection-only configuration. K=1 falls below BOTH, which is the
open question in the submission checklist (item 1.4): one node per class should
not be far worse than one class mean in the same aligned space. The figure shows
that rather than cropping it out.

Panel (b) shows the same sweep for all nine configurations, confirming the K=1
collapse is systematic and not an artefact of one run.
"""

import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.chdir(ROOT)

BLUE = "#2a78d6"
RED = "#e34948"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8983"
GRID = "#e4e3df"
SURFACE = "#ffffff"

PRETTY_D = {"imagenet_r": "ImageNet-R", "tinyimagenet": "TinyImageNet", "objectnet": "ObjectNet"}
PRETTY_B = {"dinov3": "DINOv3", "resnet50": "ResNet50", "siglip2": "SigLIP2"}
FOCUS = "objectnet/dinov3"


def main():
    with open("results/parsed/all_results.json") as f:
        parsed = json.load(f)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 4.8), facecolor=SURFACE)

    # ── Panel A: ObjectNet / DINOv3 with reference lines ──────────────────
    r = parsed[FOCUS]
    ks = [d["k"] for d in r["ablation"]["k_sweep"]]
    acc = [d["aia"] for d in r["ablation"]["k_sweep"]]
    x = np.arange(len(ks))
    comp = {c["step"]: c["aia"] for c in r["ablation"]["component"]}
    ncm = comp["Standard NCM"]
    glob = comp["+Analytic ETF (beta=0, global only)"]

    axA.set_facecolor(SURFACE)
    axA.axhline(glob, color=RED, linestyle="--", linewidth=1.6, zorder=2)
    axA.text(len(ks) - 1.02, glob + 1.4, f"Global projection only, one mean per class ({glob:.1f}%)",
             color=RED, fontsize=9, ha="right", va="bottom")
    axA.axhline(ncm, color=INK2, linestyle=":", linewidth=1.6, zorder=2)
    axA.text(len(ks) - 1.02, ncm + 1.4, f"Standard NCM ({ncm:.1f}%)",
             color=INK2, fontsize=9, ha="right", va="bottom")

    axA.plot(x, acc, "-", color=BLUE, linewidth=2.2, zorder=3, solid_capstyle="round")
    axA.plot(x, acc, "o", color=BLUE, markersize=9, markeredgecolor=SURFACE,
             markeredgewidth=2, zorder=4)
    for xi, a in zip(x, acc):
        axA.annotate(f"{a:.1f}%", (xi, a), textcoords="offset points", xytext=(0, 13),
                     ha="center", fontsize=9.5, color=BLUE, fontweight="semibold")

    axA.annotate("$K{=}1$ falls below both references\n(unresolved: see checklist 1.4)",
                 (0, acc[0]), textcoords="offset points", xytext=(24, 6), ha="left",
                 fontsize=8.8, color=MUTED, linespacing=1.35,
                 arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9,
                                 shrinkA=6, shrinkB=6))

    axA.set_xticks(x)
    axA.set_xticklabels([str(k) for k in ks])
    axA.set_ylim(20, 92)
    axA.set_xlim(-0.35, len(ks) - 0.65)
    axA.set_xlabel("Maximum nodes per class ($K$)", fontsize=10.5, color=INK)
    axA.set_ylabel("Average incremental accuracy (%)", fontsize=10.5, color=INK)
    axA.set_title(f"(a)  {PRETTY_D['objectnet']} / {PRETTY_B['dinov3']}",
                  fontsize=11.5, color=INK, loc="left", pad=10)

    # ── Panel B: all nine configurations, relative to their own K=128 ─────
    axB.set_facecolor(SURFACE)
    for key, rec in parsed.items():
        sweep = rec["ablation"]["k_sweep"]
        a = np.array([d["aia"] for d in sweep])
        rel = a - a[-1]
        focus = key == FOCUS
        axB.plot(np.arange(len(a)), rel, "-o",
                 color=BLUE if focus else INK2,
                 alpha=1.0 if focus else 0.28,
                 linewidth=2.2 if focus else 1.4,
                 markersize=7 if focus else 4,
                 markeredgecolor=SURFACE, markeredgewidth=1.2 if focus else 0.8,
                 zorder=4 if focus else 3)
    axB.axhline(0, color=INK2, linewidth=1.1, zorder=2)
    axB.annotate(f"{PRETTY_D['objectnet']} / {PRETTY_B['dinov3']}",
                 (0, parsed[FOCUS]["ablation"]["k_sweep"][0]["aia"]
                  - parsed[FOCUS]["ablation"]["k_sweep"][-1]["aia"]),
                 textcoords="offset points", xytext=(14, -4), ha="left",
                 fontsize=9.5, color=BLUE, fontweight="semibold")
    axB.text(len(ks) - 1.05, -2.0, "all nine configurations", ha="right", va="top",
             fontsize=9.5, color=MUTED)

    axB.set_xticks(np.arange(len(ks)))
    axB.set_xticklabels([str(k) for k in ks])
    axB.set_xlim(-0.35, len(ks) - 0.65)
    axB.set_xlabel("Maximum nodes per class ($K$)", fontsize=10.5, color=INK)
    axB.set_ylabel("Accuracy relative to $K{=}128$  (pp)", fontsize=10.5, color=INK)
    axB.set_title("(b)  The $K{=}1$ collapse is systematic",
                  fontsize=11.5, color=INK, loc="left", pad=10)

    for ax in (axA, axB):
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9.5, length=0)

    fig.tight_layout(pad=1.4)
    pathlib.Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/k_sweep_ablation.png", dpi=220, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig("figures/k_sweep_ablation.pdf", facecolor=SURFACE, bbox_inches="tight")

    print(f"{PRETTY_D['objectnet']}/{PRETTY_B['dinov3']}: " +
          ", ".join(f"K={k}: {a:.1f}%" for k, a in zip(ks, acc)))
    print(f"  reference -- NCM {ncm:.1f}%, global-only {glob:.1f}%")
    print("\nK=1 accuracy by configuration (all below their global-only row):")
    for key, rec in parsed.items():
        s = rec["ablation"]
        print(f"  {key:26} K=1 {s['k_sweep'][0]['aia']:5.1f}%   "
              f"global-only {s['component'][1]['aia']:5.1f}%")
    print("\nWrote figures/k_sweep_ablation.{png,pdf}")


if __name__ == "__main__":
    main()
