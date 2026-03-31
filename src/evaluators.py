# evaluators.py
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestCentroid

# Importing your specific ICML metric helpers
from src.helper import (
    build_accuracy_matrix,
    compute_average_accuracy,
    compute_average_forgetting,
)

from .config import Config

# ==============================================================================
# GRAPH EVALUATION
# ==============================================================================


def evaluate_graph(
    model, test_features, test_labels, mode="soft", historical_matrix=None
):
    print("\n📊 --- Running Graph Evaluation (Task-by-Task) ---")

    n_chunks = Config.N_CHUNKS
    mask_clean = [True] * n_chunks

    # Occlusion: Mask bottom 50%
    half = n_chunks // 2
    mask_occluded = [True] * half + [False] * (n_chunks - half)

    # 1. Run the Task-Separated Loops
    clean_task_accs = _run_eval_loop(
        model, test_features, test_labels, mask_clean, "CLEAN", mode=mode
    )
    occ_task_accs = _run_eval_loop(
        model, test_features, test_labels, mask_occluded, "OCCLUDED", mode=mode
    )

    # 2. Compute ICML Metrics
    current_row = np.array(clean_task_accs).reshape(1, -1)
    current_row_occ = np.array(occ_task_accs).reshape(1, -1)

    eval_matrix = historical_matrix if historical_matrix is not None else current_row

    final_acc = compute_average_accuracy(eval_matrix)
    forgetting = compute_average_forgetting(eval_matrix)

    print(
        f"   🏆 Graph Metrics -> Final Acc: {final_acc * 100:.2f}% | Forgetting: {forgetting * 100:.2f}%"
    )

    # 3. Plotting
    _plot_results(clean_task_accs, occ_task_accs)

    return {
        "clean_tasks": clean_task_accs,
        "occluded_tasks": occ_task_accs,
        "final_accuracy": final_acc,
        "forgetting": forgetting,
    }


def _run_eval_loop(model, features, labels, mask, name, mode="soft"):
    accuracies = []
    # print(f"   Evaluating: {name}...")

    for task_id in range(Config.N_TASKS):
        start_cls = task_id * Config.CLASSES_PER_TASK
        end_cls = (task_id + 1) * Config.CLASSES_PER_TASK

        task_mask = (labels >= start_cls) & (labels < end_cls)
        task_idxs = torch.where(task_mask)[0]

        if len(task_idxs) == 0:
            accuracies.append(0.0)
            continue

        t_feats = features[task_idxs]
        t_lbls = labels[task_idxs]

        batch_accs = []
        bs = 100
        for i in range(0, len(t_feats), bs):
            b_in = t_feats[i : i + bs]
            b_lbl = t_lbls[i : i + bs]
            preds = model.predict(b_in, mask, mode=mode)
            batch_accs.append((preds == b_lbl).float().sum().item())

        task_acc = sum(batch_accs) / len(t_feats)
        accuracies.append(task_acc)

    return accuracies


# ==============================================================================
# HYBRID EVALUATION
# ==============================================================================


def predict_dual_system(model, feature_vector, alpha=0.6):
    """
    Bio-Inspired Entropy-Aware Fusion with Logit Calibration.
    
    1. Calibrate: Rescale System 1 (Density) and System 2 (Prototypes) 
       logits to the same range using a robust Layer-Norm-like scaling.
    2. Entropy-Aware: Calculate confusion of the prototype branch.
    3. Fusion: Dynamic weighted blend.
    """
    # Raw logit extraction
    node_logits = model.predict_node_logits(feature_vector)   # (B, C)
    proto_logits = model.predict_proto_logits(feature_vector) # (B, C)

    # Logit Calibration: Normalize both branches to zero-mean and unit-variance per sample.
    # This prevents one system from "shouting over" the other due to logit-scale mismatch.
    def calibrate(logits):
        mu = logits.mean(dim=1, keepdim=True)
        std = logits.std(dim=1, keepdim=True) + 1e-8
        return (logits - mu) / std

    node_cal = calibrate(node_logits)
    proto_cal = calibrate(proto_logits)

    # Temperature-scaled probabilities for calibrated logits
    # We use a standard temperature of 1.0 because the calibration std=1.0.
    prob_node = F.softmax(node_cal, dim=1)
    prob_proto = F.softmax(proto_cal, dim=1)

    # 1. Calculate Prototype Branch Uncertainty (Normalised Entropy)
    C = prob_proto.shape[1]
    entropy = -torch.sum(prob_proto * torch.log(prob_proto + 1e-10), dim=1)
    max_entropy = np.log(C)
    uncertainty = (entropy / max_entropy).unsqueeze(1) # (B, 1)

    # 2. Dynamic Alpha Shift: 
    # For DINOv2, we stay conservative (alpha=0.2), shifting toward 0.5 under confusion.
    # For SigLIP, we start at alpha=0.6, shifting toward 0.8.
    dynamic_alpha = alpha + (min(0.8, alpha + 0.3) - alpha) * uncertainty 

    final_probs = (dynamic_alpha * prob_node) + ((1 - dynamic_alpha) * prob_proto)
    return torch.argmax(final_probs, dim=1)


def evaluate_hybrid_system(
    model,
    test_features,
    test_labels,
    alpha=0.6,
    historical_matrix=None,
):
    print("\n📊 --- Running Hybrid System Evaluation ---")

    hybrid_task_accs = _run_hybrid_eval_loop(model, test_features, test_labels, alpha)

    current_row = np.array(hybrid_task_accs).reshape(1, -1)
    eval_matrix = historical_matrix if historical_matrix is not None else current_row

    final_acc = compute_average_accuracy(eval_matrix)
    forgetting = compute_average_forgetting(eval_matrix)

    print(
        f"   🏆 Hybrid Metrics -> Final Acc: {final_acc * 100:.2f}% | Forgetting: {forgetting * 100:.2f}%"
    )

    return {
        "tasks": hybrid_task_accs,
        "final_accuracy": final_acc,
        "forgetting": forgetting,
    }


def _run_hybrid_eval_loop(model, features, labels, alpha):
    accuracies = []
    for task_id in range(Config.N_TASKS):
        start_cls = task_id * Config.CLASSES_PER_TASK
        end_cls = (task_id + 1) * Config.CLASSES_PER_TASK
        task_mask = (labels >= start_cls) & (labels < end_cls)
        task_idxs = torch.where(task_mask)[0]

        if len(task_idxs) == 0:
            accuracies.append(0.0)
            continue

        t_feats = features[task_idxs]
        t_lbls = labels[task_idxs]

        batch_accs = []
        bs = 100
        for i in range(0, len(t_feats), bs):
            b_in = t_feats[i : i + bs]
            b_lbl = t_lbls[i : i + bs]
            preds = predict_dual_system(model, b_in, alpha=alpha)
            batch_accs.append((preds == b_lbl).float().sum().item())

        task_acc = sum(batch_accs) / len(t_feats)
        accuracies.append(task_acc)

    return accuracies


# ==============================================================================
# PLOTTING UTILS
# ==============================================================================


def _plot_results(clean_accs, occ_accs):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 6))

    n_tasks = len(clean_accs)
    task_range = range(1, n_tasks + 1)

    plt.plot(
        task_range,
        [a * 100 for a in clean_accs],
        "o-",
        label="Clean Input",
        linewidth=2,
    )
    plt.plot(
        task_range,
        [a * 100 for a in occ_accs],
        "x--",
        label="50% Occluded",
        linewidth=2,
    )

    plt.title(f"Graph Memory Robustness")
    plt.xlabel("Task ID")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.xticks(task_range)

    plt.savefig("outputs/results_plot.png")
    plt.savefig("outputs/results_plot.svg", format="svg")  # SVG Save
    print(f"📈 Plot saved to outputs/results_plot.png & .svg")


def plot_memory_trace(memory_trace, dataset_name="dataset"):
    """
    Two-panel figure showing how memory evolves block-by-block during training.

    Top panel — Memory in MB
        Green line : peak memory right after each online step (before pruning)
        Red line   : post-sleep memory after each consolidation (after pruning)
        Shaded fill: memory recovered by consolidation each block
        Grey band  : constant baseline — prototypes + projection matrix

    Bottom panel — Episodic node count
        Green line : nodes after online step
        Red line   : nodes after consolidation
        Shaded fill: nodes pruned each block

    The sawtooth pattern reveals the grow-then-prune rhythm of the bio cycle.
    """
    if not memory_trace:
        print("⚠️  Memory trace is empty — nothing to plot.")
        return

    online = [s for s in memory_trace if s["phase"] == "after_online"]
    consol = [s for s in memory_trace if s["phase"] == "after_consolidation"]

    if not online:
        print("⚠️  No 'after_online' snapshots found.")
        return

    # Use block index as x so both phases align on the same tick
    online_x = [s["block"] for s in online]
    consol_x = [s["block"] for s in consol]

    online_mem = [s["total_mb"] for s in online]
    consol_mem = [s["total_mb"] for s in consol]
    online_nodes = [s["n_nodes"] for s in online]
    consol_nodes = [s["n_nodes"] for s in consol]

    # Constant floor: prototype + projection memory (take from first snapshot)
    floor_mb = online[0]["proto_mb"] + online[0]["proj_mb"]

    # ── x-axis tick labels: block index with sample count every ~10 blocks ──
    step = max(1, len(online_x) // 10)
    tick_positions = online_x[::step]
    tick_labels = [
        f"{s['block']}\n({s['samples_seen'] // 1000}k)" for s in online[::step]
    ]

    fig, (ax_mem, ax_nodes) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=False, gridspec_kw={"hspace": 0.45}
    )

    # ── TOP: Memory ──────────────────────────────────────────────────────────
    ax_mem.plot(
        online_x,
        online_mem,
        "o-",
        color="#2ca02c",
        linewidth=1.8,
        markersize=4,
        alpha=0.85,
        label="After online step (peak)",
    )
    if consol_mem:
        ax_mem.plot(
            consol_x,
            consol_mem,
            "s-",
            color="#d62728",
            linewidth=1.8,
            markersize=4,
            alpha=0.85,
            label="After consolidation (post-sleep)",
        )
        # Shade the gap only where both series share a block index
        shared_blocks = sorted(set(online_x) & set(consol_x))
        online_dict = {s["block"]: s["total_mb"] for s in online}
        consol_dict = {s["block"]: s["total_mb"] for s in consol}
        shared_peak = [online_dict[b] for b in shared_blocks]
        shared_post = [consol_dict[b] for b in shared_blocks]
        ax_mem.fill_between(
            shared_blocks,
            shared_post,
            shared_peak,
            alpha=0.12,
            color="#2ca02c",
            label="Memory freed by consolidation",
        )

    # Grey band for constant floor
    ax_mem.axhline(
        floor_mb,
        color="grey",
        linewidth=1.2,
        linestyle="--",
        alpha=0.6,
        label=f"Constant base (proto + proj ≈ {floor_mb:.2f} MB)",
    )

    ax_mem.set_ylabel("Memory (MB)", fontsize=11)
    ax_mem.set_title(
        f"Memory Footprint During Training — {dataset_name}",
        fontsize=13,
        fontweight="bold",
    )
    ax_mem.set_xticks(tick_positions)
    ax_mem.set_xticklabels(tick_labels, fontsize=8)
    ax_mem.set_xlabel("Block  (samples seen)", fontsize=10)
    ax_mem.legend(fontsize=9, loc="upper left")
    ax_mem.grid(True, linestyle="--", alpha=0.3)

    # Annotate final values
    ax_mem.annotate(
        f"  {online_mem[-1]:.2f} MB",
        xy=(online_x[-1], online_mem[-1]),
        fontsize=8,
        color="#2ca02c",
        va="center",
    )
    if consol_mem:
        ax_mem.annotate(
            f"  {consol_mem[-1]:.2f} MB",
            xy=(consol_x[-1], consol_mem[-1]),
            fontsize=8,
            color="#d62728",
            va="center",
        )

    # ── BOTTOM: Node count ────────────────────────────────────────────────────
    ax_nodes.plot(
        online_x,
        online_nodes,
        "o-",
        color="#2ca02c",
        linewidth=1.8,
        markersize=4,
        alpha=0.85,
        label="After online step",
    )
    if consol_nodes:
        ax_nodes.plot(
            consol_x,
            consol_nodes,
            "s-",
            color="#d62728",
            linewidth=1.8,
            markersize=4,
            alpha=0.85,
            label="After consolidation",
        )
        shared_on_nodes = [online_dict[b] for b in shared_blocks]  # reuse dict trick
        # recompute with node counts
        online_node_dict = {s["block"]: s["n_nodes"] for s in online}
        consol_node_dict = {s["block"]: s["n_nodes"] for s in consol}
        shared_peak_n = [online_node_dict[b] for b in shared_blocks]
        shared_post_n = [consol_node_dict[b] for b in shared_blocks]
        ax_nodes.fill_between(
            shared_blocks,
            shared_post_n,
            shared_peak_n,
            alpha=0.12,
            color="#2ca02c",
            label="Nodes pruned by consolidation",
        )

    ax_nodes.set_ylabel("Episodic node count", fontsize=11)
    ax_nodes.set_xlabel("Block  (samples seen)", fontsize=10)
    ax_nodes.set_xticks(tick_positions)
    ax_nodes.set_xticklabels(tick_labels, fontsize=8)
    ax_nodes.legend(fontsize=9, loc="upper left")
    ax_nodes.grid(True, linestyle="--", alpha=0.3)

    # Annotate final node counts
    ax_nodes.annotate(
        f"  {online_nodes[-1]}",
        xy=(online_x[-1], online_nodes[-1]),
        fontsize=8,
        color="#2ca02c",
        va="center",
    )
    if consol_nodes:
        ax_nodes.annotate(
            f"  {consol_nodes[-1]}",
            xy=(consol_x[-1], consol_nodes[-1]),
            fontsize=8,
            color="#d62728",
            va="center",
        )

    os.makedirs("outputs", exist_ok=True)
    out_base = f"outputs/memory_trace_{dataset_name}"
    plt.savefig(f"{out_base}.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{out_base}.svg", format="svg", bbox_inches="tight")
    plt.close()
    print(f"✅ Memory trace saved → {out_base}.png / .svg")


def plot_memory_comparison(
    memory_trace,
    n_classes,
    feature_dim=384,
    dataset_name="dataset",
    bio_accuracy=None,
    ncm_accuracy=None,
    replay_accuracy=None,
):
    """
    Compares memory requirements of three approaches over training time.

    Naive Replay Buffer  — stores every feature vector seen (upper bound)
    Bio-Inspired Graph   — actual measured memory from the training trace
    NCM class means      — stores only one mean prototype per class (lower bound)

    Layout
    ------
    Main axes  : log-scale y, all three methods visible across 3 orders of magnitude
    Inset axes : linear-scale zoom into the NCM–Bio region (the real tradeoff)
    Stats box  : final memory values and compression ratios

    Parameters
    ----------
    memory_trace    : list of dicts from _snapshot_memory (learner.py)
    n_classes       : total number of classes in the benchmark
    feature_dim     : DINOv2 embedding dimension (default 384)
    dataset_name    : used in the title and output filename
    bio_accuracy    : optional float, annotated on the bio line
    ncm_accuracy    : optional float, annotated on the ncm line
    replay_accuracy : optional float, annotated on the replay line
    """
    if not memory_trace:
        print("⚠️  Memory trace is empty — nothing to plot.")
        return

    # ── Extract post-consolidation snapshots (the stable, post-sleep footprint)
    consol = [s for s in memory_trace if s["phase"] == "after_consolidation"]
    if not consol:
        consol = memory_trace  # fallback: no consolidation was run

    bio_x = np.array([s["samples_seen"] for s in consol])
    bio_y = np.array([s["total_mb"] for s in consol])

    total_samples = int(max(s["samples_seen"] for s in memory_trace))
    bytes_per_vec = feature_dim * 4  # float32

    # ── Theoretical baselines over the full x range
    x_full = np.linspace(0, total_samples, 600)

    # Naive replay: one full feature vector kept per training sample
    replay_y = x_full * bytes_per_vec / (1024**2)

    # NCM: one mean vector per class, classes introduced proportionally to samples
    ncm_classes = np.clip(n_classes * x_full / total_samples, 0, n_classes)
    ncm_y = ncm_classes * bytes_per_vec / (1024**2)

    # ── Final values and compression ratios
    replay_final = total_samples * bytes_per_vec / (1024**2)
    ncm_final = n_classes * bytes_per_vec / (1024**2)
    bio_final = float(bio_y[-1]) if len(bio_y) else 0.0

    ratio_bio_vs_replay = replay_final / bio_final if bio_final > 0 else float("inf")
    ratio_bio_vs_ncm = bio_final / ncm_final if ncm_final > 0 else float("inf")

    # ── Colours
    C_REPLAY = "#d62728"  # red
    C_BIO = "#1f77b4"  # blue
    C_NCM = "#2ca02c"  # green

    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 1, 1)

    # ── Main plot (log scale) ─────────────────────────────────────────────────
    ax.plot(
        x_full,
        replay_y,
        "-",
        color=C_REPLAY,
        linewidth=2.2,
        zorder=3,
        label=f"Naive Replay Buffer  (stores every vector)",
    )
    ax.plot(
        bio_x,
        bio_y,
        "o-",
        color=C_BIO,
        linewidth=2.2,
        markersize=4,
        zorder=5,
        label=f"Bio-Inspired Graph — Ours  (post-sleep)",
    )
    ax.plot(
        x_full,
        ncm_y,
        "--",
        color=C_NCM,
        linewidth=2.0,
        zorder=3,
        label=f"NCM class means  (one vector per class)",
    )

    # ax.set_yscale("log")
    ax.set_xlim(0, total_samples * 1.05)
    ax.set_xlabel("Training samples seen", fontsize=12)
    ax.set_ylabel("Memory  (MB, log scale)", fontsize=12)
    ax.set_title(
        f"Memory Requirements — {dataset_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.3, which="both")

    # x-tick labels — scale-aware so small datasets don't show "0k" everywhere
    x_ticks = np.linspace(0, total_samples, 6)

    def _fmt_samples(v):
        if v == 0:
            return "0"
        if total_samples >= 50_000:
            return f"{int(v / 1000)}k"
        if total_samples >= 1_000:
            return f"{v / 1000:.1f}k"
        return str(int(v))

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([_fmt_samples(v) for v in x_ticks])

    # Annotate final values on the right edge
    x_ann = total_samples * 1.01
    ax.annotate(
        f" {replay_final:.1f} MB",
        xy=(total_samples, replay_final),
        fontsize=8.5,
        color=C_REPLAY,
        va="center",
    )
    ax.annotate(
        f" {bio_final:.1f} MB",
        xy=(total_samples, bio_final),
        fontsize=8.5,
        color=C_BIO,
        va="center",
    )
    ax.annotate(
        f" {ncm_final:.3f} MB",
        xy=(total_samples, ncm_y[-1]),
        fontsize=8.5,
        color=C_NCM,
        va="center",
    )

    # Optional accuracy annotations on the lines at mid-point
    mid = total_samples // 2
    mid_idx = np.searchsorted(x_full, mid)
    if replay_accuracy is not None:
        ax.annotate(
            f"acc: {replay_accuracy * 100:.1f}%",
            xy=(x_full[mid_idx], replay_y[mid_idx]),
            xytext=(0, 12),
            textcoords="offset points",
            fontsize=7.5,
            color=C_REPLAY,
            ha="center",
        )
    if ncm_accuracy is not None:
        ax.annotate(
            f"acc: {ncm_accuracy * 100:.1f}%",
            xy=(x_full[mid_idx], ncm_y[mid_idx]),
            xytext=(0, -16),
            textcoords="offset points",
            fontsize=7.5,
            color=C_NCM,
            ha="center",
        )
    if bio_accuracy is not None:
        ax.annotate(
            f"acc: {bio_accuracy * 100:.1f}%",
            xy=(bio_x[len(bio_x) // 2], bio_y[len(bio_y) // 2]),
            xytext=(0, 12),
            textcoords="offset points",
            fontsize=7.5,
            color=C_BIO,
            ha="center",
        )

    # ── Stats box ────────────────────────────────────────────────────────────
    stats_lines = [
        "Final memory",
        "─" * 26,
        f"Replay Buffer :  {replay_final:>7.1f} MB",
        f"Bio Graph     :  {bio_final:>7.1f} MB",
        f"NCM           :  {ncm_final:>7.3f} MB",
        "─" * 26,
        f"Bio is {ratio_bio_vs_replay:>5.1f}× smaller than Replay",
        f"NCM is {ratio_bio_vs_ncm:>5.1f}× smaller than Bio",
    ]
    stats_text = "\n".join(stats_lines)
    ax.text(
        0.015,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="grey"
        ),
    )

    ax.legend(fontsize=9.5, loc="lower right")

    # ── Inset: linear zoom into NCM–Bio range ────────────────────────────────
    # Only draw if the two are meaningfully different
    if bio_final > ncm_final * 2:
        y_top = bio_final * 1.35
        # position: right side, lower-middle
        ax_in = ax.inset_axes([0.54, 0.08, 0.42, 0.40])

        ax_in.plot(x_full, replay_y, "-", color=C_REPLAY, linewidth=1.4, alpha=0.35)
        ax_in.plot(bio_x, bio_y, "o-", color=C_BIO, linewidth=1.6, markersize=3)
        ax_in.plot(x_full, ncm_y, "--", color=C_NCM, linewidth=1.4)

        ax_in.set_ylim(0, y_top)
        ax_in.set_xlim(0, total_samples)
        ax_in.set_xticks(x_ticks)
        ax_in.set_xticklabels(
            [_fmt_samples(v) for v in x_ticks],
            fontsize=6.5,
        )
        ax_in.set_ylabel("Memory (MB)", fontsize=7)
        ax_in.set_title("Zoom: NCM vs Bio", fontsize=7.5, pad=3)
        ax_in.yaxis.set_tick_params(labelsize=6.5)
        ax_in.grid(True, linestyle="--", alpha=0.25)

        # Shade Bio–NCM gap in the inset
        ax_in.fill_between(
            x_full,
            ncm_y,
            np.minimum(replay_y, y_top),
            where=(np.minimum(replay_y, y_top) >= ncm_y),
            alpha=0.06,
            color=C_BIO,
        )

    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    out_base = f"outputs/memory_comparison_{dataset_name}"
    plt.savefig(f"{out_base}.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{out_base}.svg", format="svg", bbox_inches="tight")
    plt.close()
    print(f"✅ Memory comparison saved → {out_base}.png / .svg")


def plot_memory_with_errors(method_names, memory_means, memory_stds):
    """
    Plots a bar chart comparing memory usage of different methods with error bars.
    Useful for TQM vs Replay comparison.
    """
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(8, 6))

    x_pos = np.arange(len(method_names))

    # Plot Bar with Error Caps
    plt.bar(
        x_pos,
        memory_means,
        yerr=memory_stds,
        align="center",
        alpha=0.7,
        ecolor="black",
        capsize=10,
        color=["#1f77b4", "#ff7f0e"],
    )

    plt.ylabel("Memory Usage (MB)")
    plt.xticks(x_pos, method_names)
    plt.title("Memory Efficiency Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add labels on top
    for i, v in enumerate(memory_means):
        plt.text(
            i,
            v + memory_stds[i] + 0.5,
            f"{v:.1f}MB",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("outputs/memory_comparison.png")
    plt.savefig("outputs/memory_comparison.svg", format="svg")
    print("🧠 Memory plot saved to outputs/memory_comparison.svg")


# ==============================================================================
# ICML BENCHMARKS (ROBUSTNESS)
# ==============================================================================


def apply_feature_occlusion(features, occlusion_ratio):
    if occlusion_ratio <= 0.0:
        return features.clone() if torch.is_tensor(features) else features.copy()

    X_occ = features.clone() if torch.is_tensor(features) else features.copy()
    N, D = X_occ.shape
    num_masked = int(D * occlusion_ratio)

    for i in range(N):
        mask_indices = np.random.choice(D, num_masked, replace=False)
        X_occ[i, mask_indices] = 0.0

    return X_occ


def run_icml_occlusion_benchmark(
    graph_model, X_train, y_train, X_test, y_test, dataset_name, alpha=0.9
):
    """
    Generates Table 2 and Figure 1.
    INCLUDES LINEAR UPPER BOUND (SGDClassifier).
    """
    print("\n" + "=" * 50)
    print(" 🛡️ RUNNING ICML OCCLUSION BENCHMARK")
    print("=" * 50)

    ratios = [0.0, 0.1, 0.3, 0.5, 0.75]

    tqm_accs, ncm_accs, lin_accs = [], [], []

    X_train_np = X_train.cpu().numpy() if torch.is_tensor(X_train) else X_train
    y_train_np = y_train.cpu().numpy() if torch.is_tensor(y_train) else y_train
    X_test_np = X_test.cpu().numpy() if torch.is_tensor(X_test) else X_test
    y_test_np = y_test.cpu().numpy() if torch.is_tensor(y_test) else y_test

    # 1. Train Baselines
    print("   ⚙️  Training NCM and Linear Upper Bound...")
    ncm = NearestCentroid()
    ncm.fit(X_train_np, y_train_np)

    linear = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42)
    linear.fit(X_train_np, y_train_np)

    # 2. Run Sweep
    for r in ratios:
        # Occlude
        X_test_occ_np = apply_feature_occlusion(X_test_np, r)
        X_test_occ_tensor = torch.tensor(X_test_occ_np, dtype=torch.float32).to(
            graph_model.device
        )

        # NCM
        ncm_acc = accuracy_score(y_test_np, ncm.predict(X_test_occ_np))
        ncm_accs.append(ncm_acc)

        # Linear (Upper Bound)
        lin_acc = accuracy_score(y_test_np, linear.predict(X_test_occ_np))
        lin_accs.append(lin_acc)

        # TQM (hybrid dual-memory system)
        tqm_preds = (
            predict_dual_system(graph_model, X_test_occ_tensor, alpha=alpha)
            .cpu()
            .numpy()
        )
        tqm_acc = accuracy_score(y_test_np, tqm_preds)
        tqm_accs.append(tqm_acc)

        print(
            f"   [Occ {int(r * 100)}%] TQM: {tqm_acc:.1%} | NCM: {ncm_acc:.1%} | Linear: {lin_acc:.1%}"
        )

    # --- GENERATE FIGURE 1 (SVG + PNG) ---
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(7, 5))

    # Plot TQM (Hero)
    plt.plot(
        ratios,
        [a * 100 for a in tqm_accs],
        "o-",
        color="#2ca02c",
        linewidth=3,
        markersize=8,
        label="TQM (Ours)",
    )
    # Plot NCM (Baseline)
    plt.plot(
        ratios,
        [a * 100 for a in ncm_accs],
        "s--",
        color="#d62728",
        linewidth=2,
        markersize=6,
        label="Global NCM",
    )
    # Plot Linear (Upper Bound)
    plt.plot(
        ratios,
        [a * 100 for a in lin_accs],
        "^:",
        color="#1f77b4",
        linewidth=2,
        markersize=6,
        label="Linear Upper Bound",
    )

    plt.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
    plt.title("Robustness to Partial Observability", fontsize=14, fontweight="bold")
    plt.xlabel("Occlusion Ratio", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plt.savefig(f"outputs/fig1_occlusion_{dataset_name}.png", dpi=300)
    plt.savefig(f"outputs/fig1_occlusion_{dataset_name}.svg", format="svg")  # SVG
    print("✅ Figure 1 saved (PNG & SVG)")

    # Return metrics for aggregation in main.py
    return tqm_accs, ncm_accs, lin_accs


# ==============================================================================
# ICML FIGURE 2: QUALITATIVE t-SNE
# ==============================================================================


def generate_figure2_tsne(graph_model, X_test, y_test, dataset_name, target_class=0):
    print("\n🎨 Generating Figure 2 (Qualitative t-SNE)...")

    X_test_np = X_test.cpu().numpy() if torch.is_tensor(X_test) else X_test
    y_test_np = y_test.cpu().numpy() if torch.is_tensor(y_test) else y_test

    # 1. Select Query
    class_indices = np.where(y_test_np == target_class)[0]
    if len(class_indices) == 0:
        return
    query_idx = class_indices[0]
    query_clean = X_test_np[query_idx]
    query_occ = apply_feature_occlusion(query_clean.reshape(1, -1), 0.75)[0]

    # 2. Gather Points
    if hasattr(graph_model, "_get_proto"):
        ncm_centroid = graph_model._get_proto(target_class).detach().cpu().numpy()
    else:
        ncm_centroid = graph_model.prototypes[target_class].detach().cpu().numpy()
    bg_samples = X_test_np[class_indices[:100]]
    num_bg = len(bg_samples)

    # 3. Reconstruct TQM Nodes
    num_tqm = min(num_bg, 50)
    graph_model.eval()
    with torch.no_grad():
        bg_tensor = torch.tensor(bg_samples[:num_tqm], dtype=torch.float32).to(
            graph_model.device
        )
        codes = graph_model.quantize(bg_tensor)

        codebooks = graph_model.codebooks
        # BioEpisodicGraph exposes a single global node bank; ContinualGraph exposes chunk codebooks.
        if len(codebooks) == 1:
            tqm_nodes = codebooks[0][codes[:, 0]].cpu().numpy()
        else:
            n_parts = min(len(codebooks), codes.shape[1])
            tqm_nodes_list = []
            for c in range(n_parts):
                vecs = codebooks[c][codes[:, c]]
                tqm_nodes_list.append(vecs)
            tqm_nodes = torch.cat(tqm_nodes_list, dim=1).cpu().numpy()

    # 4. Concatenate & t-SNE
    all_vecs = np.vstack(
        [
            bg_samples,
            tqm_nodes,
            ncm_centroid.reshape(1, -1),
            query_clean.reshape(1, -1),
            query_occ.reshape(1, -1),
        ]
    )
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_2d = tsne.fit_transform(all_vecs)

    # 5. Plot
    idx_tqm_end = num_bg + num_tqm
    idx_ncm = idx_tqm_end

    plt.figure(figsize=(8, 8))
    # Background
    plt.scatter(
        tsne_2d[:num_bg, 0],
        tsne_2d[:num_bg, 1],
        c="gray",
        alpha=0.1,
        label="Class Samples",
    )
    # TQM Nodes
    plt.scatter(
        tsne_2d[num_bg:idx_tqm_end, 0],
        tsne_2d[num_bg:idx_tqm_end, 1],
        c="#2ca02c",
        s=50,
        alpha=0.6,
        label="TQM Visual Words",
    )
    # NCM Centroid
    plt.scatter(
        tsne_2d[idx_ncm, 0],
        tsne_2d[idx_ncm, 1],
        c="#d62728",
        s=200,
        marker="X",
        edgecolors="black",
        label="Global NCM",
    )
    # Queries
    plt.scatter(
        tsne_2d[idx_ncm + 1, 0],
        tsne_2d[idx_ncm + 1, 1],
        c="blue",
        s=150,
        marker="*",
        edgecolors="white",
        label="Clean Query",
    )
    plt.scatter(
        tsne_2d[idx_ncm + 2, 0],
        tsne_2d[idx_ncm + 2, 1],
        c="orange",
        s=150,
        marker="*",
        edgecolors="black",
        label="Occluded Query",
    )

    plt.legend(loc="upper right")
    plt.title("TQM Manifold Density vs. Global Centroid")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.savefig(f"outputs/tsne_manifold_{dataset_name}.png", dpi=300)
    plt.savefig(f"outputs/tsne_manifold_{dataset_name}.svg", format="svg")  # SVG
    print("✅ t-SNE saved (PNG & SVG)")


def compare_interpretability(model, features, labels, dataset, num_samples=2000):
    print(f"\n🧪 Starting Interpretability Visualization (Using Trained Model)...")

    num_samples = min(num_samples, len(features))
    X = torch.tensor(features[:num_samples], dtype=torch.float32).to(Config.DEVICE)
    y = labels[:num_samples]

    np.random.seed(42)
    valid_classes, counts = np.unique(y, return_counts=True)
    target_cls = valid_classes[np.argmax(counts)]

    cls_indices = np.where(y == target_cls)[0]
    query_idx = np.random.choice(cls_indices)
    query_vec_clean = X[query_idx].unsqueeze(0)

    model.eval()
    prototypes_indices = []
    with torch.no_grad():
        codes = model.quantize(query_vec_clean)
        codebooks = model.codebooks
        if len(codebooks) == 1:
            active_code = codes[0, 0].item()
            code_vec = codebooks[0][active_code].unsqueeze(0)
            dists = torch.cdist(X, code_vec).flatten()
            prototypes_indices.append(torch.argmin(dists).item())
        else:
            n_chunks = Config.N_CHUNKS
            mask_occluded = [True] * (n_chunks // 2) + [False] * (
                n_chunks - (n_chunks // 2)
            )
            for c in range(n_chunks):
                if not mask_occluded[c]:
                    continue

                active_code = codes[0, c].item()
                chunk_dim = Config.CHUNK_DIM
                start = c * chunk_dim
                end = (c + 1) * chunk_dim
                dataset_chunk = X[:, start:end]
                code_vec = codebooks[c][active_code].unsqueeze(0)

                dists = torch.cdist(dataset_chunk, code_vec).flatten()
                nearest_idx = torch.argmin(dists).item()
                prototypes_indices.append(nearest_idx)

    def get_img(idx):
        img_tensor, _ = dataset[idx]
        img = img_tensor.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        return np.clip(img, 0, 1)

    fig, axes = plt.subplots(1, len(prototypes_indices) + 1, figsize=(15, 5))

    q_img = get_img(query_idx)
    h, w, _ = q_img.shape
    q_img[h // 2 :, :, :] = 0

    axes[0].imshow(q_img)
    axes[0].set_title(f"Query\n(Class {target_cls})")
    axes[0].axis("off")

    for i, proto_idx in enumerate(prototypes_indices):
        p_img = get_img(proto_idx)
        axes[i + 1].imshow(p_img)
        axes[i + 1].set_title(f"Vote from\nChunk {i}\n(Class {y[proto_idx]})")
        color = "green" if y[proto_idx] == target_cls else "red"
        for spine in axes[i + 1].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        axes[i + 1].axis("off")

    plt.suptitle(f"Explainable Inference: Voting with Visual Words", fontsize=16)
    plt.tight_layout()
    plt.savefig("outputs/interpretability_real.png")
    print("✅ Saved REAL interpretability plot to outputs/interpretability_real.png")


# def apply_feature_occlusion(features, p_occlusion):
#     if p_occlusion <= 0.0: return features
#     X_occ = features.copy()
#     N, D = X_occ.shape
#     n_mask = int(D * p_occlusion)
#     if n_mask > 0: X_occ[:, -n_mask:] = 0.0
#     return X_occ

# def run_occlusion_experiment(graph, X_train, y_train, X_test, y_test, evaluate_graph_fn):
#     print("\n🛡️ Starting Occlusion Robustness Stress Test...")

#     X_train_np = X_train if isinstance(X_train, np.ndarray) else X_train.cpu().numpy()
#     y_train_np = y_train if isinstance(y_train, np.ndarray) else y_train.cpu().numpy()
#     X_test_np  = X_test  if isinstance(X_test, np.ndarray) else X_test.cpu().numpy()
#     y_test_np  = y_test  if isinstance(y_test, np.ndarray) else y_test.cpu().numpy()

#     ncm = NearestCentroid()
#     ncm.fit(X_train_np, y_train_np)

#     linear = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
#     linear.fit(X_train_np, y_train_np)

#     levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
#     acc_npgm, acc_ncm, acc_lin = [], [], []

#     print(f"   Testing levels: {levels}")

#     for p in levels:
#         X_test_corrupt_np = apply_feature_occlusion(X_test_np, p)
#         score_ncm = ncm.score(X_test_corrupt_np, y_test_np)
#         score_lin = linear.score(X_test_corrupt_np, y_test_np)

#         device = getattr(graph, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#         X_test_corrupt_tensor = torch.tensor(X_test_corrupt_np, dtype=torch.float32).to(device)
#         y_test_tensor = torch.tensor(y_test_np, dtype=torch.long).to(device)

#         graph_metrics = evaluate_graph_fn(graph, X_test_corrupt_tensor, y_test_tensor)
#         score_npgm = graph_metrics['final_accuracy']

#         acc_ncm.append(score_ncm)
#         acc_lin.append(score_lin)
#         acc_npgm.append(score_npgm)

#         print(f"   [Occlusion {int(p*100)}%] NPGM: {score_npgm:.2%} | NCM: {score_ncm:.2%} | Linear: {score_lin:.2%}")

#     plt.figure(figsize=(8, 6))
#     plt.plot(levels, [x * 100 for x in acc_npgm], 'o-', linewidth=3, label='TQM (Ours)')
#     plt.plot(levels, [x * 100 for x in acc_ncm], 's--', linewidth=2, label='NCM (Baseline)')
#     plt.plot(levels, [x * 100 for x in acc_lin], 'x:', linewidth=2, label='Linear (Parametric)')

#     plt.title("Robustness to Occlusion (The 'Hard Test')", fontsize=14)
#     plt.xlabel("Percentage of Features Masked", fontsize=12)
#     plt.ylabel("Accuracy (%)", fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend(fontsize=11)

#     path = 'outputs/robustness_curve.png'
#     plt.savefig(path)
#     print(f"✅ Robustness curve saved to {path}")

#     return acc_npgm, acc_ncm, acc_lin

# ==============================================================================
# ICML TABLE 2 & FIGURE 1: QUANTITATIVE OCCLUSION
# ==============================================================================

# def apply_feature_occlusion(features, occlusion_ratio):
#     """
#     Simulates Spatial CutOut Occlusion on DINOv2 features.
#     If features are patch-aggregated, zeroing out random subsets mimics spatial masking.
#     """
#     if occlusion_ratio <= 0.0:
#         return features.clone() if torch.is_tensor(features) else features.copy()

#     X_occ = features.clone() if torch.is_tensor(features) else features.copy()
#     N, D = X_occ.shape
#     num_masked = int(D * occlusion_ratio)

#     # Mask random dimensions for each sample to simulate random spatial CutOut
#     for i in range(N):
#         mask_indices = np.random.choice(D, num_masked, replace=False)
#         X_occ[i, mask_indices] = 0.0

#     return X_occ


def run_icml_occlusion_benchmark(
    graph_model, X_train, y_train, X_test, y_test, dataset_name, alpha=0.9
):
    """
    Generates Table 2 and Figure 1 for the ICML Paper.
    Compares NCM Baseline vs. TQM Graph under [0%, 10%, 30%, 50%, 75%] occlusion.
    """
    print("\n" + "=" * 50)
    print(" 🛡️ RUNNING ICML OCCLUSION BENCHMARK (Table 2 & Fig 1)")
    print("=" * 50)

    # Standard ICML Occlusion Ratios
    ratios = [0.0, 0.1, 0.3, 0.5, 0.75]

    tqm_accs = []
    ncm_accs = []

    # Convert Data to NumPy for Sklearn
    X_train_np = X_train.cpu().numpy() if torch.is_tensor(X_train) else X_train
    y_train_np = y_train.cpu().numpy() if torch.is_tensor(y_train) else y_train
    X_test_np = X_test.cpu().numpy() if torch.is_tensor(X_test) else X_test
    y_test_np = y_test.cpu().numpy() if torch.is_tensor(y_test) else y_test

    # THE FIX: Fit Standard NCM properly using the training data
    ncm = NearestCentroid()
    ncm.fit(X_train_np, y_train_np)

    # Run Benchmark
    for r in ratios:
        # 1. Apply Occlusion
        X_test_occ_np = apply_feature_occlusion(X_test_np, r)
        X_test_occ_tensor = torch.tensor(X_test_occ_np, dtype=torch.float32).to(
            graph_model.device
        )

        # 2. Evaluate NCM Baseline
        ncm_preds = ncm.predict(X_test_occ_np)
        ncm_acc = accuracy_score(y_test_np, ncm_preds)
        ncm_accs.append(ncm_acc)

        # 3. Evaluate TQM (hybrid dual-memory system)
        tqm_preds = (
            predict_dual_system(graph_model, X_test_occ_tensor, alpha=alpha)
            .cpu()
            .numpy()
        )
        tqm_acc = accuracy_score(y_test_np, tqm_preds)
        tqm_accs.append(tqm_acc)

        print(
            f"   [Occlusion {int(r * 100)}%] -> TQM: {tqm_acc * 100:.1f}% | NCM: {ncm_acc * 100:.1f}%"
        )

    # --- GENERATE TABLE 2 (LaTeX) ---
    print("\n📊 --- LATEX FOR TABLE 2 ---")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{l c c c c c}")
    print(r"\toprule")
    print(
        r"\textbf{Method} & \textbf{0\% (Clean)} & \textbf{10\% Mask} & \textbf{30\% Mask} & \textbf{50\% Mask} & \textbf{75\% Mask} \\"
    )
    print(r"\midrule")
    ncm_str = " & ".join([f"{a * 100:.1f}\\%" for a in ncm_accs])
    tqm_str = " & ".join([f"{a * 100:.1f}\\%" for a in tqm_accs])
    print(f"Standard NCM & {ncm_str} \\\\")
    print(f"\\textbf{{TQM (Ours)}} & \\textbf{{{tqm_str}}} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(
        r"\caption{Performance under spatial occlusion. TQM retains superior robustness.}"
    )
    print(r"\end{table}")

    # --- GENERATE FIGURE 1 (Line Graph) ---
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(
        ratios,
        [a * 100 for a in tqm_accs],
        "o-",
        color="#2ca02c",
        linewidth=3,
        markersize=8,
        label="TQM (Ours)",
    )
    plt.plot(
        ratios,
        [a * 100 for a in ncm_accs],
        "s--",
        color="#d62728",
        linewidth=2,
        markersize=6,
        label="Standard NCM",
    )

    plt.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
    plt.text(0.51, 20, "50% Occlusion\n(Abstract Claim)", color="gray", fontsize=9)

    plt.title("Fig 1: Accuracy vs. Occlusion Ratio", fontsize=14, fontweight="bold")
    plt.xlabel("Occlusion Ratio", fontsize=12)
    plt.ylabel("Classification Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/fig1_occlusion_line.png", dpi=300)
    print("✅ Figure 1 saved to 'outputs/fig1_occlusion_line.png'")


# ==============================================================================
# ICML FIGURE 2: QUALITATIVE t-SNE
# ==============================================================================

# In evaluators.py

# def generate_figure2_tsne(graph_model, ncm_centroids, X_test, y_test, target_class=0):
#     """
#     Figure 2: Shows that an Occluded Query drifts away from the Global NCM Centroid,
#     but stays within the manifold boundary of the TQM visual words.
#     """
#     print("\n🎨 Generating Figure 2 (Qualitative t-SNE Visualization)...")

#     X_test_np = X_test.cpu().numpy() if torch.is_tensor(X_test) else X_test
#     y_test_np = y_test.cpu().numpy() if torch.is_tensor(y_test) else y_test

#     # 1. Pick a single clean query from the target class
#     class_indices = np.where(y_test_np == target_class)[0]
#     if len(class_indices) == 0:
#         print(f"⚠️ No test samples found for class {target_class}. Skipping Figure 2.")
#         return

#     query_idx = class_indices[0]
#     query_clean = X_test_np[query_idx]

#     # 2. Create the 50% Occluded version
#     query_occ = apply_feature_occlusion(query_clean.reshape(1, -1), 0.75)[0]

#     # 3. Gather components to plot
#     ncm_centroid = ncm_centroids[target_class].cpu().numpy()

#     # Cap background samples at 100
#     background_samples = X_test_np[class_indices[:100]]
#     num_bg = len(background_samples)

#     # Reconstruct TQM Visual Words (Up to 50 nodes)
#     num_tqm = min(num_bg, 50)
#     graph_model.eval()
#     with torch.no_grad():
#         bg_tensor = torch.tensor(background_samples[:num_tqm], dtype=torch.float32).to(graph_model.device)
#         codes = graph_model.quantize(bg_tensor)

#         tqm_nodes_list = []
#         for c in range(Config.N_CHUNKS):
#             chunk_vectors = graph_model.codebooks[c][codes[:, c]]
#             tqm_nodes_list.append(chunk_vectors)

#         tqm_nodes = torch.cat(tqm_nodes_list, dim=1).cpu().numpy()

#     # 4. Concatenate everything for t-SNE
#     all_vectors = np.vstack([
#         background_samples,  # Length: num_bg
#         tqm_nodes,           # Length: num_tqm
#         ncm_centroid,        # Length: 1
#         query_clean,         # Length: 1
#         query_occ            # Length: 1
#     ])

#     # 5. Run t-SNE (Project 384d -> 2d)
#     tsne = TSNE(n_components=2, perplexity=min(30, len(all_vectors)-1), random_state=42)
#     tsne_2d = tsne.fit_transform(all_vectors)

#     # --- DYNAMIC INDICES FOR PLOTTING ---
#     idx_bg_end = num_bg
#     idx_tqm_end = idx_bg_end + num_tqm
#     idx_ncm = idx_tqm_end
#     idx_query = idx_ncm + 1
#     idx_occ = idx_ncm + 2

#     # 6. Plotting

#     plt.figure(figsize=(8, 6))

#     # Background Samples (Light Blue)
#     plt.scatter(tsne_2d[0:idx_bg_end, 0], tsne_2d[0:idx_bg_end, 1], c='lightblue', alpha=0.4, label='Class Distribution')

#     # TQM Graph Nodes (Green Triangles)
#     plt.scatter(tsne_2d[idx_bg_end:idx_tqm_end, 0], tsne_2d[idx_bg_end:idx_tqm_end, 1], c='green', marker='^', s=80, edgecolors='black', label='TQM Graph Nodes')

#     # NCM Global Centroid (Red Star)
#     plt.scatter(tsne_2d[idx_ncm, 0], tsne_2d[idx_ncm, 1], c='red', marker='*', s=300, edgecolors='black', label='NCM Global Centroid')

#     # Clean Query (Blue Circle)
#     plt.scatter(tsne_2d[idx_query, 0], tsne_2d[idx_query, 1], c='blue', marker='o', s=150, edgecolors='white', label='Clean Query')

#     # Occluded Query (Orange Circle)
#     plt.scatter(tsne_2d[idx_occ, 0], tsne_2d[idx_occ, 1], c='orange', marker='X', s=200, edgecolors='black', label='75% Occluded Query')

#     # Draw Arrows to show the shift
#     plt.annotate('', xy=tsne_2d[idx_occ], xytext=tsne_2d[idx_query],
#                  arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

#     plt.title("Fig 2: TQM Nodes Absorb Occlusion Shift better than Global NCM", fontsize=14, fontweight='bold')
#     plt.legend(loc='best')
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('outputs/fig2_tsne_visualization.png', dpi=300)
#     print("✅ Figure 2 saved to 'outputs/fig2_tsne_visualization.png'")

# In src/evaluators.py


def run_alpha_ablation(graph_model, X_test, y_test, dataset_name):
    """
    Generates Table 3 (Ablation) and Figure 3 (Alpha Sensitivity).
    Tests the Hybrid System with alpha ranging from 0.0 (Pure Prototype) to 1.0 (Pure Episodic).
    """
    print("\n" + "=" * 50)
    print(" 🔬 RUNNING ALPHA ABLATION (Table 3 & Fig 3)")
    print("=" * 50)

    # We test these specific values for the graph
    # 0.0 = Pure NCM (System 2)
    # 1.0 = Pure Graph (System 1)
    # 0.6 = Our proposed Hybrid
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    accuracies = []

    # Ensure tensors
    device = graph_model.device
    X_test = (
        torch.tensor(X_test, dtype=torch.float32).to(device)
        if not torch.is_tensor(X_test)
        else X_test.to(device)
    )
    y_test = (
        torch.tensor(y_test, dtype=torch.long).to(device)
        if not torch.is_tensor(y_test)
        else y_test.to(device)
    )

    print(f"{'Alpha':<10} | {'Mode':<20} | {'Accuracy':<10}")
    print("-" * 45)

    for alpha in alphas:
        # Re-use your existing dual predictor
        preds = predict_dual_system(graph_model, X_test, alpha=alpha)
        acc = (preds == y_test).float().mean().item()
        accuracies.append(acc)

        mode_name = ""
        if alpha == 0.0:
            mode_name = "(Pure NCM)"
        elif alpha == 1.0:
            mode_name = "(Pure Graph)"
        elif alpha == 0.6:
            mode_name = "(Proposed TQM)"

        print(f"{alpha:<10.1f} | {mode_name:<20} | {acc * 100:.2f}%")

    # --- GENERATE TABLE 3 (LaTeX) ---
    print("\n📊 --- LATEX FOR TABLE 3 ---")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{l c c}")
    print(r"\toprule")
    print(r"\textbf{Method} & \textbf{Configuration} & \textbf{Accuracy} \\")
    print(r"\midrule")

    # Extract key rows for the table
    acc_ncm = accuracies[0]  # alpha 0.0 — pure prototype
    acc_hyb = accuracies[6]  # alpha 0.6 (or whichever performs best)
    acc_gph = accuracies[-1]  # alpha 1.0 — pure episodic

    print(f"Pure Prototype (NCM) & $\\alpha = 0.0$ & {acc_ncm * 100:.2f}\\% \\\\")
    print(f"Pure Episodic (Graph) & $\\alpha = 1.0$ & {acc_gph * 100:.2f}\\% \\\\")
    print(
        f"\\textbf{{Hybrid TQM}} & $\\mathbf{{\\alpha = 0.6}}$ & \\textbf{{{acc_hyb * 100:.2f}\\%}} \\\\"
    )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(
        r"\caption{Ablation of dual-memory fusion. The hybrid approach outperforms individual components.}"
    )
    print(r"\end{table}")

    # --- GENERATE FIGURE 3 (Bar Chart) ---
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(8, 5))

    # Bar colors: Highlight the winner
    colors = ["gray"] * len(alphas)
    best_idx = np.argmax(accuracies)
    colors[best_idx] = "#2ca02c"  # Green for best
    colors[0] = "#d62728"  # Red for NCM
    colors[-1] = "#1f77b4"  # Blue for Graph

    bars = plt.bar(
        [str(a) for a in alphas], [a * 100 for a in accuracies], color=colors, alpha=0.8
    )

    plt.title(
        "Fig 3: Sensitivity to Fusion Parameter ($\\alpha$)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Alpha (0=NCM, 1=Graph)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(min(accuracies) * 100 - 5, max(accuracies) * 100 + 5)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    # Add text labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(f"outputs/fig3_alpha_sensitivity_{dataset_name}.png", dpi=300)
    print(f"✅ Figure 3 saved to 'outputs/fig3_alpha_sensitivity_{dataset_name}.png'")
