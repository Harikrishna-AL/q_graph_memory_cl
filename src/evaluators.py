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
    total_train_samples=None,
):
    """
    Compares memory requirements of three approaches over training time.

    Naive Replay Buffer  — stores every feature vector seen (upper bound)
    MAYA Graph           — actual measured memory from the training trace
    NCM class means      — stores only one mean prototype per class (lower bound)

    Layout
    ------
    Main axes  : log-scale y, all three methods visible across 3 orders of magnitude
    Inset axes : linear-scale zoom into the NCM–MAYA region (the real tradeoff)
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

    total_samples_trace = int(max(s["samples_seen"] for s in memory_trace))
    total_samples = total_train_samples if total_train_samples else total_samples_trace
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
    C_MAYA = "#1f77b4"  # blue
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
        color=C_MAYA,
        linewidth=2.2,
        markersize=4,
        zorder=5,
        label=f"MAYA — Ours  (post-sleep)",
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
        color=C_MAYA,
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
            color=C_MAYA,
            ha="center",
        )

    # ── Stats box ────────────────────────────────────────────────────────────
    stats_lines = [
        "Final memory",
        "─" * 26,
        f"Replay Buffer :  {replay_final:>7.1f} MB",
        f"MAYA          :  {bio_final:>7.1f} MB",
        f"NCM           :  {ncm_final:>7.3f} MB",
        "─" * 26,
        f"MAYA is {ratio_bio_vs_replay:>5.1f}× smaller than Replay",
        f"NCM is {ratio_bio_vs_ncm:>5.1f}× smaller than MAYA",
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
        ax_in.plot(bio_x, bio_y, "o-", color=C_MAYA, linewidth=1.6, markersize=3)
        ax_in.plot(x_full, ncm_y, "--", color=C_NCM, linewidth=1.4)

        ax_in.set_ylim(0, y_top)
        ax_in.set_xlim(0, total_samples)
        ax_in.set_xticks(x_ticks)
        ax_in.set_xticklabels(
            [_fmt_samples(v) for v in x_ticks],
            fontsize=6.5,
        )
        ax_in.set_ylabel("Memory (MB)", fontsize=7)
        ax_in.set_title("Zoom: NCM vs MAYA", fontsize=7.5, pad=3)
        ax_in.yaxis.set_tick_params(labelsize=6.5)
        ax_in.grid(True, linestyle="--", alpha=0.25)

        ax_in.fill_between(
            x_full,
            ncm_y,
            np.minimum(replay_y, y_top),
            where=(np.minimum(replay_y, y_top) >= ncm_y),
            alpha=0.06,
            color=C_MAYA,
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


# ==============================================================================
# PAPER PLOTS: Pareto, Forgetting, Per-Backbone Bar
# ==============================================================================

def plot_pareto_accuracy_memory(results_summary, dataset_name="dataset"):
    """
    Scatter plot of Accuracy (%) vs Memory (MB) for all methods.
    Draws the Pareto frontier connecting non-dominated points.

    Parameters
    ----------
    results_summary : dict  {stage_key: (aia_fraction, mem_mb, ...)}
    dataset_name    : str   used in title and filename
    """
    LABELS = {
        "0": "Vanilla NCM", "1": "NCM", "2": "Replay",
        "2b": "ER+MLP", "3": "Node-Replay", "4": "N-Node Sweep",
        "5": "MAYA", "5b": "MAYA+Linear",
    }
    COLORS = {
        "0": "#9e9e9e", "1": "#2ca02c", "2": "#d62728",
        "2b": "#e377c2", "3": "#ff7f0e", "4": "#bcbd22",
        "5": "#1f77b4", "5b": "#17becf",
    }
    MARKERS = {
        "0": "v", "1": "^", "2": "s",
        "2b": "D", "3": "p", "4": "h",
        "5": "*", "5b": "P",
    }

    if not results_summary:
        print("⚠️  No results to plot Pareto curve.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    points = []  # (mem, acc, key)
    for k, v in results_summary.items():
        aia, mem = v[0], v[1]
        acc_pct = aia * 100
        marker = MARKERS.get(k, "o")
        color = COLORS.get(k, "#333333")
        label = LABELS.get(k, k)
        sz = 220 if k == "5" else 120  # MAYA gets a bigger marker
        ax.scatter(mem, acc_pct, s=sz, marker=marker, color=color,
                   edgecolors="black", linewidths=0.8, zorder=5, label=label)
        ax.annotate(f"  {label}\\n  {acc_pct:.1f}%  |  {mem:.1f} MB",
                    xy=(mem, acc_pct), fontsize=7.5, color=color,
                    va="bottom", ha="left")
        points.append((mem, acc_pct, k))

    # Draw Pareto frontier (lower memory, higher accuracy = better)
    points.sort(key=lambda p: p[0])  # sort by memory ascending
    pareto = []
    best_acc = -1
    for mem, acc, k in points:
        if acc > best_acc:
            pareto.append((mem, acc))
            best_acc = acc
    if len(pareto) >= 2:
        px, py = zip(*pareto)
        ax.plot(px, py, "--", color="#555555", linewidth=1.5, alpha=0.6,
                label="Pareto Frontier", zorder=2)
        ax.fill_between(px, py, min(py) - 2, alpha=0.04, color="#1f77b4")

    ax.set_xlabel("Memory (MB)", fontsize=13)
    ax.set_ylabel("Average Incremental Accuracy (%)", fontsize=13)
    ax.set_title(f"Accuracy vs Memory — {dataset_name}", fontsize=15, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    out = f"outputs/pareto_{dataset_name}"
    plt.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out}.svg", format="svg", bbox_inches="tight")
    plt.close()
    print(f"✅ Pareto plot saved → {out}.png / .svg")


def plot_forgetting_curves(forgetting_data, dataset_name="dataset"):
    """
    Line chart showing per-task accuracy degradation over training.

    Parameters
    ----------
    forgetting_data : dict  {method_name: historical_matrix}
        historical_matrix is shape (N_TASKS, N_TASKS) where entry [t, k] is
        accuracy on task k after training through task t.
    dataset_name    : str
    """
    if not forgetting_data:
        print("⚠️  No forgetting data to plot.")
        return

    METHOD_COLORS = {
        "NCM": "#2ca02c", "Replay": "#d62728", "ER+MLP": "#e377c2",
        "MAYA": "#1f77b4", "MAYA+Linear": "#17becf",
    }

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 9),
                                          gridspec_kw={"hspace": 0.35})

    # ── TOP: AIA over tasks (average accuracy after each task) ────────────
    for name, hist in forgetting_data.items():
        hist = np.array(hist)
        n_tasks = hist.shape[0]
        aia_curve = []
        for t in range(n_tasks):
            # Average accuracy on tasks 0..t after training through task t
            accs = [hist[t, k] for k in range(t + 1)]
            aia_curve.append(np.mean(accs) * 100)
        color = METHOD_COLORS.get(name, None)
        ax_top.plot(range(1, n_tasks + 1), aia_curve, "o-", label=name,
                    color=color, linewidth=2, markersize=5)

    ax_top.set_xlabel("Tasks Learned", fontsize=12)
    ax_top.set_ylabel("Average Incremental Accuracy (%)", fontsize=12)
    ax_top.set_title(f"AIA Curve — {dataset_name}", fontsize=14, fontweight="bold")
    ax_top.legend(fontsize=10, loc="lower left")
    ax_top.grid(True, linestyle="--", alpha=0.3)

    # ── BOTTOM: Per-task forgetting (acc on task 0 after each step) ────────
    for name, hist in forgetting_data.items():
        hist = np.array(hist)
        n_tasks = hist.shape[0]
        # Show forgetting of the FIRST task across training
        task0_acc = [hist[t, 0] * 100 for t in range(n_tasks)]
        color = METHOD_COLORS.get(name, None)
        ax_bot.plot(range(1, n_tasks + 1), task0_acc, "s--", label=f"{name} (Task 1)",
                    color=color, linewidth=1.8, markersize=4, alpha=0.85)

    ax_bot.set_xlabel("Tasks Learned", fontsize=12)
    ax_bot.set_ylabel("Accuracy on Task 1 (%)", fontsize=12)
    ax_bot.set_title(f"Forgetting of Task 1 — {dataset_name}", fontsize=14, fontweight="bold")
    ax_bot.legend(fontsize=10, loc="lower left")
    ax_bot.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    out = f"outputs/forgetting_{dataset_name}"
    plt.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out}.svg", format="svg", bbox_inches="tight")
    plt.close()
    print(f"✅ Forgetting curves saved → {out}.png / .svg")


def plot_backbone_gain_over_ncm(backbone_results, dataset_name="dataset"):
    """
    Grouped bar chart showing MAYA's gain (Δ%) over NCM for each backbone.

    Parameters
    ----------
    backbone_results : dict  {backbone_name: {"ncm_aia": float, "maya_aia": float,
                                               "ncm_mem": float, "maya_mem": float}}
    dataset_name     : str
    """
    if not backbone_results:
        print("⚠️  No backbone results to plot.")
        return

    backbones = list(backbone_results.keys())
    ncm_accs = [backbone_results[b]["ncm_aia"] * 100 for b in backbones]
    maya_accs = [backbone_results[b]["maya_aia"] * 100 for b in backbones]
    gains = [m - n for m, n in zip(maya_accs, ncm_accs)]

    x = np.arange(len(backbones))
    w = 0.32

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={"width_ratios": [3, 2], "wspace": 0.35})

    # ── LEFT: Grouped bar chart (NCM vs MAYA) ─────────────────────────────
    bars_ncm = ax1.bar(x - w / 2, ncm_accs, w, label="NCM", color="#2ca02c",
                       edgecolor="black", linewidth=0.6, alpha=0.85)
    bars_maya = ax1.bar(x + w / 2, maya_accs, w, label="MAYA", color="#1f77b4",
                        edgecolor="black", linewidth=0.6, alpha=0.85)

    # Value labels on bars
    for bar in bars_ncm:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_maya:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([b.replace("dinov2", "DINOv2").replace("siglip", "SigLIP")
                          .replace("resnet50", "ResNet-50").replace("clip", "CLIP")
                          for b in backbones], fontsize=11)
    ax1.set_ylabel("AIA (%)", fontsize=12)
    ax1.set_title(f"NCM vs MAYA — {dataset_name}", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)
    # Set y-axis to start near the lowest value for better visibility
    y_min = max(0, min(ncm_accs + maya_accs) - 5)
    ax1.set_ylim(y_min, max(ncm_accs + maya_accs) + 4)

    # ── RIGHT: Gain bar chart (Δ%) ────────────────────────────────────────
    colors = ["#1f77b4" if g >= 0 else "#d62728" for g in gains]
    bars_g = ax2.bar(x, gains, w * 1.5, color=colors, edgecolor="black",
                     linewidth=0.6, alpha=0.85)
    for bar, g in zip(bars_g, gains):
        va = "bottom" if g >= 0 else "top"
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.15 if g >= 0 else -0.15),
                 f"{g:+.1f}%", ha="center", va=va, fontsize=9, fontweight="bold")

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([b.replace("dinov2", "DINOv2").replace("siglip", "SigLIP")
                          .replace("resnet50", "ResNet-50").replace("clip", "CLIP")
                          for b in backbones], fontsize=11)
    ax2.set_ylabel("MAYA Gain over NCM (Δ%)", fontsize=12)
    ax2.set_title(f"MAYA Improvement — {dataset_name}", fontsize=14, fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    out = f"outputs/backbone_gain_{dataset_name}"
    plt.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out}.svg", format="svg", bbox_inches="tight")
    plt.close()
    print(f"✅ Backbone gain chart saved → {out}.png / .svg")
