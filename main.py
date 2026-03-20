import argparse
import random

import numpy as np
import torch

from src.baselines import run_baselines, run_statistical_baselines, run_naive_replay_baseline, run_rehearsal_mlp_baseline
from src.config import Config
from src.data_utils import get_dataloader
from src.evaluators import (
    compare_interpretability,
    evaluate_graph,
    evaluate_hybrid_system,
    generate_figure2_tsne,
    plot_memory_comparison,
    plot_memory_trace,
    predict_dual_system,
    run_alpha_ablation,
    run_icml_occlusion_benchmark,
    _run_hybrid_eval_loop,
    compute_average_accuracy,
    compute_average_forgetting,
)
from src.learner import train_bio_graph
from src.model import (
    extract_features,
    load_cached_features,
    load_backbone,
    load_dino,  # kept for backward compat; internally calls load_backbone()
    save_cached_features,
)

# ==============================================================================
# 🛠️ HELPER FUNCTIONS (Memory & Seeding)
# ==============================================================================


def set_seed(seed):
    """Ensures reproducibility across PyTorch, NumPy, and Python."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_memory_usage(model):
    """Calculates the memory footprint of the model state in MB."""
    mem_bytes = 0
    # 1. Episodic Nodes
    if hasattr(model, "nodes"):
        mem_bytes += model.nodes.element_size() * model.nodes.nelement()
    # 2. Long-term Prototypes
    if hasattr(model, "_build_proto_tensor"):
        protos, _ = model._build_proto_tensor()
        if protos is not None:
            mem_bytes += protos.element_size() * protos.nelement()
    elif hasattr(model, "prototypes"):
        mem_bytes += model.prototypes.element_size() * model.prototypes.nelement()
    # 3. Support for legacy ContinualGraph if needed
    if hasattr(model, "codebooks"):
        for cb in model.codebooks:
            mem_bytes += cb.element_size() * cb.nelement()

    mb_size = mem_bytes / (1024 * 1024)
    return mb_size


def split_stream_and_val(features, labels, val_ratio):
    """Per-task holdout split so model-selection never touches test data."""
    if val_ratio <= 0.0:
        return features, labels, None, None

    stream_idx, val_idx = [], []
    for task_id in range(Config.N_TASKS):
        start_cls = task_id * Config.CLASSES_PER_TASK
        end_cls = (task_id + 1) * Config.CLASSES_PER_TASK
        task_mask = (labels >= start_cls) & (labels < end_cls)
        task_idxs = np.where(task_mask)[0]
        if len(task_idxs) == 0:
            continue

        np.random.shuffle(task_idxs)
        n_val = int(len(task_idxs) * val_ratio)
        if len(task_idxs) > 1:
            n_val = min(max(n_val, 1), len(task_idxs) - 1)
        else:
            n_val = 0

        val_idx.extend(task_idxs[:n_val])
        stream_idx.extend(task_idxs[n_val:])

    if len(val_idx) == 0:
        return features, labels, None, None

    return (
        features[stream_idx],
        labels[stream_idx],
        features[val_idx],
        labels[val_idx],
    )


# ==============================================================================
# 🧪 SINGLE EXPERIMENT RUNNER
# ==============================================================================


def run_single_experiment(seed, features, labels, args, run_benchmarks=False):
    """
    Runs one full experiment: Data Split -> Train -> Eval.
    Returns: accuracy (float), memory (MB)
    """
    print(f"\n🌱 --- Starting Run with Seed: {seed} ---")
    set_seed(seed)

    # 1. Prepare Data Split (Dependent on Seed)
    train_indices = []
    test_indices = []

    for task_id in range(Config.N_TASKS):
        start_cls = task_id * Config.CLASSES_PER_TASK
        end_cls = (task_id + 1) * Config.CLASSES_PER_TASK
        task_mask = (labels >= start_cls) & (labels < end_cls)
        task_idxs = np.where(task_mask)[0]

        # Shuffle deterministically based on current set_seed
        np.random.shuffle(task_idxs)
        split_point = int(len(task_idxs) * Config.TRAIN_TEST_SPLIT)

        train_indices.extend(task_idxs[:split_point])
        test_indices.extend(task_idxs[split_point:])

    X_train = features[train_indices]
    y_train = labels[train_indices]
    X_test = features[test_indices]
    y_test = labels[test_indices]

    # 2. Internal train/validation split for robust model selection
    X_stream, y_stream, X_val, y_val = split_stream_and_val(
        X_train,
        y_train,
        val_ratio=args.val_ratio,
    )
    print(
        f"   📦 Stream samples: {len(X_stream)} | Val samples: {0 if X_val is None else len(X_val)}"
    )

    # 3. Train Bio-Inspired Graph (new idea)
    print("   🚀 Training Bio-Inspired Graph (Hippocampus + Cortex)...")
    
    bio_model = None
    historical_matrix = []
    memory_trace = []
    
    if getattr(args, "pure_cil", False):
        print("\n   🌀 --- PURE CIL MODE SCENARIO ---")
        for task_id in range(Config.N_TASKS):
            print(f"\n   📚 Training Task {task_id + 1}/{Config.N_TASKS}")
            start_cls = task_id * Config.CLASSES_PER_TASK
            end_cls = (task_id + 1) * Config.CLASSES_PER_TASK
            
            # Extract only this task from X_stream
            task_mask = (y_stream >= start_cls) & (y_stream < end_cls)
            X_stream_task = X_stream[task_mask]
            y_stream_task = y_stream[task_mask]
            
            # Train model sequentially
            bio_model, tr = train_bio_graph(
                X_stream_task,
                y_stream_task,
                shuffle_stream=args.shuffle_stream,
                consolidate_every=args.consolidate_every,
                lambda_val=args.consolidation_lambda,
                model=bio_model,
                verbose=False
            )
            memory_trace.extend(tr)
            
            # Tune alpha on validation set up to current task
            best_alpha = args.alpha
            if X_val is not None and len(X_val) > 0:
                t_val_mask = (y_val < end_cls)
                val_tensor = torch.tensor(X_val[t_val_mask], dtype=torch.float32).to(Config.DEVICE)
                val_labels = torch.tensor(y_val[t_val_mask], dtype=torch.long).to(Config.DEVICE)
                
                alpha_grid = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
                best_val_acc = -1.0
                for alpha in alpha_grid:
                    preds = predict_dual_system(bio_model, val_tensor, alpha=alpha)
                    val_acc = (preds == val_labels).float().mean().item()
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_alpha = alpha
                print(f"   🔎 Tuned best alpha={best_alpha:.2f} (Val Acc: {best_val_acc*100:.1f}%)")
            
            # Pure CIL Evaluate using the tuned alpha
            test_tensor = torch.tensor(X_test, dtype=torch.float32).to(Config.DEVICE)
            test_labels = torch.tensor(y_test, dtype=torch.long).to(Config.DEVICE)
            hybrid_task_accs = _run_hybrid_eval_loop(bio_model, test_tensor, test_labels, alpha=best_alpha)
            
            # Slice only up to current task
            current_accs = hybrid_task_accs[: task_id + 1]
            padded = current_accs + [0.0] * (Config.N_TASKS - len(current_accs))
            historical_matrix.append(padded)
            
            task_avg = sum(current_accs) / len(current_accs)
            print(f"   -> AIA after Task {task_id + 1}: {task_avg*100:.2f}% | Breakdown: {[f'{x*100:.1f}%' for x in current_accs]}")
    else:
        bio_model, tr = train_bio_graph(
            X_stream,
            y_stream,
            shuffle_stream=args.shuffle_stream,
            consolidate_every=args.consolidate_every,
            lambda_val=args.consolidation_lambda,
        )
        memory_trace.extend(tr)

    # Plot memory trace immediately after training so it's always saved,
    # not just during the benchmark run.
    plot_memory_trace(memory_trace, dataset_name=args.dataset)

    # Compare memory footprint against the two extreme baselines:
    # naive replay (every feature vector) and pure NCM (one mean per class).
    # Accuracy annotations are filled in after evaluation below.
    _n_classes = Config.N_TASKS * Config.CLASSES_PER_TASK
    _feature_dim = Config.FEATURE_DIM  # set by load_backbone(); backbone-agnostic

    # 4. Measure Memory Usage
    bio_mem = get_model_memory_usage(bio_model)
    mem_usage = bio_mem
    print(f"   🧠 Bio-Graph Memory Footprint: {bio_mem:.2f} MB")

    # 5. Tune alpha: episodic (System 1) vs prototype (System 2)
    selected_alpha = args.alpha
    if X_val is not None and len(X_val) > 0:
        val_tensor = torch.tensor(X_val, dtype=torch.float32).to(Config.DEVICE)
        val_labels = torch.tensor(y_val, dtype=torch.long).to(Config.DEVICE)
        alpha_grid = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
        best_alpha, best_val_acc = selected_alpha, -1.0
        for alpha in alpha_grid:
            preds = predict_dual_system(bio_model, val_tensor, alpha=alpha)
            val_acc = (preds == val_labels).float().mean().item()
            print(f"   🔎 Val alpha={alpha:.2f} -> acc={val_acc * 100:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_alpha = alpha
        selected_alpha = best_alpha
        print(
            f"   ✅ Selected alpha: {selected_alpha:.2f} | Val Acc: {best_val_acc * 100:.2f}%"
        )
    else:
        print(f"   ⚠️  No validation split available; using alpha={selected_alpha:.2f}")

    # 7. Evaluate Bio System
    test_tensor = torch.tensor(X_test, dtype=torch.float32).to(Config.DEVICE)
    test_labels = torch.tensor(y_test).to(Config.DEVICE)

    if getattr(args, "pure_cil", False):
        historical_matrix_np = np.array(historical_matrix)
        hybrid_acc = compute_average_accuracy(historical_matrix_np)
        forgetting = compute_average_forgetting(historical_matrix_np)
        print(f"\n   🏆 Pure CIL Metrics -> AIA: {hybrid_acc * 100:.2f}% | Forgetting: {forgetting * 100:.2f}%")
    else:
        hybrid_results = evaluate_hybrid_system(
            bio_model, test_tensor, test_labels, alpha=selected_alpha
        )
        hybrid_acc = hybrid_results["final_accuracy"]
        print(f"   ✅ Run Result: {hybrid_acc * 100:.2f}%")

    # Now that we have the final accuracy we can annotate the comparison plot.
    # NCM accuracy = alpha=0.0 (pure prototype, no episodic nodes).
    ncm_acc_val = None
    if X_val is not None and len(X_val) > 0:
        val_tensor_cmp = torch.tensor(X_val, dtype=torch.float32).to(Config.DEVICE)
        val_labels_cmp = torch.tensor(y_val, dtype=torch.long).to(Config.DEVICE)
        ncm_preds = predict_dual_system(bio_model, val_tensor_cmp, alpha=0.0)
        ncm_acc_val = (ncm_preds == val_labels_cmp).float().mean().item()
    plot_memory_comparison(
        memory_trace,
        n_classes=_n_classes,
        feature_dim=_feature_dim,
        dataset_name=args.dataset,
        bio_accuracy=hybrid_acc,
        ncm_accuracy=ncm_acc_val,
    )

    # 8. Run Expensive Benchmarks (Only on the first seed/main run)
    if run_benchmarks:
        print("\n   📉 Running ICML Occlusion Benchmark...")
        run_icml_occlusion_benchmark(
            bio_model,
            X_stream,
            y_stream,
            test_tensor,
            test_labels,
            dataset_name=args.dataset,
            alpha=selected_alpha,
        )

        print("   🎨 Generating t-SNE...")
        generate_figure2_tsne(
            bio_model,
            test_tensor,
            test_labels,
            target_class=0,
            dataset_name=args.dataset,
        )

        print("   🎚️ Running Alpha Ablation...")
        run_alpha_ablation(
            bio_model, test_tensor, test_labels, dataset_name=args.dataset
        )

        print("   🔍 Checking Interpretability...")
        # compare_interpretability(model, features, labels, args.dataset)

        # Run Baselines (Optional: only needed once to compare)
        print("   📏 Running Baselines...")
        run_baselines(X_train, y_train, X_test, y_test)

        print("   📊 Running Statistical Baselines...")
        run_statistical_baselines(X_train, y_train, X_test, y_test)

        print("   📼 Running Naive Replay Buffer Baseline...")
        replay_results = run_naive_replay_baseline(X_train, y_train, X_test, y_test)

        print("   🔁 Running Rehearsal MLP Baseline (ER + MLP)...")
        rehearsal_results = run_rehearsal_mlp_baseline(X_train, y_train, X_test, y_test)

        # ── Summary comparison ──────────────────────────────────────────────
        print("\n" + "─" * 57)
        print("   📊 MEMORY vs ACCURACY COMPARISON")
        print("─" * 57)
        print(f"   {'Method':<30} {'Acc':>8}  {'Memory':>12}")
        print(f"   {'─'*30} {'─'*8}  {'─'*12}")
        print(f"   {'Naive Replay (max-cosine)':<30} {replay_results['accuracy']*100:>7.2f}%  {replay_results['memory_mb']:>10.2f} MB")
        print(f"   {'Rehearsal MLP (ER+MLP)':<30} {rehearsal_results['accuracy']*100:>7.2f}%  {rehearsal_results['memory_mb']:>10.2f} MB")
        print(f"   {'Bio-Graph (Ours)':<30} {hybrid_acc*100:>7.2f}%  {bio_mem:>10.2f} MB")
        ratio_vs_naive    = replay_results['memory_mb']    / bio_mem if bio_mem > 0 else float('inf')
        ratio_vs_rehearsal = rehearsal_results['memory_mb'] / bio_mem if bio_mem > 0 else float('inf')
        print(f"   {'─'*30} {'─'*8}  {'─'*12}")
        print(f"   Bio-Graph is {ratio_vs_naive:.1f}× smaller than Naive Replay")
        print(f"   Bio-Graph is {ratio_vs_rehearsal:.1f}× smaller than Rehearsal NCM")
        print("─" * 57)

    return hybrid_acc, mem_usage


# ==============================================================================
# 🏁 MAIN
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="Graph Memory Continual Learning")
    parser.add_argument(
        "--use_train", action="store_true", default=True, help="Use full TRAIN set"
    )
    parser.add_argument(
        "--words", type=int, default=512, help="Words per task per chunk"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2",
        choices=["dinov2", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                 "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4"],
        help="Feature extraction backbone (default: dinov2).",
    )
    parser.add_argument(
        "--dataset", type=str, default="tinyimagenet", help="dataset name"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 1, 2026], help="Seeds for variance"
    )
    parser.add_argument(
        "--imode", type=str, default="soft", help="inference mode: soft/hard"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.9, help="hybrid fusion weight for graph branch"
    )
    parser.add_argument(
        "--shuffle_stream",
        action="store_true",
        help="shuffle training stream before episodic graph updates",
    )
    parser.add_argument(
        "--consolidate_every",
        type=int,
        default=1,
        help="run graph consolidation every N blocks; 0 disables consolidation",
    )
    parser.add_argument(
        "--consolidation_lambda",
        type=float,
        default=0.1,
        help="prototype blending factor used during consolidation",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="per-task validation ratio for selecting alpha in bio-only mode",
    )
    parser.add_argument(
        "--bio_proto_weight",
        type=float,
        default=0.55,
        help="fusion weight for prototype branch inside Bio graph",
    )
    parser.add_argument(
        "--bio_node_temp",
        type=float,
        default=0.08,
        help="temperature for episodic node logits",
    )
    parser.add_argument(
        "--bio_proto_temp",
        type=float,
        default=0.10,
        help="temperature for prototype logits",
    )
    parser.add_argument(
        "--bio_merge_threshold",
        type=float,
        default=0.30,
        help="distance threshold for merging new nodes into existing same-class nodes",
    )
    parser.add_argument(
        "--bio_max_nodes_per_class",
        type=int,
        default=64,
        help="upper bound for episodic nodes kept per class",
    )
    parser.add_argument(
        "--bio_kmeans_per_class",
        type=int,
        default=8,
        help="max class-conditional clusters added per block",
    )
    parser.add_argument(
        "--bio-discrim-consolidation",
        dest="bio_use_discrim_consolidation",
        action="store_true",
        help="enable discriminative sleep-phase consolidation loss",
    )
    parser.add_argument(
        "--no-bio-discrim-consolidation",
        dest="bio_use_discrim_consolidation",
        action="store_false",
        help="disable discriminative sleep-phase consolidation loss",
    )
    parser.set_defaults(bio_use_discrim_consolidation=True)
    parser.add_argument(
        "--bio_disc_steps",
        type=int,
        default=150,
        help=(
            "Adam steps for cosine-CE discriminative prototype optimization "
            "during the sleep phase (more steps → closer to linear-probe quality)"
        ),
    )
    parser.add_argument(
        "--bio_disc_lr",
        type=float,
        default=0.01,
        help=(
            "initial learning rate for the Adam + cosine-annealing optimizer "
            "used in discriminative prototype optimization"
        ),
    )
    parser.add_argument(
        "--bio_disc_margin",
        type=float,
        default=0.15,
        help="(legacy, unused) margin that was used by the old margin-SGD step",
    )
    parser.add_argument(
        "--bio_disc_neg_weight",
        type=float,
        default=1.0,
        help="(legacy, unused) neg-weight that was used by the old margin-SGD step",
    )
    parser.add_argument(
        "--bio-mahalanobis",
        dest="bio_use_mahalanobis",
        action="store_true",
        help="use diagonal Mahalanobis scoring for prototype branch",
    )
    parser.add_argument(
        "--no-bio-mahalanobis",
        dest="bio_use_mahalanobis",
        action="store_false",
        help="use cosine scoring for prototype branch",
    )
    parser.set_defaults(bio_use_mahalanobis=False)
    parser.add_argument(
        "--bio_var_eps",
        type=float,
        default=1e-3,
        help="minimum diagonal variance for Mahalanobis stability",
    )
    parser.add_argument(
        "--bio_uncertainty_momentum",
        type=float,
        default=0.95,
        help="EMA momentum for class uncertainty estimates",
    )
    parser.add_argument(
        "--bio_dynamic_budget_floor",
        type=float,
        default=0.25,
        help="minimum per-class memory floor as fraction of max nodes/class",
    )
    parser.add_argument(
        "--bio-projection",
        dest="bio_use_projection",
        action="store_true",
        help="enable learned sleep-phase metric projection",
    )
    parser.add_argument(
        "--pure-cil",
        action="store_true",
        help="evaluate model in a pure class-incremental learning setup (task-by-task AIA)",
    )
    parser.add_argument(
        "--no-bio-projection",
        dest="bio_use_projection",
        action="store_false",
        help="disable learned sleep-phase metric projection",
    )
    parser.set_defaults(bio_use_projection=True)
    parser.add_argument(
        "--bio_proj_dim",
        type=int,
        default=128,
        help="projection dimensionality for bio metric learning",
    )
    parser.add_argument(
        "--bio_proj_steps",
        type=int,
        default=30,
        help="sleep-phase optimization steps for projection",
    )
    parser.add_argument(
        "--bio_proj_lr",
        type=float,
        default=0.03,
        help="learning rate for projection optimization",
    )
    parser.add_argument(
        "--bio_proj_margin",
        type=float,
        default=0.20,
        help="hard-negative margin for projection learning",
    )
    parser.add_argument(
        "--bio_proj_ortho_reg",
        type=float,
        default=1e-2,
        help="orthogonality regularization on projection matrix",
    )
    args = parser.parse_args()

    Config.WORDS_PER_TASK = args.words
    Config.BACKBONE = args.backbone  # must be set before load_backbone()
    Config.BIO_PROTO_WEIGHT = args.bio_proto_weight
    Config.BIO_NODE_TEMP = args.bio_node_temp
    Config.BIO_PROTO_TEMP = args.bio_proto_temp
    Config.BIO_MERGE_THRESHOLD = args.bio_merge_threshold
    Config.BIO_MAX_NODES_PER_CLASS = args.bio_max_nodes_per_class
    Config.BIO_KMEANS_PER_CLASS = args.bio_kmeans_per_class
    Config.BIO_USE_DISCRIM_CONSOLIDATION = args.bio_use_discrim_consolidation
    Config.BIO_DISC_STEPS = args.bio_disc_steps
    Config.BIO_DISC_LR = args.bio_disc_lr
    Config.BIO_DISC_MARGIN = args.bio_disc_margin
    Config.BIO_DISC_NEG_WEIGHT = args.bio_disc_neg_weight
    Config.BIO_USE_MAHALANOBIS = args.bio_use_mahalanobis
    Config.BIO_VAR_EPS = args.bio_var_eps
    Config.BIO_UNCERTAINTY_MOMENTUM = args.bio_uncertainty_momentum
    Config.BIO_DYNAMIC_BUDGET_FLOOR = args.bio_dynamic_budget_floor
    Config.BIO_USE_PROJECTION = args.bio_use_projection
    Config.BIO_PROJ_DIM = args.bio_proj_dim
    Config.BIO_PROJ_STEPS = args.bio_proj_steps
    Config.BIO_PROJ_LR = args.bio_proj_lr
    Config.BIO_PROJ_MARGIN = args.bio_proj_margin
    Config.BIO_PROJ_ORTHO_REG = args.bio_proj_ortho_reg

    print("=== 🧠 Graph Memory Project: ICML Robustness Eval ===")

    # 1. Load Features ONCE (Static)
    dataset, loader = get_dataloader(
        dataset_name=args.dataset, use_train_set=args.use_train
    )
    features, labels = load_cached_features(
        dataset_name=args.dataset, use_train=args.use_train
    )

    if features is None:
        backbone = load_backbone()
        features, labels = extract_features(backbone, loader)
        save_cached_features(features, labels, args.dataset, args.use_train)
    else:
        # Even when loading from cache we need FEATURE_DIM to be set correctly
        from src.model import _BACKBONE_DIMS
        Config.FEATURE_DIM = _BACKBONE_DIMS.get(Config.BACKBONE.lower().strip(), 384)
        print("⚡ Features loaded from cache.")

    # 2. Run Experiments across Seeds
    accuracies = []
    memories = []

    print(f"\n🚀 Launching Experiments on Seeds: {args.seeds}")

    for i, seed in enumerate(args.seeds):
        # Run benchmarks only on the FIRST seed (to save time)
        do_benchmarks = i == 0

        acc, mem = run_single_experiment(
            seed, features, labels, args, run_benchmarks=do_benchmarks
        )

        accuracies.append(acc * 100)
        memories.append(mem)

    # 3. Final Aggregated Report
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_mem = np.mean(memories)

    print("\n" + "=" * 50)
    print(f"📊 FINAL ICML TABLE RESULTS ({len(args.seeds)} runs)")
    print("=" * 50)
    print(f"🏆 Hybrid Accuracy:  {mean_acc:.2f} ± {std_acc:.2f} %")
    print(f"🧠 Avg Memory Usage: {mean_mem:.2f} MB")
    print("=" * 50)
    print("Copy these numbers directly into Table 1 and Table 2.")


if __name__ == "__main__":
    main()
