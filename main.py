import argparse
import torch
import numpy as np
import random
import sys
from src.data_utils import get_dataloader
from src.model import load_dino, extract_features, save_cached_features, load_cached_features
from src.learner import train_task_free_graph
from src.evaluators import (evaluate_graph, 
                            evaluate_hybrid_system, 
                            run_icml_occlusion_benchmark, 
                            generate_figure2_tsne,
                            run_alpha_ablation,
                            compare_interpretability)
from src.baselines import run_baselines, run_statistical_baselines
from src.config import Config

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

def get_tqm_memory_usage(model):
    """Calculates the memory footprint of the TQM model state in MB."""
    mem_bytes = 0
    # 1. Codebooks
    for cb in model.codebooks:
        mem_bytes += cb.element_size() * cb.nelement()
    # 2. Class Priors
    mem_bytes += model.class_counts.element_size() * model.class_counts.nelement()
    # 3. Feature Counts (Sparse Dictionary)
    mem_bytes += sys.getsizeof(model.feature_counts)
    for key, val in model.feature_counts.items():
        mem_bytes += sys.getsizeof(key)
        if isinstance(val, torch.Tensor):
            mem_bytes += val.element_size() * val.nelement()
    # 4. Edge Counts
    mem_bytes += sys.getsizeof(model.edge_counts)
    for key, val in model.edge_counts.items():
        mem_bytes += sys.getsizeof(key) + sys.getsizeof(val)
        
    mb_size = mem_bytes / (1024 * 1024)
    return mb_size

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

    # 2. Train Graph (System 1)
    print("   🚀 Training Graph Memory...")
    model = train_task_free_graph(X_train, y_train)
    
    # 3. Measure Memory Usage
    mem_usage = get_tqm_memory_usage(model)
    print(f"   🧠 TQM Memory Footprint: {mem_usage:.2f} MB")

    # 4. Build System 2 (NCM Centroids)
    print("   🏗️  Building System 2 (NCM)...")
    n_classes = Config.N_TASKS * Config.CLASSES_PER_TASK
    centroids = torch.zeros((n_classes, 384), device=Config.DEVICE)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(Config.DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(Config.DEVICE)
    
    for c in range(n_classes):
        mask = (y_train_tensor == c)
        if mask.sum() > 0:
            centroids[c] = X_train_tensor[mask].mean(dim=0)
            
    # 5. Evaluate Hybrid System
    test_tensor = torch.tensor(X_test, dtype=torch.float32).to(Config.DEVICE)
    test_labels = torch.tensor(y_test).to(Config.DEVICE)
    
    hybrid_results = evaluate_hybrid_system(model, centroids, test_tensor, test_labels, alpha=0.9)
    hybrid_acc = hybrid_results['final_accuracy']
    
    print(f"   ✅ Run Result: {hybrid_acc*100:.2f}%")

    # 6. Run Expensive Benchmarks (Only on the first seed/main run)
    if run_benchmarks:
        print("\n   📉 Running ICML Occlusion Benchmark...")
        run_icml_occlusion_benchmark(model, X_train, y_train, test_tensor, test_labels, dataset_name = args.dataset)
        
        print("   🎨 Generating t-SNE...")
        generate_figure2_tsne(model, centroids, test_tensor, test_labels, target_class=0, dataset_name = args.dataset)
        
        print("   🎚️ Running Alpha Ablation...")
        run_alpha_ablation(model, centroids, test_tensor, test_labels, dataset_name = args.dataset)
        
        print("   🔍 Checking Interpretability...")
        # compare_interpretability(model, features, labels, args.dataset)
        
        # Run Baselines (Optional: only needed once to compare)
        print("   📏 Running Baselines...")
        run_baselines(X_train, y_train, X_test, y_test)

        print("   📊 Running Statistical Baselines...")
        run_statistical_baselines(X_train, y_train, X_test, y_test)

    return hybrid_acc, mem_usage

# ==============================================================================
# 🏁 MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Graph Memory Continual Learning")
    parser.add_argument("--use_train", action="store_true", default=True, help="Use full TRAIN set")
    parser.add_argument("--words", type=int, default=512, help="Words per task per chunk")
    parser.add_argument("--dataset", type=str, default='tinyimagenet', help="dataset name")
    parser.add_argument("--seeds", nargs='+', type=int, default=[42, 1, 2026], help="Seeds for variance")
    parser.add_argument("--imode", type=str, default='soft', help="inference mode: soft/hard")
    args = parser.parse_args()
    
    Config.WORDS_PER_TASK = args.words
    
    print("=== 🧠 Graph Memory Project: ICML Robustness Eval ===")
    
    # 1. Load Features ONCE (Static)
    dataset, loader = get_dataloader(dataset_name=args.dataset, use_train_set=args.use_train)
    features, labels = load_cached_features(dataset_name=args.dataset, use_train=args.use_train)

    if features is None:
        dino = load_dino()
        features, labels = extract_features(dino, loader)
        save_cached_features(features, labels, args.dataset, args.use_train)
    else:
        print("⚡ Features loaded from cache.")

    # 2. Run Experiments across Seeds
    accuracies = []
    memories = []
    
    print(f"\n🚀 Launching Experiments on Seeds: {args.seeds}")
    
    for i, seed in enumerate(args.seeds):
        # Run benchmarks only on the FIRST seed (to save time)
        do_benchmarks = (i == 0)
        
        acc, mem = run_single_experiment(seed, features, labels, args, run_benchmarks=do_benchmarks)
        
        accuracies.append(acc * 100)
        memories.append(mem)

    # 3. Final Aggregated Report
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_mem = np.mean(memories)
    
    print("\n" + "="*50)
    print(f"📊 FINAL ICML TABLE RESULTS ({len(args.seeds)} runs)")
    print("="*50)
    print(f"🏆 Hybrid Accuracy:  {mean_acc:.2f} ± {std_acc:.2f} %")
    print(f"🧠 Avg Memory Usage: {mean_mem:.2f} MB")
    print("="*50)
    print("Copy these numbers directly into Table 1 and Table 2.")

if __name__ == "__main__":
    main()