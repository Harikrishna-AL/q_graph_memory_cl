import argparse
import time
import numpy as np
import torch
import json
import os

from src.config import Config
from src.model import load_cached_features, _BACKBONE_DIMS
from src.data_utils import get_dataloader
from src.evaluators import compute_average_accuracy
from main import set_seed

from src.comparisons import StreamingSLDA, AnalyticCL, RanPAC

def parse_args():
    parser = argparse.ArgumentParser(description="Run SOTA Comparisons (SLDA, ACL, RanPAC)")
    parser.add_argument("--dataset", type=str, default="tinyimagenet", help="cifar100, tinyimagenet, imagenet-r, objectnet")
    parser.add_argument("--backbone", type=str, default="dinov3", help="dinov3, siglip2, resnet50")
    parser.add_argument("--ranpac_dim", type=int, default=10000, help="Projection dimension for RanPAC")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("==================================================")
    print(" 🏆 RUNNING SOTA BASELINE COMPARISONS 🏆")
    print(f" Dataset: {args.dataset.upper()} | Backbone: {args.backbone.upper()}")
    print("==================================================\n")
    
    Config.DATASET = args.dataset
    Config.BACKBONE = args.backbone
    set_seed(42)
    
    # 1. Load Features
    # Fallback for old names if needed, but we assume cached features exist
    Config.FEATURE_DIM = _BACKBONE_DIMS.get(args.backbone.lower().strip(), 384)
    if "resnet50" in args.backbone.lower():
        Config.FEATURE_DIM = 2048
    if "dinov3" in args.backbone.lower():
        Config.FEATURE_DIM = 384
    if "siglip" in args.backbone.lower():
        Config.FEATURE_DIM = 768

    features, labels = load_cached_features(args.dataset, use_train=True)
    if features is None or labels is None:
        raise ValueError("Cached features not found. Run main extraction first.")
        
    actual_dim = features.shape[1]
    if actual_dim != Config.FEATURE_DIM:
        Config.FEATURE_DIM = actual_dim
        
    # Reload dataloader to ensure Config.CLASSES_PER_TASK is updated
    _, _ = get_dataloader(args.dataset, use_train_set=True)
    N_TASKS = Config.N_TASKS
    CPT = Config.CLASSES_PER_TASK
    
    # Ensure dense labels
    unique_labels = np.unique(labels)
    label_map = {old_val: i for i, old_val in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])
    num_classes = len(unique_labels)
    
    # Create CIL Stream splits (exact replica of run_paper_story logic)
    train_indices = []
    test_indices = []
    
    for task_id in range(N_TASKS):
        start_cls = task_id * CPT
        end_cls = (task_id + 1) * CPT
        task_mask = (labels >= start_cls) & (labels < end_cls)
        task_idxs = np.where(task_mask)[0]
        
        np.random.shuffle(task_idxs)
        split_point = int(len(task_idxs) * 0.8)
        train_indices.extend(task_idxs[:split_point])
        test_indices.extend(task_idxs[split_point:])
        
    X_train_full = torch.tensor(features[train_indices], dtype=torch.float32).to(Config.DEVICE)
    y_train_full = torch.tensor(labels[train_indices], dtype=torch.long).to(Config.DEVICE)
    X_test_full = torch.tensor(features[test_indices], dtype=torch.float32).to(Config.DEVICE)
    y_test_full = torch.tensor(labels[test_indices], dtype=torch.long).to(Config.DEVICE)
    
    print(f"✅ Loaded {len(X_train_full)} train and {len(X_test_full)} test samples.")
    print(f"✅ Feature Dim: {Config.FEATURE_DIM} | Classes: {num_classes} | Tasks: {N_TASKS}")
    
    # 2. Initialize Learners
    learners = {
        "SLDA": StreamingSLDA(Config.FEATURE_DIM, num_classes, device=Config.DEVICE),
        "ACL": AnalyticCL(Config.FEATURE_DIM, num_classes, device=Config.DEVICE),
        "RanPAC": RanPAC(Config.FEATURE_DIM, num_classes, projection_dim=args.ranpac_dim, device=Config.DEVICE)
    }
    
    results_history = {name: [] for name in learners.keys()}
    
    # 3. Stream Loop
    for task_id in range(N_TASKS):
        print(f"\n📚 --- Task {task_id+1}/{N_TASKS} ---")
        start_cls = task_id * CPT
        end_cls = (task_id + 1) * CPT
        
        train_mask = (y_train_full >= start_cls) & (y_train_full < end_cls)
        X_curr, y_curr = X_train_full[train_mask], y_train_full[train_mask]
        
        test_mask = (y_test_full < end_cls)
        X_te, y_te = X_test_full[test_mask], y_test_full[test_mask]
        
        for name, learner in learners.items():
            start_time = time.time()
            learner.update(X_curr, y_curr)
            preds = learner.predict(X_te)
            
            # Evaluate per-task accuracy
            accs = []
            for t in range(task_id + 1):
                t_mask = (y_te >= t*CPT) & (y_te < (t+1)*CPT)
                if t_mask.any():
                    acc = (preds[t_mask] == y_te[t_mask]).float().mean().item()
                else:
                    acc = 0.0
                accs.append(acc)
                
            # Pad with zeros for unseen tasks
            padded = accs + [0.0] * (N_TASKS - len(accs))
            results_history[name].append(padded)
            
            print(f"   [{name}] Step Time: {time.time()-start_time:.2f}s | Current Acc: {accs[-1]*100:.1f}%")
            
    # 4. Final Summary
    print("\n==================================================")
    print(" 🎯 FINAL COMPARISON RESULTS ")
    print("==================================================")
    
    final_metrics = {}
    for name, history in results_history.items():
        hist_np = np.array(history)
        aia = compute_average_accuracy(hist_np)
        mem = learners[name].memory_mb()
        final_metrics[name] = {"AIA": aia, "Memory_MB": mem}
        print(f"{name:<15} | AIA: {aia*100:5.2f}% | Mem: {mem:6.1f} MB")
    print("==================================================")
    
    # Save to JSON
    os.makedirs("results/comparisons", exist_ok=True)
    out_path = f"results/comparisons/sota_{args.dataset}_{args.backbone}.json"
    with open(out_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    print(f"💾 Results saved to {out_path}")

if __name__ == "__main__":
    main()
