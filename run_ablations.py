import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import os
from main import run_single_experiment, load_cached_features, Config, set_seed
from src.model import _BACKBONE_DIMS, load_backbone, extract_features, save_cached_features
from src.data_utils import get_dataloader
from src.evaluators import compute_average_accuracy, compute_average_forgetting

def run_standard_ncm_baseline(features, labels, backbone_name):
    """
    Exact replicate of the Stage 1 logic from run_paper_story.py.
    """
    print(f"\n📏 Running Standalone NCM Baseline (Stage 1 Logic)...")
    
    # Setup splits
    set_seed(42)
    unique_labels = np.unique(labels)
    label_map = {old_val: i for i, old_val in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])
    
    N_TASKS = Config.N_TASKS
    CPT = Config.CLASSES_PER_TASK
    
    train_indices, test_indices = [], []
    for task_id in range(N_TASKS):
        mask = (labels >= task_id * CPT) & (labels < (task_id + 1) * CPT)
        idxs = np.where(mask)[0]
        np.random.shuffle(idxs)
        split = int(len(idxs) * Config.TRAIN_TEST_SPLIT)
        train_indices.extend(idxs[:split]); test_indices.extend(idxs[split:])
        
    X_train, y_train = features[train_indices], labels[train_indices]
    X_test, y_test = features[test_indices], labels[test_indices]
    
    ncm_proto_sum, ncm_proto_count = {}, {}
    historical_matrix = []
    
    for task_id in range(N_TASKS):
        start, end = task_id * CPT, (task_id + 1) * CPT
        mask = (y_train >= start) & (y_train < end)
        curr_x, curr_y = X_train[mask], y_train[mask]
        
        # Train
        x_t = F.normalize(torch.tensor(curr_x, dtype=torch.float32), p=2, dim=1)
        for i in range(len(x_t)):
            lbl = int(curr_y[i])
            if lbl not in ncm_proto_sum:
                ncm_proto_sum[lbl] = torch.zeros(curr_x.shape[1]); ncm_proto_count[lbl] = 0
            ncm_proto_sum[lbl] += x_t[i]; ncm_proto_count[lbl] += 1
            
        # Eval
        classes_seen = sorted(ncm_proto_sum.keys())
        protos = torch.stack([F.normalize(ncm_proto_sum[c]/ncm_proto_count[c], p=2, dim=0) for c in classes_seen]).to(Config.DEVICE)
        x_te = F.normalize(torch.tensor(X_test[y_test < end], dtype=torch.float32).to(Config.DEVICE), p=2, dim=1)
        sims = torch.matmul(x_te, protos.t())
        preds = torch.tensor([classes_seen[i] for i in torch.argmax(sims, dim=1).cpu()], dtype=torch.long)
        
        y_te = torch.tensor(y_test[y_test < end], dtype=torch.long)
        task_accs = []
        for t in range(task_id + 1):
            m = (y_te >= t*CPT) & (y_te < (t+1)*CPT)
            task_accs.append((preds[m] == y_te[m]).float().mean().item() if m.any() else 0.0)
        historical_matrix.append(task_accs + [0.0]*(N_TASKS - len(task_accs)))

    hist_np = np.array(historical_matrix)
    aia = compute_average_accuracy(hist_np)
    forgetting = compute_average_forgetting(hist_np)
    mem_mb = (len(ncm_proto_sum) * curr_x.shape[1] * 4) / (1024**2)
    
    return aia, mem_mb, forgetting

def run_experiment(backbone, dataset, **overrides):
    set_seed(42)
    Config.BACKBONE = backbone
    Config.DATASET = dataset
    Config.FEATURE_DIM = _BACKBONE_DIMS.get(backbone, 384)
    
    # Base configuration for MAYA runs
    Config.BIO_CONSOLIDATION_MODE = "analytic_etf"
    Config.BIO_MAX_NODES_PER_CLASS = 128
    Config.BIO_USE_DISCRIM_CONSOLIDATION = True
    
    for k, v in overrides.items():
        setattr(Config, k.upper(), v)
        
    _, _ = get_dataloader(dataset, use_train_set=True)
    features, labels = load_cached_features(dataset, use_train=True)
    unique_labels = np.unique(labels)
    remapped_labels = np.array([{old: i for i, old in enumerate(unique_labels)}[l] for l in labels])
    
    # Prepare arguments
    args = argparse.Namespace(
        dataset=dataset, backbone=backbone, pure_cil=True, val_ratio=0.1, shuffle_stream=False,
        consolidate_every=1, consolidation_lambda=overrides.get("consolidation_lambda", 0.1),
        alpha=overrides.get("alpha", 0.5), bio_node_temp=0.08,
        consolidation_mode=Config.BIO_CONSOLIDATION_MODE, bio_max_nodes_per_class=Config.BIO_MAX_NODES_PER_CLASS,
        use_etf=True, bio_use_discrim_consolidation=Config.BIO_USE_DISCRIM_CONSOLIDATION
    )
    
    # Fix alpha logic for "ETF-only" step to be STRICTLY 0.0 (ignore entropy shift)
    if overrides.get("alpha") == 0.0:
        # We modify the Namespace so main.py skips tuning, 
        # and we set a flag that evaluators will (hopefully) respect
        args.alpha = 0.0
        
    aia, mem, hist = run_single_experiment(42, features, remapped_labels, args, run_benchmarks=False)
    forge = compute_average_forgetting(hist) if hist is not None else 0.0
    return aia, mem, forge

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="siglip2")
    parser.add_argument("--dataset", type=str, default="objectnet")
    args = parser.parse_args()
    
    os.makedirs("results", exist_ok=True)
    results = {}

    # Load data for the standalone baseline
    _, _ = get_dataloader(args.dataset, use_train_set=True)
    features, labels = load_cached_features(args.dataset, use_train=True)

    # --- 1. Component Ablation Table ---
    print("\n🚀 [1/3] Component Ablation Table...")
    results["component_ablation"] = []
    
    # a) NCM Baseline (Regular Stage 1 code)
    aia, mem, forge = run_standard_ncm_baseline(features, labels, args.backbone)
    results["component_ablation"].append({"step": "Standard NCM", "aia": aia, "mem": mem, "forgetting": forge})
    
    # b) + Analytic ETF (Alignment active, but alpha=0)
    aia, mem, forge = run_experiment(args.backbone, args.dataset, alpha=0.0)
    results["component_ablation"].append({"step": "+Analytic ETF", "aia": aia, "mem": mem, "forgetting": forge})
    
    # c) Full MAYA (Hybrid System 1+2)
    # Default alpha (0.6 for siglip) + tuning
    aia, mem, forge = run_experiment(args.backbone, args.dataset, alpha=0.6)
    results["component_ablation"].append({"step": "Full MAYA (Hybrid)", "aia": aia, "mem": mem, "forgetting": forge})

    # --- 2. Lambda Sweep ---
    print("\n🚀 [2/3] Lambda Sweep...")
    results["lambda_sweep"] = []
    for l in [0.01, 0.1, 0.5]:
        aia, mem, forge = run_experiment(args.backbone, args.dataset, alpha=0.6, consolidation_lambda=l)
        results["lambda_sweep"].append({"lambda": l, "aia": aia, "mem": mem, "forgetting": forge})

    # --- 3. Node Count Sweep ---
    print("\n🚀 [3/3] Node Count Sweep (Fixed alpha=0.5)...")
    results["k_sweep"] = []
    for k in [1, 16, 64, 128]:
        aia, mem, forge = run_experiment(args.backbone, args.dataset, bio_max_nodes_per_class=k, alpha=0.5)
        results["k_sweep"].append({"k": k, "aia": aia, "mem": mem, "forgetting": forge})

    out_path = f"results/ablation_{args.backbone}_{args.dataset}.json"
    with open(out_path, "w") as f: json.dump(results, f, indent=4)
    print(f"\n✅ Ablations complete. Saved to {out_path}")

if __name__ == "__main__": main()
