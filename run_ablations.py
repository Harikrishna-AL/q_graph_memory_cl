import argparse
import json
import numpy as np
import torch
import os
from main import run_single_experiment, load_cached_features, Config, set_seed
from src.model import _BACKBONE_DIMS, load_backbone, extract_features, save_cached_features
from src.data_utils import get_dataloader

def run_experiment(backbone, dataset, **overrides):
    # --- STEP 1: FORCE RESET CONFIG ---
    set_seed(42)
    Config.BACKBONE = backbone
    Config.DATASET = dataset
    Config.FEATURE_DIM = _BACKBONE_DIMS.get(backbone, 384)
    
    # Defaults (Ensure these match the intended MAYA base state)
    Config.BIO_CONSOLIDATION_MODE = "analytic_etf"
    Config.BIO_DYNAMIC_BUDGET_FLOOR = 0.25
    Config.BIO_USE_PROJECTION = False
    Config.BIO_USE_MAHALANOBIS = False
    Config.BIO_USE_DISCRIM_CONSOLIDATION = True
    Config.BIO_MAX_NODES_PER_CLASS = 128
    
    # --- STEP 2: APPLY UPPERCASE OVERRIDES (The Fix) ---
    for k, v in overrides.items():
        if k == "alpha": continue
        # Convert lowercase 'bio_var' to uppercase 'BIO_VAR'
        upper_k = k.upper()
        setattr(Config, upper_k, v)
        print(f"   🔧 Override: Config.{upper_k} = {v}")
        
    # --- STEP 3: SYNC DATA ---
    _, _ = get_dataloader(dataset, use_train_set=True)
    N_TASKS = Config.N_TASKS
    CPT = Config.CLASSES_PER_TASK
    print(f"🧪 Tasks: {N_TASKS} | CPT: {CPT}")

    features, labels = load_cached_features(dataset, use_train=True)
    
    # Remap labels to dense 0-N
    unique_labels = np.unique(labels)
    label_map = {old_val: i for i, old_val in enumerate(unique_labels)}
    remapped_labels = np.array([label_map[l] for l in labels])
    
    # --- STEP 4: CONSTRUCT ARGS ---
    # These must reflect the CURRENT Config state
    default_alpha = 0.2 if "dinov2" in backbone.lower() else 0.6
    node_temp = 0.12 if "dinov2" in backbone.lower() else 0.08

    args = argparse.Namespace(
        dataset=dataset,
        backbone=backbone,
        pure_cil=True,
        val_ratio=0.1,
        shuffle_stream=False,
        consolidate_every=1,
        consolidation_lambda=getattr(Config, "CONSOLIDATION_LAMBDA", 0.1),
        alpha=overrides.get("alpha", default_alpha),
        bio_node_temp=node_temp,
        consolidation_mode=Config.BIO_CONSOLIDATION_MODE,
        bio_use_projection=Config.BIO_USE_PROJECTION,
        bio_use_mahalanobis=Config.BIO_USE_MAHALANOBIS,
        bio_dynamic_budget_floor=Config.BIO_DYNAMIC_BUDGET_FLOOR,
        bio_max_nodes_per_class=Config.BIO_MAX_NODES_PER_CLASS,
        use_etf=True,
        pap_weight=1.0,
        align_dim=256,
        subspace_rank=10,
        bio_use_discrim_consolidation=Config.BIO_USE_DISCRIM_CONSOLIDATION
    )
    
    # Final pass: ensure all overrides are in args too
    for k, v in overrides.items():
        setattr(args, k, v)
    
    # Run
    aia, mem = run_single_experiment(42, features, remapped_labels, args, run_benchmarks=False)
    return aia, mem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="dinov2_giant")
    parser.add_argument("--dataset", type=str, default="objectnet")
    args = parser.parse_args()
    
    os.makedirs("results", exist_ok=True)
    results = {}

    # --- 1. Component Ablation Table ---
    print("\n🚀 [1/3] Component Ablation Table...")
    results["component_ablation"] = []
    
    # a) NCM Baseline
    # Forces alpha=0 and disables sleep-phase refinement
    aia, mem = run_experiment(args.backbone, args.dataset, alpha=0.0, bio_use_discrim_consolidation=False)
    results["component_ablation"].append({"step": "NCM Baseline", "aia": aia, "mem": mem})
    
    # b) + Analytic ETF
    # System 2 only, but with refinement enabled
    aia, mem = run_experiment(args.backbone, args.dataset, alpha=0.0, bio_use_discrim_consolidation=True)
    results["component_ablation"].append({"step": "+Analytic ETF", "aia": aia, "mem": mem})
    
    # c) Full MAYA (Hybrid)
    # The default state: Hybrid System 1+2
    aia, mem = run_experiment(args.backbone, args.dataset, bio_use_discrim_consolidation=True)
    results["component_ablation"].append({"step": "Full MAYA (Hybrid)", "aia": aia, "mem": mem})

    # --- 2. Lambda Sensitivity Sweep ---
    print("\n🚀 [2/3] Lambda Sensitivity Sweep...")
    results["lambda_sweep"] = []
    for l in [0.001, 0.01, 0.1, 0.5, 0.9]:
        aia, mem = run_experiment(args.backbone, args.dataset, consolidation_lambda=l)
        results["lambda_sweep"].append({"lambda": l, "aia": aia, "mem": mem})

    # --- 3. Node Count Sweep ---
    print("\n🚀 [3/3] Node Count Sweep (K ablation)...")
    results["k_sweep"] = []
    for k in [1, 16, 32, 64, 128, 256]:
        # Overriding the uppercase BIO_MAX_NODES_PER_CLASS
        aia, mem = run_experiment(args.backbone, args.dataset, bio_max_nodes_per_class=k)
        results["k_sweep"].append({"k": k, "aia": aia, "mem": mem})

    # Final Save
    out_path = f"results/ablation_{args.backbone}_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ All MAYA ablations complete. Results saved to {out_path}")

if __name__ == "__main__":
    main()
