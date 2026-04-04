import argparse
import json
import numpy as np
import torch
import os
from main import run_single_experiment, load_cached_features, Config, set_seed
from src.model import _BACKBONE_DIMS, load_backbone, extract_features, save_cached_features
from src.data_utils import get_dataloader

def run_experiment(backbone, dataset, **overrides):
    # Reset Config to base state for this backbone/dataset
    set_seed(42)
    Config.BACKBONE = backbone
    Config.DATASET = dataset
    Config.FEATURE_DIM = _BACKBONE_DIMS.get(backbone, 384)
    
    # Defaults from the user's specific command
    Config.BIO_CONSOLIDATION_MODE = "analytic_etf"
    Config.BIO_DYNAMIC_BUDGET_FLOOR = 0.25
    Config.BIO_USE_PROJECTION = False
    Config.BIO_USE_MAHALANOBIS = False
    
    # Apply experiment-specific overrides
    for k, v in overrides.items():
        if k == "alpha": continue # alpha handled in namespace
        setattr(Config, k, v)
        
    # Load or Extract Features
    features, labels = load_cached_features(dataset, use_train=True)
    if features is None:
        _, train_loader = get_dataloader(dataset, use_train_set=True)
        backbone_model = load_backbone()
        features, labels = extract_features(backbone_model, train_loader)
        save_cached_features(features, labels, dataset, True)

    # Re-run dataloader to sync N_TASKS and CPT
    _, _ = get_dataloader(dataset, use_train_set=True)
    N_TASKS = Config.N_TASKS
    CPT = Config.CLASSES_PER_TASK

    # ── LABEL REMAPPING (Matches run_paper_story.py exactly) ──
    unique_labels = np.unique(labels)
    label_map = {old_val: i for i, old_val in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])
    
    # Backbone-aware defaults from run_paper_story.py
    default_alpha = 0.2 if "dinov2" in backbone.lower() else 0.6
    node_temp = 0.12 if "dinov2" in backbone.lower() else 0.08

    args = argparse.Namespace(
        dataset=dataset,
        backbone=backbone,
        pure_cil=True,
        val_ratio=0.1,
        shuffle_stream=False,
        consolidate_every=1,
        consolidation_lambda=0.1,
        alpha=overrides.get("alpha", default_alpha),
        bio_node_temp=node_temp,
        consolidation_mode=Config.BIO_CONSOLIDATION_MODE,
        bio_use_projection=Config.BIO_USE_PROJECTION,
        bio_use_mahalanobis=Config.BIO_USE_MAHALANOBIS,
        bio_dynamic_budget_floor=Config.BIO_DYNAMIC_BUDGET_FLOOR,
        use_etf=True, # Implicit in analytic_etf
        pap_weight=1.0,
        align_dim=256,
        subspace_rank=10
    )
    
    # Re-map any remaining overrides to args
    for k, v in overrides.items():
        setattr(args, k, v)
    
    aia, mem = run_single_experiment(42, features, labels, args, run_benchmarks=False)
    return aia, mem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="dinov2_giant")
    parser.add_argument("--dataset", type=str, default="objectnet")
    args = parser.parse_args()
    
    os.makedirs("results", exist_ok=True)
    results = {}

    # --- 1. Component Ablation Table ---
    # NCM -> +ETF -> +Episodic Graph -> +Projection -> Full MAYA
    print("\n🚀 [1/4] Component Ablation Table...")
    results["component_ablation"] = []
    
    # a) NCM Baseline
    aia, mem = run_experiment(args.backbone, args.dataset, alpha=0.0, bio_use_discrim_consolidation=False)
    results["component_ablation"].append({"step": "NCM", "aia": aia, "mem": mem})
    
    # b) + ETF Alignment (using System 2 only)
    aia, mem = run_experiment(args.backbone, args.dataset, alpha=0.0, bio_use_discrim_consolidation=True)
    results["component_ablation"].append({"step": "+ETF", "aia": aia, "mem": mem})
    
    # c) + Episodic Graph (the default hybrid)
    aia, mem = run_experiment(args.backbone, args.dataset, alpha=0.5)
    results["component_ablation"].append({"step": "+Episodic Graph", "aia": aia, "mem": mem})
    
    # d) + Projection (Enable projection matrix)
    aia, mem = run_experiment(args.backbone, args.dataset, alpha=0.5, bio_use_projection=True)
    results["component_ablation"].append({"step": "+Projection (Full MAYA)", "aia": aia, "mem": mem})

    # --- 2. Projection Matrix Ablation ---
    print("\n🚀 [2/4] Projection Matrix Ablation...")
    results["projection_ablation"] = []
    for use_proj in [True, False]:
        aia, mem = run_experiment(args.backbone, args.dataset, bio_use_projection=use_proj)
        results["projection_ablation"].append({"use_projection": use_proj, "aia": aia, "mem": mem})

    # --- 3. Lambda Sensitivity Sweep (RLA Ridge) ---
    # We sweep 'consolidation_lambda' as the weight of the new info vs old mean
    print("\n🚀 [3/4] Lambda Sensitivity Sweep...")
    results["lambda_sweep"] = []
    for l in [0.001, 0.01, 0.1, 0.5, 0.9]:
        aia, mem = run_experiment(args.backbone, args.dataset, consolidation_lambda=l)
        results["lambda_sweep"].append({"lambda": l, "aia": aia, "mem": mem})

    # --- 4. Node Count Sweep (K Ablation) ---
    print("\n🚀 [4/4] Node Count Sweep (K ablation)...")
    results["k_sweep"] = []
    for k in [1, 16, 32, 64, 128, 256]:
        aia, mem = run_experiment(args.backbone, args.dataset, bio_max_nodes_per_class=k)
        results["k_sweep"].append({"k": k, "aia": aia, "mem": mem})

    # Final Save
    out_path = f"results/ablation_{args.backbone}_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ All MAYA ablations complete. Results saved to {out_path}")

if __name__ == "__main__":
    main()
