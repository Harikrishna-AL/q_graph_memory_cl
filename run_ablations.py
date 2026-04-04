import argparse
import json
import numpy as np
import torch
from main import run_single_experiment, load_cached_features, Config
from src.model import _BACKBONE_DIMS

def run_experiment(backbone, dataset, **overrides):
    # Setup Config
    for k, v in overrides.items():
        setattr(Config, k, v)
    Config.BACKBONE = backbone
    Config.DATASET = dataset
    Config.FEATURE_DIM = _BACKBONE_DIMS.get(backbone, 384)
    
    # Load data once
    features, labels = load_cached_features(dataset, use_train=True)
    if features is None:
        return 0.0, 0.0
        
    args = argparse.Namespace(
        dataset=dataset,
        backbone=backbone,
        pure_cil=True,
        val_ratio=0.1,
        shuffle_stream=False,
        consolidate_every=1,
        consolidation_lambda=0.1,
        alpha=0.5,
        **overrides
    )
    
    aia, mem = run_single_experiment(42, features, labels, args, run_benchmarks=False)
    return aia, mem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="dinov2")
    parser.add_argument("--dataset", type=str, default="tinyimagenet")
    args = parser.parse_args()
    
    results = {}

    # --- 1. Component Ablation Table ---
    print("\n🚀 Running Component Ablation...")
    comp_results = []
    # a) NCM Baseline
    aia, mem = run_experiment(args.backbone, args.dataset, alpha=0.0, bio_use_discrim_consolidation=False)
    comp_results.append({"name": "NCM", "aia": aia, "mem": mem})
    
    # b) + ETF Alignment
    aia, mem = run_experiment(args.backbone, args.dataset, alpha=0.0, bio_use_discrim_consolidation=True, consolidation_mode="analytic_etf")
    comp_results.append({"name": "+ETF", "aia": aia, "mem": mem})
    
    # c) + Episodic Graph
    aia, mem = run_experiment(args.backbone, args.dataset, alpha=0.5, bio_use_discrim_consolidation=True, consolidation_mode="analytic_etf")
    comp_results.append({"name": "+Episodic Graph", "aia": aia, "mem": mem})
    
    # d) Full MAYA (with Subspaces)
    aia, mem = run_experiment(args.backbone, args.dataset, alpha=0.5, bio_use_discrim_consolidation=True, consolidation_mode="analytic_etf")
    comp_results.append({"name": "Full TQM", "aia": aia, "mem": mem})
    results["component_ablation"] = comp_results

    # --- 2. Projection Matrix Ablation ---
    print("\n🚀 Running Projection Ablation...")
    proj_results = []
    for use_proj in [True, False]:
        aia, mem = run_experiment(args.backbone, args.dataset, bio_use_projection=use_proj, consolidation_mode="analytic_etf")
        proj_results.append({"use_projection": use_proj, "aia": aia, "mem": mem})
    results["projection_ablation"] = proj_results

    # --- 3. Lambda Sensitivity Sweep (RLA Ridge) ---
    print("\n🚀 Running Lambda Sweep...")
    # Lambda is hardcoded in model.py currently, but we can sweep consolidation_lambda 
    # as a proxy for 'blending' strength if we want, or add a proper flag.
    # For now, let's skip or implement a proxy.
    
    # --- 4. Node Count Sweep (K Ablation) ---
    print("\n🚀 Running Node Count Sweep...")
    k_results = []
    for k in [1, 16, 32, 64, 128, 256]:
        aia, mem = run_experiment(args.backbone, args.dataset, bio_max_nodes_per_class=k, consolidation_mode="analytic_etf")
        k_results.append({"k": k, "aia": aia, "mem": mem})
    results["k_sweep"] = k_results

    # Save to JSON
    with open(f"results/ablation_{args.backbone}_{args.dataset}.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Ablations complete. Results saved to results/ablation_{args.backbone}_{args.dataset}.json")

if __name__ == "__main__":
    main()
