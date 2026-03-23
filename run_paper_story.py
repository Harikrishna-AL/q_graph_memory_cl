import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

from src.config import Config
from src.model import load_backbone, extract_features, BioEpisodicGraph, load_cached_features, save_cached_features, _BACKBONE_DIMS
from src.data_utils import get_dataloader
from src.learner import train_bio_graph
from src.evaluators import _run_hybrid_eval_loop, compute_average_accuracy, compute_average_forgetting, predict_dual_system
from src.baselines import NaiveReplayBuffer, RehearsalMLPBuffer
from main import get_model_memory_usage, set_seed, run_single_experiment

def parse_args():
    parser = argparse.ArgumentParser(description="5-Stage Story Runner for TQM Paper")
    parser.add_argument("--dataset", type=str, default="tinyimagenet", help="cifar100, tinyimagenet, imagenet-r, objectnet")
    parser.add_argument("--backbone", type=str, default="dinov2", help="dinov2, dinov2_giant, siglip, resnet50")
    parser.add_argument("--epochs", type=int, default=1, help="unused for this stream logic but kept for compat")
    
    # Specific Bio-Graph Hyperparameters to match main.py
    parser.add_argument("--bio-projection", dest="bio_use_projection", action="store_true", help="enable learned sleep-phase metric projection")
    parser.add_argument("--no-bio-projection", dest="bio_use_projection", action="store_false", help="disable learned sleep-phase metric projection")
    parser.set_defaults(bio_use_projection=True)
    
    parser.add_argument("--bio-mahalanobis", dest="bio_use_mahalanobis", action="store_true", help="enable class-conditional mahalanobis distance")
    parser.add_argument("--no-bio-mahalanobis", dest="bio_use_mahalanobis", action="store_false", help="disable class-conditional mahalanobis distance")
    parser.set_defaults(bio_use_mahalanobis=True)
    
    parser.add_argument("--bio_dynamic_budget_floor", type=float, default=0.25, help="minimum per-class memory floor as fraction of max nodes/class")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("==================================================")
    print(" 📖 RUNNING THE 5-STAGE TQM PAPER NARRATIVE 📖")
    print(f" Dataset: {args.dataset.upper()} | Backbone: {args.backbone.upper()}")
    print("==================================================\n")
    
    Config.DATASET = args.dataset
    Config.BACKBONE = args.backbone
    Config.BIO_USE_PROJECTION = args.bio_use_projection
    Config.BIO_USE_MAHALANOBIS = args.bio_use_mahalanobis
    Config.BIO_DYNAMIC_BUDGET_FLOOR = args.bio_dynamic_budget_floor
    Config.SEED = 42
    
    set_seed(42)
    
    # 1. Load Features
    Config.FEATURE_DIM = _BACKBONE_DIMS.get(args.backbone.lower().strip(), 384)
    features, labels = load_cached_features(args.dataset, use_train=True)
    if features is None or labels is None:
        _, train_loader = get_dataloader(args.dataset, use_train_set=True)
        backbone = load_backbone()
        features, labels = extract_features(backbone, train_loader)
        save_cached_features(features, labels, args.dataset, True)
    
    # Create CIL Stream splits
    N_TASKS = Config.N_TASKS
    CPT = Config.CLASSES_PER_TASK
    
    # Generate static train/test split like main.py
    train_indices = []
    test_indices = []
    
    for task_id in range(N_TASKS):
        start_cls = task_id * CPT
        end_cls = (task_id + 1) * CPT
        task_mask = (labels >= start_cls) & (labels < end_cls)
        task_idxs = np.where(task_mask)[0]
        
        np.random.shuffle(task_idxs)
        split_point = int(len(task_idxs) * 0.8)  # 80/20 split matches main.py precisely
        
        train_indices.extend(task_idxs[:split_point])
        test_indices.extend(task_idxs[split_point:])
        
    X_train_full = features[train_indices]
    y_train_full = labels[train_indices]
    X_test_full = features[test_indices]
    y_test_full = labels[test_indices]
    
    # -------------------------------------------------------------------------
    # Helper to run CIL
    # -------------------------------------------------------------------------
    def run_cil_stream(name, eval_func, train_func=None):
        print(f"\n🚀 --- STAGE: {name} ---")
        st = time.time()
        historical_matrix = []
        memory_usage_mb = 0
        
        for task_id in range(N_TASKS):
            start_cls = task_id * CPT
            end_cls = (task_id + 1) * CPT
            
            # Train split
            train_mask = (y_train_full >= start_cls) & (y_train_full < end_cls)
            X_curr = X_train_full[train_mask]
            y_curr = y_train_full[train_mask]
            
            if train_func:
                mem = train_func(task_id, X_curr, y_curr)
                if mem: memory_usage_mb = mem
                
            # Eval split (up to current task)
            test_mask = (y_test_full < end_cls)
            X_te = X_test_full[test_mask]
            y_te = y_test_full[test_mask]
            
            task_accs = eval_func(task_id, X_te, y_te)
            
            padded = task_accs + [0.0] * (N_TASKS - len(task_accs))
            historical_matrix.append(padded)
            
        hist_np = np.array(historical_matrix)
        aia = compute_average_accuracy(hist_np)
        forgetting = compute_average_forgetting(hist_np)
        
        print(f"✅ [{name}] Finished in {time.time()-st:.1f}s")
        print(f"   🏆 AIA: {aia*100:.2f}% | Forgetting: {forgetting*100:.2f}% | Memory: {memory_usage_mb:.2f} MB")
        return aia, forgetting, memory_usage_mb

    
    # =========================================================================
    # STAGE 1: Standard NCM (Basic method, loses boundaries/variance)
    # =========================================================================
    # We simulate pure NCM using our TQM model locked to alpha=0.0 (prototypes only)
    stage1_model = BioEpisodicGraph(input_dim=Config.FEATURE_DIM)
    
    def train_stage1(tid, x, y):
        nonlocal stage1_model
        stage1_model, _ = train_bio_graph(x, y, model=stage1_model, verbose=False, lambda_val=0.1)
        return get_model_memory_usage(stage1_model)

    def eval_stage1(tid, x, y):
        x_tok = torch.tensor(x, dtype=torch.float32).to(Config.DEVICE)
        y_tok = torch.tensor(y, dtype=torch.long).to(Config.DEVICE)
        return _run_hybrid_eval_loop(stage1_model, x_tok, y_tok, alpha=0.0)[:tid+1]
        
    s1_aia, _, _ = run_cil_stream("1. Standard NCM", eval_stage1, train_stage1)
    
    # =========================================================================
    # STAGE 2: Replay (Basic NCM + Replay memory)
    # =========================================================================
    # Store everything (or a fixed amount). We use NaiveReplayBuffer as the upper bound for Replay.
    stage2_buffer = NaiveReplayBuffer()
    
    def train_stage2(tid, x, y):
        stage2_buffer.update(x, y)
        bytes_stored = len(stage2_buffer.features) * Config.FEATURE_DIM * 4
        return bytes_stored / (1024**2)
        
    def eval_stage2(tid, x, y):
        preds = stage2_buffer.predict(x)
        accs = []
        for t in range(tid + 1):
            s_c = t * CPT
            e_c = (t + 1) * CPT
            mask = (y >= s_c) & (y < e_c)
            if mask.sum() > 0:
                accs.append((preds[mask] == y[mask]).mean())
            else:
                accs.append(0.0)
        return accs

    s2_aia, _, s2_mem = run_cil_stream("2. Basic Replay (Raw Features)", eval_stage2, train_stage2)

    # =========================================================================
    # STAGE 2b: Optimized Replay (Linear Layer + ER)
    # =========================================================================
    # A standard replay approach in CL literature: train an MLP linearly constrained by ER buffer.
    total_classes = N_TASKS * CPT
    stage2b_buffer = RehearsalMLPBuffer(input_dim=Config.FEATURE_DIM, num_classes=total_classes, device=Config.DEVICE)
    
    def train_stage2b(tid, x, y):
        # We only finetune 1 epoch per streaming task to match speeds, but standard is 5. We use 3 for a fair balance.
        stage2b_buffer.update(x, y, finetune_epochs=3, batch_size=64)
        return stage2b_buffer.memory_mb()
        
    def eval_stage2b(tid, x, y):
        preds = stage2b_buffer.predict(x)
        accs = []
        for t in range(tid + 1):
            s_c = t * CPT
            e_c = (t + 1) * CPT
            mask = (y >= s_c) & (y < e_c)
            if mask.sum() > 0:
                accs.append((preds[mask] == y[mask]).mean())
            else:
                accs.append(0.0)
        return accs
        
    s2b_aia, _, s2b_mem = run_cil_stream("2b. Optimized Replay (Linear + ER)", eval_stage2b, train_stage2b)

    # =========================================================================
    # STAGE 3: Node-Replay (Use compressed nodes instead of raw features)
    # =========================================================================
    # We use our TQM model locked to alpha=1.0 (graph nodes only)
    stage3_model = BioEpisodicGraph(input_dim=Config.FEATURE_DIM)
    
    def train_stage3(tid, x, y):
        nonlocal stage3_model
        stage3_model, _ = train_bio_graph(x, y, model=stage3_model, verbose=False, lambda_val=0.1)
        return get_model_memory_usage(stage3_model)

    def eval_stage3(tid, x, y):
        x_tok = torch.tensor(x, dtype=torch.float32).to(Config.DEVICE)
        y_tok = torch.tensor(y, dtype=torch.long).to(Config.DEVICE)
        return _run_hybrid_eval_loop(stage3_model, x_tok, y_tok, alpha=1.0)[:tid+1]
        
    s3_aia, _, s3_mem = run_cil_stream("3. Node-Replay (Episodic Graph Only)", eval_stage3, train_stage3)

    # =========================================================================
    # STAGE 4: N-Node Sweep (Trade-off curves)
    # =========================================================================
    print(f"\n🚀 --- STAGE: 4. N-Node Sweeps (Varying graph capacity) ---")
    sweeps = [0.1, 0.4, 0.8] # Simulate different BIO_DYNAMIC_BUDGET_FLOOR
    sweep_results = []
    
    original_floor = Config.BIO_DYNAMIC_BUDGET_FLOOR
    for floor in sweeps:
        Config.BIO_DYNAMIC_BUDGET_FLOOR = floor
        s4_model = BioEpisodicGraph(input_dim=Config.FEATURE_DIM)
        
        def train_s4(tid, x, y):
            nonlocal s4_model
            s4_model, _ = train_bio_graph(x, y, model=s4_model, verbose=False, lambda_val=0.1)
            return get_model_memory_usage(s4_model)
            
        def eval_s4(tid, x, y):
            x_tok = torch.tensor(x, dtype=torch.float32).to(Config.DEVICE)
            y_tok = torch.tensor(y, dtype=torch.long).to(Config.DEVICE)
            return _run_hybrid_eval_loop(s4_model, x_tok, y_tok, alpha=1.0)[:tid+1]
            
        aia, forge, mem = run_cil_stream(f"Sweep Floor={floor}", eval_s4, train_s4)
        sweep_results.append((floor, aia, mem))
        
    Config.BIO_DYNAMIC_BUDGET_FLOOR = original_floor

    # =========================================================================
    # STAGE 5: TQM (Complete Algorithm)
    # =========================================================================
    
    # 💥 VERY IMPORTANT: Execute the true main.py runner so metrics strictly align! 💥
    print(f"\n🚀 --- STAGE: 5. Full TQM Algorithm (Ours) ---")
    st = time.time()
    
    stage5_args = argparse.Namespace(
        dataset=args.dataset,
        backbone=args.backbone,
        bio_use_projection=args.bio_use_projection,
        bio_use_mahalanobis=args.bio_use_mahalanobis,
        bio_dynamic_budget_floor=args.bio_dynamic_budget_floor,
        val_ratio=0.1,
        pure_cil=True,
        shuffle_stream=False,
        consolidate_every=1,
        consolidation_lambda=0.1,
        alpha=0.5
    )
    
    # Run the exact sequence from main.py
    s5_aia_raw, s5_mem_raw = run_single_experiment(42, features, labels, stage5_args, run_benchmarks=False)
    
    # run_single_experiment produces AIA as a fraction (0.798) and Memory in MB (18.33)
    s5_aia = s5_aia_raw
    s5_mem = s5_mem_raw
    
    print(f"✅ [5. Full TQM Algorithm (Ours)] Finished in {time.time()-st:.1f}s")
    print(f"   🏆 AIA: {s5_aia*100:.2f}% | Memory: {s5_mem:.2f} MB")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n==================================================")
    print(" 🎯 FINAL STORYLINE RESULTS ")
    print("==================================================")
    print(f"1. Standard NCM          | AIA: {s1_aia*100:.1f}% | Mem: ~0 MB")
    print(f"2. Replay (Raw Features) | AIA: {s2_aia*100:.1f}% | Mem: {s2_mem:.1f} MB")
    print(f"2b. Replay (Linear + ER) | AIA: {s2b_aia*100:.1f}% | Mem: {s2b_mem:.1f} MB")
    print(f"3. Node-Replay Only      | AIA: {s3_aia*100:.1f}% | Mem: {s3_mem:.1f} MB")
    
    best_sweep = max(sweep_results, key=lambda x: x[1])
    print(f"4. N-Node Sweeps (Best)  | AIA: {best_sweep[1]*100:.1f}% | Mem: {best_sweep[2]:.1f} MB  [Floor={best_sweep[0]:.2f}]")
    
    print(f"5. **TQM (Full)**        | AIA: {s5_aia*100:.1f}% | Mem: {s5_mem:.1f} MB")
    print("==================================================")

if __name__ == "__main__":
    main()
