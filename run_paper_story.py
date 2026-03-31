import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

from src.config import Config
from src.model import load_backbone, extract_features, BioEpisodicGraph, load_cached_features, save_cached_features, _BACKBONE_DIMS
from src.data_utils import get_dataloader
from src.learner import train_bio_graph
from src.evaluators import _run_hybrid_eval_loop, compute_average_accuracy, compute_average_forgetting, predict_dual_system
from src.baselines import NaiveReplayBuffer, RehearsalMLPBuffer, NodeBankReplayBuffer
from main import get_model_memory_usage, set_seed, run_single_experiment

def parse_args():
    parser = argparse.ArgumentParser(description="5-Stage Story Runner for TQM Paper")
    parser.add_argument("--dataset", type=str, default="tinyimagenet", help="cifar100, tinyimagenet, imagenet-r, objectnet")
    parser.add_argument("--backbone", type=str, default="dinov2", help="dinov2, dinov2_giant, siglip, resnet50")
    parser.add_argument("--epochs", type=int, default=1, help="unused for this stream logic but kept for compat")
    
    # Specific Bio-Graph Hyperparameters
    parser.add_argument("--bio-projection", dest="bio_use_projection", action="store_true", help="enable learned sleep-phase metric projection")
    parser.add_argument("--no-bio-projection", dest="bio_use_projection", action="store_false", help="disable learned sleep-phase metric projection")
    parser.set_defaults(bio_use_projection=True)
    
    parser.add_argument("--bio-mahalanobis", dest="bio_use_mahalanobis", action="store_true", help="enable class-conditional mahalanobis distance")
    parser.add_argument("--no-bio-mahalanobis", dest="bio_use_mahalanobis", action="store_false", help="disable class-conditional mahalanobis distance")
    parser.set_defaults(bio_use_mahalanobis=True)
    
    parser.add_argument("--bio_dynamic_budget_floor", type=float, default=0.25, help="minimum per-class memory floor as fraction of max nodes/class")
    
    parser.add_argument("--bio-discrim-consolidation", dest="bio_use_discrim_consolidation", action="store_true", help="enable discriminative sleep-phase consolidation loss")
    parser.add_argument("--no-bio-discrim-consolidation", dest="bio_use_discrim_consolidation", action="store_false", help="disable discriminative sleep-phase consolidation loss")
    parser.set_defaults(bio_use_discrim_consolidation=True)

    parser.add_argument("--consolidation_mode", type=str, default="sgd", choices=["sgd", "analytic", "nc_align", "analytic_etf"],
                        help="Refinement mode: 'sgd', 'analytic', 'nc_align', or 'analytic_etf'")
    
    parser.add_argument("--pap_weight", type=float, default=1.0, help="Weight for Push term in nc_align mode")
    parser.add_argument("--align_dim", type=int, default=256, help="Dimensionality of alignment layer")
    parser.add_argument("--subspace_rank", type=int, default=10, help="Rank for class manifold PCA")

    parser.add_argument("--use_etf", action="store_true", help="Anchor prototypes to a fixed Equiangular Tight Frame (ETF)")

    parser.add_argument("--stages", nargs="+", type=str, default=["1", "2", "2b", "3", "4", "5", "5b"], 
                        help="Stages to run: 1 (NCM), 2 (Replay), 2b (ER+MLP), 3 (Nodes), 4 (Sweeps), 5 (TQM Full), 5b (TQM+Linear)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("==================================================")
    print(" 📖 RUNNING THE 5-STAGE TQM PAPER NARRATIVE 📖")
    print(f" Dataset: {args.dataset.upper()} | Backbone: {args.backbone.upper()}")
    print(f" Requested Stages: {args.stages}")
    print("==================================================\n")
    
    Config.DATASET = args.dataset
    Config.BACKBONE = args.backbone
    Config.BIO_USE_PROJECTION = args.bio_use_projection
    Config.BIO_USE_MAHALANOBIS = args.bio_use_mahalanobis
    Config.BIO_DYNAMIC_BUDGET_FLOOR = args.bio_dynamic_budget_floor
    Config.BIO_USE_DISCRIM_CONSOLIDATION = args.bio_use_discrim_consolidation
    Config.BIO_CONSOLIDATION_MODE = args.consolidation_mode
    Config.BIO_PAP_WEIGHT = args.pap_weight
    Config.BIO_ALIGN_DIM = args.align_dim
    Config.BIO_USE_ETF = args.use_etf
    Config.SEED = 42
    
    # ── Backbone-Aware Alpha & Temp Selection ────────────────────────────────
    default_alpha = 0.6  # Default (more episodic weight)
    node_temp = 0.08     # Default (more "hard" voting)
    if "dinov2" in Config.BACKBONE.lower():
        default_alpha = 0.2  # DINOv2 (System 2 is very strong)
        node_temp = 0.12     # "Softer" nodes to prevent overriding prototypes
    print(f"🧪 Auto-tuned Alpha for {Config.BACKBONE}: {default_alpha}")
    print(f"🧪 Adjusted Node Temp for {Config.BACKBONE}: {node_temp}")

    set_seed(42)
    
    # 1. Load Features
    Config.FEATURE_DIM = _BACKBONE_DIMS.get(args.backbone.lower().strip(), 384)
    features, labels = load_cached_features(args.dataset, use_train=True)
    if features is None or labels is None:
        _, train_loader = get_dataloader(args.dataset, use_train_set=True)
        backbone = load_backbone()
        features, labels = extract_features(backbone, train_loader)
        save_cached_features(features, labels, args.dataset, True)
    
    actual_dim = features.shape[1]
    print(f"DEVICE: ", Config.DEVICE)
    print(f"📐 Feature shape: {features.shape} (actual_dim={actual_dim}, Config.FEATURE_DIM={Config.FEATURE_DIM})")
    if actual_dim != Config.FEATURE_DIM:
        print(f"⚠️  DIMENSION MISMATCH! Auto-correcting to {actual_dim}")
        Config.FEATURE_DIM = actual_dim
    
    # Create CIL Stream splits
    N_TASKS = Config.N_TASKS
    CPT = Config.CLASSES_PER_TASK
    
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
        
    X_train_full = features[train_indices]
    y_train_full = labels[train_indices]
    X_test_full = features[test_indices]
    y_test_full = labels[test_indices]
    
    results_summary = {}

    def run_cil_stream(name, eval_func, train_func=None):
        print(f"\n🚀 --- STAGE: {name} ---")
        st = time.time()
        historical_matrix = []
        memory_usage_mb = 0
        for task_id in range(N_TASKS):
            start_cls = task_id * CPT
            end_cls = (task_id + 1) * CPT
            train_mask = (y_train_full >= start_cls) & (y_train_full < end_cls)
            X_curr, y_curr = X_train_full[train_mask], y_train_full[train_mask]
            if train_func:
                mem = train_func(task_id, X_curr, y_curr)
                if mem: memory_usage_mb = mem
            test_mask = (y_test_full < (task_id + 1) * CPT)
            X_te, y_te = X_test_full[test_mask], y_test_full[test_mask]
            task_accs = eval_func(task_id, X_te, y_te)
            padded = task_accs + [0.0] * (N_TASKS - len(task_accs))
            historical_matrix.append(padded)
        hist_np = np.array(historical_matrix)
        aia = compute_average_accuracy(hist_np)
        print(f"✅ [{name}] AIA: {aia*100:.2f}% | Mem: {memory_usage_mb:.2f} MB")
        return aia, memory_usage_mb

    # STAGE 1
    if "1" in args.stages:
        ncm_proto_sum, ncm_proto_count = {}, {}
        def train_s1(tid, x, y):
            x_t = F.normalize(torch.tensor(x, dtype=torch.float32), p=2, dim=1)
            for i in range(len(x_t)):
                lbl = int(y[i])
                if lbl not in ncm_proto_sum:
                    ncm_proto_sum[lbl] = torch.zeros(Config.FEATURE_DIM)
                    ncm_proto_count[lbl] = 0
                ncm_proto_sum[lbl] += x_t[i]
                ncm_proto_count[lbl] += 1
            return (len(ncm_proto_sum) * Config.FEATURE_DIM * 4) / (1024**2)
        def eval_s1(tid, x, y):
            classes_seen = sorted(ncm_proto_sum.keys())
            protos = torch.stack([F.normalize(ncm_proto_sum[c]/ncm_proto_count[c], p=2, dim=0) for c in classes_seen]).to(Config.DEVICE)
            x_t = F.normalize(torch.tensor(x, dtype=torch.float32).to(Config.DEVICE), p=2, dim=1)
            sims = torch.matmul(x_t, protos.t())
            preds = torch.tensor([classes_seen[i] for i in torch.argmax(sims, dim=1).cpu()], dtype=torch.long)
            y_t = torch.tensor(y, dtype=torch.long)
            accs = []
            for t in range(tid + 1):
                mask = (y_t >= t*CPT) & (y_t < (t+1)*CPT)
                accs.append((preds[mask] == y_t[mask]).float().mean().item() if mask.any() else 0.0)
            return accs
        res = run_cil_stream("1. Standard NCM", eval_s1, train_s1)
        results_summary["1"] = res

    # STAGE 2
    if "2" in args.stages:
        buf2 = NaiveReplayBuffer()
        def train_s2(tid, x, y):
            buf2.update(x, y)
            return (len(buf2.features) * Config.FEATURE_DIM * 4) / (1024**2)
        def eval_s2(tid, x, y):
            preds = buf2.predict(x)
            y_np = np.array(y)
            return [ (preds[(y_np >= t*CPT) & (y_np < (t+1)*CPT)] == y_np[(y_np >= t*CPT) & (y_np < (t+1)*CPT)]).mean() if ((y_np >= t*CPT) & (y_np < (t+1)*CPT)).any() else 0.0 for t in range(tid+1) ]
        res = run_cil_stream("2. Basic Replay", eval_s2, train_s2)
        results_summary["2"] = res

    # STAGE 2b
    if "2b" in args.stages:
        buf2b = RehearsalMLPBuffer(input_dim=Config.FEATURE_DIM, num_classes=N_TASKS*CPT, device=Config.DEVICE)
        def train_s2b(tid, x, y):
            buf2b.update(x, y, finetune_epochs=3, batch_size=64)
            return buf2b.memory_mb()
        def eval_s2b(tid, x, y):
            preds = buf2b.predict(x)
            y_np = np.array(y)
            return [ (preds[(y_np >= t*CPT) & (y_np < (t+1)*CPT)] == y_np[(y_np >= t*CPT) & (y_np < (t+1)*CPT)]).mean() if ((y_np >= t*CPT) & (y_np < (t+1)*CPT)).any() else 0.0 for t in range(tid+1) ]
        res = run_cil_stream("2b. Optimized Replay", eval_s2b, train_s2b)
        results_summary["2b"] = res

    # STAGE 3
    if "3" in args.stages:
        s3_model = BioEpisodicGraph(input_dim=Config.FEATURE_DIM)
        def train_s3(tid, x, y):
            nonlocal s3_model
            s3_model, _ = train_bio_graph(x, y, model=s3_model, verbose=False)
            return get_model_memory_usage(s3_model)
        def eval_s3(tid, x, y):
            return _run_hybrid_eval_loop(s3_model, torch.tensor(x, dtype=torch.float32).to(Config.DEVICE), torch.tensor(y, dtype=torch.long).to(Config.DEVICE), alpha=1.0)[:tid+1]
        res = run_cil_stream("3. Node-Replay Only", eval_s3, train_s3)
        results_summary["3"] = res

    # STAGE 4
    if "4" in args.stages:
        sweeps = [0.1, 0.4, 0.8]
        best_aia, best_mem, best_f = 0, 0, 0
        for f in sweeps:
            Config.BIO_DYNAMIC_BUDGET_FLOOR = f
            m = BioEpisodicGraph(input_dim=Config.FEATURE_DIM)
            def t_s4(tid, x, y): 
                nonlocal m; m, _ = train_bio_graph(x, y, model=m, verbose=False); return get_model_memory_usage(m)
            def e_s4(tid, x, y): 
                return _run_hybrid_eval_loop(m, torch.tensor(x, dtype=torch.float32).to(Config.DEVICE), torch.tensor(y, dtype=torch.long).to(Config.DEVICE), alpha=1.0)[:tid+1]
            aia, mem = run_cil_stream(f"Sweep Floor={f}", e_s4, t_s4)
            if aia > best_aia: best_aia, best_mem, best_f = aia, mem, f
        results_summary["4"] = (best_aia, best_mem, best_f)

    # STAGE 5
    if "5" in args.stages:
        print(f"\n🚀 --- STAGE: 5. Full TQM Algorithm ---")
        s5_args = argparse.Namespace(dataset=args.dataset, backbone=args.backbone, bio_use_projection=args.bio_use_projection,
            bio_use_mahalanobis=args.bio_use_mahalanobis, bio_use_discrim_consolidation=args.bio_use_discrim_consolidation,
            bio_dynamic_budget_floor=args.bio_dynamic_budget_floor, val_ratio=0.1, pure_cil=True, shuffle_stream=False,
            consolidate_every=1, consolidation_lambda=0.1, alpha=default_alpha, bio_node_temp=node_temp,
            consolidation_mode=args.consolidation_mode, use_etf=args.use_etf,
            pap_weight=args.pap_weight, align_dim=args.align_dim, subspace_rank=args.subspace_rank)
        aia, mem = run_single_experiment(42, features, labels, s5_args, run_benchmarks=False)
        results_summary["5"] = (aia, mem)

    # STAGE 5b
    if "5b" in args.stages:
        m5b = BioEpisodicGraph(input_dim=Config.FEATURE_DIM)
        buf5b = NodeBankReplayBuffer(input_dim=Config.FEATURE_DIM, num_classes=int(np.max(labels))+1, device=Config.DEVICE)
        def t_s5b(tid, x, y):
            nonlocal m5b; m5b, _ = train_bio_graph(x, y, model=m5b, verbose=False)
            if m5b.nodes.shape[0] > 0: buf5b.update_from_nodes(m5b.nodes.detach(), m5b.node_labels.detach(), finetune_epochs=5, batch_size=256)
            return get_model_memory_usage(m5b) + (sum(p.numel()*p.element_size() for p in buf5b.model.parameters())/(1024**2))
        def e_s5b(tid, x, y):
            p = buf5b.predict(x); y_np = np.array(y)
            return [ float(np.mean(p[(y_np >= t*CPT) & (y_np < (t+1)*CPT)] == y_np[(y_np >= t*CPT) & (y_np < (t+1)*CPT)])) if ((y_np >= t*CPT) & (y_np < (t+1)*CPT)).any() else 0.0 for t in range(tid+1) ]
        res = run_cil_stream("5b. TQM + Linear Head", e_s5b, t_s5b)
        results_summary["5b"] = res

    # FINAL SUMMARY
    print("\n==================================================")
    print(" 🎯 FINAL STORYLINE RESULTS ")
    print("==================================================")
    for k in ["1", "2", "2b", "3", "4", "5", "5b"]:
        if k in results_summary:
            label = {"1":"NCM", "2":"Replay", "2b":"ER+MLP", "3":"Nodes", "4":"Sweeps", "5":"TQM Full", "5b":"TQM+Linear"}[k]
            aia, mem = results_summary[k][:2]
            print(f"{k}. {label:<20} | AIA: {aia*100:.2f}% | Mem: {mem:.1f} MB" + (f" [F={results_summary[k][2]:.2f}]" if k=="4" else ""))
    print("==================================================")

if __name__ == "__main__":
    main()
