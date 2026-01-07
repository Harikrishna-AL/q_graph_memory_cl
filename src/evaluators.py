import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from .config import Config

def evaluate_graph(graph, test_features, test_labels):
    print("\n📊 --- Running Dual Evaluation (Clean vs Occluded) ---")
    
    # Defined Masks
    mask_clean = [True] * 8
    mask_occluded = [True]*4 + [False]*4
    
    results = {}
    
    results['clean'] = _run_eval_loop(graph, test_features, test_labels, mask_clean, "CLEAN")
    results['occluded'] = _run_eval_loop(graph, test_features, test_labels, mask_occluded, "OCCLUDED")
    
    _plot_results(results['clean'], results['occluded'])
    return results

def _run_eval_loop(graph, features, labels, mask, name):
    accuracies = []
    print(f"   Evaluating: {name}...")
    
    for task_id in range(Config.N_TASKS):
        start_cls = task_id * Config.CLASSES_PER_TASK
        end_cls = (task_id + 1) * Config.CLASSES_PER_TASK
        
        # Filter Test Set for this Task
        task_mask = (labels >= start_cls) & (labels < end_cls)
        task_idxs = torch.where(task_mask)[0]
        
        if len(task_idxs) == 0: continue
        
        t_feats = features[task_idxs]
        t_lbls = labels[task_idxs]
        
        # Batch Inference
        batch_accs = []
        bs = 100
        for i in range(0, len(t_feats), bs):
            b_in = t_feats[i:i+bs]
            b_lbl = t_lbls[i:i+bs]
            
            winner_idx = graph.predict(b_in, mask)
            preds = graph.labels[winner_idx]
            batch_accs.append((preds == b_lbl).float().sum().item())
            
        acc = sum(batch_accs) / len(t_feats)
        accuracies.append(acc)
        
    avg = sum(accuracies) / len(accuracies)
    print(f"   >> Average {name} Acc: {avg*100:.2f}%")
    return accuracies

def _plot_results(clean_accs, occ_accs):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 21), [a*100 for a in clean_accs], 'o-', label='Clean Input', linewidth=2)
    plt.plot(range(1, 21), [a*100 for a in occ_accs], 'x--', label='50% Occluded', linewidth=2)
    
    plt.title(f"Graph Memory Robustness\nClean Avg: {np.mean(clean_accs)*100:.1f}% | Occluded Avg: {np.mean(occ_accs)*100:.1f}%")
    plt.xlabel("Task ID")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    save_path = "outputs/results_plot.png"
    plt.savefig(save_path)
    print(f"📈 Plot saved to {save_path}")