import argparse
import torch
import numpy as np
from src.data_utils import get_dataloader
from src.model import load_dino, extract_features
from src.learner import train_continual_graph, train_task_free_graph
from src.evaluators import evaluate_graph
from src.baselines import run_baselines  # <--- NEW IMPORT
from src.config import Config

def main():
    parser = argparse.ArgumentParser(description="Graph Memory Continual Learning")
    parser.add_argument("--use_train", action="store_true", default=True, help="Use full TRAIN set")
    parser.add_argument("--words", type=int, default=512, help="Words per task per chunk")
    args = parser.parse_args()
    
    Config.WORDS_PER_TASK = args.words
    
    print("=== 🧠 Graph Memory Project ===")
    
    # 1. Data & Features
    dataset, loader = get_dataloader(use_train_set=args.use_train)
    dino = load_dino()
    features, labels = extract_features(dino, loader)
    
    # 2. Train Your Graph
    # graph, test_feats, test_lbls = train_continual_graph(features, labels)
    graph, test_feats, test_lbls = train_task_free_graph(features, labels)
    
    # 3. Eval Your Graph
    graph_results = evaluate_graph(graph, test_feats, test_lbls)
    
    # 4. Run Baselines (NEW STEP)
    # We need to reconstruct the training set for the baselines
    # (Since learner.py splits it, we need to grab the memory indices again or just use the graph's stored leaves? 
    #  Actually, let's just use the full 'features' array for a rough upper-bound comparison, 
    #  OR strictly split it again to be fair. To be fair, let's use the same Split.)
    
    print("\n--- Preparing Data for Baselines ---")
    # Re-creating the exact split used in learner.py to be scientifically fair
    train_indices = []
    test_indices = []
    
    # We perform the same split logic as in learner.py
    for task_id in range(Config.N_TASKS):
        start_cls = task_id * Config.CLASSES_PER_TASK
        end_cls = (task_id + 1) * Config.CLASSES_PER_TASK
        task_mask = (labels >= start_cls) & (labels < end_cls)
        task_idxs = np.where(task_mask)[0]
        
        np.random.seed(Config.SEED)
        np.random.shuffle(task_idxs)
        split_point = int(len(task_idxs) * Config.TRAIN_TEST_SPLIT)
        
        train_indices.extend(task_idxs[:split_point])
        test_indices.extend(task_idxs[split_point:])
        
    X_train = features[train_indices]
    y_train = labels[train_indices]
    X_test = features[test_indices]
    y_test = labels[test_indices]
    
    # Run
    baseline_results = run_baselines(X_train, y_train, X_test, y_test)
    
    # Final Comparison Print
    print("\n🏆 FINAL SCOREBOARD (Clean Accuracy)")
    print(f"   Graph Memory (Ours): {np.mean(graph_results['clean'])*100:.2f}%")
    print(f"   NCM Baseline:        {baseline_results['NCM']*100:.2f}%")
    print(f"   Linear Baseline:     {baseline_results['Linear']*100:.2f}%")

if __name__ == "__main__":
    main()