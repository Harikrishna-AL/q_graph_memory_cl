import argparse
import torch
import numpy as np
from src.data_utils import get_dataloader
from src.model import load_dino, extract_features, save_cached_features, load_cached_features
from src.learner import run_sequential_linear_probe, train_continual_graph, train_task_free_graph
# Import the new function here 👇
from src.evaluators import evaluate_graph, compare_interpretability, run_occlusion_experiment
from src.baselines import run_baselines
from src.config import Config

def main():
    parser = argparse.ArgumentParser(description="Graph Memory Continual Learning")
    parser.add_argument("--use_train", action="store_true", default=True, help="Use full TRAIN set")
    parser.add_argument("--words", type=int, default=512, help="Words per task per chunk")
    parser.add_argument("--imode", type=str, default='soft', help="Inference mode for evaluation")
    parser.add_argument("--dataset", type=str, default='tinyimagenet', help="Dataset to use: tinyimagenet or cifar100   ")
    args = parser.parse_args()
    
    Config.WORDS_PER_TASK = args.words
    
    print("=== 🧠 Graph Memory Project ===")
    
    # 1. Data & Features
    dataset, loader = get_dataloader(dataset_name=args.dataset, use_train_set=args.use_train)

    features, labels = load_cached_features(dataset_name=args.dataset, use_train=args.use_train)

    if features is None:
        dino = load_dino()
        features, labels = extract_features(dino, loader)
        save_cached_features(features, labels, args.dataset, args.use_train)
    else:
        print("⚡ Skipping feature extraction")

    # --- Prepare Data Split for Comparisons ---
    print("\n--- Preparing Data for Baselines & Robustness ---")
    train_indices = []
    test_indices = []
    
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
        
    # 2. Train Your Graph
    model = train_task_free_graph(X_train, y_train)
    
    # 3. Eval
    print("--- Using Mode:", args.imode, "---")
    
    # Convert Test Data to Tensor
    test_tensor = torch.tensor(X_test, dtype=torch.float32).to(Config.DEVICE)
    test_labels = torch.tensor(y_test).to(Config.DEVICE)
    
    graph_results = evaluate_graph(model, test_tensor, test_labels, mode=args.imode)
    
    # 4. Run Baselines (Clean)
    baseline_results = run_baselines(X_train, y_train, X_test, y_test)
    
    # 5. Run Robustness Experiment (Occlusion) <--- NEW STEP
    # We pass 'evaluate_graph' so the function can re-use your inference logic
    # run_occlusion_experiment(
    #     graph, X_train, y_train, X_test, y_test, 
    #     evaluate_graph_fn=evaluate_graph
    # )
    
    # 6. Interpretability Check
    compare_interpretability(model, features, labels, dataset)

    # Final Print
    print("\n🏆 FINAL SCOREBOARD (Clean Accuracy)")
    print(f"   Graph Memory (Ours): {np.mean(graph_results['clean'])*100:.2f}%")
    print(f"   NCM Baseline:        {baseline_results['NCM']*100:.2f}%")
    print(f"   Linear Baseline:     {baseline_results['Linear']*100:.2f}%")

if __name__ == "__main__":
    main()