import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from .config import Config
from sklearn.cluster import KMeans
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier

def evaluate_graph(model, test_features, test_labels, mode='soft'):
    print("\n📊 --- Running Dual Evaluation (Clean vs Occluded) ---")
    
    n_chunks = Config.N_CHUNKS
    
    # Dynamic Masks
    mask_clean = [True] * n_chunks
    
    # Occlusion: Mask bottom 50%
    half = n_chunks // 2
    mask_occluded = [True] * half + [False] * (n_chunks - half)
    
    results = {}
    
    results['clean'] = _run_eval_loop(model, test_features, test_labels, mask_clean, "CLEAN", mode=mode)
    results['occluded'] = _run_eval_loop(model, test_features, test_labels, mask_occluded, "OCCLUDED", mode=mode)
    
    # Plotting (Optional)
    _plot_results(results['clean'], results['occluded'])
    
    return results

def _run_eval_loop(model, features, labels, mask, name, mode='soft'):
    accuracies = []
    print(f"   Evaluating: {name}...")
    
    for task_id in range(Config.N_TASKS):
        start_cls = task_id * Config.CLASSES_PER_TASK
        end_cls = (task_id + 1) * Config.CLASSES_PER_TASK
        
        task_mask = (labels >= start_cls) & (labels < end_cls)
        task_idxs = torch.where(task_mask)[0]
        
        if len(task_idxs) == 0: continue
        
        t_feats = features[task_idxs]
        t_lbls = labels[task_idxs]
        
        batch_accs = []
        bs = 100
        for i in range(0, len(t_feats), bs):
            b_in = t_feats[i:i+bs]
            b_lbl = t_lbls[i:i+bs]
            
            # Predict returns CLASS LABELS directly now
            preds = model.predict(b_in, mask, mode=mode)
            
            # Compare directly! No lookup needed.
            batch_accs.append((preds == b_lbl).float().sum().item())
            
        acc = sum(batch_accs) / len(t_feats)
        accuracies.append(acc)
        
    avg = sum(accuracies) / len(accuracies)
    print(f"   >> Average {name} Acc: {avg*100:.2f}%")
    return accuracies

def _plot_results(clean_accs, occ_accs):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # --- FIX: Dynamic Range based on actual data length ---
    n_tasks = len(clean_accs)
    task_range = range(1, n_tasks + 1)
    
    plt.plot(task_range, [a*100 for a in clean_accs], 'o-', label='Clean Input', linewidth=2)
    plt.plot(task_range, [a*100 for a in occ_accs], 'x--', label='50% Occluded', linewidth=2)
    
    plt.title(f"Graph Memory Robustness\nClean Avg: {np.mean(clean_accs)*100:.1f}% | Occluded Avg: {np.mean(occ_accs)*100:.1f}%")
    plt.xlabel("Task ID")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Ensure integer ticks on x-axis
    plt.xticks(task_range)
    
    save_path = "outputs/results_plot.png"
    plt.savefig(save_path)
    print(f"📈 Plot saved to {save_path}")

def compare_interpretability(model, features, labels, dataset, num_samples=2000):
    """
    Real Interpretability Test:
    1. Takes a query image.
    2. Occludes it.
    3. Asks the TRAINED Bayes Model to vote.
    4. Shows the 'Visual Words' (Prototypes) that cast the strongest votes.
    """
    print(f"\n🧪 Starting Interpretability Visualization (Using Trained Model)...")
    
    # 1. Setup Data
    num_samples = min(num_samples, len(features))
    # Ensure features are on the correct device for the model
    X = torch.tensor(features[:num_samples], dtype=torch.float32).to(Config.DEVICE)
    y = labels[:num_samples]
    
    # 2. Pick a Query (Ensure we pick a class the model knows)
    np.random.seed(42)
    # Find a class with enough examples
    valid_classes, counts = np.unique(y, return_counts=True)
    target_cls = valid_classes[np.argmax(counts)] # Pick most common class
    
    # Find indices of this class
    cls_indices = np.where(y == target_cls)[0]
    query_idx = np.random.choice(cls_indices)
    
    query_vec_clean = X[query_idx].unsqueeze(0) # (1, 384)
    
    # 3. Create Occlusion (Bottom 50%)
    n_chunks = Config.N_CHUNKS
    mask_occluded = [True] * (n_chunks // 2) + [False] * (n_chunks - (n_chunks // 2))
    
    print(f"   Query Index: {query_idx} | Class: {target_cls} | Mask: {mask_occluded}")

    # 4. Get Model's "Thinking"
    # We want to know WHICH words voted for the winner.
    # We essentially run 'predict' manually to extract internal states.
    
    model.eval()
    with torch.no_grad():
        # A. Quantize
        # (1, n_chunks)
        codes = model.quantize(query_vec_clean) 
        
        # B. Get the prototypes (Visual Words) for the ACTIVE chunks
        # We want to visualize what the model "sees" in the top half.
        
        # Find the nearest neighbor images in our dataset to these active codes
        # This is a reverse-lookup: "Who else looks like Code #42?"
        
        prototypes_indices = []
        
        for c in range(n_chunks):
            if not mask_occluded[c]: 
                continue # Skip occluded parts
            
            # The code the model assigned to this chunk
            active_code = codes[0, c].item()
            
            # Find closest match in the dataset for this specific chunk
            # (In a real paper, you'd show the centroid, but showing a real image is clearer)
            
            # Get all features for this chunk
            chunk_dim = Config.CHUNK_DIM
            start = c * chunk_dim
            end = (c + 1) * chunk_dim
            dataset_chunk = X[:, start:end] # (N, 64)
            
            # Get the vector for the active code from the model
            code_vec = model.codebooks[c][active_code].unsqueeze(0) # (1, 64)
            
            # Find sample in X closest to this code_vec
            dists = torch.cdist(dataset_chunk, code_vec).flatten()
            nearest_idx = torch.argmin(dists).item()
            
            prototypes_indices.append(nearest_idx)

    # 5. Visualization
    def get_img(idx):
        img_tensor, _ = dataset[idx]
        img = img_tensor.permute(1, 2, 0).numpy()
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        return np.clip(img, 0, 1)

    fig, axes = plt.subplots(1, len(prototypes_indices) + 1, figsize=(15, 5))
    
    # Show Query (Occluded)
    q_img = get_img(query_idx)
    h, w, _ = q_img.shape
    q_img[h//2:, :, :] = 0 # Visual black box
    
    axes[0].imshow(q_img)
    axes[0].set_title(f"Query\n(Class {target_cls})")
    axes[0].axis('off')
    
    # Show the "Votes" (The Prototypes)
    for i, proto_idx in enumerate(prototypes_indices):
        p_img = get_img(proto_idx)
        
        # Highlight the chunk we are looking at? 
        # Optional: Apply a mask to the prototype too to show "I matched the head"
        # For now, just show the whole image
        
        axes[i+1].imshow(p_img)
        axes[i+1].set_title(f"Vote from\nChunk {i}\n(Class {y[proto_idx]})")
        
        # Color border green if it voted correctly
        if y[proto_idx] == target_cls:
            # Add green border effect (simple way via spine color)
            for spine in axes[i+1].spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        else:
            for spine in axes[i+1].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
                
        axes[i+1].axis('off')
        
    plt.suptitle(f"Explainable Inference: Voting with Visual Words", fontsize=16)
    plt.tight_layout()
    plt.savefig('outputs/interpretability_real.png')
    print("✅ Saved REAL interpretability plot to outputs/interpretability_real.png")

# --- Helper: Apply Occlusion ---
def apply_feature_occlusion(features, p_occlusion):
    """
    Zeroes out the bottom p% of feature dimensions to simulate occlusion.
    """
    if p_occlusion <= 0.0:
        return features
        
    X_occ = features.copy()
    N, D = X_occ.shape
    # Calculate how many features to zero out
    n_mask = int(D * p_occlusion)
    
    # Mask the last n_mask dimensions (Bottom-up occlusion proxy)
    if n_mask > 0:
        X_occ[:, -n_mask:] = 0.0
        
    return X_occ

# --- Main Robustness Experiment ---
def run_occlusion_experiment(graph, X_train, y_train, X_test, y_test, evaluate_graph_fn):
    """
    Compares NPGM (Graph), NCM, and Linear Probe under increasing occlusion.
    """
    print("\n🛡️ Starting Occlusion Robustness Stress Test...")
    
    # 1. Train Baselines on Clean Data (On the fly)
    # Baselines (sklearn) need NumPy (CPU)
    print("   Training Baselines (NCM & Linear)...")
    
    # Ensure inputs are Numpy for Sklearn
    X_train_np = X_train if isinstance(X_train, np.ndarray) else X_train.cpu().numpy()
    y_train_np = y_train if isinstance(y_train, np.ndarray) else y_train.cpu().numpy()
    X_test_np  = X_test  if isinstance(X_test, np.ndarray) else X_test.cpu().numpy()
    y_test_np  = y_test  if isinstance(y_test, np.ndarray) else y_test.cpu().numpy()

    # NCM Baseline
    ncm = NearestCentroid()
    ncm.fit(X_train_np, y_train_np)
    
    # Linear Probe (Parametric Baseline)
    linear = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
    linear.fit(X_train_np, y_train_np)

    # 2. Define Occlusion Levels
    levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    acc_npgm = []
    acc_ncm = []
    acc_lin = []

    print(f"   Testing levels: {levels}")

    for p in levels:
        # Create corrupted test set (Numpy)
        X_test_corrupt_np = apply_feature_occlusion(X_test_np, p)
        
        # A. Evaluate Baselines (Expect Numpy)
        score_ncm = ncm.score(X_test_corrupt_np, y_test_np)
        score_lin = linear.score(X_test_corrupt_np, y_test_np)
        
        # B. Evaluate Graph (Expects Tensor!) -- THIS IS THE FIX --
        # Convert Numpy back to Tensor and move to correct device
        # Note: We assume 'graph' has a .device attribute or we default to 'cuda'/'cpu'
        device = getattr(graph, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        X_test_corrupt_tensor = torch.tensor(X_test_corrupt_np, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test_np, dtype=torch.long).to(device)
        
        # Pass TENSORS to your graph evaluator
        graph_metrics = evaluate_graph_fn(graph, X_test_corrupt_tensor, y_test_tensor)
        
        # Handle dict vs scalar return
        if isinstance(graph_metrics, dict):
            val = graph_metrics.get('clean', 0.0)
            score_npgm = np.mean(val) if isinstance(val, (list, np.ndarray)) else val
        else:
            score_npgm = graph_metrics

        # Store
        acc_ncm.append(score_ncm)
        acc_lin.append(score_lin)
        acc_npgm.append(score_npgm)
        
        print(f"   [Occlusion {int(p*100)}%] NPGM: {score_npgm:.2%} | NCM: {score_ncm:.2%} | Linear: {score_lin:.2%}")

    # 3. Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(levels, [x * 100 for x in acc_npgm], 'o-', linewidth=3, label='NPGM (Ours)')
    plt.plot(levels, [x * 100 for x in acc_ncm], 's--', linewidth=2, label='NCM (Baseline)')
    plt.plot(levels, [x * 100 for x in acc_lin], 'x:', linewidth=2, label='Linear (Parametric)')
    
    plt.title("Robustness to Occlusion (The 'Hard Test')", fontsize=14)
    plt.xlabel("Percentage of Features Masked", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    path = 'outputs/robustness_curve.png'
    plt.savefig(path)
    print(f"✅ Robustness curve saved to {path}")
    
    return acc_npgm, acc_ncm, acc_lin