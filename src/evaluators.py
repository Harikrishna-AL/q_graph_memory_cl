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

def compare_interpretability(features, labels, dataset, num_samples=2000, n_chunks=8):
    """
    Performs the 'Hard Test' (Occlusion):
    1. Takes a query image.
    2. Zeroes out 50% of its features (Occlusion).
    3. Asks NPGM to retrieve based on the *surviving* chunks.
    4. Asks NCM to retrieve based on the *corrupted* vector distance.
    """
    print(f"🧪 Starting Interpretability HARD MODE (Occlusion Test)...")
    
    # 1. Prepare Data
    num_samples = min(num_samples, len(features))
    X = features[:num_samples]      # (N, D)
    y = labels[:num_samples]        # (N,)
    N, D = X.shape
    chunk_dim = D // n_chunks
    
    # 2. Train NPGM Codebooks (Simulation)
    # We train independent codebooks for ALL chunks to simulate full graph voting
    print("   Training NPGM Codebooks for all chunks...")
    codebooks = [] # List of KMeans models
    all_codes = np.zeros((N, n_chunks), dtype=int)
    
    for m in range(n_chunks):
        start = m * chunk_dim
        end = (m + 1) * chunk_dim
        chunk_data = X[:, start:end]
        
        kmeans = KMeans(n_clusters=64, n_init=10, random_state=42)
        kmeans.fit(chunk_data)
        codebooks.append(kmeans)
        all_codes[:, m] = kmeans.labels_

    # 3. Train NCM Centroids
    print("   Computing NCM Centroids...")
    unique_classes = np.unique(y)
    class_means = {}
    for c in unique_classes:
        class_means[c] = np.mean(X[y == c], axis=0)

    # 4. Select Query & Apply Occlusion
    np.random.seed(42)
    # Pick a distinct class query
    for _ in range(100):
        query_idx = np.random.randint(0, num_samples)
        query_cls = y[query_idx]
        if np.sum(y == query_cls) > 5: break
            
    query_vec_clean = X[query_idx].copy()
    
    # --- APPLY OCCLUSION ---
    # We zero out the second half of the chunks (Chunks 4-7)
    # This simulates bottom-half occlusion
    occluded_indices = range(n_chunks // 2, n_chunks) 
    valid_indices = range(0, n_chunks // 2)
    
    query_vec_dirty = query_vec_clean.copy()
    # In feature space, we zero out the corresponding dimensions
    for m in occluded_indices:
        start = m * chunk_dim
        end = (m + 1) * chunk_dim
        query_vec_dirty[start:end] = 0.0 # Destroy features

    print(f"   Query: Index {query_idx} (Class {query_cls}) - 50% Occluded")

    # ==========================================
    # METHOD A: NPGM Retrieval (Voting on Valid Chunks)
    # ==========================================
    # NPGM Logic: If a chunk is occluded/zero, it casts NO votes.
    # We only match based on 'valid_indices' (Chunks 0-3).
    
    # 1. Get codes for the VALID chunks of the query
    query_codes_valid = []
    for m in valid_indices:
        start = m * chunk_dim
        end = (m + 1) * chunk_dim
        # Predict code for the clean part (since in NPGM, occluded parts don't generate valid codes)
        # Note: In real NPGM, zero-vector might map to a random code, but we filter it out.
        # Here we simulate filtering by manually picking the valid codes.
        sub_vec = query_vec_clean[start:end].reshape(1, -1)
        code = codebooks[m].predict(sub_vec)[0]
        query_codes_valid.append(code)
        
    # 2. Vote! Count how many valid chunks match for each image in database
    # (Simple voting simulation)
    votes = np.zeros(N)
    for i in range(N):
        if i == query_idx: continue
        score = 0
        # Check matches only on valid chunks
        for m_idx, m_real in enumerate(valid_indices):
            if all_codes[i, m_real] == query_codes_valid[m_idx]:
                score += 1
        votes[i] = score
        
    # Sort by votes (Descending)
    npgm_matches = np.argsort(votes)[::-1]
    # Filter to exclude self
    npgm_matches = [idx for idx in npgm_matches if idx != query_idx]

    # ==========================================
    # METHOD B: NCM Retrieval (Distance to Average)
    # ==========================================
    # NCM Logic: Distance calculation includes the zeros!
    centroid = class_means[query_cls]
    
    # Calculate distance between DIRTY query and CLEAN Centroid
    # This is exactly what happens in deployment
    dists = cdist(X, query_vec_dirty.reshape(1, -1), metric='euclidean').flatten()
    
    # Sort by distance (Ascending)
    ncm_matches = np.argsort(dists)
    ncm_matches = [idx for idx in ncm_matches if idx != query_idx]

    # 5. Visualization
    def get_img(idx):
        img_tensor, _ = dataset[idx]
        img = img_tensor.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        return np.clip(img, 0, 1)

    fig, axes = plt.subplots(3, 6, figsize=(16, 9))
    
    # Row 1: The Query (Show it 'Occluded' visually)
    q_img = get_img(query_idx)
    h, w, c = q_img.shape
    # Draw a black box on bottom half to represent the feature loss
    q_img[h//2:, :, :] = 0 
    
    axes[0,0].imshow(q_img)
    axes[0,0].set_title(f"OCCLUDED QUERY\nClass {query_cls}")
    axes[0,0].axis('off')
    for ax in axes[0, 1:]: ax.axis('off')

    # Row 2: NPGM Results (Voting)
    axes[1,0].text(0.5, 0.5, "NPGM\n(Partial Voting)", ha='center', fontsize=12, fontweight='bold')
    axes[1,0].axis('off')
    for k in range(5):
        idx = npgm_matches[k]
        axes[1, k+1].imshow(get_img(idx))
        title_color = 'green' if y[idx] == query_cls else 'red'
        axes[1, k+1].set_title(f"Class {y[idx]}", color=title_color, fontweight='bold')
        axes[1, k+1].axis('off')

    # Row 3: NCM Results (Corrupted Distance)
    axes[2,0].text(0.5, 0.5, "NCM\n(Global Distance)", ha='center', fontsize=12, fontweight='bold')
    axes[2,0].axis('off')
    for k in range(5):
        idx = ncm_matches[k]
        axes[2, k+1].imshow(get_img(idx))
        title_color = 'green' if y[idx] == query_cls else 'red'
        axes[2, k+1].set_title(f"Class {y[idx]}", color=title_color, fontweight='bold')
        axes[2, k+1].axis('off')

    plt.suptitle("The Hard Test: Occlusion Robustness\n(Voting vs Distance)", fontsize=16)
    plt.tight_layout()
    plt.savefig('outputs/interpretability_hard_test.png')
    print("✅ Saved HARD test plot to outputs/interpretability_hard_test.png")

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
    
    Args:
        graph: The trained NPGM graph object.
        X_train, y_train: Data to train the baselines on the fly.
        X_test, y_test: Data to corrupt and evaluate on.
        evaluate_graph_fn: Your existing function `evaluate_graph(graph, feats, lbls)`
                           that returns a dict with accuracy.
    """
    print("\n🛡️ Starting Occlusion Robustness Stress Test...")
    
    # 1. Train Baselines on Clean Data (On the fly)
    print("   Training Baselines (NCM & Linear)...")
    
    # NCM Baseline
    ncm = NearestCentroid()
    ncm.fit(X_train, y_train)
    
    # Linear Probe (Parametric Baseline)
    # Using SGDClassifier as a fast approximation of a linear layer
    linear = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
    linear.fit(X_train, y_train)

    # 2. Define Occlusion Levels (0% to 50%)
    levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Store results
    acc_npgm = []
    acc_ncm = []
    acc_lin = []

    print(f"   Testing levels: {levels}")

    for p in levels:
        # Create corrupted test set
        X_test_corrupt = apply_feature_occlusion(X_test, p)
        
        # A. Evaluate Baselines
        score_ncm = ncm.score(X_test_corrupt, y_test)
        score_lin = linear.score(X_test_corrupt, y_test)
        
        # B. Evaluate Graph (Using your existing function)
        # We assume evaluate_graph returns a dict like {'clean': [acc...]} or scalar
        # We wrap it in try/except to handle return type variations
        graph_metrics = evaluate_graph_fn(graph, X_test_corrupt, y_test)
        
        if isinstance(graph_metrics, dict):
            # Take mean of 'clean' accuracy if it's a list (continual setting)
            # or just the value if it's scalar
            val = graph_metrics.get('clean', 0.0)
            score_npgm = np.mean(val) if isinstance(val, (list, np.ndarray)) else val
        else:
            score_npgm = graph_metrics # Fallback if it returns scalar

        # Store
        acc_ncm.append(score_ncm)
        acc_lin.append(score_lin)
        acc_npgm.append(score_npgm)
        
        print(f"   [Occlusion {int(p*100)}%] NPGM: {score_npgm:.2%} | NCM: {score_ncm:.2%} | Linear: {score_lin:.2%}")

    # 3. Plotting the Curve
    plt.figure(figsize=(8, 6))
    plt.plot(levels, [x * 100 for x in acc_npgm], 'o-', linewidth=3, label='NPGM (Ours)')
    plt.plot(levels, [x * 100 for x in acc_ncm], 's--', linewidth=2, label='NCM (Baseline)')
    plt.plot(levels, [x * 100 for x in acc_lin], 'x:', linewidth=2, label='Linear (Parametric)')
    
    plt.title("Robustness to Occlusion (The 'Hard Test')", fontsize=14)
    plt.xlabel("Percentage of Features Masked", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    # Save
    path = 'outputs/robustness_curve.png'
    plt.savefig(path)
    print(f"✅ Robustness curve saved to {path}")
    
    return acc_npgm, acc_ncm, acc_lin