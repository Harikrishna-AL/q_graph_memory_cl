import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from .config import Config
from sklearn.cluster import KMeans
from collections import defaultdict
from scipy.spatial.distance import cdist

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

def compare_interpretability(model, dataset, num_samples=2000, n_chunks=8):
    """
    Compares NPGM (Visual Words) vs NCM (Centroids) for concept retrieval.
    """
    print(f"🧪 Starting Interpretability Showdown (N={num_samples})...")
    device = model.device
    
    # --- 1. Extract Features & Prepare Data ---
    print("   Extracting features...")
    all_features = []
    all_images = []
    all_labels = []
    
    # We collect a subset to act as our "Database"
    for i in range(num_samples):
        img, label = dataset[i]
        with torch.no_grad():
            # Get the full 384-dim vector
            z = model.encoder(img.unsqueeze(0).to(device)).cpu().numpy()[0]
        all_features.append(z)
        all_images.append(img)
        all_labels.append(label)
        
    X = np.array(all_features)      # (N, D)
    y = np.array(all_labels)        # (N,)
    N, D = X.shape
    chunk_dim = D // n_chunks
    
    # --- 2. Train NPGM (Quantization) on the fly ---
    # We simulate your Codebooks using K-Means on the first chunk
    print("   Training NPGM Codebooks (Simulation)...")
    # Let's focus on Chunk 0 (arbitrary attribute) for the demo
    chunk_0_data = X[:, :chunk_dim]
    
    # Create Codebook for Chunk 0
    kmeans = KMeans(n_clusters=64, n_init=10, random_state=42)
    kmeans.fit(chunk_0_data)
    codes = kmeans.labels_ # The "Visual Word" assigned to each image for Chunk 0
    centroids_pq = kmeans.cluster_centers_

    # --- 3. Train NCM (Class Means) ---
    print("   Computing NCM Centroids...")
    unique_classes = np.unique(y)
    class_means = {}
    for c in unique_classes:
        class_means[c] = np.mean(X[y == c], axis=0)

    # --- 4. Select a Query Image ---
    # Pick an interesting image (e.g. index 10)
    query_idx = 10 
    query_cls = y[query_idx]
    query_vec = X[query_idx]
    
    print(f"   Query Image: Index {query_idx} (Class {query_cls})")

    # ==========================================
    # METHOD A: NPGM Retrieval (Exact Code Match)
    # ==========================================
    # 1. Get the code for the query image on Chunk 0
    query_code = codes[query_idx]
    
    # 2. Find ALL other images that have this SAME code
    # (Matches based on Shared Vocabulary)
    npgm_matches = [i for i, c in enumerate(codes) if c == query_code and i != query_idx]
    
    # ==========================================
    # METHOD B: NCM Retrieval (Distance to Average)
    # ==========================================
    # 1. Get the Centroid for the Query's Class
    centroid = class_means[query_cls]
    
    # 2. Look at ONLY Chunk 0 of the Centroid (Fair comparison)
    centroid_chunk = centroid[:chunk_dim]
    
    # 3. Find images whose Chunk 0 is closest to the *Centroid's* Chunk 0
    # (Matches based on Distance to Average)
    dists = cdist(chunk_0_data, centroid_chunk.reshape(1, -1), metric='euclidean').flatten()
    # Sort by distance
    ncm_matches = np.argsort(dists)
    ncm_matches = [i for i in ncm_matches if i != query_idx] # Remove self if present

    # --- 5. Visualization ---
    def denorm(tensor):
        img = tensor.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        return np.clip(img, 0, 1)

    fig, axes = plt.subplots(3, 6, figsize=(16, 9))
    
    # Row 1: The Query
    axes[0,0].imshow(denorm(all_images[query_idx]))
    axes[0,0].set_title(f"QUERY\nClass {query_cls}")
    axes[0,0].axis('off')
    # Hide the rest of row 1
    for ax in axes[0, 1:]: ax.axis('off')

    # Row 2: NPGM Results
    axes[1,0].text(0.5, 0.5, "NPGM\n(Shared Code)", ha='center', fontsize=12, fontweight='bold')
    axes[1,0].axis('off')
    for k in range(5):
        if k < len(npgm_matches):
            idx = npgm_matches[k]
            axes[1, k+1].imshow(denorm(all_images[idx]))
            axes[1, k+1].set_title(f"Class {y[idx]}")
            axes[1, k+1].axis('off')

    # Row 3: NCM Results
    axes[2,0].text(0.5, 0.5, "NCM\n(Dist to Mean)", ha='center', fontsize=12, fontweight='bold')
    axes[2,0].axis('off')
    for k in range(5):
        if k < len(ncm_matches):
            idx = ncm_matches[k]
            axes[2, k+1].imshow(denorm(all_images[idx]))
            axes[2, k+1].set_title(f"Class {y[idx]}")
            axes[2, k+1].axis('off')

    plt.suptitle("Interpretability: Discrete Codes (NPGM) vs Continuous Means (NCM)", fontsize=16)
    plt.tight_layout()
    plt.savefig('outputs/interpretability_comparison.png')
    print("✅ Saved comparison plot to outputs/interpretability_comparison.png")

# --- Usage ---
# compare_interpretability(model, train_loader.dataset)