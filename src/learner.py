import time
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from .config import Config
from .model import ContinualGraph

def train_continual_graph(features, labels):
    print(f"\n🚀 Starting {Config.N_TASKS}-Task Benchmark (Split: {int(Config.TRAIN_TEST_SPLIT*100)}/{int((1-Config.TRAIN_TEST_SPLIT)*100)})")
    
    # Initialize Storage
    codebooks = [np.zeros((0, Config.CHUNK_DIM)) for _ in range(Config.N_CHUNKS)]
    graph_indices = []
    graph_labels_list = []
    
    # To accumulate unseen test data
    test_data = {"features": [], "labels": []}
    
    total_start = time.time()
    
    for task_id in range(Config.N_TASKS):
        start_cls = task_id * Config.CLASSES_PER_TASK
        end_cls = (task_id + 1) * Config.CLASSES_PER_TASK
        
        # 1. Filter Data for Task
        task_mask = (labels >= start_cls) & (labels < end_cls)
        task_idxs = np.where(task_mask)[0]
        
        # 2. Shuffle & Split
        np.random.seed(Config.SEED)
        np.random.shuffle(task_idxs)
        
        split_point = int(len(task_idxs) * Config.TRAIN_TEST_SPLIT)
        memory_idxs = task_idxs[:split_point]
        query_idxs = task_idxs[split_point:]
        
        memory_data = features[memory_idxs]
        
        # Store Labels for Graph and Features for Testing
        graph_labels_list.append(labels[memory_idxs])
        test_data["features"].append(features[query_idxs])
        test_data["labels"].append(labels[query_idxs])
        
        print(f"[Task {task_id+1}/{Config.N_TASKS}] Memory: {len(memory_idxs)} | Test: {len(query_idxs)}...", end="")
        
        # 3. Learn Vocabulary (K-Means)
        # Use int32 to prevent overflow
        task_quantized = np.zeros((len(memory_data), Config.N_CHUNKS), dtype=np.int32)
        
        for c in range(Config.N_CHUNKS):
            sub_vecs = memory_data[:, c*Config.CHUNK_DIM : (c+1)*Config.CHUNK_DIM]
            
            kmeans = MiniBatchKMeans(
                n_clusters=Config.WORDS_PER_TASK, 
                n_init=10, 
                batch_size=256, 
                random_state=Config.SEED
            )
            kmeans.fit(sub_vecs)
            
            # Update Codebook
            current_offset = codebooks[c].shape[0]
            if codebooks[c].shape[0] == 0:
                codebooks[c] = kmeans.cluster_centers_
            else:
                codebooks[c] = np.vstack([codebooks[c], kmeans.cluster_centers_])
            
            # Predict & Offset
            preds = kmeans.predict(sub_vecs).astype(np.int32)
            task_quantized[:, c] = preds + current_offset
            
        graph_indices.append(task_quantized)
        print(f" Done.")

    print(f"✅ Training Complete in {time.time()-total_start:.1f}s")
    
    # 4. Assemble Graph
    full_indices = np.vstack(graph_indices)
    full_labels = torch.tensor(np.concatenate(graph_labels_list)).to(Config.DEVICE)
    
    # Integrity Check
    assert full_indices.shape[0] == full_labels.shape[0], "Graph Wiring/Labels Mismatch!"
    
    graph = ContinualGraph(codebooks, full_indices, full_labels)
    
    # Prepare Test Set
    test_tensor = torch.tensor(np.concatenate(test_data["features"]), dtype=torch.float32).to(Config.DEVICE)
    test_labels = torch.tensor(np.concatenate(test_data["labels"])).to(Config.DEVICE)
    
    return graph, test_tensor, test_labels

def train_task_free_graph(features, labels, buffer_size=1000):
    print(f"\n🌊 Starting Task-Free Stream Learning...")
    
    # Global Storage
    codebooks = [np.zeros((0, Config.CHUNK_DIM)) for _ in range(Config.N_CHUNKS)]
    graph_indices = []
    graph_labels_list = []
    
    # 1. Shuffle ALL data to simulate a random chaotic stream
    # (In a real robot, this happens naturally as time passes)
    perm = np.random.permutation(len(features))
    stream_features = features[perm]
    stream_labels = labels[perm]
    
    total_images = len(stream_features)
    
    # 2. Process in "Blocks" (Simulating waking/sleeping cycles)
    block_id = 0
    for start_idx in range(0, total_images, buffer_size):
        end_idx = min(start_idx + buffer_size, total_images)
        
        # Get the "Block" of data
        block_data = stream_features[start_idx:end_idx]
        block_lbls = stream_labels[start_idx:end_idx]
        
        if len(block_data) < 100: break # Skip tiny final batches
        
        print(f"   [Block {block_id}] Processing stream items {start_idx}-{end_idx}...", end="")
        
        # 3. Learn Local Vocabulary (Just for this block)
        # We append these new words to the global codebook
        block_quantized = np.zeros((len(block_data), Config.N_CHUNKS), dtype=np.int32)
        
        for c in range(Config.N_CHUNKS):
            sub_vecs = block_data[:, c*Config.CHUNK_DIM : (c+1)*Config.CHUNK_DIM]
            
            # Run K-Means only on this new experience
            # We use a smaller K (e.g., 64) because the batch is smaller (1000 imgs)
            kmeans = MiniBatchKMeans(n_clusters=64, n_init=3, batch_size=256)
            kmeans.fit(sub_vecs)
            
            # --- CRITICAL: APPEND-ONLY LOGIC ---
            # 1. Get the offset (Current size of global codebook)
            global_offset = codebooks[c].shape[0]
            
            # 2. Add new words to global codebook
            codebooks[c] = np.vstack([codebooks[c], kmeans.cluster_centers_])
            
            # 3. Map images to these NEW words (plus the offset)
            # Note: In a smarter version, we would check if old words fit better, 
            # but for pure speed, we assume this block introduces new concepts.
            preds = kmeans.predict(sub_vecs).astype(np.int32)
            block_quantized[:, c] = preds + global_offset
            
        # Store memories
        graph_indices.append(block_quantized)
        graph_labels_list.append(block_lbls)
        
        print(" Done.")
        block_id += 1

    # ... Assembly Logic (same as before) ...
    full_indices = np.vstack(graph_indices)
    full_labels = torch.tensor(np.concatenate(graph_labels_list)).to(Config.DEVICE)
    
    graph = ContinualGraph(codebooks, full_indices, full_labels)
    print(f"✅ Stream Training Complete.")
    # Prepare ALL data as test set (since no tasks)
    test_tensor = torch.tensor(features, dtype=torch.float32).to(Config.DEVICE)
    test_labels = torch.tensor(labels).to(Config.DEVICE)
    return graph, test_tensor, test_labels