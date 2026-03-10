import time

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDClassifier

from .bayes import StreamingNaiveBayes
from .config import Config
from .helper import build_hub_neighbors, build_hubs
from .model import BioEpisodicGraph, ContinualGraph


def train_continual_graph(features, labels):
    print(
        f"\n🚀 Starting {Config.N_TASKS}-Task Benchmark (Split: {int(Config.TRAIN_TEST_SPLIT * 100)}/{int((1 - Config.TRAIN_TEST_SPLIT) * 100)})"
    )

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

        print(
            f"[Task {task_id + 1}/{Config.N_TASKS}] Memory: {len(memory_idxs)} | Test: {len(query_idxs)}...",
            end="",
        )

        # 3. Learn Vocabulary (K-Means)
        # Use int32 to prevent overflow
        task_quantized = np.zeros((len(memory_data), Config.N_CHUNKS), dtype=np.int32)

        for c in range(Config.N_CHUNKS):
            sub_vecs = memory_data[:, c * Config.CHUNK_DIM : (c + 1) * Config.CHUNK_DIM]

            kmeans = MiniBatchKMeans(
                n_clusters=Config.WORDS_PER_TASK,
                n_init=10,
                batch_size=256,
                random_state=Config.SEED,
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

    print(f"✅ Training Complete in {time.time() - total_start:.1f}s")

    # 4. Assemble Graph
    full_indices = np.vstack(graph_indices)
    full_labels = torch.tensor(np.concatenate(graph_labels_list)).to(Config.DEVICE)

    # Integrity Check
    assert full_indices.shape[0] == full_labels.shape[0], (
        "Graph Wiring/Labels Mismatch!"
    )

    hub_indices, hub_labels = build_hubs(full_indices, full_labels)
    neighbors = build_hub_neighbors(hub_indices)

    graph = ContinualGraph(codebooks, hub_indices, hub_labels, neighbors)

    # Prepare Test Set
    test_tensor = torch.tensor(
        np.concatenate(test_data["features"]), dtype=torch.float32
    ).to(Config.DEVICE)
    test_labels = torch.tensor(np.concatenate(test_data["labels"])).to(Config.DEVICE)

    return graph, test_tensor, test_labels


def train_task_free_graph(features, labels, buffer_size=1000):
    # Note: features/labels here should ALREADY be the training split from main.py
    print(f"\n🌊 Starting Task-Free Stream Learning (Naive Bayes)...")

    # 1. Initialize Codebooks (Empty)
    codebooks = [np.zeros((0, Config.CHUNK_DIM)) for _ in range(Config.N_CHUNKS)]

    # 2. We need to initialize Bayes Model LATER, once we have some codebooks,
    # OR we initialize it with placeholders.
    # Let's initialize it at the end for clean export, or incrementally if we had fixed codebooks.
    # Actually, for Append-Only, we need to track counts incrementally.
    # But Bayes needs the `codebooks` for inference.
    # Strategy: We will update a temporary count dictionary, and build the object at the end.

    # Actually, let's just Instantiate it with empty codebooks and update them later.
    bayes_model = StreamingNaiveBayes(
        Config.N_CHUNKS, [np.zeros((1, Config.CHUNK_DIM))] * Config.N_CHUNKS
    )

    # Shuffle Stream
    perm = np.random.permutation(len(features))
    stream_features = features[perm]
    stream_labels = labels[perm]

    total_images = len(stream_features)
    block_id = 0

    for start_idx in range(0, total_images, buffer_size):
        end_idx = min(start_idx + buffer_size, total_images)

        block_data = stream_features[start_idx:end_idx]
        block_lbls = stream_labels[start_idx:end_idx]

        if len(block_data) < 50:
            break

        print(f"   [Block {block_id}] Processing {len(block_data)} items...", end="")

        block_quantized = np.zeros((len(block_data), Config.N_CHUNKS), dtype=np.int32)

        for c in range(Config.N_CHUNKS):
            sub_vecs = block_data[:, c * Config.CHUNK_DIM : (c + 1) * Config.CHUNK_DIM]

            # Local K-Means (Append-Only)
            kmeans = MiniBatchKMeans(
                n_clusters=Config.WORDS_PER_TASK, n_init=3, batch_size=256
            )
            kmeans.fit(sub_vecs)

            # Offset
            global_offset = codebooks[c].shape[0]

            # Append Codebooks
            codebooks[c] = np.vstack([codebooks[c], kmeans.cluster_centers_])

            # Quantize
            preds = kmeans.predict(sub_vecs).astype(np.int32)
            block_quantized[:, c] = preds + global_offset

        # Update Bayes Model
        codes_tensor = torch.tensor(block_quantized, dtype=torch.long)
        lbls_tensor = torch.tensor(block_lbls, dtype=torch.long)

        bayes_model.partial_fit(codes_tensor, lbls_tensor)

        print(" Done.")
        block_id += 1

    print(f"✅ Stream Training Complete.")

    # UPDATE the Bayes Model with the final full Codebooks so it can predict correctly
    bayes_model.codebooks = [
        torch.tensor(cb, dtype=torch.float32).to(Config.DEVICE) for cb in codebooks
    ]

    # bayes_model.diffuse_graph(alpha=0.5)

    return bayes_model


# Inside learner.py

# def train_task_free_graph(features, labels, buffer_size=1000, split_ratio=0.8):
#     print(f"\n🌊 Starting Task-Free Stream Learning (Online adaptation)...")

#     # 1. SPLIT DATA FIRST (Crucial Step to prevent Leakage)
#     total_samples = len(features)
#     split_idx = int(total_samples * split_ratio)

#     # Shuffle data
#     np.random.seed(Config.SEED)
#     perm = np.random.permutation(total_samples)
#     shuffled_features = features[perm]
#     shuffled_labels = labels[perm]

#     # DIVIDE: Robot sees 'train', Robot NEVER sees 'test'
#     train_features = shuffled_features[:split_idx] # The Stream
#     train_labels   = shuffled_labels[:split_idx]

#     test_features  = shuffled_features[split_idx:] # The Hidden Test Set
#     test_labels    = shuffled_labels[split_idx:]

#     print(f"   📉 Data Split: {len(train_features)} Training (Stream) | {len(test_features)} Unseen Test")

#     # 2. Initialize Online K-Means
#     global_kmeans = [
#         MiniBatchKMeans(
#             n_clusters=Config.WORDS_PER_TASK,
#             random_state=Config.SEED,
#             batch_size=256,
#             n_init=3
#         )
#         for _ in range(Config.N_CHUNKS)
#     ]

#     graph_indices = []
#     graph_labels_list = []

#     # 3. Process ONLY the Training Stream
#     total_train = len(train_features)
#     block_id = 0

#     for start_idx in range(0, total_train, buffer_size):
#         end_idx = min(start_idx + buffer_size, total_train)

#         block_data = train_features[start_idx:end_idx]
#         block_lbls = train_labels[start_idx:end_idx]

#         if len(block_data) < 50: break

#         print(f"   [Block {block_id}] Processing {len(block_data)} items...", end="")

#         block_quantized = np.zeros((len(block_data), Config.N_CHUNKS), dtype=np.int32)

#         for c in range(Config.N_CHUNKS):
#             start_d = c * Config.CHUNK_DIM
#             end_d   = (c + 1) * Config.CHUNK_DIM
#             sub_vecs = block_data[:, start_d : end_d]

#             # Update Knowledge (Learn)
#             global_kmeans[c].partial_fit(sub_vecs)

#             # Apply Knowledge (Quantize)
#             preds = global_kmeans[c].predict(sub_vecs).astype(np.int32)
#             block_quantized[:, c] = preds

#         graph_indices.append(block_quantized)
#         graph_labels_list.append(block_lbls)

#         print(" Done.")
#         block_id += 1

#     # 4. Extract Final Codebooks
#     final_codebooks = [km.cluster_centers_ for km in global_kmeans]

#     # 5. Build Graph
#     print("   Building Graph Topology...")
#     if len(graph_indices) > 0:
#         full_indices = np.vstack(graph_indices)
#         full_labels = torch.tensor(np.concatenate(graph_labels_list)).to(Config.DEVICE)

#         hubs_indices, hub_labels = build_hubs(full_indices, full_labels)
#         neighbors = build_hub_neighbors(hubs_indices)

#         graph = ContinualGraph(final_codebooks, hubs_indices, hub_labels, neighbors)
#     else:
#         raise ValueError("Graph failed to train: No data processed.")

#     # 6. RETURN ONLY THE HIDDEN TEST SET
#     # This ensures your 'evaluate_graph' function is strictly testing generalization.
#     test_tensor = torch.tensor(test_features, dtype=torch.float32).to(Config.DEVICE)
#     test_labels_tensor = torch.tensor(test_labels).to(Config.DEVICE)

#     return graph, test_tensor, test_labels_tensor


def run_sequential_linear_probe(features, labels, n_tasks=20):
    print("\n📉 --- Running Sequential Linear Probe (The 'Forgetting' Test) ---")

    # We use SGDClassifier because it supports incremental learning (partial_fit)
    clf = SGDClassifier(loss="log_loss", random_state=42)
    classes = np.unique(labels)

    accuracies = []

    # Simulate 20 Tasks
    chunk_size = len(features) // n_tasks

    for i in range(n_tasks):
        # 1. Get Task Data
        start = i * chunk_size
        end = (i + 1) * chunk_size
        X_task = features[start:end]
        y_task = labels[start:end]

        # 2. Train ONLY on this task (Simulating streaming)
        clf.partial_fit(X_task, y_task, classes=classes)

        # 3. Test on ALL Data (Global Accuracy)
        # In a real scenario, we check if it remembers Task 1
        current_acc = clf.score(features, labels)
        accuracies.append(current_acc)
        print(f"   Task {i + 1}/{n_tasks}: Global Accuracy = {current_acc * 100:.2f}%")

    print(f"   ❌ Final Sequential Linear Probe Accuracy: {accuracies[-1] * 100:.2f}%")
    return accuracies[-1]


def train_bio_graph(
    features,
    labels,
    buffer_size=1000,
    shuffle_stream=False,
    consolidate_every=0,
    lambda_val=0.1,
):
    """
    Simulates the Bio-Inspired learning process:
    ONLINE: Rapid recording into an episodic graph + running averages.
    OFFLINE: Sleep-like consolidation, essence transfer, and pruning.
    """
    if consolidate_every > 0:
        print(
            f"\n🧠 Starting Bio-Inspired Online Learning with Offline Consolidation..."
        )
    else:
        print(f"\n🧠 Starting Bio-Inspired Online Learning (Consolidation Disabled)...")

    n_classes = Config.N_TASKS * Config.CLASSES_PER_TASK
    model = BioEpisodicGraph(input_dim=384, n_classes=n_classes)

    # Preserve incoming order by default; global shuffling hurts episodic purity.
    if shuffle_stream:
        perm = np.random.permutation(len(features))
        features = features[perm]
        labels = labels[perm]

    stream_features = torch.tensor(features, dtype=torch.float32).to(Config.DEVICE)
    stream_labels = torch.tensor(labels, dtype=torch.long).to(Config.DEVICE)

    total_images = len(stream_features)

    # We treat each buffer block as a "Day" (Online Phase) followed by "Sleep" (Offline Phase)
    for i, start_idx in enumerate(range(0, total_images, buffer_size)):
        end_idx = min(start_idx + buffer_size, total_images)

        batch_feat = stream_features[start_idx:end_idx]
        batch_lbl = stream_labels[start_idx:end_idx]

        # 1. ONLINE PHASE: Rapid Hippocampal recording
        model.online_step(batch_feat, batch_lbl)

        # 2. OFFLINE CONSOLIDATION PHASE: Transfer to Long-term Memory & Pruning
        # In a real scenario, this might happen less frequently, but we do it per block here.
        if consolidate_every > 0 and (i + 1) % consolidate_every == 0:
            model.consolidate(lambda_val=lambda_val)

            print(
                f"   [Consolidation] Processed {end_idx}/{total_images} samples. Graph size: {len(model.nodes)} nodes."
            )

    # Final consolidation pass for the tail block(s).
    if consolidate_every > 0:
        model.consolidate(lambda_val=lambda_val)

    print(f"✅ Bio-Graph Training Complete.")
    return model
