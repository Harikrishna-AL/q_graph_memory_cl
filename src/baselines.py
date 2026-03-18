import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
from .config import Config
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Simple MLP Class
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1), # Small dropout for stability
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)

def run_baselines(train_features, train_labels, test_features, test_labels):
    print("\n⚖️  --- Running Baselines for Comparison ---")
    results = {}
    
    # 1. Nearest Class Mean (NCM)
    print("   1. Training NCM (Nearest Class Mean)...")
    start = time.time()
    ncm = NearestCentroid()
    ncm.fit(train_features, train_labels)
    ncm_preds = ncm.predict(test_features)
    ncm_acc = accuracy_score(test_labels, ncm_preds)
    results['NCM'] = ncm_acc
    print(f"      >> NCM Accuracy: {ncm_acc*100:.2f}% (Time: {time.time()-start:.2f}s)")
    
    # 2. MLP Baseline (Linear Layers + ReLU)
    print("   2. Training MLP (Linear + ReLU)...")
    start = time.time()
    
    # Prepare Data
    device = Config.DEVICE
    X_train = torch.tensor(train_features, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_labels, dtype=torch.long).to(device)
    X_test  = torch.tensor(test_features, dtype=torch.float32).to(device)
    
    # Init Model
    # DINOv2 dimension is 384 for Small, but check actual dim
    input_dim = train_features.shape[1] 
    num_classes = len(np.unique(train_labels))
    
    model = SimpleMLP(input_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train Loop (Fast: 10 Epochs is usually enough for pre-trained features)
    model.train()
    batch_size = 256
    
    for epoch in range(10): 
        perm = torch.randperm(X_train.size(0))
        epoch_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i : i + batch_size]
            batch_X, batch_y = X_train[idx], y_train[idx]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
    mlp_acc = accuracy_score(test_labels, preds)
    results['Linear'] = mlp_acc # Key kept as 'Linear' for compatibility
    print(f"      >> MLP Accuracy: {mlp_acc*100:.2f}% (Time: {time.time()-start:.2f}s)")

    # 2b. Testing MLP on OCCLUDED Data
    print("   2b. Testing MLP on OCCLUDED Data...")
    
    # Simulate Occlusion (Zeroing out last 50% of dimensions)
    # 384 dims -> 6 chunks of 64. Occluding 3 chunks = 192 dims.
    cutoff = input_dim // 2
    X_test_occ = X_test.clone()
    X_test_occ[:, cutoff:] = 0.0 # Zero out top half
    
    model.eval()
    with torch.no_grad():
        logits_occ = model(X_test_occ)
        preds_occ = torch.argmax(logits_occ, dim=1).cpu().numpy()
        
    mlp_acc_occ = accuracy_score(test_labels, preds_occ)
    print(f"      >> MLP Occluded Accuracy: {mlp_acc_occ*100:.2f}%")
    
    return results

def run_statistical_baselines(X_train, y_train, X_test, y_test):
    print("\n📉 --- Running Additional Statistical Baselines ---")
    
    # Ensure CPU numpy
    X_tr = X_train.cpu().numpy() if torch.is_tensor(X_train) else X_train
    y_tr = y_train.cpu().numpy() if torch.is_tensor(y_train) else y_train
    X_te = X_test.cpu().numpy() if torch.is_tensor(X_test) else X_test
    y_te = y_test.cpu().numpy() if torch.is_tensor(y_test) else y_test

    results = {}

    # 1. k-NN (The "Memory Upper Bound")
    # Stores ALL data. TQM should be close to this but efficient.
    print("   Running k-NN (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr, y_tr)
    acc_knn = knn.score(X_te, y_te)
    results['k-NN'] = acc_knn
    print(f"   >> k-NN Accuracy: {acc_knn*100:.2f}%")

    # 2. Deep SLDA (Streaming Linear Discriminant Analysis)
    # The strongest statistical competitor. Assumes 1 Gaussian per class.
    print("   Running Deep SLDA...")
    try:
        slda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        slda.fit(X_tr, y_tr)
        acc_slda = slda.score(X_te, y_te)
        results['SLDA'] = acc_slda
        print(f"   >> SLDA Accuracy: {acc_slda*100:.2f}%")
    except Exception as e:
        print(f"   !! SLDA Failed (Singular Matrix?): {e}")
        results['SLDA'] = 0.0

    return results


# ==============================================================================
# NAIVE REPLAY BUFFER BASELINE
# ==============================================================================


class NaiveReplayBuffer:
    """
    Stores every training feature vector verbatim (never compresses).

    Inference: for each query, return the class whose stored vectors have
    the highest max cosine similarity — exactly the pseudo-code spec.

    Memory: n_samples × feature_dim × 4 bytes  (unbounded growth).
    """

    def __init__(self):
        self.features: list[np.ndarray] = []   # one (D,) array per sample
        self.labels:   list[int]        = []

    # ------------------------------------------------------------------
    # "Training" — just append; no compression whatsoever
    # ------------------------------------------------------------------
    def update(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Add a batch of feature vectors and their labels to the buffer."""
        features = np.array(features, dtype=np.float32)
        labels   = np.array(labels,   dtype=np.int64)
        for f, l in zip(features, labels):
            self.features.append(f)
            self.labels.append(int(l))

    # ------------------------------------------------------------------
    # Inference — max cosine similarity per class
    # ------------------------------------------------------------------
    def predict(self, queries: np.ndarray) -> np.ndarray:
        """
        queries : (N, D)  L2-normalised feature vectors
        returns : (N,)    predicted class labels
        """
        if not self.features:
            raise RuntimeError("Buffer is empty — call update() first.")

        bank   = np.array(self.features, dtype=np.float32)  # (M, D)
        labels = np.array(self.labels,   dtype=np.int64)     # (M,)

        # L2-normalise both sides for cosine similarity
        bank_n    = bank    / (np.linalg.norm(bank,    axis=1, keepdims=True) + 1e-8)
        queries_n = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)

        # Cosine sims: (N, M)
        sims = queries_n @ bank_n.T

        unique_classes = np.unique(labels)
        # (N, C) matrix of per-class max similarity
        class_scores = np.full((len(queries), len(unique_classes)), -1.0, dtype=np.float32)
        for i, c in enumerate(unique_classes):
            mask = labels == c
            class_scores[:, i] = sims[:, mask].max(axis=1)

        best = np.argmax(class_scores, axis=1)
        return unique_classes[best]

    # ------------------------------------------------------------------
    # Memory footprint
    # ------------------------------------------------------------------
    def memory_mb(self) -> float:
        """Returns the exact memory consumed by the stored float32 vectors."""
        if not self.features:
            return 0.0
        n, d = len(self.features), len(self.features[0])
        return n * d * 4 / (1024 ** 2)


def run_naive_replay_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    batch_size: int = 512,
) -> dict:
    """
    Train and evaluate the naive replay buffer.

    Simulates the streaming scenario: feeds X_train in arrival order
    (block by block) into the buffer, then evaluates on X_test.

    Returns a dict with keys: accuracy, memory_mb.
    """
    print("\n📼 --- Naive Replay Buffer Baseline ---")

    buf = NaiveReplayBuffer()

    # Simulate streaming: feed data block by block (order = arrival order)
    n = len(X_train)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        buf.update(X_train[start:end], y_train[start:end])

    mem = buf.memory_mb()
    print(f"   💾 Buffer size after training: {len(buf.features):,} vectors "
          f"| Memory: {mem:.2f} MB")

    # Inference in batches to avoid OOM on large datasets
    all_preds = []
    for start in range(0, len(X_test), batch_size):
        end = min(start + batch_size, len(X_test))
        preds = buf.predict(X_test[start:end])
        all_preds.append(preds)

    all_preds = np.concatenate(all_preds)
    acc = float(np.mean(all_preds == y_test))
    print(f"   🏆 Naive Replay Accuracy: {acc * 100:.2f}%  |  Memory: {mem:.2f} MB")

    return {"accuracy": acc, "memory_mb": mem}



# ==============================================================================
# REHEARSAL MLP BASELINE  (Experience Replay + MLP)
# ==============================================================================


class RehearsalMLPBuffer:
    """
    Experience Replay with an MLP classifier.

    Training (streaming):
        After each new block the buffer grows, then the SAME MLP is
        fine-tuned for `finetune_epochs` epochs on the combined
        (buffer + current batch) data — warm-started from the previous
        weights so it doesn't restart cold each time.

    Inference: forward pass through the final MLP state.

    Memory: n_samples × D × 4 bytes  (same as naive replay).

    This is the canonical ER-Replay baseline in the CL literature:
    every stored sample is replayed alongside each new block, so the
    model sees all past data at every update step.
    """

    def __init__(self, input_dim: int, num_classes: int, device, lr: float = 5e-4):
        self.device = device
        self.input_dim   = input_dim
        self.num_classes = num_classes

        # Buffer
        self.features: list[np.ndarray] = []
        self.labels:   list[int]        = []

        # MLP — same architecture as SimpleMLP in run_baselines()
        self.model = SimpleMLP(input_dim, num_classes).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    def update(self, new_features: np.ndarray, new_labels: np.ndarray,
               finetune_epochs: int = 5, batch_size: int = 256) -> None:
        """
        Add new_features/new_labels to the buffer, then fine-tune the MLP
        on the full (buffer + current batch) for `finetune_epochs` epochs.
        """
        new_features = np.array(new_features, dtype=np.float32)
        new_labels   = np.array(new_labels,   dtype=np.int64)
        for f, l in zip(new_features, new_labels):
            self.features.append(f)
            self.labels.append(int(l))

        # Build tensors from the FULL buffer (includes current batch)
        X = torch.tensor(np.array(self.features), dtype=torch.float32).to(self.device)
        y = torch.tensor(np.array(self.labels),   dtype=torch.long).to(self.device)

        self.model.train()
        n = len(X)
        for _ in range(finetune_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = perm[start: start + batch_size]
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X[idx]), y[idx])
                loss.backward()
                self.optimizer.step()
        self.model.eval()

    # ------------------------------------------------------------------
    def predict(self, queries: np.ndarray) -> np.ndarray:
        q = torch.tensor(queries, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(q)
        return torch.argmax(logits, dim=1).cpu().numpy()

    # ------------------------------------------------------------------
    def memory_mb(self) -> float:
        if not self.features:
            return 0.0
        n, d = len(self.features), len(self.features[0])
        return n * d * 4 / (1024 ** 2)


def run_rehearsal_mlp_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    block_size: int = 512,
    finetune_epochs: int = 5,
    lr: float = 5e-4,
) -> dict:
    """
    Train and evaluate the rehearsal MLP baseline.

    Streams X_train in blocks of `block_size`. After each block the
    MLP is fine-tuned on buffer + current batch for `finetune_epochs`
    epochs (warm-started from the previous weights).

    Returns a dict with keys: accuracy, memory_mb.
    """
    print("\n🔁 --- Rehearsal MLP Baseline (ER + MLP) ---")

    device      = Config.DEVICE
    input_dim   = X_train.shape[1]
    num_classes = int(np.max(y_train)) + 1

    buf = RehearsalMLPBuffer(input_dim, num_classes, device, lr=lr)

    n_blocks = (len(X_train) + block_size - 1) // block_size
    for i, start in enumerate(range(0, len(X_train), block_size)):
        end = min(start + block_size, len(X_train))
        buf.update(X_train[start:end], y_train[start:end],
                   finetune_epochs=finetune_epochs, batch_size=256)
        if (i + 1) % max(1, n_blocks // 5) == 0:
            print(f"   [Block {i+1}/{n_blocks}] Buffer: {end:,} samples "
                  f"| Memory: {buf.memory_mb():.2f} MB")

    mem = buf.memory_mb()
    print(f"   💾 Final buffer: {len(buf.features):,} vectors | Memory: {mem:.2f} MB")

    # Inference in batches
    all_preds = []
    for start in range(0, len(X_test), 512):
        end = min(start + 512, len(X_test))
        preds = buf.predict(X_test[start:end])
        all_preds.append(preds)

    all_preds = np.concatenate(all_preds)
    acc = float(np.mean(all_preds == y_test))
    print(f"   🏆 Rehearsal MLP Accuracy:  {acc * 100:.2f}%  |  Memory: {mem:.2f} MB")

    return {"accuracy": acc, "memory_mb": mem}