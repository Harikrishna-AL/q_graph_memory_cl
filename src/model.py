import torch
import torch.nn as nn
import numpy as np
from .config import Config
import os
import numpy as np
from src.config import Config

CACHE_DIR = "cache"

def get_cache_paths(use_train):
    split = "train" if use_train else "val"
    model = Config.DINO_MODEL.replace("/", "_")
    os.makedirs(CACHE_DIR, exist_ok=True)

    feat_path = f"{CACHE_DIR}/{model}_{split}_features.npy"
    lbl_path  = f"{CACHE_DIR}/{model}_{split}_labels.npy"
    return feat_path, lbl_path


def load_cached_features(use_train):
    feat_path, lbl_path = get_cache_paths(use_train)

    if os.path.exists(feat_path) and os.path.exists(lbl_path):
        print("💾 Loading cached DINO features...")
        features = np.load(feat_path, mmap_mode="r")  # memory-efficient
        labels = np.load(lbl_path)
        return features, labels

    return None, None

def save_cached_features(features, labels, use_train):
    feat_path, lbl_path = get_cache_paths(use_train)
    np.save(feat_path, features)
    np.save(lbl_path, labels)
    print(f"✅ Cached features saved to {feat_path}")

def load_dino():
    print(f"🦖 Loading {Config.DINO_MODEL}...")
    dino = torch.hub.load(Config.DINO_REPO, Config.DINO_MODEL)
    dino.to(Config.DEVICE)
    dino.eval()
    return dino

def extract_features(dino, dataloader):
    print("🔍 Extracting Features (this may take a moment)...")
    all_feats, all_lbls = [], []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(Config.DEVICE)
            f = dino(imgs).cpu().numpy()
            # L2 Normalize
            f = f / np.linalg.norm(f, axis=1, keepdims=True)
            all_feats.append(f)
            all_lbls.append(lbls.numpy())
    
    features = np.concatenate(all_feats)
    labels = np.concatenate(all_lbls)
    return features, labels

class ContinualGraph(nn.Module):
    def __init__(self, codebooks, hub_indices, graph_labels, neighbors, input_dim=384):
        super().__init__()
        self.n_chunks = len(codebooks)
        self.device = Config.DEVICE
        
        # 1. Leaf Nodes (Visual Words)
        self.leaves = nn.ParameterList([
            nn.Parameter(torch.tensor(cb, dtype=torch.float32).to(self.device), requires_grad=False) 
            for cb in codebooks
        ])
        
        # 2. Hub Nodes (Wiring)
        self.wiring = torch.tensor(hub_indices, dtype=torch.long).to(self.device)
        self.labels = torch.tensor(graph_labels, dtype=torch.long).to(self.device)

        # 3. Vectorized Neighbors (Adjacency Matrix)
        # We convert the list of neighbors into a Sparse Adjacency Matrix for fast diffusion
        n_hubs = len(hub_indices)
        indices = []
        values = []
        
        print("⚡ Building Adjacency Matrix for Diffusion...")
        for i, neigh_list in enumerate(neighbors):
            for n_idx in neigh_list:
                indices.append([i, n_idx])
                values.append(1.0)
        
        if len(indices) > 0:
            indices = torch.tensor(indices, dtype=torch.long).t().to(self.device)
            values = torch.tensor(values, dtype=torch.float32).to(self.device)
            # Create Sparse Matrix (Hubs x Hubs)
            self.adj_matrix = torch.sparse_coo_tensor(indices, values, (n_hubs, n_hubs))
        else:
            self.adj_matrix = None

    def diffuse(self, energy, steps=1):
        """
        Fast Matrix Multiplication Diffusion
        energy: (Batch, Hubs)
        """
        if self.adj_matrix is None:
            return energy
            
        for _ in range(steps):
            # Sparse MatMul: (Hubs, Hubs) @ (Hubs, Batch) -> (Hubs, Batch)
            # We transpose energy for calculation: (B, H) -> (H, B)
            smoothed = torch.sparse.mm(self.adj_matrix, energy.t()).t()
            
            # Add to original (Residual Connection)
            energy = energy + smoothed
            
        return energy

    def readout(self, energy):
        """
        Aggregate hub scores → label scores (Vectorized)
        """
        # Create a mapping matrix from Hubs -> Classes
        # This is faster than the loop
        batch_sz = energy.shape[0]
        n_classes = Config.N_TASKS * Config.CLASSES_PER_TASK # Or find max label
        
        # Simple Loop (Safe for now)
        unique_labels = torch.unique(self.labels)
        scores = torch.zeros((batch_sz, len(unique_labels)), device=self.device)
        
        for i, lbl in enumerate(unique_labels):
            # Find all hubs belonging to this class
            mask = (self.labels == lbl)
            # Sum their energy
            if mask.any():
                scores[:, i] = energy[:, mask].sum(dim=1)

        best_indices = torch.argmax(scores, dim=1)
        return unique_labels[best_indices]
            
    def predict(self, input_features, mask, mode="soft", top_k=3):
        # 1. Setup Input
        if not isinstance(input_features, torch.Tensor):
            input_features = torch.tensor(input_features, dtype=torch.float32)
        input_features = input_features.to(self.device)
        
        # 2. Chunking (No Rotation for now - Safety First!)
        chunks = input_features.chunk(self.n_chunks, dim=1)
        
        batch_sz = input_features.shape[0]
        n_hubs = self.wiring.shape[0]
        energy = torch.zeros((batch_sz, n_hubs), device=self.device)

        # 3. Accumulate Votes
        for c in range(self.n_chunks):
            if not mask[c]: continue

            leaf_bank = self.leaves[c] # (K, D)
            chunk = chunks[c]          # (B, D)

            # Similarity: (B, K)
            sims = torch.matmul(chunk, leaf_bank.t())

            if mode == "hard":
                # Top-K Voting
                vals, topk_indices = sims.topk(top_k, dim=1) # (B, k)
                
                # Expand for Broadcast Comparison with Hubs
                # Hubs have 1 code per chunk: self.wiring[:, c] -> (H)
                hub_codes = self.wiring[:, c] 
                
                # Check: Does Hub's code exist in input's Top-K?
                # (B, k, 1) == (1, 1, H) -> (B, k, H)
                matches = (topk_indices.unsqueeze(2) == hub_codes.view(1, 1, -1))
                
                # If any of the top-k matched, we get a hit
                hits = matches.any(dim=1).float()
                energy += hits
            
            else:
                # Soft Voting (Direct Energy Transfer) - SAFER BASELINE
                # We pull the specific scalar similarity for the Hub's chosen word
                # 1. Get code for every hub: (H)
                hub_codes = self.wiring[:, c]
                # 2. Select those columns from sims: (B, K) -> (B, H)
                # Note: This index select can be memory heavy, strictly dot product is better:
                hub_leaves = leaf_bank[hub_codes] # (H, D)
                energy += torch.matmul(chunk, hub_leaves.t())

        # 4. Diffusion (Optional: Try steps=0 first if debugging)
        if mode == "hard":
            energy = self.diffuse(energy, steps=1)

        # 5. Readout
        # return self.readout(energy)
        return torch.argmax(energy, dim=1)