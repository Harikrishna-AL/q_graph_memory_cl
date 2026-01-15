import torch
import torch.nn as nn
import numpy as np
from .config import Config

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
    def __init__(self, codebooks, hub_indices, graph_labels, input_dim=384):
        super().__init__()
        self.n_chunks = len(codebooks)
        self.device = Config.DEVICE
        
        # 1. Leaf Nodes (Visual Words) -> Move to Device
        self.leaves = nn.ParameterList([
            nn.Parameter(torch.tensor(cb, dtype=torch.float32).to(self.device), requires_grad=False) 
            for cb in codebooks
        ])
        
        # 2. Hub Nodes (Wiring) -> Move to Device
        self.wiring = torch.tensor(hub_indices, dtype=torch.long).to(self.device)
        self.labels = graph_labels # This is usually a list/numpy array, so it stays on CPU
            
    def predict(self, input_features, mask, mode='soft', top_k=3):
        """
        input_features: (Batch, 384)
        mask: List of bools, length 8 (True = Visible, False = Occluded)
        mode: 'soft' (Dot Product) or 'hard' (Discrete Voting)
        """
        # Ensure input is a Tensor on the correct device
        if not isinstance(input_features, torch.Tensor):
            input_features = torch.tensor(input_features, dtype=torch.float32)
        input_features = input_features.to(self.device)

        batch_sz = input_features.shape[0]
        n_hubs = self.wiring.shape[0]
        
        # Energy accumulator on Device
        energy = torch.zeros((batch_sz, n_hubs), device=self.device) 
        chunks = input_features.chunk(self.n_chunks, dim=1)
        
        for c in range(self.n_chunks):
            # ROBUSTNESS CHECK: Skip masked chunks
            if not mask[c]:
                continue
                
            leaf_bank = self.leaves[c] # (K, Chunk_Dim)
            input_chunk = chunks[c]    # (Batch, Chunk_Dim)

            if mode == 'hard':
                # --- HARD VOTING (Top-K) ---
                # 1. Quantize: Find Top-K nearest Visual Word indices
                # (Batch, 48) @ (48, K) -> (Batch, K)
                sims = torch.matmul(input_chunk, leaf_bank.t()) 
                
                # Get Top-K indices: (Batch, top_k)
                _, topk_indices = sims.topk(top_k, dim=1)
                
                # 2. Vote: Check if Hub's code exists in Top-K
                # Hub Wiring for this chunk: (Num_Hubs,)
                hub_codes = self.wiring[:, c] 
                
                # Broadcasting Match:
                # (Batch, top_k, 1) == (1, 1, Num_Hubs) -> (Batch, top_k, Num_Hubs)
                matches = (topk_indices.unsqueeze(2) == hub_codes.view(1, 1, -1))
                
                # If ANY of the Top-K words match, it's a hit
                hits = matches.any(dim=1).float() 
                
                energy += hits

            else:
                # --- SOFT VOTING ---
                hub_leaves = leaf_bank[self.wiring[:, c]] # (Hubs, 48)
                energy += torch.matmul(input_chunk, hub_leaves.t())
        
        return torch.argmax(energy, dim=1)