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
    def __init__(self, codebooks, hub_indices, graph_labels):
        super().__init__()
        self.n_chunks = len(codebooks)
        self.device = Config.DEVICE
        
        # Leaf Nodes (Visual Words)
        self.leaves = nn.ParameterList([
            nn.Parameter(torch.tensor(cb, dtype=torch.float32).to(self.device), requires_grad=False) 
            for cb in codebooks
        ])
        
        # Hub Nodes (Wiring)
        # Use int64 (long) for indices
        self.wiring = torch.tensor(hub_indices, dtype=torch.long).to(self.device)
        self.labels = graph_labels

        R = torch.nn.init.orthogonal_(torch.empty(input_dim, input_dim))
        self.register_buffer('rotation_matrix', R)

    def predict(self, input_features, mask, mode='soft'):
        """
        input_features: (Batch, 384)
        mask: List of bools, length 8 (True = Visible, False = Occluded)
        mode: 'soft' (Dot Product) or 'hard' (Discrete Voting)
        """
        batch_sz = input_features.shape[0]
        n_hubs = self.wiring.shape[0]
        
        # We accumulate "Energy" (Votes) for each Hub
        energy = torch.zeros((batch_sz, n_hubs), device=self.device)

        rotated_features = torch.matmul(input_features, self.rotation_matrix)
        
        # 2. Slice the *rotated* features into chunks
        chunks = rotated_features.chunk(self.n_chunks, dim=1)
        # Split input into M chunks
        # chunks = input_features.chunk(self.n_chunks, dim=1)
        
        for c in range(self.n_chunks):
            # 1. ROBUSTNESS CHECK: If mask is False, we completely IGNORE this chunk.
            # (NCM cannot do this; it sees 'zeros' and calculates distance on them)
            if not mask[c]:
                continue
                
            leaf_bank = self.leaves[c] # Shape: (K, 48)
            input_chunk = chunks[c]    # Shape: (Batch, 48)

            if mode == 'hard':
                # --- HARD VOTING (The "Graph" Way) ---
                # 1. Quantize: Find the nearest Visual Word index for the input
                #    (Batch, 48) @ (48, K) -> (Batch, K)
                sims = torch.matmul(input_chunk, leaf_bank.t()) 
                input_codes = torch.argmax(sims, dim=1) # (Batch,) containing integers [0...255]
                
                # 2. Vote: Does this Input Code match the Hub's Code?
                #    Hub Wiring for this chunk: (Num_Hubs,)
                hub_codes = self.wiring[:, c] 
                
                #    Compare: (Batch, 1) == (1, Num_Hubs) -> (Batch, Num_Hubs) Boolean Mask
                matches = (input_codes.unsqueeze(1) == hub_codes.unsqueeze(0))
                
                # 3. Add Votes (1.0 for match, 0.0 for no match)
                energy += matches.float()

            else:
                # --- SOFT VOTING (The "Similarity" Way) ---
                # This is your old code. Good for Clean Accuracy, less unique.
                hub_leaves = leaf_bank[self.wiring[:, c]] # (Hubs, 48)
                energy += torch.matmul(input_chunk, hub_leaves.t())
        
        return torch.argmax(energy, dim=1)