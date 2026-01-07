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

    def predict(self, input_features, mask):
        """
        input_features: (Batch, 384)
        mask: List of bools, length 8 (True = Visible, False = Occluded)
        """
        batch_sz = input_features.shape[0]
        n_hubs = self.wiring.shape[0]
        energy = torch.zeros((batch_sz, n_hubs), device=self.device)
        
        # Split input into chunks
        chunks = input_features.chunk(self.n_chunks, dim=1)
        
        for c in range(self.n_chunks):
            if mask[c]:
                leaf_bank = self.leaves[c]
                # Gather the leaf vectors corresponding to every hub's c-th connection
                hub_leaves = leaf_bank[self.wiring[:, c]] # (Hubs, Chunk_Dim)
                
                # Dot product: (Batch, Chunk_Dim) @ (Chunk_Dim, Hubs) -> (Batch, Hubs)
                energy += torch.matmul(chunks[c], hub_leaves.t())
        
        return torch.argmax(energy, dim=1)