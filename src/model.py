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
    def __init__(self, codebooks, hub_indices, graph_labels, adjacency, input_dim=384):
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

        self.adjacency = torch.tensor(adjacency, dtype=torch.float32).to(self.device)

    def diffuse(self, energy):
        A = self.adjacency
        alpha = Config.DIFFUSION_ALPHA
        propagated = torch.matmul(energy, A)
        return alpha * energy + (1 - alpha) * propagated

    def readout(self, energy):
        """
        Aggregate hub scores → label scores
        """
        unique_labels = torch.unique(self.labels)
        scores = []

        for lbl in unique_labels:
            mask = (self.labels == lbl)
            scores.append(energy[:, mask].sum(dim=1))

        scores = torch.stack(scores, dim=1)
        return unique_labels[torch.argmax(scores, dim=1)]
            
    def predict(self, input_features, mask, mode="hard", top_k=3):
        if not isinstance(input_features, torch.Tensor):
            input_features = torch.tensor(input_features, dtype=torch.float32)

        input_features = input_features.to(self.device)
        B = input_features.shape[0]
        H = self.wiring.shape[0]

        energy = torch.zeros((B, H), device=self.device)

        rotated = input_features @ self.rotation_matrix
        chunks = rotated.chunk(self.n_chunks, dim=1)

        for c in range(self.n_chunks):
            if not mask[c]:
                continue

            leaf_bank = self.leaves[c]
            chunk = chunks[c]

            sims = torch.matmul(chunk, leaf_bank.t())
            _, topk = sims.topk(top_k, dim=1)

            hub_codes = self.wiring[:, c]

            matches = (topk.unsqueeze(2) == hub_codes.view(1, 1, -1))
            hits = matches.any(dim=1).float()

            energy += hits

        # 🔁 Diffusion
        energy = self.diffuse(energy)

        # 🏷 Label-level decision
        return self.readout(energy)