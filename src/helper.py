import numpy as np
import torch
from .config import Config

def build_hubs(quantized_codes, labels):
    """
    quantized_codes: (N, n_chunks)
    labels: (N,)
    """
    hubs = {}
    hub_labels = []

    for code, lbl in zip(quantized_codes, labels):
        key = tuple(code.tolist())
        if key not in hubs:
            hubs[key] = len(hubs)
            hub_labels.append(lbl)

    hub_indices = torch.tensor(np.array(list(hubs.keys())))
    hub_labels = torch.tensor(np.array(hub_labels)).to(Config.DEVICE)

    return hub_indices, hub_labels


def build_adjacency(hub_indices):
    """
    Hebbian-style co-activation graph
    """
    H, C = hub_indices.shape
    adj = torch.zeros((H, H), dtype=torch.float32).to(Config.DEVICE)

    for i in range(H):
        matches = (hub_indices == hub_indices[i]).sum(axis=1)
        neighbors = np.where(matches >= Config.EDGE_MATCH_THRESHOLD)[0]

        for j in neighbors:
            if i != j:
                adj[i, j] += 1

    # sparsify
    for i in range(H):
        keep = np.argsort(adj[i])[-Config.MAX_NEIGHBORS:]
        mask = np.ones(H, dtype=bool)
        mask[keep] = False
        adj[i][mask] = 0

    # normalize
    adj /= (adj.sum(axis=1, keepdims=True) + 1e-6)

    return adj
