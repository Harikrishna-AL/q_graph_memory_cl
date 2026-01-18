import numpy as np
import torch
from .config import Config

def build_hubs(quantized_codes, labels):
    """
    quantized_codes: (N, n_chunks)  [numpy array]
    labels: (N,)                    [torch tensor, possibly on GPU]
    """
    hubs = {}
    hub_labels = []

    # Ensure labels are on CPU for iteration (cheap, 1D)
    labels_cpu = labels.detach().cpu()

    for code, lbl in zip(quantized_codes, labels_cpu):
        key = tuple(code.tolist())
        if key not in hubs:
            hubs[key] = len(hubs)
            hub_labels.append(lbl.item())  # <-- scalar, not tensor

    hub_indices = torch.tensor(
        np.array(list(hubs.keys())),
        dtype=torch.long,
        device=Config.DEVICE
    )

    hub_labels = torch.tensor(
        hub_labels,
        dtype=torch.long,
        device=Config.DEVICE
    )

    return hub_indices, hub_labels


def build_hub_neighbors(hub_indices, k=8):
    """
    Build sparse hub neighbors based on Hamming similarity
    hub_indices: (H, C)
    Returns: list of neighbor indices per hub
    """
    hub_indices = hub_indices.cpu().numpy()
    H, C = hub_indices.shape

    neighbors = [[] for _ in range(H)]

    for i in range(H):
        # Hamming similarity
        matches = (hub_indices == hub_indices[i]).sum(axis=1)

        # Exclude self
        matches[i] = -1

        # Top-k neighbors
        nn = np.argsort(matches)[-k:]
        neighbors[i] = nn.tolist()

    return neighbors

