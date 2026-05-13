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

def compute_average_accuracy(accuracy_matrix):
    """
    Computes Average Final Accuracy (A_T) after learning all T tasks.
    Formula: A_T = (1/T) * sum_{j=1}^T R_{T,j}
    
    Args:
        accuracy_matrix (np.ndarray): A T x T matrix where R[i,j] is the 
                                      accuracy on task j after training on task i.
    Returns:
        float: The average final accuracy.
    """
    # The last row represents the model's accuracy on all tasks after training on the final task.
    final_row = accuracy_matrix[-1, :]
    return np.mean(final_row)

def compute_average_forgetting(accuracy_matrix):
    """
    Computes Average Forgetting (F) across all tasks (excluding the last one).
    Forgetting for task j is the difference between its peak accuracy and its final accuracy.
    Formula: F = (1 / T-1) * sum_{j=1}^{T-1} ( max_{i < T} R_{i,j} - R_{T,j} )
    
    Args:
        accuracy_matrix (np.ndarray): A T x T matrix.
    Returns:
        float: The average forgetting. Lower is better.
    """
    T = accuracy_matrix.shape[0]
    
    if T <= 1:
        return 0.0  # No forgetting possible with only one task

    forgetting_list = []
    
    # We only compute forgetting for tasks 0 to T-2 (the last task hasn't been forgotten yet)
    for j in range(T - 1):
        # Best accuracy recorded for task j at any point before the final state
        best_acc = np.max(accuracy_matrix[:T, j])
        # Final accuracy for task j
        final_acc = accuracy_matrix[-1, j]
        
        forgetting_list.append(best_acc - final_acc)
        
    return np.mean(forgetting_list)

def build_accuracy_matrix(model, all_test_features, all_test_labels, current_task_id, mode='soft'):
    """
    Runs evaluation on all tasks seen so far to populate row `i` of the Accuracy Matrix R.
    Call this inside your sequential training loop after finishing each task.
    """
    row_accuracies = []
    for task_id in range(Config.N_TASKS):
        start_cls = task_id * Config.CLASSES_PER_TASK
        end_cls = (task_id + 1) * Config.CLASSES_PER_TASK
        
        task_mask = (all_test_labels >= start_cls) & (all_test_labels < end_cls)
        task_idxs = torch.where(task_mask)[0]
        
        if len(task_idxs) == 0:
            row_accuracies.append(0.0)
            continue
            
        t_feats = all_test_features[task_idxs]
        t_lbls = all_test_labels[task_idxs]
        
        preds = model.predict(t_feats, mask=None, mode=mode)
        acc = (preds == t_lbls).float().mean().item()
        row_accuracies.append(acc)
        
    return row_accuracies

def build_accuracy_matrix_row(model, all_test_features, all_test_labels, mode='soft'):
    """
    Evaluates the model on ALL tasks to populate ONE row of the Accuracy Matrix R.
    """
    row_accuracies = []
    for task_id in range(Config.N_TASKS):
        start_cls = task_id * Config.CLASSES_PER_TASK
        end_cls = (task_id + 1) * Config.CLASSES_PER_TASK
        
        task_mask = (all_test_labels >= start_cls) & (all_test_labels < end_cls)
        task_idxs = torch.where(task_mask)[0]
        
        if len(task_idxs) == 0:
            row_accuracies.append(0.0)
            continue
            
        t_feats = all_test_features[task_idxs]
        t_lbls = all_test_labels[task_idxs]
        
        preds = model.predict(t_feats, mask=None, mode=mode)
        acc = (preds == t_lbls).float().mean().item()
        row_accuracies.append(acc)
        
    return row_accuracies