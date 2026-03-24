import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config

from .config import Config

CACHE_DIR = "cache"

# Mapping: backbone name -> output feature dimension
_BACKBONE_DIMS = {
    "dinov2": 384,       # ViT-S/14
    "dinov2_giant": 1536,# ViT-g/14
    "siglip": 1152,      # vit_so400m_patch14_siglip_384
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet50_ssl": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
    "efficientnet_b0": 1280,
    "efficientnet_b1": 1280,
    "efficientnet_b2": 1408,
    "efficientnet_b3": 1536,
    "efficientnet_b4": 1792,
}


def get_cache_paths(dataset_name, use_train):
    """Return (feat_path, lbl_path) for the current backbone + dataset + split."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    split = "train" if use_train else "val"
    # Use backbone name so DINOv2 and ResNet caches never collide
    backbone = Config.BACKBONE.lower().replace("/", "_")
    dname = dataset_name.lower().strip()

    feat_path = f"{CACHE_DIR}/{dname}_{backbone}_{split}_features.npy"
    lbl_path  = f"{CACHE_DIR}/{dname}_{backbone}_{split}_labels.npy"

    return feat_path, lbl_path


def load_cached_features(dataset_name, use_train):
    feat_path, lbl_path = get_cache_paths(dataset_name, use_train)

    if os.path.exists(feat_path) and os.path.exists(lbl_path):
        backbone = Config.BACKBONE
        print(
            f"💾 Loading cached {backbone} features for {dataset_name} "
            f"({'Train' if use_train else 'Test'})..."
        )
        try:
            features = np.load(feat_path, mmap_mode="r")  # memory-efficient
            labels = np.load(lbl_path)
            return features, labels
        except Exception as e:
            print(f"⚠️  Cache file corrupted or incompatible. Re-extracting. Error: {e}")
            return None, None

    print(f"⚠️  No cache found at {feat_path}")
    return None, None


# 3. Update: Pass dataset_name through
def save_cached_features(features, labels, dataset_name, use_train):
    feat_path, lbl_path = get_cache_paths(dataset_name, use_train)

    # Ensure inputs are numpy
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    np.save(feat_path, features)
    np.save(lbl_path, labels)
    print(f"✅ Cached {len(features)} features to {feat_path}")


# ---------------------------------------------------------------------------
# ResNet feature extractor wrapper
# ---------------------------------------------------------------------------

class _ResNetExtractor(nn.Module):
    """
    Wraps a torchvision ResNet and returns average-pooled penultimate features
    (before the final classification head) so the API matches DINOv2.
    """
    def __init__(self, resnet):
        super().__init__()
        # Keep everything except the final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        # Output shape: (B, C, 1, 1) -> flatten to (B, C)
        return self.backbone(x).flatten(1)


class _EfficientNetExtractor(nn.Module):
    """
    Wraps a torchvision EfficientNet and returns average-pooled penultimate features
    (before the final classification head).
    """
    def __init__(self, efficientnet):
        super().__init__()
        # EfficientNet has 'features' (conv blocks) and 'classifier' (dropout + linear)
        # We also need the avgpool layer which sits between them.
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)


# ---------------------------------------------------------------------------
# Unified backbone loader
# ---------------------------------------------------------------------------

def load_backbone():
    """
    Load the feature-extraction backbone specified by Config.BACKBONE.

    Supported values
    ----------------
    "dinov2"              DINOv2 ViT-S/14  (384-d, default)
    "resnet18"            512-d
    "resnet34"            512-d
    "resnet50"            2048-d
    "resnet101"           2048-d
    "resnet152"           2048-d

    Sets Config.FEATURE_DIM as a side-effect so the rest of the pipeline
    is automatically dimension-aware.
    """
    backbone_name = Config.BACKBONE.lower().strip()

    if backbone_name not in _BACKBONE_DIMS:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Choose one of: {list(_BACKBONE_DIMS.keys())}"
        )

    # Set global feature dimension
    Config.FEATURE_DIM = _BACKBONE_DIMS[backbone_name]

    if backbone_name == "dinov2":
        print(f"🦖 Loading DINOv2 ({Config.DINO_MODEL}, dim={Config.FEATURE_DIM})...")
        model = torch.hub.load(Config.DINO_REPO, Config.DINO_MODEL)
    elif backbone_name == "dinov2_giant":
        print(f"🦖 Loading DINOv2 Giant (dinov2_vitg14_reg, dim={Config.FEATURE_DIM})...")
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    elif backbone_name == "siglip":
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for SigLIP models. Run: pip install timm")
        print(f"👀 Loading SigLIP SO400m (dim={Config.FEATURE_DIM})...")
        model = timm.create_model('vit_so400m_patch14_siglip_384.webli', pretrained=True, num_classes=0)
    elif backbone_name == "resnet50_ssl":
        print(f"🧬 Loading Self-Supervised ResNet50 (DINO Contrastive, dim={Config.FEATURE_DIM})...")
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    else:
        import torchvision.models as tvm
        print(f"🏗️  Loading {backbone_name} (dim={Config.FEATURE_DIM})...")
        constructor = getattr(tvm, backbone_name)
        weights_name = "IMAGENET1K_V1"
        if "efficientnet" in backbone_name:
            # e.g., EfficientNet_B0_Weights.IMAGENET1K_V1
            weights_name = "DEFAULT"
        
        vision_model = constructor(weights=weights_name)
        if "resnet" in backbone_name:
            model = _ResNetExtractor(vision_model)
        elif "efficientnet" in backbone_name:
            model = _EfficientNetExtractor(vision_model)
        else:
            raise ValueError(f"Unknown backbone architecture class: {backbone_name}")

    model.to(Config.DEVICE)
    model.eval()
    return model


# Keep the old name as an alias so existing call-sites in main.py still work
def load_dino():
    """Deprecated alias for load_backbone(). Use load_backbone() instead."""
    return load_backbone()


def extract_features(backbone, dataloader):
    """Run `backbone` over `dataloader` and return L2-normalised features + labels."""
    print("🔍 Extracting Features (this may take a moment)...")
    all_feats, all_lbls = [], []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], (tuple, list)) and len(batch[0]) >= 2:
                # default_collate completely aborted; batch is `[(img, lbl, ...), (img, lbl, ...)]`
                imgs = [item[0] for item in batch]
                lbls = [item[1] for item in batch]
            elif isinstance(batch, dict):
                imgs = batch.get("image", batch.get("images", batch.get("img", None)))
                lbls = batch.get("label", batch.get("labels", batch.get("target", None)))
            else:
                imgs, lbls = batch[0], batch[1]
                
            # Deep unpack and resolve tuples/lists
            if isinstance(imgs, (tuple, list)):
                if len(imgs) > 0 and isinstance(imgs[0], torch.Tensor):
                    # Filter out any non-tensors just in case!
                    imgs = torch.stack([x for x in imgs if isinstance(x, torch.Tensor)])
                elif len(imgs) > 0:
                    imgs = imgs[0]
            
            if isinstance(lbls, (tuple, list)):
                if len(lbls) > 0 and isinstance(lbls[0], torch.Tensor):
                    lbls = torch.stack([x for x in lbls if isinstance(x, torch.Tensor)])
                elif len(lbls) > 0:
                    lbls = torch.tensor(lbls)
            
            if not isinstance(imgs, torch.Tensor):
                raise ValueError(f"CRITICAL ERROR: 'imgs' is of type {type(imgs)} inside subset DataLoader instead of a Tensor. Batch structure: {type(batch)}")

            imgs = imgs.to(Config.DEVICE)
            
            # 🔥 Accelerate throughput by 3-4x using Mixed Precision (GPU Tensor Cores)
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            if device_type == "cpu":
                # CPU autocast only supports bfloat16 but often isn't globally available, so bypass it cleanly
                f = backbone(imgs)
            else:
                with torch.autocast(device_type="cuda"):
                    f = backbone(imgs)
            
            f = f.float().cpu().numpy()
            # L2 Normalize
            f = f / np.linalg.norm(f, axis=1, keepdims=True)
            all_feats.append(f)
            if torch.is_tensor(lbls):
                all_lbls.append(lbls.cpu().numpy())
            else:
                all_lbls.append(np.array(lbls))

    features = np.concatenate(all_feats)
    labels = np.concatenate(all_lbls)
    return features, labels


class ContinualGraph(nn.Module):
    def __init__(self, codebooks, hub_indices, graph_labels, neighbors, input_dim=384):
        super().__init__()
        self.n_chunks = len(codebooks)
        self.device = Config.DEVICE

        # 1. Leaf Nodes (Visual Words)
        self.leaves = nn.ParameterList(
            [
                nn.Parameter(
                    torch.tensor(cb, dtype=torch.float32).to(self.device),
                    requires_grad=False,
                )
                for cb in codebooks
            ]
        )

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

        # Simple Loop (Safe for now)
        unique_labels = torch.unique(self.labels)
        scores = torch.zeros((batch_sz, len(unique_labels)), device=self.device)

        for i, lbl in enumerate(unique_labels):
            # Find all hubs belonging to this class
            mask = self.labels == lbl
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
            if not mask[c]:
                continue

            leaf_bank = self.leaves[c]  # (K, D)
            chunk = chunks[c]  # (B, D)

            # Similarity: (B, K)
            sims = torch.matmul(chunk, leaf_bank.t())

            if mode == "hard":
                # Top-K Voting
                vals, topk_indices = sims.topk(top_k, dim=1)  # (B, k)

                # Expand for Broadcast Comparison with Hubs
                # Hubs have 1 code per chunk: self.wiring[:, c] -> (H)
                hub_codes = self.wiring[:, c]

                # Check: Does Hub's code exist in input's Top-K?
                # (B, k, 1) == (1, 1, H) -> (B, k, H)
                matches = topk_indices.unsqueeze(2) == hub_codes.view(1, 1, -1)

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
                hub_leaves = leaf_bank[hub_codes]  # (H, D)
                energy += torch.matmul(chunk, hub_leaves.t())

        # 4. Diffusion (Optional: Try steps=0 first if debugging)
        if mode == "hard":
            energy = self.diffuse(energy, steps=1)

        # 5. Readout
        # return self.readout(energy)
        return torch.argmax(energy, dim=1)


class BioEpisodicGraph(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.device = Config.DEVICE
        self.input_dim = input_dim

        # ── Open-world prototype memory (dicts keyed by class_id) ──────────
        # No n_classes needed — new classes are discovered lazily from the stream.
        self._proto_sum   = {}   # class_id -> (D,) sum tensor (unnormalised)
        self._proto_count = {}   # class_id -> float count
        self._proto_m2    = {}   # class_id -> (D,) Welford M2 for online variance
        self._class_unc   = {}   # class_id -> float EMA uncertainty

        proj_dim = int(min(getattr(Config, "BIO_PROJ_DIM", 128), input_dim))
        proj_init = torch.eye(input_dim, device=self.device)[:, :proj_dim]
        self.register_buffer("proj_matrix", proj_init.contiguous())

        # Episodic Graph (Hippocampus) — already dynamic
        self.nodes = torch.zeros((0, input_dim), device=self.device)
        self.node_labels = torch.empty(0, dtype=torch.long, device=self.device)

        # Activation histories (tracked per node)
        self.node_freq = torch.zeros(0, device=self.device)
        self.node_strength = torch.zeros(0, device=self.device)
        self.node_consistency = torch.zeros(0, device=self.device)
        self.node_fidelity = torch.zeros(0, device=self.device)

        # Bio readout controls
        self.proto_weight = getattr(Config, "BIO_PROTO_WEIGHT", 0.55)
        self.node_temp = getattr(Config, "BIO_NODE_TEMP", 0.08)
        self.proto_temp = getattr(Config, "BIO_PROTO_TEMP", 0.10)
        self.merge_threshold = getattr(Config, "BIO_MERGE_THRESHOLD", 0.30)
        self.max_nodes_per_class = getattr(Config, "BIO_MAX_NODES_PER_CLASS", 64)
        self.kmeans_per_class = getattr(Config, "BIO_KMEANS_PER_CLASS", 8)
        self.use_mahalanobis = getattr(Config, "BIO_USE_MAHALANOBIS", True)
        self.var_eps = getattr(Config, "BIO_VAR_EPS", 1e-3)
        self.uncert_momentum = getattr(Config, "BIO_UNCERTAINTY_MOMENTUM", 0.95)
        self.dynamic_budget_floor = getattr(Config, "BIO_DYNAMIC_BUDGET_FLOOR", 0.25)
        self.use_projection = getattr(Config, "BIO_USE_PROJECTION", True)
        self.proj_margin = getattr(Config, "BIO_PROJ_MARGIN", 0.20)
        self.proj_lr = getattr(Config, "BIO_PROJ_LR", 0.03)
        self.proj_steps = getattr(Config, "BIO_PROJ_STEPS", 30)
        self.proj_ortho_reg = getattr(Config, "BIO_PROJ_ORTHO_REG", 1e-2)

    # ── Open-world prototype helpers ─────────────────────────────────────

    @property
    def seen_classes(self):
        """Sorted list of class IDs seen so far."""
        return sorted(self._proto_count.keys())

    def _ensure_class(self, lbl: int):
        """Lazily initialise dict entries for a new class."""
        if lbl not in self._proto_sum:
            d = self.input_dim
            self._proto_sum[lbl]   = torch.zeros(d, device=self.device)
            self._proto_count[lbl] = 0.0
            self._proto_m2[lbl]    = torch.zeros(d, device=self.device)
            self._class_unc[lbl]   = 0.0

    def _get_proto(self, lbl: int) -> torch.Tensor:
        """Return the L2-normalised running mean for class lbl."""
        self._ensure_class(lbl)
        count = self._proto_count[lbl]
        if count == 0:
            return torch.zeros(self.input_dim, device=self.device)
        mean = self._proto_sum[lbl] / count
        return F.normalize(mean, p=2, dim=0)

    def _build_proto_tensor(self):
        """
        Build (C_seen x D) prototype matrix and a list of corresponding class IDs.
        Returns: (proto_tensor, class_order_list)
        """
        classes = self.seen_classes
        if not classes:
            return None, []
        protos = torch.stack([self._get_proto(c) for c in classes])  # (C, D)
        return protos, classes

    # ── Welford running stats helpers ────────────────────────────────────

    @property
    def prototypes_counts(self):
        """1-D tensor of counts indexed by class ID (for compat). Zero for unseen classes."""
        if not self._proto_count:
            return torch.zeros(0, device=self.device)
        max_cls = max(self._proto_count.keys())
        counts = torch.zeros(max_cls + 1, device=self.device)
        for c, n in self._proto_count.items():
            counts[c] = float(n)
        return counts

    def _append_node(self, node_vec, label):
        self.nodes = torch.cat([self.nodes, node_vec.view(1, -1)], dim=0)
        self.node_labels = torch.cat(
            [self.node_labels, torch.tensor([label], device=self.device)], dim=0
        )
        self.node_freq = torch.cat(
            [self.node_freq, torch.tensor([1.0], device=self.device)]
        )
        self.node_strength = torch.cat(
            [self.node_strength, torch.tensor([1.0], device=self.device)]
        )
        self.node_consistency = torch.cat(
            [self.node_consistency, torch.tensor([1.0], device=self.device)]
        )
        self.node_fidelity = torch.cat(
            [self.node_fidelity, torch.tensor([1.0], device=self.device)]
        )

    def _norm01(self, x):
        return (x - x.min()) / (x.max() - x.min() + 1e-6)

    def _project(self, x, proj_matrix=None):
        if not self.use_projection:
            return x
        p = self.proj_matrix if proj_matrix is None else proj_matrix
        return x @ p

    def _prototype_variance(self):
        """
        Return a (C_seen x D) tensor of per-class per-dimension variance,
        built from the Welford M2 accumulators stored in _proto_m2.
        Indexed by the same order as seen_classes.
        """
        classes = self.seen_classes
        if not classes:
            return torch.full((0, self.input_dim), self.var_eps, device=self.device)
        rows = []
        for c in classes:
            count = self._proto_count.get(c, 0.0)
            m2    = self._proto_m2.get(c, torch.zeros(self.input_dim, device=self.device))
            denom = max(1.0, count - 1.0)
            var   = torch.clamp(m2 / denom, min=self.var_eps)
            rows.append(var)
        return torch.stack(rows)  # (C_seen, D)

    def _class_budgets(self):
        if self.nodes.shape[0] == 0:
            return {}

        active = torch.unique(self.node_labels)
        n_active = int(active.numel())
        if n_active == 0:
            return {}

        per_class_cap = int(self.max_nodes_per_class)
        total_budget = max(n_active, n_active * per_class_cap)
        base = max(1, int(per_class_cap * self.dynamic_budget_floor))
        base_total = min(total_budget, base * n_active)
        remaining = total_budget - base_total

        # Variance: build a (C_seen, D) tensor then index into it
        proto_var_matrix = self._prototype_variance()  # (C_seen, D)
        cls_list = self.seen_classes  # same order as _prototype_variance
        cls_to_idx = {c: i for i, c in enumerate(cls_list)}

        # uncertainty tensors aligned with `active`
        u_var_list, u_ema_list = [], []
        for lbl_t in active:
            lbl = int(lbl_t.item())
            idx = cls_to_idx.get(lbl)
            if idx is not None:
                u_var_list.append(proto_var_matrix[idx].mean())
            else:
                u_var_list.append(torch.tensor(0.0, device=self.device))
            u_ema_list.append(torch.tensor(self._class_unc.get(lbl, 0.0), device=self.device))

        u_var = self._norm01(torch.stack(u_var_list))
        u_ema = self._norm01(torch.stack(u_ema_list))
        uncertainty = 0.5 * u_var + 0.5 * u_ema
        denom = uncertainty.sum().item()
        if denom <= 0:
            uncertainty = torch.ones_like(uncertainty) / max(1, len(uncertainty))
        else:
            uncertainty = uncertainty / uncertainty.sum()

        extra_float = remaining * uncertainty
        extra = torch.floor(extra_float).long()
        remainder = int(remaining - int(extra.sum().item()))
        if remainder > 0:
            frac = extra_float - extra.float()
            top_idx = torch.topk(frac, k=min(remainder, len(frac))).indices
            extra[top_idx] += 1

        budgets = {}
        for i, lbl in enumerate(active):
            budgets[int(lbl.item())] = base + int(extra[i].item())
        return budgets

    def _cap_nodes_per_class(self, node_priority=None):
        if self.nodes.shape[0] == 0:
            return

        budgets = self._class_budgets()
        keep_mask = torch.zeros(
            self.nodes.shape[0], dtype=torch.bool, device=self.device
        )
        for lbl in torch.unique(self.node_labels):
            idxs = torch.where(self.node_labels == lbl)[0]
            class_budget = budgets.get(int(lbl.item()), int(self.max_nodes_per_class))
            if idxs.numel() <= class_budget:
                keep_mask[idxs] = True
                continue

            if node_priority is not None:
                score = node_priority[idxs]
            else:
                proto = self._get_proto(int(lbl.item()))
                dist = torch.norm(self.nodes[idxs] - proto, dim=1)
                uniq = self._norm01(dist)
                fidelity = self.node_fidelity[idxs] / (self.node_freq[idxs] + 1e-6)
                strength = self.node_strength[idxs] / (self.node_freq[idxs] + 1e-6)
                score = (
                    0.5 * self._norm01(fidelity)
                    + 0.3 * self._norm01(strength)
                    + 0.2 * uniq
                )

            topk = torch.topk(score, k=class_budget).indices
            keep_mask[idxs[topk]] = True

        self.nodes = self.nodes[keep_mask]
        self.node_labels = self.node_labels[keep_mask]
        self.node_freq = self.node_freq[keep_mask]
        self.node_strength = self.node_strength[keep_mask]
        self.node_consistency = self.node_consistency[keep_mask]
        self.node_fidelity = self.node_fidelity[keep_mask]

    def online_step(self, batch_features, batch_labels):
        """
        ONLINE PHASE: Rapidly record episodes and update running prototypes.
        """
        batch_features = batch_features.to(self.device)
        batch_labels = batch_labels.to(self.device)
        batch_features = F.normalize(batch_features, p=2, dim=1)

        # 1. Welford online update for running prototype (mean + variance)
        for i in range(len(batch_features)):
            feat = batch_features[i]
            lbl  = int(batch_labels[i].item())
            self._ensure_class(lbl)

            n_new  = self._proto_count[lbl] + 1.0
            mean_prev = self._proto_sum[lbl] / max(1.0, self._proto_count[lbl])

            delta  = feat - mean_prev
            new_sum = self._proto_sum[lbl] + feat
            mean_new = new_sum / n_new
            delta2 = feat - mean_new

            self._proto_sum[lbl]   = new_sum
            self._proto_count[lbl] = n_new
            self._proto_m2[lbl]   += delta * delta2

            if n_new > 1:
                sim = F.cosine_similarity(
                    feat.view(1, -1), mean_new.view(1, -1), dim=1
                ).item()
                uncert = max(0.0, 1.0 - sim)
                self._class_unc[lbl] = (
                    self.uncert_momentum * self._class_unc[lbl]
                    + (1.0 - self.uncert_momentum) * uncert
                )

        # 2. Update Activation History for EXISTING nodes
        if self.nodes.shape[0] > 0:
            dists = torch.cdist(batch_features, self.nodes)
            min_dists, nearest_nodes = torch.min(dists, dim=1)

            for i in range(len(batch_features)):
                node_idx = nearest_nodes[i]
                self.node_freq[node_idx] += 1
                self.node_strength[node_idx] += torch.exp(-min_dists[i])
                if self.node_labels[node_idx] == batch_labels[i]:
                    self.node_fidelity[node_idx] += 1

            unique_hits = torch.unique(nearest_nodes)
            self.node_consistency[unique_hits] += 1

        # 3. Class-conditional K-Means node extraction (reduces label-noise)
        from sklearn.cluster import KMeans

        unique_batch_labels = torch.unique(batch_labels)
        for lbl_t in unique_batch_labels:
            lbl = int(lbl_t.item())
            class_feats = batch_features[batch_labels == lbl_t]
            if class_feats.shape[0] == 0:
                continue

            n_clusters = min(self.kmeans_per_class, class_feats.shape[0])
            if class_feats.shape[0] >= 12:
                n_clusters = max(2, n_clusters)
            else:
                n_clusters = 1

            class_np = class_feats.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=Config.SEED)
            kmeans.fit(class_np)

            new_nodes = torch.tensor(
                kmeans.cluster_centers_, dtype=torch.float32, device=self.device
            )
            new_nodes = F.normalize(new_nodes, p=2, dim=1)

            same_label_idxs = torch.where(self.node_labels == lbl_t)[0]
            if same_label_idxs.numel() == 0:
                for node_vec in new_nodes:
                    self._append_node(node_vec, lbl)
                continue

            existing = self.nodes[same_label_idxs]
            dists = torch.cdist(new_nodes, existing)
            min_dists, min_pos = torch.min(dists, dim=1)

            for i in range(new_nodes.shape[0]):
                node_vec = new_nodes[i]
                if min_dists[i] < self.merge_threshold:
                    target_idx = same_label_idxs[min_pos[i]]
                    self.nodes[target_idx] = F.normalize(
                        0.7 * self.nodes[target_idx] + 0.3 * node_vec, p=2, dim=0
                    )
                    self.node_freq[target_idx] += 1.0
                    self.node_strength[target_idx] += 1.0
                    self.node_consistency[target_idx] += 1.0
                    self.node_fidelity[target_idx] += 1.0
                else:
                    self._append_node(node_vec, lbl)

        self._cap_nodes_per_class()

    def _discriminative_proto_optimization(self, omega):
        """
        Sleep-phase discriminative prototype optimization.

        Treats prototypes as a cosine-softmax linear classifier and updates them
        with weighted cross-entropy, using every retained episodic node as a
        training sample.  This is mathematically equivalent to training a linear
        layer on the retained node bank, which closes the gap between NCM
        (class-mean prototypes) and a fully-supervised linear probe.

        Key properties:
        - All seen classes are considered simultaneously per gradient step →
          proper multi-class discrimination, not pairwise margin tricks.
        - Node importance weights (omega) down-weight noisy / redundant nodes.
        - Temperature matches BIO_PROTO_TEMP so the optimization objective
          exactly aligns with the inference scoring path in predict_proto_logits.
        - If a metric projection is enabled, both nodes and prototypes are
          embedded through the CURRENT (fixed) projection before scoring,
          keeping optimization consistent with inference.
        - Warm-start from current prototype values → stable convergence.
        """
        active_labels = torch.unique(self.node_labels)
        n_active = active_labels.numel()
        if n_active < 2 or self.nodes.shape[0] < n_active:
            return

        n_steps = max(1, int(getattr(Config, "BIO_DISC_STEPS", 150)))
        lr = float(getattr(Config, "BIO_DISC_LR", 0.01))
        temp = float(getattr(Config, "BIO_PROTO_TEMP", 0.10))

        # Local label index mapping  [0, n_active)
        label_to_pos = {int(lbl.item()): i for i, lbl in enumerate(active_labels)}
        node_label_pos = torch.tensor(
            [label_to_pos[int(lbl.item())] for lbl in self.node_labels],
            dtype=torch.long,
            device=self.device,
        )

        # Prepare node embeddings — fixed during this step, not learned
        nodes_normed = F.normalize(self.nodes.detach(), p=2, dim=1)
        if self.use_projection:
            proj = self.proj_matrix.detach()  # treat current projection as fixed
            nodes_embed = F.normalize(nodes_normed @ proj, p=2, dim=1)
        else:
            proj = None
            nodes_embed = nodes_normed  # (N, D)

        # Importance weights: higher-omega nodes drive the gradient more
        weights = omega.detach() + 1e-6
        weights = weights / weights.sum()

        # Warm-start from current prototype values
        proto_list = [self._get_proto(int(lbl.item())) for lbl in active_labels]
        proto_vars = torch.stack(proto_list).detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([proto_vars], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps, eta_min=lr * 0.05
        )

        for _ in range(n_steps):
            opt.zero_grad()

            # Project + normalize prototypes — matches the predict_proto_logits path
            if proj is not None:
                proto_embed = F.normalize(proto_vars @ proj, p=2, dim=1)
            else:
                proto_embed = F.normalize(proto_vars, p=2, dim=1)

            # Cosine-softmax cross-entropy: same objective as a normalized linear layer
            logits = torch.matmul(nodes_embed, proto_embed.t()) / temp  # (N, C_active)
            ce = F.cross_entropy(logits, node_label_pos, reduction="none")  # (N,)
            loss = (ce * weights).sum()

            loss.backward()
            opt.step()
            scheduler.step()

        # Write back: unit-normalized to match cosine scoring at inference time
        normed_protos = F.normalize(proto_vars.detach(), p=2, dim=1)
        for i, lbl_t in enumerate(active_labels):
            lbl = int(lbl_t.item())
            count = max(1.0, self._proto_count[lbl])
            self._proto_sum[lbl] = normed_protos[i] * count

    def consolidate(self, lambda_val=0.1):
        """
        OFFLINE CONSOLIDATION: Transfer essence to prototypes and prune graph.
        """
        if len(self.nodes) == 0:
            return

        # STEP 1: Score every node
        # Ω(v) = Frequency × Strength × Consistency × Fidelity
        omega = (
            self._norm01(self.node_freq)
            * self._norm01(self.node_strength)
            * self._norm01(self.node_consistency)
            * self._norm01(self.node_fidelity)
        )

        # STEP 2 & 3: Compute consolidated prototype and update NCM
        unique_labels = torch.unique(self.node_labels)
        for lbl in unique_labels:
            mask = self.node_labels == lbl
            v_class = self.nodes[mask]
            omega_class = omega[mask]

            if omega_class.sum() > 0:
                mu_consolidated = (omega_class.unsqueeze(1) * v_class).sum(
                    dim=0
                ) / omega_class.sum()
                old_proto = self._get_proto(int(lbl.item()))
                new_proto = (1 - lambda_val) * old_proto + lambda_val * mu_consolidated
                
                # Write back into dicts
                # Normalise keeping total count same: just assign new_proto * count to sum
                count = max(1.0, self._proto_count[int(lbl.item())])
                self._proto_sum[int(lbl.item())] = new_proto * count

        # STEP 3.5: Discriminative prototype optimization (cosine-softmax CE)
        # Replaces the old margin-SGD loop.  See _discriminative_proto_optimization
        # for the full rationale; the short version: this is equivalent to training
        # a normalized linear classifier on the retained node bank, which is provably
        # the right objective for closing the gap with a linear probe baseline.
        if (
            getattr(Config, "BIO_USE_DISCRIM_CONSOLIDATION", True)
            and len(unique_labels) > 0
        ):
            self._discriminative_proto_optimization(omega)

        # STEP 3.6: Sleep-phase metric projection learning (hard-negative contrastive)
        if self.use_projection and len(unique_labels) > 0:
            active_labels = torch.unique(self.node_labels)
            if active_labels.numel() > 1 and self.nodes.shape[0] > 0:
                label_to_pos = {
                    int(lbl.item()): i for i, lbl in enumerate(active_labels)
                }
                node_label_pos = torch.tensor(
                    [label_to_pos[int(lbl.item())] for lbl in self.node_labels],
                    dtype=torch.long,
                    device=self.device,
                )

                nodes = self.nodes.detach()
                proto_list = [self._get_proto(int(lbl.item())) for lbl in active_labels]
                protos = torch.stack(proto_list).detach()
                weights = omega.detach() + 1e-6

                p_var = self.proj_matrix.detach().clone().requires_grad_(True)
                opt_p = torch.optim.Adam([p_var], lr=float(self.proj_lr))
                p_margin = float(self.proj_margin)
                p_steps = max(1, int(self.proj_steps))
                ortho_reg = float(self.proj_ortho_reg)

                for _ in range(p_steps):
                    opt_p.zero_grad()

                    z_nodes = F.normalize(self._project(nodes, p_var), p=2, dim=1)
                    z_proto = F.normalize(self._project(protos, p_var), p=2, dim=1)
                    sims = torch.matmul(z_nodes, z_proto.t())  # (N, C_active)

                    pos = sims.gather(1, node_label_pos.view(-1, 1)).squeeze(1)
                    cls_mask = F.one_hot(
                        node_label_pos, num_classes=active_labels.numel()
                    ).bool()
                    neg = sims.masked_fill(cls_mask, -1e4).max(dim=1).values
                    loss_rank = (
                        torch.relu(p_margin + neg - pos) * weights
                    ).sum() / weights.sum()
                    loss_pull = ((1.0 - pos) * weights).sum() / weights.sum()

                    gram = torch.matmul(p_var.t(), p_var)
                    ident = torch.eye(gram.shape[0], device=self.device)
                    loss_ortho = torch.mean((gram - ident) ** 2)

                    loss = loss_rank + 0.5 * loss_pull + ortho_reg * loss_ortho
                    loss.backward()
                    opt_p.step()

                self.proj_matrix.copy_(p_var.detach())

        # STEP 4: Prune redundant nodes
        # Build a proto tensor matched to node_labels
        node_protos = torch.stack([self._get_proto(int(lbl.item())) for lbl in self.node_labels])
        dist_to_proto = torch.norm(
            self.nodes - node_protos, dim=1
        )
        uniqueness = self._norm01(dist_to_proto)

        # # Keep if score is high OR uniqueness is high
        # keep_mask = (omega > omega.median()) | (uniqueness > uniqueness.median())
        score_thresh = omega.mean() * 0.5
        unique_thresh = uniqueness.mean() * 0.5

        keep_mask = (omega > score_thresh) | (uniqueness > unique_thresh)

        pre_priority = 0.7 * omega + 0.3 * uniqueness
        self.nodes = self.nodes[keep_mask]
        self.node_labels = self.node_labels[keep_mask]
        self.node_freq = self.node_freq[keep_mask]
        self.node_strength = self.node_strength[keep_mask]
        self.node_consistency = self.node_consistency[keep_mask]
        self.node_fidelity = self.node_fidelity[keep_mask]

        post_priority = pre_priority[keep_mask]
        self._cap_nodes_per_class(node_priority=post_priority)

        # RESET activation histories for next task
        self.node_freq = torch.zeros(len(self.nodes), device=self.device)
        self.node_strength = torch.zeros(len(self.nodes), device=self.device)
        self.node_consistency = torch.zeros(len(self.nodes), device=self.device)
        self.node_fidelity = torch.zeros(len(self.nodes), device=self.device)

    def predict_proto_logits(self, batch_features):
        """
        System 2 — Cortex (Long-term Prototype Memory).
        Returns raw, unscaled logits from the prototype branch only.
        """
        if not isinstance(batch_features, torch.Tensor):
            batch_features = torch.tensor(batch_features, dtype=torch.float32)
        batch_features = batch_features.to(self.device)
        batch_features = F.normalize(batch_features, p=2, dim=1)

        batch_size = batch_features.shape[0]
        max_cls = max(self.seen_classes) if self.seen_classes else -1
        if max_cls < 0:
            return torch.full((batch_size, 1), -1e4, device=self.device)
            
        C = max_cls + 1
        proto_logits = torch.full((batch_size, C), -1e4, device=self.device)

        proto_tensor, classes = self._build_proto_tensor()
        
        if self.use_mahalanobis and not self.use_projection:
            var = self._prototype_variance()  # (C_seen, D)
            diff = batch_features.unsqueeze(1) - proto_tensor.unsqueeze(0)
            logits_seen = -0.5 * torch.sum((diff * diff) / var.unsqueeze(0), dim=2)
        else:
            batch_embed = F.normalize(self._project(batch_features), p=2, dim=1)
            proto_bank = F.normalize(self._project(proto_tensor), p=2, dim=1)
            logits_seen = torch.matmul(batch_embed, proto_bank.t())

        for i, cls_id in enumerate(classes):
            proto_logits[:, cls_id] = logits_seen[:, i]

        return proto_logits

    def predict_node_logits(self, batch_features):
        """
        System 1 — Hippocampus (Episodic Node Memory).
        Returns raw, unscaled logits from the episodic node branch only.
        """
        if not isinstance(batch_features, torch.Tensor):
            batch_features = torch.tensor(batch_features, dtype=torch.float32)
        batch_features = batch_features.to(self.device)
        batch_features = F.normalize(batch_features, p=2, dim=1)
        batch_embed = F.normalize(self._project(batch_features), p=2, dim=1)

        batch_size = batch_features.shape[0]
        max_cls = max(self.seen_classes) if self.seen_classes else -1
        if max_cls < 0:
            return torch.full((batch_size, 1), -1e4, device=self.device)
            
        C = max_cls + 1
        graph_logits = torch.full((batch_size, C), -1e4, device=self.device)

        if self.nodes.shape[0] > 0:
            node_bank = F.normalize(self._project(self.nodes), p=2, dim=1)
            logits_per_node = torch.matmul(batch_embed, node_bank.t())
            for lbl in torch.unique(self.node_labels):
                idxs = torch.where(self.node_labels == lbl)[0]
                graph_logits[:, int(lbl.item())] = logits_per_node[:, idxs].max(dim=1).values

        return graph_logits

    def predict_logits(self, batch_features):
        """
        Internal fused prediction: combines System 1 (episodic) and System 2 (prototype).

        Uses the model's own proto_weight, node_temp, and proto_temp to blend the two
        branches.  Call predict_node_logits / predict_proto_logits directly if you need
        the individual signals for an external dual-system fusion.
        """
        if not isinstance(batch_features, torch.Tensor):
            batch_features = torch.tensor(batch_features, dtype=torch.float32)
        batch_features = batch_features.to(self.device)

        # Guard: nothing learned yet
        if self.nodes.shape[0] == 0 and torch.sum(self.prototypes_counts) == 0:
            out_dim = max(self.seen_classes) + 1 if self.seen_classes else 1
            return torch.zeros(
                (batch_features.shape[0], out_dim), device=self.device
            )

        node_logits = self.predict_node_logits(batch_features)
        proto_logits = self.predict_proto_logits(batch_features)

        graph_weight = 1.0 - self.proto_weight
        return graph_weight * (node_logits / self.node_temp) + self.proto_weight * (
            proto_logits / self.proto_temp
        )

    @property
    def codebooks(self):
        """
        Mock codebooks property to satisfy the visualization script.
        It treats the entire episodic node bank as a single 'codebook'.
        """
        # We return it as a list containing one element to mimic [codebook_chunk_1, ...]
        return [self.nodes]

    def quantize(self, x):
        """
        Modified quantize to return the correct shape for the visualization loop.
        The eval script expects: (Batch, Num_Chunks)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)

        if self.nodes.shape[0] == 0:
            return torch.zeros((x.shape[0], 1), dtype=torch.long, device=self.device)

        # 1. Compute similarity/distance to all episodic nodes
        dists = torch.cdist(x, self.nodes)
        indices = torch.min(dists, dim=1).indices

        # 2. Reshape to (Batch, 1) because the script expects one index per 'chunk'
        return indices.unsqueeze(1)
