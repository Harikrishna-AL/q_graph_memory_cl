import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch.mps
except ImportError:
    pass

from src.config import Config

from .config import Config

CACHE_DIR = "cache"

# Mapping: backbone name -> output feature dimension
_BACKBONE_DIMS = {
    "dinov2": 384,       # ViT-S/14
    "dinov2_giant": 1536,# ViT-g/14
    "dinov3": 1024,      # dinov3_vitl16 (Large)
    "siglip": 1152,      # vit_so400m_patch14_siglip_384
    "siglip2": 1152,     # ViT-SO400M-14-SigLIP2
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet50_ssl": 2048,
    "resnet50_clip": 2048,
    "resnet50_lightly_simclr": 2048,
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


class AlignmentLayer(nn.Module):
    """
    The Alignment Layer (NCPTM-CIL) with Residual Connection.
    Acts as a 'learned nudge' to move frozen backbone features 
    into a perfect ETF configuration without destroying their 
    original discriminative power.
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        # Use a Residual path to preserve backbone quality
        # If d_in != d_out, we use a projection for the identity path
        self.use_projection = (d_in != d_out)
        if self.use_projection:
            self.proj = nn.Linear(d_in, d_out, bias=False)
            
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.LeakyReLU(0.1),
            nn.Linear(d_in, d_out)
        )
        
        # Initialize the MLP to be a 'small correction' initially
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        identity = self.proj(x) if self.use_projection else x
        # Increased Authority: 0.5 nudge allows the model to actually 
        # fix backbone overlaps rather than just 'suggest' a change.
        out = identity + 0.5 * self.net(x)
        return F.normalize(out, p=2, dim=1)


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
    elif backbone_name == "dinov3":
        print(f"🦖 Loading DINOv3 Large (dinov3_vitl16, dim={Config.FEATURE_DIM})...")
        model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl16')
    elif backbone_name == "siglip":
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for SigLIP models. Run: pip install timm")
        print(f"👀 Loading SigLIP SO400m (dim={Config.FEATURE_DIM})...")
        model = timm.create_model('vit_so400m_patch14_siglip_384.webli', pretrained=True, num_classes=0)
    elif backbone_name == "siglip2":
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for SigLIP 2 models. Run: pip install timm")
        print(f"👀 Loading SigLIP 2 SO400m (dim={Config.FEATURE_DIM})...")
        # Accessing SigLIP 2 via timm's HF Hub integration
        model = timm.create_model('hf-hub:timm/ViT-SO400M-14-SigLIP2', pretrained=True, num_classes=0)
    elif backbone_name == "resnet50_ssl":
        print(f"🧬 Loading Self-Supervised ResNet50 (DINO Contrastive, dim={Config.FEATURE_DIM})...")
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    elif backbone_name == "resnet50_clip":
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for resnet50_clip. Run: pip install timm")
        print(f"🎯 Loading CLIP ResNet50 (resnet50_clip.openai, dim={Config.FEATURE_DIM})...")
        model = timm.create_model('resnet50_clip.openai', pretrained=True, num_classes=0)
    elif backbone_name == "resnet50_lightly_simclr":
        print(f"🤖 Loading Lightly-AI SimCLR ResNet50 (dim={Config.FEATURE_DIM})...")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub is required to download from Lightly. Run: pip install huggingface_hub")
        
        weights_path = hf_hub_download(repo_id="lightly-ai/simclrv1-imagenet1k-resnet50-1x", filename="resnet50-1x.pth")
        import torchvision.models as tvm
        vision_model = tvm.resnet50(weights=None)
        state_dict = torch.load(weights_path, map_location="cpu")
        
        # PyTorch Lightning heavily nests weights inside a 'state_dict' key
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        # Strip generic SSL encoder prefixes to isolate the pure ResNet weights
        clean_sd = {}
        for k, v in state_dict.items():
            new_k = k
            for prefix in ["backbone.", "resnet.", "model.backbone.", "model.resnet.", "encoder.", "model."]:
                if new_k.startswith(prefix):
                    new_k = new_k.replace(prefix, "")
            clean_sd[new_k] = v
                
        missing, unexpected = vision_model.load_state_dict(clean_sd, strict=False)
        
        # ResNet50 has exactly ~320 keys (weights/biases/buffers). If missing is 320, we failed to map!
        loaded_keys = len(vision_model.state_dict()) - len(missing)
        print(f"   => Successfully mapped {loaded_keys} ResNet layers from Lightly checkpoint!")
        if loaded_keys == 0:
            print("   => CRITICAL ERROR: 0 layers mapped! The model is randomly initialized!")
            
        model = _ResNetExtractor(vision_model)
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

    # Enable Multi-GPU Parallelism if more than one GPU is detected
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"🚀 Detected {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = torch.nn.DataParallel(model)

    return model


# Keep the old name as an alias so existing call-sites in main.py still work
def load_dino():
    """Deprecated alias for load_backbone(). Use load_backbone() instead."""
    return load_backbone()


def extract_features(backbone, dataloader):
    """Run `backbone` over `dataloader` and return L2-normalised features + labels."""
    print("🔍 Extracting Features (this may take a moment)...")
    
    # Enable loading of truncated/corrupted images (common in massive datasets)
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    all_feats, all_lbls = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
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
                    print(f"⚠️ Skipping batch {batch_idx}: 'imgs' is not a Tensor")
                    continue

                imgs = imgs.to(Config.DEVICE)
                
                # 🔥 Accelerate throughput by 3-4x using Mixed Precision (GPU Tensor Cores)
                device_type = "cuda" if torch.cuda.is_available() else "cpu"
                use_amp = device_type == "cuda" and ("dinov2" in Config.BACKBONE.lower() or "siglip" in Config.BACKBONE.lower())
                
                if use_amp:
                    with torch.autocast(device_type="cuda"):
                        f = backbone(imgs)
                else:
                    f = backbone(imgs)
                
                f = f.float().cpu().numpy()
                
                # L2 Normalize natively with a pure zero-division protection just in case a crop is completely black
                f_norm = np.linalg.norm(f, axis=1, keepdims=True)
                f_norm[f_norm == 0] = 1e-8 # mathematically secure the normalization scalar
                f = f / f_norm
                
                all_feats.append(f)
                if torch.is_tensor(lbls):
                    all_lbls.append(lbls.cpu().numpy())
                else:
                    all_lbls.append(np.array(lbls))
                    
                if (batch_idx + 1) % 100 == 0:
                    print(f"   Processed batch {batch_idx + 1}...")
                    
            except (OSError, RuntimeError) as e:
                print(f"⚠️  Skipping corrupted batch {batch_idx}: {e}")
                continue

    if not all_feats:
        raise ValueError("No features were extracted. Check your dataset and backbone.")

    features = np.concatenate(all_feats)
    labels = np.concatenate(all_lbls)
    
    unique_found = len(np.unique(labels))
    print(f"✅ Extraction Complete: {len(features)} samples, {unique_found} unique classes found.")
    if unique_found <= 1 and len(features) > 100:
        print("⚠️  CRITICAL WARNING: Only one class found in extraction! This usually means the Dataloader or Dataset folder structure is incorrect.")
        
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
        self._class_subspaces = {} # class_id -> (D, K) orthonormal basis tensors
        
        # ── RLA: Recursive Linear Alignment Accumulators ───────────────────
        # A: (D, D) - Sum of outer products (X^T X)
        # B: (D, D_align) - Correlation with ETF (X^T W_etf)
        d_in = input_dim
        d_out = getattr(Config, "BIO_ALIGN_DIM", 256)
        self.register_buffer("_RLA_A", torch.zeros((d_in, d_in), device=self.device))
        self.register_buffer("_RLA_B", torch.zeros((d_in, d_out), device=self.device))
        self.register_buffer("_RLA_P", torch.eye(d_in, device=self.device)[:, :d_out])

        # Pre-generate Simplex ETF matrix for maximally separated prototypes
        # Both nc_align and analytic_etf require an ETF target frame.
        mode = getattr(Config, "BIO_CONSOLIDATION_MODE", "sgd")
        if getattr(Config, "BIO_USE_ETF", False) or mode in ["nc_align", "analytic_etf"]:
            # For nc_align/analytic_etf, the ETF should match the ALIGN_DIM
            etf_dim = getattr(Config, "BIO_ALIGN_DIM", 256) if mode in ["nc_align", "analytic_etf"] else input_dim
            self._etf_matrix = self._generate_simplex_etf(Config.BIO_ETF_MAX_CLASSES, etf_dim)
        else:
            self._etf_matrix = None

        # Alignment Layer (NC-Alignment mode)
        self.align_layer = AlignmentLayer(input_dim, getattr(Config, "BIO_ALIGN_DIM", 256)).to(self.device)

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

    def _generate_simplex_etf(self, k, d):
        """
        Generates a Simplex Equiangular Tight Frame (ETF).
        Vertices are maximally separated on the hypersphere.
        """
        # A Simplex ETF for K classes in D dimensions requires D >= K-1
        # If D < K-1, we generate a 'Near-ETF' using a random orthogonal projection
        if d < k - 1:
            # Random frame - compute on CPU to avoid MPS svd issues
            W = torch.randn(d, k, device="cpu")
            U, S, V = torch.svd(W)
            return V.to(self.device) # (k, d)
        
        # Standard Simplex ETF construction
        # W = sqrt(k/(k-1)) * (I - 1/k 11^T)
        # Compute on CPU for stability and to avoid MPS warnings
        I = torch.eye(k, device="cpu")
        ones = torch.ones((k, k), device="cpu")
        M = I - (1.0 / k) * ones
        
        # Extract the k-1 non-zero eigenvectors
        U, S, V = torch.svd(M)
        # V is (k, k). Columns corresponding to non-zero eigenvalues form the ETF basis.
        # We take the first 'd' dimensions (or k-1 if d is larger)
        basis = V[:, :min(d, k-1)] # (k, d_eff)
        
        # Unit normalize and move back to target device
        return F.normalize(basis.to(self.device), p=2, dim=1) # (K, D)

    def _apply_etf_anchoring(self):
        """
        Optimal Prototype-to-ETF Alignment using the Hungarian Algorithm.
        """
        if self._etf_matrix is None:
            return

        classes = self.seen_classes
        if not classes:
            return

        # 1. Build current prototype matrix
        protos = torch.stack([self._get_proto(c) for c in classes]) # (C_seen, D)
        
        # 2. Slice the ETF matrix to match the current number of classes
        # (For progressive learning, we always match the seen classes to the best 'slots')
        etf_vertices = self._etf_matrix[:len(classes)] # (C_seen, D)

        # 3. Compute cost matrix (Distance between prototypes and ETF vertices)
        # We use cosine distance (1 - similarity)
        dist_matrix = 1.0 - torch.matmul(protos, etf_vertices.t()) # (C_seen, C_seen)

        # 4. Hungarian Matching
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(dist_matrix.cpu().numpy())

        # 5. Anchor: Move prototypes to their assigned ETF vertices
        # In the 'Ideal' Neural Collapse state (NC2), the prototypes ARE the ETF vertices.
        # For DINOv2, we trust the features enough to do Hard Anchoring (1.0).
        backbone = Config.BACKBONE.lower().strip()
        blend_weight = 1.0 if "dinov2" in backbone else 0.8
        
        for i, class_idx in enumerate(row_ind):
            lbl = classes[class_idx]
            target_vertex = etf_vertices[col_ind[i]]
            
            if blend_weight >= 1.0:
                refined_proto = target_vertex
            else:
                old_proto = self._get_proto(lbl)
                refined_proto = F.normalize((1.0 - blend_weight) * old_proto + blend_weight * target_vertex, p=2, dim=0)
            
            count = max(1.0, self._proto_count[lbl])
            self._proto_sum[lbl] = refined_proto * count

    def _update_class_subspaces(self, k=10):
        """
        Compute a low-rank orthonormal basis (subspace) for each class 
        using its episodic node bank. This captures the local manifold 
        variation rather than just a single point.
        
        Subspaces are ALWAYS computed in the backbone space to provide 
        a robust generative signal (System 1) complementary to the 
        aligned discriminative signal (System 2).
        """
        unique_labels = torch.unique(self.node_labels)
        
        for lbl_t in unique_labels:
            lbl = int(lbl_t.item())
            mask = self.node_labels == lbl_t
            class_nodes = self.nodes[mask] # (N_nodes, D_backbone)
            
            if class_nodes.shape[0] < 2:
                continue
                
            # SYSTEM 1 Manifold Calculation
            mode = getattr(Config, "BIO_CONSOLIDATION_MODE", "sgd")
            
            if mode == "analytic_etf" and self._RLA_P is not None:
                # Use aligned space for manifolds
                class_nodes = F.normalize(torch.matmul(class_nodes, self._RLA_P), p=2, dim=1)
                classes_list = self.seen_classes
                label_to_idx = {l: i for i, l in enumerate(classes_list)}
                mu_c = self._etf_matrix[label_to_idx[lbl]]
            else:
                # Use generative backbone space
                mu_c = self._get_proto(lbl)
            
            # Orthonormal basis via SVD on CPU
            centered_nodes = (class_nodes - mu_c).detach().cpu()
            
            try:
                k_eff = min(k, class_nodes.shape[0] - 1)
                # Exact SVD on CPU
                U, S, Vh = torch.linalg.svd(centered_nodes, full_matrices=False)
                
                # Principal components: (D_backbone, k_eff)
                V_basis = Vh[:k_eff].t()
                
                self._class_subspaces[lbl] = V_basis.to(self.device)
            except Exception as e:
                continue

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
            
            # 1.5 Update RLA Accumulators (X^T X and X^T W_etf)
            if getattr(Config, "BIO_CONSOLIDATION_MODE", "sgd") == "analytic_etf":
                if self._etf_matrix is not None:
                    # w_target: (1, D_align)
                    # We map class label to the specific ETF slot
                    classes_list = self.seen_classes
                    label_to_idx = {l: i for i, l in enumerate(classes_list)}
                    idx = label_to_idx.get(lbl, -1)
                    
                    if idx >= 0:
                        w_target = self._etf_matrix[idx].unsqueeze(0)
                        feat_uns = feat.unsqueeze(0)
                        
                        # Accumulate
                        self._RLA_A += torch.matmul(feat_uns.t(), feat_uns)
                        self._RLA_B += torch.matmul(feat_uns.t(), w_target)

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
        
        # Adjust temperature for high-dimensional backbones (SigLIP/ResNet)
        # Higher dimensions lead to lower dot-products, so we increase temp slightly.
        backbone = Config.BACKBONE.lower().strip()
        if "siglip" in backbone or "resnet" in backbone or "efficientnet" in backbone:
            temp = max(temp, 0.15)

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
        opt = torch.optim.Adam([proto_vars], lr=lr, weight_decay=1e-4)
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
            # Label smoothing prevents the prototypes from being pushed too far by outlier nodes.
            logits = torch.matmul(nodes_embed, proto_embed.t()) / temp  # (N, C_active)
            ce = F.cross_entropy(logits, node_label_pos, reduction="none", label_smoothing=0.1)  # (N,)
            loss = (ce * weights).sum()

            loss.backward()
            opt.step()
            scheduler.step()

        # Write back: unit-normalized to match cosine scoring at inference time.
        # We use a backbone-aware momentum blend with the original NCM mean for stability.
        # For DINOv2 (highly reliable), we trust the refinement more (0.8).
        # For SigLIP/ResNet, we stay conservative (0.5) to prevent drift.
        normed_protos = F.normalize(proto_vars.detach(), p=2, dim=1)
        
        blend_weight = 0.5
        backbone = Config.BACKBONE.lower().strip()
        if "dinov2" in backbone:
            blend_weight = 0.8
            
        for i, lbl_t in enumerate(active_labels):
            lbl = int(lbl_t.item())
            count = max(1.0, self._proto_count[lbl])
            
            old_proto = self._get_proto(lbl)
            refined_proto = F.normalize((1.0 - blend_weight) * old_proto + blend_weight * normed_protos[i], p=2, dim=0)
            self._proto_sum[lbl] = refined_proto * count

    def _nc_alignment_optimization(self, omega):
        """
        Pull-and-Push (PAP) Alignment Optimization with Task-Specific Reset.
        
        Following NCPTM-CIL, we reset the alignment layer every task and 
        re-train it on the entire episodic memory to find a fresh, globally 
        consistent mapping to the ETF.
        """
        active_labels = torch.unique(self.node_labels)
        if active_labels.numel() < 2 or self.nodes.shape[0] < 2:
            return

        # 1. Task-Specific Reset: Initialize fresh parameters
        # This prevents interference from previous task gradients.
        d_in = self.input_dim
        d_out = getattr(Config, "BIO_ALIGN_DIM", 256)
        self.align_layer = AlignmentLayer(d_in, d_out).to(self.device)

        # Ensure ETF exists
        if self._etf_matrix is None or self._etf_matrix.shape[1] != d_out:
            self._etf_matrix = self._generate_simplex_etf(Config.BIO_ETF_MAX_CLASSES, d_out)

        # 2. Setup Optimization
        lr = float(getattr(Config, "BIO_DISC_LR", 0.01))
        n_steps = max(1, int(getattr(Config, "BIO_DISC_STEPS", 150)))
        pap_weight = float(getattr(Config, "BIO_PAP_WEIGHT", 1.0))
        
        # Add Weight Decay to keep the mapping 'gentle'
        opt = torch.optim.Adam(self.align_layer.parameters(), lr=lr, weight_decay=1e-4)
        
        classes = self.seen_classes
        etf_vertices = self._etf_matrix[:len(classes)] # (C_seen, D_align)
        label_to_idx = {lbl: i for i, lbl in enumerate(classes)}
        
        # 3. Training Data: Stochastic pseudo-sampling to prevent OOM
        # New classes get full resolution, old classes get a stable summary.
        nodes = self.nodes.detach()
        node_labels = self.node_labels
        n_pseudo_new = int(getattr(Config, "BIO_PSEUDO_SAMPLES", 512))
        n_pseudo_old = 64 # Reduced footprint for historical classes
        
        classes = self.seen_classes
        # Identify most recent task labels (last CPT labels)
        CPT = getattr(Config, "CLASSES_PER_TASK", 10)
        recent_threshold = max(classes) - CPT + 1 if classes else 0
        
        pseudo_feats_list = []
        pseudo_labels_list = []
        
        for lbl in classes:
            mu = self._get_proto(lbl)
            count = max(1.0, self._proto_count[lbl])
            std = torch.sqrt(self._proto_m2[lbl] / count + 1e-6) if lbl in self._proto_m2 else 0.02
            
            # Stochastic budget: more for recent, less for old
            budget = n_pseudo_new if lbl >= recent_threshold else n_pseudo_old
            
            noise = torch.randn((budget, self.input_dim), device=self.device)
            samples = mu.unsqueeze(0) + std.unsqueeze(0) * noise
            pseudo_feats_list.append(samples)
            pseudo_labels_list.append(torch.full((budget,), lbl, dtype=torch.long, device=self.device))
            
        pseudo_feats = torch.cat(pseudo_feats_list, dim=0)
        pseudo_labels = torch.cat(pseudo_labels_list, dim=0)
        train_feats = torch.cat([nodes, pseudo_feats], dim=0)
        train_labels = torch.cat([node_labels, pseudo_labels], dim=0)
        
        # Free memory immediately
        del pseudo_feats_list, pseudo_labels_list
        
        # ... rest of weights logic ...
        node_weights = (omega.detach() + 1e-6)
        node_weights = node_weights / (node_weights.sum() + 1e-6)
        pseudo_weights = torch.full((len(pseudo_feats),), 1.0 / len(pseudo_feats), device=self.device)
        all_weights = torch.cat([0.5 * node_weights, 0.5 * pseudo_weights], dim=0)
        
        label_to_idx = {lbl: i for i, lbl in enumerate(classes)}
        target_indices = torch.tensor([label_to_idx[int(l.item())] for l in train_labels], device=self.device)
        target_vertices = etf_vertices[target_indices]

        # Reduce steps for speed 
        n_steps_nc = 100 
        self.align_layer.train()
        for _ in range(n_steps_nc):
            opt.zero_grad()
            aligned_feats = self.align_layer(train_feats + 0.005 * torch.randn_like(train_feats)) 
            dot_pos = (aligned_feats * target_vertices).sum(dim=1)
            pull = torch.mean((dot_pos - 1.0)**2 * all_weights * len(train_feats))
            sims = torch.matmul(aligned_feats, etf_vertices.t())
            K = len(classes)
            pos_indices = target_indices.unsqueeze(1)
            pos_diff_sq = torch.gather((sims - (-1.0/(K-1.0)))**2, 1, pos_indices)
            push = ((sims - (-1.0/(K-1.0)))**2).sum() - pos_diff_sq.sum()
            push = push / (len(train_feats) * (K - 1.0))
            loss = pull + pap_weight * push
            loss.backward()
            opt.step()
        
        self.align_layer.eval()
        # Final cleanup
        del train_feats, train_labels, target_vertices, target_indices, all_weights
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _analytic_prototype_refinement(self):
        """
        Analytic Boundary Refinement (Closed-Form).
        
        Instead of SGD, we use Linear Discriminant Analysis (LDA) theory to 
        analytically compute the optimal discriminative prototypes.
        """
        if self.nodes.shape[0] < 2:
            return

        # 1. Calculate Global Covariance of the episodic node bank
        all_nodes = self.nodes.detach() # (N_total, D)
        mu_global = all_nodes.mean(dim=0, keepdim=True)
        centered_nodes = all_nodes - mu_global
        
        D = self.input_dim
        ridge = 1e-3
        sigma = torch.matmul(centered_nodes.t(), centered_nodes) / (all_nodes.shape[0] - 1)
        sigma = sigma + ridge * torch.eye(D, device=self.device)
        
        # 2. Compute Precision Matrix
        try:
            precision = torch.inverse(sigma)
        except:
            precision = torch.linalg.pinv(sigma)

        # 3. Refine Prototypes: mu_refined = Precision * mu_raw
        classes = self.seen_classes
        for lbl in classes:
            raw_proto = self._get_proto(lbl)
            refined_proto = torch.matmul(precision, raw_proto)
            
            count = max(1.0, self._proto_count[lbl])
            # Use same blend logic as SGD for consistency if desired, or just overwrite
            # Here we follow the blend logic for stability
            blend_weight = 0.8 if "dinov2" in Config.BACKBONE.lower() else 0.5
            old_proto = self._get_proto(lbl)
            normed_refined = F.normalize(refined_proto, p=2, dim=0)
            final_proto = F.normalize((1.0 - blend_weight) * old_proto + blend_weight * normed_refined, p=2, dim=0)
            self._proto_sum[lbl] = final_proto * count

    def _recursive_linear_alignment(self):
        """
        Closed-form Recursive Linear Alignment (RLA).
        Solves P = (A + lambda*I)^-1 * B
        """
        A = self._RLA_A
        B = self._RLA_B
        
        # Ridge regularization
        ridge = 1e-3
        D = A.shape[0]
        reg_A = A + ridge * torch.eye(D, device=self.device)
        
        # Solve for P
        try:
            self._RLA_P = torch.linalg.solve(reg_A, B)
        except:
            # Fallback to pseudo-inverse
            self._RLA_P = torch.matmul(torch.linalg.pinv(reg_A), B)

    def consolidate(self, lambda_val=0.1):
        """
        OFFLINE CONSOLIDATION: Transfer essence to prototypes and prune graph.
        """
        if len(self.nodes) == 0:
            return

        # STEP 1: Score every node (Importance weights)
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

        # STEP 3.5: Discriminative prototype optimization
        if (
            getattr(Config, "BIO_USE_DISCRIM_CONSOLIDATION", True)
            and len(unique_labels) > 0
        ):
            mode = getattr(Config, "BIO_CONSOLIDATION_MODE", "sgd")
            if mode == "analytic":
                self._analytic_prototype_refinement()
            elif mode == "analytic_etf":
                self._recursive_linear_alignment()
            elif mode == "nc_align":
                self._nc_alignment_optimization(omega)
            else:
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

        # Update manifold subspaces after pruning
        rank = getattr(Config, "BIO_SUBSPACE_RANK", 10)
        self._update_class_subspaces(k=rank)

        # STEP 3.7: Equiangular Tight Frame (ETF) Anchoring
        if getattr(Config, "BIO_USE_ETF", False):
            self._apply_etf_anchoring()

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

        mode = getattr(Config, "BIO_CONSOLIDATION_MODE", "sgd")
        
        if mode == "analytic_etf" and self._RLA_P is not None:
            # RLA Path: Linear Projection -> ETF Matrix
            aligned_feats = F.normalize(torch.matmul(batch_features, self._RLA_P), p=2, dim=1)
            classes = self.seen_classes
            etf_vertices = self._etf_matrix[:len(classes)] # (C_seen, D_align)
            logits_seen = torch.matmul(aligned_feats, etf_vertices.t())
            for i, cls_id in enumerate(classes):
                proto_logits[:, cls_id] = logits_seen[:, i]
            return proto_logits

        if mode == "nc_align" and self._etf_matrix is not None:
            # NC-Alignment Path: Features -> Alignment Layer -> ETF Matrix
            aligned_feats = self.align_layer(batch_features) # (B, D_align)
            classes = self.seen_classes
            etf_vertices = self._etf_matrix[:len(classes)] # (C_seen, D_align)
            logits_seen = torch.matmul(aligned_feats, etf_vertices.t())
            
            for i, cls_id in enumerate(classes):
                proto_logits[:, cls_id] = logits_seen[:, i]
            return proto_logits

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
        device = self.device
        batch_size = batch_features.shape[0]
        
        mode = getattr(Config, "BIO_CONSOLIDATION_MODE", "sgd")

        if self.nodes.shape[0] > 0:
            if mode == "nc_align":
                query_embed = self.align_layer(batch_features) 
                node_bank = self.align_layer(self.nodes)       
            elif mode == "analytic_etf" and self._RLA_P is not None:
                query_embed = F.normalize(torch.matmul(batch_features, self._RLA_P), p=2, dim=1)
                node_bank = F.normalize(torch.matmul(self.nodes, self._RLA_P), p=2, dim=1)
            else:
                query_embed = F.normalize(self._project(batch_features), p=2, dim=1)
                node_bank = F.normalize(self._project(self.nodes), p=2, dim=1)
            
            # (B, N_total_nodes)
            logits_per_node = torch.matmul(query_embed, node_bank.t())

            # ── Vectorized Log-Sum-Exp (Manifold Density) ────────────────────
            density_temp = self.node_temp * 0.5
            scaled_sims = logits_per_node / density_temp
            
            # Expand labels for batch: (B, N_nodes)
            node_labels_expanded = self.node_labels.unsqueeze(0).expand(batch_size, -1)
            
            # Get Max per class for numerical stability in LSE
            m = torch.full((batch_size, C), -1e4, device=device)
            m.scatter_reduce_(1, node_labels_expanded, scaled_sims, reduce='amax', include_self=False)
            
            # sum(exp(x - m))
            max_per_node = m.gather(1, node_labels_expanded)
            exp_diff = torch.exp(scaled_sims - max_per_node)
            
            sum_exp = torch.zeros((batch_size, C), device=device)
            sum_exp.scatter_add_(1, node_labels_expanded, exp_diff)
            
            # density_score = (m + log(sum_exp)) * temp
            graph_logits = (m + torch.log(sum_exp + 1e-10)) * density_temp

            if self._class_subspaces:
                classes_list = self.seen_classes
                subspace_tensors = []
                mu_tensors = []
                
                # Identify active dimension and target rank
                d_align = getattr(Config, "BIO_ALIGN_DIM", 256)
                curr_dim = d_align if mode in ["nc_align", "analytic_etf"] else self.input_dim
                target_rank = getattr(Config, "BIO_SUBSPACE_RANK", 10)
                
                for lbl in range(C):
                    if lbl in self._class_subspaces:
                        V = self._class_subspaces[lbl] # (D, k_eff)
                        # Padding: ensure all subspaces have shape (curr_dim, target_rank)
                        if V.shape[1] < target_rank:
                            padding = torch.zeros((curr_dim, target_rank - V.shape[1]), device=device)
                            V = torch.cat([V, padding], dim=1)
                        subspace_tensors.append(V)
                        
                        if mode == "analytic_etf":
                            label_to_idx = {l: i for i, l in enumerate(classes_list)}
                            mu_tensors.append(self._etf_matrix[label_to_idx[lbl]])
                        else:
                            mu_tensors.append(self._get_proto(lbl))
                    else:
                        # Dummy for unseen/unstable classes
                        subspace_tensors.append(torch.zeros((curr_dim, target_rank), device=device))
                        mu_tensors.append(torch.zeros(curr_dim, device=device))
                
                V_all = torch.stack(subspace_tensors)
                Mu_all = torch.stack(mu_tensors)
                
                centered_queries = query_embed.unsqueeze(1) - Mu_all.unsqueeze(0)
                proj_coeffs = torch.einsum('bcd,cdk->bck', centered_queries, V_all)
                manifold_fit = torch.norm(proj_coeffs, dim=2)
                
                graph_logits = graph_logits + 0.5 * manifold_fit
        else:
            graph_logits = torch.full((batch_size, C), -1e4, device=self.device)

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
