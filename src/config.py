import torch


class Config:
    # Hardware
    DEVICE = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    
    # Data
    DATA_ROOT = "./data"
    DATASET_NAME = "tiny-imagenet-200"
    URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    # Model / Backbone
    # Options: "dinov2" | "resnet18" | "resnet34" | "resnet50" | "resnet101" | "resnet152"
    BACKBONE = "dinov2"

    # DINOv2-specific
    DINO_REPO = "facebookresearch/dinov2"
    DINO_MODEL = "dinov2_vits14"

    # Feature dimension — automatically updated by load_backbone() in model.py
    FEATURE_DIM = 384  # default: DINOv2 ViT-S/14

    # Continual Learning
    N_TASKS = 20
    CLASSES_PER_TASK = 10
    N_CHUNKS = 8
    CHUNK_DIM = 384 // 8

    # Hyperparameters
    WORDS_PER_TASK = 512
    BATCH_SIZE = 128
    TRAIN_TEST_SPLIT = 0.8  # 80% Memory, 20% Test
    SEED = 42

    TOP_K = 3
    DIFFUSION_ALPHA = 0.7
    EDGE_MATCH_THRESHOLD = 3  # how many chunk matches to form edge
    MAX_NEIGHBORS = 10
    GRAPH_CLASS_AGG = "mean"  # options: "sum" or "mean"

    # Bio Graph Defaults
    BIO_PROTO_WEIGHT = 0.55
    BIO_NODE_TEMP = 0.08
    BIO_PROTO_TEMP = 0.10
    BIO_MERGE_THRESHOLD = 0.30
    BIO_MAX_NODES_PER_CLASS = 64
    BIO_KMEANS_PER_CLASS = 8
    BIO_USE_DISCRIM_CONSOLIDATION = True
    BIO_DISC_STEPS = 150
    BIO_DISC_LR = 0.01
    BIO_DISC_MARGIN = 0.15
    BIO_DISC_NEG_WEIGHT = 1.0
    BIO_USE_MAHALANOBIS = True
    BIO_VAR_EPS = 1e-3
    BIO_UNCERTAINTY_MOMENTUM = 0.95
    BIO_DYNAMIC_BUDGET_FLOOR = 0.25
    BIO_USE_PROJECTION = True
    BIO_PROJ_DIM = 128
    BIO_PROJ_STEPS = 30
    BIO_PROJ_LR = 0.03
    BIO_PROJ_MARGIN = 0.20
    BIO_PROJ_ORTHO_REG = 1e-2
