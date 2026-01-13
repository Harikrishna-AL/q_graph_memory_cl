import torch

class Config:
    # Hardware
    DEVICE = torch.device("mps") if torch.backends.mps.is_available() else \
             torch.device("cuda") if torch.cuda.is_available() else \
             torch.device("cpu")
    
    # Data
    DATA_ROOT = "./data"
    DATASET_NAME = "tiny-imagenet-200"
    URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    
    # Model
    DINO_REPO = 'facebookresearch/dinov2'
    DINO_MODEL = 'dinov2_vits14'
    
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
