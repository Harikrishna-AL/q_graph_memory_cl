import os
import zipfile
import urllib.request
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, CIFAR100
from torch.utils.data import DataLoader
from .config import Config

def setup_tiny_imagenet():
    """Downloads and formats Tiny-ImageNet if not present."""
    os.makedirs(Config.DATA_ROOT, exist_ok=True)
    dataset_folder_name = "tiny-imagenet-200"
    dataset_root = os.path.join(Config.DATA_ROOT, dataset_folder_name)
    zip_path = os.path.join(Config.DATA_ROOT, f"{dataset_folder_name}.zip")
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    
    if os.path.exists(dataset_root):
        print(f"✅ Tiny-ImageNet found at {dataset_root}")
        return dataset_root

    print("⬇️  Downloading Tiny-ImageNet...")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
    
    print("📦 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(Config.DATA_ROOT)
        
    # Formatting validation folder structure for ImageFolder compatibility
    val_dir = os.path.join(dataset_root, "val")
    if os.path.exists(os.path.join(val_dir, "images")):
        print("🔧 Reformatting validation folder structure...")
        with open(os.path.join(val_dir, "val_annotations.txt"), 'r') as f:
            for line in f:
                p = line.split('\t')
                target_dir = os.path.join(val_dir, p[1])
                os.makedirs(target_dir, exist_ok=True)
                src = os.path.join(val_dir, "images", p[0])
                dst = os.path.join(target_dir, p[0])
                if os.path.exists(src):
                    os.rename(src, dst)
        if os.path.exists(os.path.join(val_dir, "images")):
            os.rmdir(os.path.join(val_dir, "images"))
    return dataset_root

def setup_core50():
    """Downloads and extracts CORe50 dataset (128x128 version)."""
    os.makedirs(Config.DATA_ROOT, exist_ok=True)
    dataset_root = os.path.join(Config.DATA_ROOT, "core50_128x128")
    zip_path = os.path.join(Config.DATA_ROOT, "core50_128x128.zip")
    url = "http://bias.csr.unibo.it/vr/core50/core50_128x128.zip"

    if os.path.exists(dataset_root):
        print(f"✅ CORe50 found at {dataset_root}")
        return dataset_root

    print("⬇️  Downloading CORe50...")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
    print("📦 Extracting CORe50...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(Config.DATA_ROOT)
    return dataset_root

def setup_imagenet_r():
    """
    Checks for ImageNet-R. 
    NOTE: Automatic download is unstable for ImageNet-R. 
    Please download 'imagenet-r.tar' manually if this fails.
    """
    os.makedirs(Config.DATA_ROOT, exist_ok=True)
    dataset_root = os.path.join(Config.DATA_ROOT, "imagenet-r")
    
    if os.path.exists(dataset_root):
        print(f"✅ ImageNet-R found at {dataset_root}")
        return dataset_root
        
    print(f"⚠️  ImageNet-R folder not found at {dataset_root}")
    print(f"👉 Please download it from https://github.com/hendrycks/imagenet-r and extract it to {Config.DATA_ROOT}/imagenet-r")
    
    # Attempt extraction if user placed the tar file there manually
    tar_path = os.path.join(Config.DATA_ROOT, "imagenet-r.tar")
    if os.path.exists(tar_path):
         print("📦 Extracting imagenet-r.tar...")
         import tarfile
         with tarfile.open(tar_path) as file:
             file.extractall(Config.DATA_ROOT)
         return dataset_root
         
    raise FileNotFoundError("ImageNet-R not found. Please download and extract it manually.")

def get_dataloader(dataset_name='tinyimagenet', use_train_set=True):
    """
    Unified loader for TinyImageNet, CIFAR-100, CORe50, or ImageNet-R.
    """
    print(f"🔄 Preparing Dataset: {dataset_name.upper()}...")
    
    # Standard Transform for DINOv2 (224x224)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if dataset_name.lower() == 'cifar100':
        root = os.path.join(Config.DATA_ROOT, 'cifar100')
        dataset = CIFAR100(root=root, train=use_train_set, transform=transform, download=True)
        Config.CLASSES_PER_TASK = 10 
        Config.N_TASKS = 10
        
    elif 'tiny' in dataset_name.lower():
        root = setup_tiny_imagenet()
        target_folder = 'train' if use_train_set else 'val'
        data_dir = os.path.join(root, target_folder)
        dataset = ImageFolder(root=data_dir, transform=transform)
        Config.CLASSES_PER_TASK = 10 
        Config.N_TASKS = 20

    elif dataset_name.lower() == 'core50':
        root = setup_core50()
        dataset = ImageFolder(root=root, transform=transform)
        # CORe50 Standard NC (New Classes) Scenario
        Config.CLASSES_PER_TASK = 5 
        Config.N_TASKS = 10
        
    elif 'r' in dataset_name.lower() or 'rendition' in dataset_name.lower():
        root = setup_imagenet_r()
        # ImageNet-R has 200 classes total
        # We assume a 10-task split (20 classes per task) or 20-task split (10 per task)
        # L2P uses 10 tasks of 20 classes.
        dataset = ImageFolder(root=root, transform=transform)
        Config.CLASSES_PER_TASK = 20
        Config.N_TASKS = 10
        
        # ImageNet-R doesn't have a train/test split. It's just one folder.
        # We usually split it manually inside the main script (80/20 split).
        print("ℹ️  Note: ImageNet-R is a single folder. Train/Test splitting happens in main.py.")
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'cifar100', 'tinyimagenet', 'core50', or 'imagenet-r'.")

    print(f"✅ Loaded {len(dataset)} images. Classes expected: {Config.CLASSES_PER_TASK * Config.N_TASKS}")
    
    loader = DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return dataset, loader