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
    # Ensure root exists
    os.makedirs(Config.DATA_ROOT, exist_ok=True)
    
    dataset_folder_name = "tiny-imagenet-200"
    dataset_root = os.path.join(Config.DATA_ROOT, dataset_folder_name)
    zip_path = os.path.join(Config.DATA_ROOT, f"{dataset_folder_name}.zip")
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    
    if os.path.exists(dataset_root):
        print(f"✅ Dataset found at {dataset_root}")
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
                # p[0] = filename, p[1] = class_label
                target_dir = os.path.join(val_dir, p[1])
                os.makedirs(target_dir, exist_ok=True)
                
                src = os.path.join(val_dir, "images", p[0])
                dst = os.path.join(target_dir, p[0])
                
                if os.path.exists(src):
                    os.rename(src, dst)
                    
        if os.path.exists(os.path.join(val_dir, "images")):
            os.rmdir(os.path.join(val_dir, "images"))
        
    return dataset_root

def get_dataloader(dataset_name='tinyimagenet', use_train_set=True):
    """
    Unified loader for TinyImageNet or CIFAR-100.
    dataset_name: 'tinyimagenet' or 'cifar100'
    """
    print(f"🔄 Preparing Dataset: {dataset_name.upper()}...")
    
    # Standard Transform for DINOv2 (224x224)
    # Upsampling CIFAR (32->224) is vital for DINOv2 performance
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if dataset_name.lower() == 'cifar100':
        # Download/Load CIFAR-100 via Torchvision
        root = os.path.join(Config.DATA_ROOT, 'cifar100')
        dataset = CIFAR100(
            root=root, 
            train=use_train_set, 
            transform=transform, 
            download=True
        )
        # Update Config globally for consistency
        # CIFAR-100 typically split into 10 tasks of 10 classes or 20 tasks of 5
        Config.CLASSES_PER_TASK = 10 
        Config.N_TASKS = 10
        
    elif 'tiny' in dataset_name.lower():
        # Load Tiny-ImageNet
        root = setup_tiny_imagenet()
        target_folder = 'train' if use_train_set else 'val'
        data_dir = os.path.join(root, target_folder)
        
        dataset = ImageFolder(root=data_dir, transform=transform)
        # TinyImageNet defaults
        Config.CLASSES_PER_TASK = 10 
        Config.N_TASKS = 20
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"✅ Loaded {len(dataset)} images. Classes: {len(dataset.classes) if hasattr(dataset, 'classes') else 100}")
    
    loader = DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, # We shuffle manually in learner.py to simulate streams
        num_workers=4,
        pin_memory=True
    )
    
    return dataset, loader