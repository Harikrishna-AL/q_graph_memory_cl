import os
import zipfile
import urllib.request
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from .config import Config

def setup_tiny_imagenet():
    """Downloads and formats Tiny-ImageNet if not present."""
    dataset_root = os.path.join(Config.DATA_ROOT, Config.DATASET_NAME)
    zip_path = os.path.join(Config.DATA_ROOT, f"{Config.DATASET_NAME}.zip")
    
    if os.path.exists(dataset_root):
        print(f"✅ Dataset found at {dataset_root}")
        return dataset_root

    print("⬇️  Downloading Tiny-ImageNet...")
    os.makedirs(Config.DATA_ROOT, exist_ok=True)
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(Config.URL, zip_path)
    
    print("📦 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(Config.DATA_ROOT)
        
    # Formatting validation folder (just in case we need it, though we use train)
    val_dir = os.path.join(dataset_root, "val")
    if os.path.exists(os.path.join(val_dir, "images")):
        print("🔧 Reformatting validation folder structure...")
        with open(os.path.join(val_dir, "val_annotations.txt"), 'r') as f:
            for line in f:
                p = line.split('\t')
                os.makedirs(os.path.join(val_dir, p[1]), exist_ok=True)
                os.rename(os.path.join(val_dir, "images", p[0]), os.path.join(val_dir, p[1], p[0]))
        os.rmdir(os.path.join(val_dir, "images"))
        
    return dataset_root

def get_dataloader(use_train_set=True):
    root = setup_tiny_imagenet()
    target_folder = 'train' if use_train_set else 'val'
    data_dir = os.path.join(root, target_folder)
    
    print(f"📂 Loading data from: {data_dir}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    return dataset, loader