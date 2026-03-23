import os
import urllib.request
import zipfile

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, ImageFolder

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
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(Config.DATA_ROOT)

    # Formatting validation folder structure for ImageFolder compatibility
    val_dir = os.path.join(dataset_root, "val")
    if os.path.exists(os.path.join(val_dir, "images")):
        print("🔧 Reformatting validation folder structure...")
        with open(os.path.join(val_dir, "val_annotations.txt"), "r") as f:
            for line in f:
                p = line.split("\t")
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
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
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
    print(
        f"👉 Please download it from https://github.com/hendrycks/imagenet-r and extract it to {Config.DATA_ROOT}/imagenet-r"
    )

    # Attempt extraction if user placed the tar file there manually
    tar_path = os.path.join(Config.DATA_ROOT, "imagenet-r.tar")
    if os.path.exists(tar_path):
        print("📦 Extracting imagenet-r.tar...")
        import tarfile

        with tarfile.open(tar_path) as file:
            file.extractall(Config.DATA_ROOT)
        return dataset_root

    raise FileNotFoundError(
        "ImageNet-R not found. Cannot auto-download due to GDrive quota issues. "
        "Please download 'imagenet-r.tar' manually."
    )

def setup_objectnet():
    """
    Checks for ObjectNet.
    Users must download ObjectNet.zip manually due to license walls.
    """
    os.makedirs(Config.DATA_ROOT, exist_ok=True)
    dataset_root = os.path.join(Config.DATA_ROOT, "objectnet")

    if os.path.exists(dataset_root) and os.path.exists(os.path.join(dataset_root, "images")):
        print(f"✅ ObjectNet found at {dataset_root}")
        return dataset_root

    print(f"⚠️  ObjectNet folder not found at {dataset_root}")
    print(f"👉 Please download it from https://objectnet.dev/ and extract it to {dataset_root}")

    zip_path = os.path.join(Config.DATA_ROOT, "objectnet-1.0.zip")
    if os.path.exists(zip_path):
        print("📦 Extracting ObjectNet zip...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_root)
        return dataset_root
        
    raise FileNotFoundError("ObjectNet not found. Cannot auto-download. Please see instructions.")

def setup_domainnet(domain=None):
    """
    Checks for DomainNet root and optional single-domain subfolder.
    Expected layout:
      data/domainnet/<domain_name>/<class_name>/*.jpg
    Supported domains:
      clipart, infograph, painting, quickdraw, real, sketch
    """
    os.makedirs(Config.DATA_ROOT, exist_ok=True)
    root = os.path.join(Config.DATA_ROOT, "domainnet")
    if not os.path.exists(root):
        raise FileNotFoundError(
            f"DomainNet not found at {root}. Please place extracted DomainNet there."
        )

    if domain is None:
        sample_entries = [
            p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))
        ]
        known = {"clipart", "infograph", "painting", "quickdraw", "real", "sketch"}
        if any(x in known for x in sample_entries):
            default_domain = "real"
            default_root = os.path.join(root, default_domain)
            if os.path.exists(default_root):
                print(
                    "ℹ️  DomainNet multi-domain root detected. "
                    "Defaulting to 'real'. "
                    "Pass --dataset domainnet_<domain> to select another domain."
                )
                print(f"✅ DomainNet ({default_domain}) found at {default_root}")
                return default_root
        print(f"✅ DomainNet found at {root}")
        return root

    domain = domain.lower().strip()
    domain_root = os.path.join(root, domain)
    if not os.path.exists(domain_root):
        raise FileNotFoundError(
            f"DomainNet domain folder not found: {domain_root}. "
            "Expected one of: clipart, infograph, painting, quickdraw, real, sketch."
        )

    print(f"✅ DomainNet ({domain}) found at {domain_root}")
    return domain_root


def get_dataloader(dataset_name="tinyimagenet", use_train_set=True):
    """
    Unified loader for TinyImageNet, CIFAR-100, CORe50, or ImageNet-R.
    """
    print(f"🔄 Preparing Dataset: {dataset_name.upper()}...")

    # Ensure correct resolution based on backbone
    is_siglip = "siglip" in Config.BACKBONE.lower()
    img_size = 384 if is_siglip else 224

    mean = [0.5, 0.5, 0.5] if is_siglip else [0.485, 0.456, 0.406]
    std = [0.5, 0.5, 0.5] if is_siglip else [0.229, 0.224, 0.225]
    interp = transforms.InterpolationMode.BICUBIC if is_siglip else transforms.InterpolationMode.BILINEAR

    # Standard Transform for Models
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=interp),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    if dataset_name.lower() == "cifar100":
        root = os.path.join(Config.DATA_ROOT, "cifar100")
        dataset = CIFAR100(
            root=root, train=use_train_set, transform=transform, download=True
        )
        Config.CLASSES_PER_TASK = 10
        Config.N_TASKS = 10

    elif "tiny" in dataset_name.lower():
        root = setup_tiny_imagenet()
        target_folder = "train" if use_train_set else "val"
        data_dir = os.path.join(root, target_folder)
        dataset = ImageFolder(root=data_dir, transform=transform)
        Config.CLASSES_PER_TASK = 10
        Config.N_TASKS = 20

    elif dataset_name.lower() == "core50":
        root = setup_core50()
        dataset = ImageFolder(root=root, transform=transform)
        # CORe50 Standard NC (New Classes) Scenario
        Config.CLASSES_PER_TASK = 5
        Config.N_TASKS = 10

    elif "domainnet" in dataset_name.lower():
        # Accept:
        #   domainnet
        #   domainnet_real / domainnet-sketch / domainnet:clipart
        dname = dataset_name.lower().strip()
        domain = None
        if "_" in dname:
            parts = dname.split("_", 1)
            if parts[0] == "domainnet" and parts[1]:
                domain = parts[1]
        elif "-" in dname:
            parts = dname.split("-", 1)
            if parts[0] == "domainnet" and parts[1]:
                domain = parts[1]
        elif ":" in dname:
            parts = dname.split(":", 1)
            if parts[0] == "domainnet" and parts[1]:
                domain = parts[1]

        root = setup_domainnet(domain=domain)
        dataset = ImageFolder(root=root, transform=transform)

        # DomainNet has 345 classes in full benchmark.
        # Use 15 classes/task for 23 tasks (345 classes).
        Config.CLASSES_PER_TASK = 15
        Config.N_TASKS = 23

        # DomainNet folder(s) often don't come with explicit train/test class folders here.
        # We split in main.py with existing 80/20 logic.
        print("ℹ️  Note: DomainNet split is handled in main.py (80/20 per task).")

    elif "r" in dataset_name.lower() or "rendition" in dataset_name.lower():
        root = setup_imagenet_r()
        # ImageNet-R has 200 classes total
        # We assume a 10-task split (20 classes per task) or 20-task split (10 per task)
        # L2P uses 10 tasks of 20 classes.
        dataset = ImageFolder(root=root, transform=transform)
        Config.CLASSES_PER_TASK = 20
        Config.N_TASKS = 10

        # ImageNet-R doesn't have a train/test split. It's just one folder.
        # We usually split it manually inside the main script (80/20 split).
        print(
            "ℹ️  Note: ImageNet-R is a single folder. Train/Test splitting happens in main.py."
        )

    elif dataset_name.lower() == "objectnet":
        root = setup_objectnet()
        img_folder = os.path.join(root, "images") if os.path.exists(os.path.join(root, "images")) else root
        dataset = ImageFolder(root=img_folder, transform=transform)
        Config.CLASSES_PER_TASK = 20
        Config.N_TASKS = 15 # 300 classes
        print("ℹ️  Note: ObjectNet handled as 15 tasks of 20 classes (first 300 classes). Train/test split handled in main.py.")

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            "Choose 'cifar100', 'tinyimagenet', 'core50', 'imagenet-r', or 'domainnet'."
        )

    print(
        f"✅ Loaded {len(dataset)} images. Classes expected: {Config.CLASSES_PER_TASK * Config.N_TASKS}"
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return dataset, loader
