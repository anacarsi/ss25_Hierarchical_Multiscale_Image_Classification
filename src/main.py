from collections import defaultdict
import os
import sys
import csv
import argparse
import requests
from tqdm import tqdm
from torch.utils.data import Subset
import torchvision.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import shutil
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageOps
from lxml import etree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.resnet import (
    ResNetClassifier,
    ResNetFeatureExtractor,
    ResNet18ClassifierSIMCLR,
)
from datasets.patch_dataset import PatchDataset
from src.datasets.mil_dataset import WSIMILTestDataset, get_mil_dataloaders
from models.mil_classifier import MILClassifier
from utils.evaluation_FROC import (
    computeEvaluationMask,
    computeITCList,
    readCSVContent,
    compute_FP_TP_Probs,
    computeFROC,
    plotFROC,
)
from sklearn.metrics import f1_score, precision_score, recall_score
from models.simclr import pretrain_simclr
import zipfile
import pandas as pd
from train import train_resnet
from utils.structure import get_latest_mil_model_path

""""
os.add_dll_directory(
    r"C:\Program Files\OpenSlide\openslide-bin-4.0.0.8-windows-x64\bin"
)
"""
import openslide


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    DEBUG = "\033[96m"
    INFO = "\033[95m"  # pink
    WARNING = "\033[93m"  # yellow
    ERROR = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


MODEL_PATH = "mil_classifier_attention_final_level3_resnet18_patch_classifier_final_level3_20250710055900.pth.pth"
BATCH_SIZE = 512  # 4 GPUS - 128 per GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Balance the dataset by limiting the number of patches per class. At level, max 7483 tumor patches and 7000 normal patches.
SAMPLES_PER_CLASS = 7480

# Base URL for the CAMELYON16 dataset
BASE_URL = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100439/"

# Patches size (not directly used in the download logic, but kept for context)
PATCH_SIZE_LEVEL_0 = 1792

# File paths for CAMELYON16
CAMELYON16_FILES = {
    "train_normal": [
        f"CAMELYON16/training/normal/normal_{i:03d}.tif" for i in range(1, 112)
    ],
    "train_tumor": [
        f"CAMELYON16/training/tumor/tumor_{i:03d}.tif" for i in range(1, 112)
    ],
    "test_images": [
        f"CAMELYON16/testing/images/test_{i:03d}.tif" for i in range(1, 51)
    ],
    "train_masks": ["CAMELYON16/training/lesion_annotations.zip"],
    "test_masks": [
        "CAMELYON16/testing/lesion_annotations.zip",
        "CAMELYON16/testing/evaluation/evaluation_python.zip",
    ],
}

DOWNLOADED_FILES = {
    "train_normal": [f"normal_{i:03d}" for i in range(1, 112)],
    "train_tumor": [f"tumor_{i:03d}" for i in range(1, 112)],
    "test_images": [f"test_{i:03d}" for i in range(1, 51)],
}


def download_file(url, destination_path):
    """
    Download a file from a URL to a destination path with a progress bar.

    Parameters:
    - url (str): The URL to download from.
    - destination_path (str): The local file path to save the downloaded file.

    Returns:
    - bool: True if download succeeded, False otherwise.
    """
    try:
        print(
            f"{bcolors.INFO}[INFO]{bcolors.ENDC} Downloading: {url} into {destination_path}"
        )
        os.makedirs(
            os.path.dirname(destination_path), exist_ok=True
        )  # Ensure destination dir exists
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with open(destination_path, "wb") as f, tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {os.path.basename(destination_path)}",
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
        print(
            f"{bcolors.INFO}[INFO]{bcolors.ENDC} Successfully downloaded {os.path.basename(destination_path)}."
        )
        return True
    except requests.exceptions.RequestException as e:
        print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Failed to download {url}: {e}")
        return False
    except Exception as e:
        print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} An unexpected error occurred: {e}")
        return False


def download_dataset(remote=False):
    """
    Download the CAMELYON16 dataset, including training, testing, and mask files.

    Parameters:
    - remote (bool): If True, download all files; if False, download only a subset for testing.
    """
    camelyon_dir = os.path.join(os.getcwd(), "data", "camelyon16")

    # Define the target directories
    train_img_dir = os.path.join(camelyon_dir, "train", "img")
    val_img_dir = os.path.join(camelyon_dir, "val", "img")
    test_img_dir = os.path.join(camelyon_dir, "test", "img")
    train_mask_dir = os.path.join(camelyon_dir, "train", "mask")
    test_mask_dir = os.path.join(camelyon_dir, "test", "mask")

    # Mapping of CAMELYON16_FILES keys to their target directories
    download_map = {
        "train_normal": train_img_dir,
        "train_tumor": train_img_dir,
        "test_images": test_img_dir,
        "train_masks": train_mask_dir,
        "test_masks": test_mask_dir,
    }

    # Apply limits for non-remote mode
    limits = {"train_normal": 39, "train_tumor": 110, "test_images": 30}

    for file_type, target_dir in download_map.items():
        files_to_download = CAMELYON16_FILES[file_type]

        # Apply limits based on file type
        if file_type in limits:
            files_to_download = files_to_download[: limits[file_type]]

        # In non-remote mode, only download one image file per category
        if not remote and file_type in ["train_normal", "train_tumor", "test_images"]:
            files_to_download = files_to_download[:1]

        for remote_file_path in files_to_download:
            file_name = os.path.basename(remote_file_path)

            if "evaluation_python" in file_name and remote:
                print(
                    f"{bcolors.INFO}[INFO]{bcolors.ENDC} Skipping download of {file_name} in remote mode."
                )
                continue

            train_img_path = os.path.join(train_img_dir, file_name)
            val_img_path = os.path.join(val_img_dir, file_name)
            test_img_path = os.path.join(test_img_dir, file_name)
            destination_path = os.path.join(target_dir, file_name)

            if any(
                os.path.exists(p) for p in [train_img_path, val_img_path, test_img_path]
            ):
                print(
                    f"{bcolors.INFO}[INFO]{bcolors.ENDC} Skipping: {file_name} already exists in train/img, val/img, or test/img."
                )
                continue
            if file_type in ["train_masks", "test_masks"] and os.path.exists(
                destination_path
            ):
                print(
                    f"{bcolors.INFO}[INFO]{bcolors.ENDC} Skipping: {file_name} already exists in {target_dir}."
                )
                continue

            url = BASE_URL + remote_file_path
            download_file(url, destination_path)

    create_validation_set()


def extract_zip(zip_path, extract_to):
    """
    Extract a zip file containing masks to the specified annotation directory.

    Parameters:
    - zip_path (str): Path to the zip file to extract.
    - extract_to (str): Directory to extract the contents to.
    """
    # Check if the path extract_to exists. If yes, check contains all elements from tumor_001.xml to tumor_050.xml.
    # If it does not contain them, delete the directory and extract again. If exists and contains all elements, skip extraction.
    expected_xmls = [f"tumor_{i:03d}.xml" for i in range(1, 51)]

    if os.path.exists(extract_to):
        existing_xmls = set(os.listdir(extract_to))
        if all(xml in existing_xmls for xml in expected_xmls):
            print(
                f"{bcolors.INFO}[INFO]{bcolors.ENDC} Directory {extract_to} already exists and contains all expected XMLs. Skipping extraction."
            )
            return
        else:
            print(
                f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Directory {extract_to} exists but is missing some XMLs. Re-extracting..."
            )
            shutil.rmtree(extract_to)
            os.makedirs(extract_to)
    else:
        os.makedirs(extract_to)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Extracted {zip_path} to {extract_to}")


def download_all_tumor_extract_patches(download=False):
    """
    Download all tumor images and extract tumor patches from them.

    Parameters:
    - download (bool): If True, download all tumor images before extracting patches.
    """
    print(
        f"{bcolors.HEADER}{bcolors.BOLD}[HEADER]{bcolors.ENDC} Download all tumor images and extract tumor patches"
    )
    camelyon_dir = os.path.join(os.getcwd(), "data", "camelyon16")
    train_img_dir = os.path.join(camelyon_dir, "train", "img")
    train_mask_dir = os.path.join(camelyon_dir, "train", "mask", "annotations")

    # Download all tumor images
    if download:
        print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Downloading all tumor images...")
        for i in range(36, 112):  # from validation images to end
            file_name = f"tumor_{i:03d}.tif"
            if os.path.exists(os.path.join(train_img_dir, file_name)):
                print(
                    f"{bcolors.INFO}[INFO]{bcolors.ENDC} Tumor image {file_name} already exists in {train_img_dir}. Skipping download."
                )
                continue
            url = BASE_URL + f"CAMELYON16/training/tumor/{file_name}"
            destination_path = os.path.join(train_img_dir, file_name)
            download_file(url, destination_path)

    # Extract only tumor patches from downloaded tumor images
    extract_patches(patch_size=224, level=3, stride=None, pad=True)


def parse_xml_mask(xml_path, level_dims, slide):
    """
    Convert an XML annotation file to a binary mask for a WSI at a given level.

    Parameters:
    - xml_path (str): Path to the XML file containing annotations.
    - level_dims (tuple): Dimensions of the WSI at the specified level (width, height).
    - slide (OpenSlide): OpenSlide object for the WSI.

    Returns:
    - PIL.Image: Binary mask image.
    """
    try:
        tree = etree.parse(xml_path)
    except etree.XMLSyntaxError as e:
        print(
            f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Error parsing XML file {xml_path}: {e}"
        )
        return None

    # Compute scaling factors based on actual dimensions
    base_dims = slide.level_dimensions[0]
    scale_x = level_dims[0] / base_dims[0]
    scale_y = level_dims[1] / base_dims[1]

    mask = Image.new("L", level_dims, 0)
    draw = ImageDraw.Draw(mask)

    for coordinates_node in tree.xpath(
        "//Annotation/Coordinates | //Annotations/Annotation/Coordinates"
    ):
        coords = []
        for coord_node in coordinates_node.findall("Coordinate"):
            try:
                x = float(coord_node.get("X"))
                y = float(coord_node.get("Y"))
                # Scale coordinates to the target level
                scaled_x = int(x * scale_x)
                scaled_y = int(y * scale_y)
                coords.append((scaled_x, scaled_y))
            except (ValueError, TypeError) as e:
                print(
                    f"{bcolors.WARNING}Warning: Could not parse coordinate (X,Y) from XML for {xml_path}: {e}{bcolors.ENDC}"
                )
                continue
        if coords:
            draw.polygon(coords, outline=255, fill=255)
    return mask


def get_dataloaders(patch_dir, batch_size=BATCH_SIZE, balanced=False):
    """
    Create PyTorch DataLoaders for training and validation patch datasets.

    Parameters:
    - patch_dir (str): Directory containing patch data.
    - batch_size (int): Batch size for DataLoader.
    - balanced (bool): Whether to balance the dataset by limiting samples per class.

    Returns:
    - tuple: (train_loader, val_loader, train_dataset, val_dataset)
    """
    train_dir = os.path.join(patch_dir, "train")
    val_dir = os.path.join(patch_dir, "val")
    train_slides = [
        d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    ]
    print(
        f"{bcolors.INFO}[INFO]{bcolors.ENDC} Found {len(train_slides)} training slides in {train_dir}."
    )
    val_slides = [
        d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))
    ]
    print(
        f"{bcolors.INFO}[INFO]{bcolors.ENDC} Found {len(val_slides)} validation slides in {val_dir}."
    )
    print(f"For example, found slide: {val_slides[0]} in {val_dir}.")
    print(
        f"{bcolors.INFO}[INFO]{bcolors.ENDC} Found {len(train_slides) + len(val_slides)} slides in {patch_dir}."
    )
    # Data augmentation for classification task
    train_transform = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(90),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = PatchDataset(
        patch_dir,
        slide_names=train_slides,
        tumor_transform=train_transform,
        normal_transform=train_transform,  # Apply augmentation to both classes in training
        balanced=balanced,
        max_samples=SAMPLES_PER_CLASS if balanced else None,
    )
    val_dataset = PatchDataset(
        patch_dir,
        slide_names=val_slides,
        tumor_transform=val_transform,
        normal_transform=val_transform,  # No augmentation for validation
    )

    # Balance the validation set w subset if there are enough samples -> subset to the minimum of tumor and normal patches
    val_labels_array = np.array(val_dataset.labels)
    tumor_indices = np.where(val_labels_array == 1)[0]
    normal_indices = np.where(val_labels_array == 0)[0]
    n_tumor = len(tumor_indices)
    n_normal = len(normal_indices)

    # if n_tumor > 0 and n_normal > 0:
    # n_min = min(n_tumor, n_normal)
    # rng = np.random.default_rng(42)
    # tumor_sel = rng.choice(tumor_indices, n_min, replace=False)
    # normal_sel = rng.choice(normal_indices, n_min, replace=False)
    # selected_indices = np.concatenate([tumor_sel, normal_sel])
    # val_dataset = Subset(val_dataset, selected_indices)
    # print(
    # f"{bcolors.INFO}[INFO]{bcolors.ENDC} Validation set balanced: {n_min} normal and {n_min} tumor patches."
    # )
    # else:
    print(
        f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Did not balance validation set: tumor patches = {n_tumor}, normal patches = {n_normal}."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset, val_dataset


def train_resnet_classifier(
    level=3,
    strategy="baseline",  # options: baseline, balanced, weighted_loss, self_supervised
    phase1_epochs=10,
    phase2_epochs=20,
):
    """
    Train a ResNet18 classifier on extracted patches with optional strategy.

    Parameters:
    - level (int): WSI level for patch extraction.
    - strategy (str): Training strategy ('baseline', 'balanced', 'weighted_loss', 'self_supervised').
    - phase1_epochs (int): Number of epochs for phase 1 (self-supervised only).
    - phase2_epochs (int): Number of epochs for phase 2 (self-supervised only).
    """
    patch_dir = os.path.join(
        os.getcwd(), "data", "camelyon16", "patches", f"level_{level}"
    )
    pretrained_simclr_path = os.path.join(
        os.getcwd(), "src", "models", f"simclr_encoder_best_level{level}.pth"
    )
    balanced = strategy == "balanced"

    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(
        patch_dir, batch_size=BATCH_SIZE, balanced=balanced
    )
    if train_loader is None:
        return

    base_model_path = f"src/models/resnet18_{strategy}_level{level}"

    all_labels = np.array(train_dataset.labels)
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    sample_counts = np.array(
        [
            counts[np.where(unique_labels == t)[0][0]] if t in unique_labels else 1
            for t in [0, 1]
        ]
    )
    class_weights = 1.0 / sample_counts
    class_weights = class_weights / np.min(class_weights)
    weight_tensor = torch.FloatTensor(class_weights).to(device)

    if strategy == "self_supervised":
        if not os.path.exists(pretrained_simclr_path):
            print("Pretraining SimCLR encoder...")
            pretrain_simclr(patch_dir, epochs=200, level=level)

        # Phase 1
        model = ResNet18ClassifierSIMCLR(
            pretrained_weights_path=pretrained_simclr_path, freeze_encoder=True
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)
        optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=3)
        train_resnet(
            model,
            train_loader,
            val_loader,
            train_dataset,
            val_dataset,
            criterion,
            optimizer,
            scheduler,
            phase1_epochs,
            base_model_path + "_phase1",
        )

        # Phase 2
        model = ResNet18ClassifierSIMCLR(
            pretrained_weights_path=base_model_path + "_phase1_best.pth",
            freeze_encoder=False,
        ).to(device)
        optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)
        train_resnet(
            model,
            train_loader,
            val_loader,
            train_dataset,
            val_dataset,
            criterion,
            optimizer,
            scheduler,
            phase2_epochs,
            base_model_path,
        )

    else:
        model = ResNetClassifier().to(device)
        criterion = (
            nn.CrossEntropyLoss(weight=weight_tensor)
            if strategy == "weighted_loss"
            else nn.CrossEntropyLoss()
        )
        optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)
        train_resnet(
            model,
            train_loader,
            val_loader,
            train_dataset,
            val_dataset,
            criterion,
            optimizer,
            scheduler,
            30,
            base_model_path,
        )


def train_mil_classifier(
    feature_level=3,
    pooling="attention",
    epochs=50,
    lr=1e-4,
    patience=10,
    model_type="resnet18",
):
    """
    Train a Multiple Instance Learning (MIL) classifier using extracted WSI features.

    Parameters:
    - feature_level (int): WSI level from which features were extracted.
    - pooling (str): Aggregation method ('attention', 'mean', 'max').
    - epochs (int): Number of training epochs.
    - lr (float): Learning rate for the MIL model.
    - patience (int): Patience for early stopping.
    - model_type (str): The name of the feature extractor model (e.g., "resnet18_patch_classifier_final" or "resnet18_patch_classifier_simclr_best").
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_base_dir_train = os.path.join(
        os.getcwd(),
        "data",
        "camelyon16",
        "features",
        f"level_{feature_level}",
        model_type,
        "train",
    )
    feature_base_dir_val = os.path.join(
        os.getcwd(),
        "data",
        "camelyon16",
        "features",
        f"level_{feature_level}",
        model_type,
        "val",
    )
    if not os.path.exists(feature_base_dir_train) or not os.path.exists(
        feature_base_dir_val
    ):
        print(
            f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Feature directory does not exist. Please run feature extraction first."
        )
        return

    train_loader, val_loader, _ = get_mil_dataloaders(
        feature_base_dir_train,
        feature_base_dir_val,
        feature_base_dir_test=None,
        batch_size=1,
    )

    feature_dim = 512 if model_type.startswith("resnet18") else 2048
    model = MILClassifier(feature_dim=feature_dim, pooling=pooling).to(device)

    # Optimizer: All parameters of the MILClassifier are trainable.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=patience // 2,
        threshold=0.001,
    )

    best_auc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    # For consistent naming
    # Remove .pth from model_type if it exists
    if model_type.endswith(".pth"):
        model_prefix = model_type[:-4]
    base_model_name = f"mil_{model_prefix}_{pooling}"

    log_filename = f"src/models/{base_model_name}_log.csv"
    log_fields = ["epoch", "train_loss", "train_auc", "val_loss", "val_auc"]
    with open(log_filename, "w", newline="") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(log_fields)

        for epoch in range(epochs):
            # ----------- Train -----------
            model.train()
            train_loss = 0.0
            train_preds, train_labels = [], []
            for bags, labels, _ in tqdm(
                train_loader, desc=f"MIL Epoch {epoch+1} Training"
            ):
                bags, labels = bags[0].to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    logits, _ = model(bags)
                    loss = criterion(logits.unsqueeze(0), labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                train_preds.append(logits.softmax(dim=-1)[1].item())
                train_labels.append(labels.item())

            train_auc = roc_auc_score(train_labels, train_preds)
            train_loss /= len(train_loader)

            # ----------- Validation -----------
            model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []
            with torch.no_grad():
                for bags, labels, _ in tqdm(
                    val_loader, desc=f"MIL Epoch {epoch+1} Validation"
                ):
                    bags, labels = bags[0].to(device), labels.to(device)
                    with torch.cuda.amp.autocast():
                        logits, _ = model(bags)
                        loss = criterion(logits.unsqueeze(0), labels)

                    val_loss += loss.item()
                    val_preds.append(logits.softmax(dim=-1)[1].item())
                    val_labels.append(labels.item())

            val_auc = roc_auc_score(val_labels, val_preds)
            val_loss /= len(val_loader)

            print(
                f"{bcolors.INFO}[Epoch {epoch+1:03d}] Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}{bcolors.ENDC}"
            )
            log_writer.writerow(
                [
                    epoch + 1,
                    f"{train_loss:.4f}",
                    f"{train_auc:.4f}",
                    f"{val_loss:.4f}",
                    f"{val_auc:.4f}",
                ]
            )
            log_file.flush()

            # Learning rate scheduler step
            scheduler.step(val_auc)

            # ----------- Early Stopping Logic -----------

            if val_auc > best_auc:
                best_auc = val_auc
                best_model_wts = copy.deepcopy(model.state_dict())
                early_stop_counter = 0
                best_model_path = f"src/models/{base_model_name}_best.pth"
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"{bcolors.INFO}[INFO]{bcolors.ENDC} Best validation AUC achieved. Model saved to {best_model_path}"
                )
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(
                        f"{bcolors.INFO}[Early Stopping]{bcolors.ENDC} No improvement for {patience} epochs. Stopping at epoch {epoch+1}."
                    )
                    break

            # Save checkpoint every 10 epochs (or adjust frequency)
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"src/models/{base_model_name}_epoch{epoch+1}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                print(
                    f"{bcolors.INFO}[INFO]{bcolors.ENDC} Checkpoint saved: {checkpoint_path}"
                )

        # Load best model weights at the end
        model.load_state_dict(best_model_wts)
        final_model_path = f"src/models/{base_model_name}_test_final.pth"
        torch.save(model.state_dict(), final_model_path)
        print(
            f"{bcolors.INFO}[INFO]{bcolors.ENDC} Training complete. Final best model saved to {final_model_path}. Best Val AUC: {best_auc:.4f}"
        )


def test_mil_classifier(feature_level, pooling="attention", model_type="resnet18"):
    """
    Test a trained MIL classifier on features extracted from WSI at a specific level.

    Parameters:
    - feature_level (int): WSI level from which features were extracted.
    - pooling (str): Aggregation method ('attention', 'mean', 'max').
    - model_type (str): The name of the feature extractor model (e.g., "resnet18").

    Returns:
    - dict: Test metrics (AUC, accuracy, precision, recall, f1_score).
    """
    # model_path = get_latest_mil_model_path()
    if model_type.endswith(".pth"):
        model_name = model_type[:-4]
    model_path = f"src/models/mil_{model_name}_{pooling}.pth"
    feature_dim = 512 if model_type.startswith("resnet18") else 2048
    model = MILClassifier(feature_dim=feature_dim, pooling=pooling)
    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    feature_dir = os.path.join(
        os.getcwd(),
        "data",
        "camelyon16",
        "features",
        f"level_{feature_level}",
        model_type,
        "test",
    )
    feature_base_dir_train = os.path.join(
        os.getcwd(),
        "data",
        "camelyon16",
        "features",
        f"level_{feature_level}",
        model_type,
        "train",
    )
    feature_base_dir_val = os.path.join(
        os.getcwd(),
        "data",
        "camelyon16",
        "features",
        f"level_{feature_level}",
        model_type,
        "val",
    )
    print(
        f"{bcolors.DEBUG}[DEBUG]{bcolors.ENDC} Testing MIL classifier with features from {feature_dir}..."
    )
    test_dataset = WSIMILTestDataset(feature_dir)
    if len(test_dataset) == 0:
        print(
            f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} No feature files found in: {feature_dir}"
        )
        return None
    train_loader, val_loader, test_loader = get_mil_dataloaders(
        feature_base_dir_train=feature_base_dir_train,
        feature_base_dir_val=feature_base_dir_val,
        feature_base_dir_test=feature_dir,
        batch_size=1,
    )
    # feature, label. wsi _ name not
    # for i, j in train_loader:
    # print(f"WSI: {j}, Label: {i}, Shape: {i.shape}")

    # for features, label, wsi_name in train_loader:
    # print(f"WSI: {wsi_name[0]}, Label: {label.item()}, Shape: {features.shape}")

    all_predictions = []
    all_true_labels = []
    wsi_names = []

    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Starting model testing...")

    with torch.no_grad():
        for features, labels, wsi_name in test_loader:
            # In our DataLoader, batch_size=1, so features will be a list containing one tensor
            features = features.squeeze(0).to(device)
            labels = labels.to(device)

            logits, _ = model(features)
            print(f"Logits: {logits.cpu().numpy()}")

            probs = torch.softmax(logits, dim=0).cpu()
            probabilities = probs[1].item()
            entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
            print(f"Entropy: {entropy:.4f}, Probabilities: {probs.tolist()}")
            predicted_label = int(
                probabilities > 0.5
            )  # Binary prediction based on 0.5 threshold

            all_predictions.append(probabilities)
            all_true_labels.append(labels.cpu().item())
            wsi_names.append(wsi_name[0])  # wsi_name is a tuple

            print(
                f"WSI: {wsi_name[0]}, True Label: {labels.cpu().item()}, Predicted Probability (Tumor): {probabilities:.4f}, Predicted Class: {predicted_label}"
            )

    if len(all_true_labels) == 0 or len(all_predictions) == 0:
        print(
            f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} No test samples found. Check your test feature directory and data preparation."
        )
        return None
    auc_score = roc_auc_score(all_true_labels, all_predictions)
    binary_predictions = [1 if p > 0.5 else 0 for p in all_predictions]
    accuracy = accuracy_score(all_true_labels, binary_predictions)
    precision = precision_score(all_true_labels, binary_predictions, zero_division=0)
    recall = recall_score(all_true_labels, binary_predictions, zero_division=0)
    f1 = f1_score(all_true_labels, binary_predictions, zero_division=0)

    print(f"\n{bcolors.OKBLUE}--- Test Results ---{bcolors.ENDC}")
    print(f"Test AUC: {auc_score:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")

    results_df = pd.DataFrame(
        {
            "WSI_Name": wsi_names,
            "True_Label": all_true_labels,
            "Predicted_Probability": all_predictions,
            "Predicted_Class": binary_predictions,
        }
    )
    results_df.to_csv(f"mil_test_results_{model_type}.csv", index=False)
    print(
        f"{bcolors.INFO}[INFO]{bcolors.ENDC} Detailed results saved to mil_test_results_{model_type}.csv"
    )

    return {
        "auc": auc_score,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def extract_patches(patch_size=224, level=3, stride=None, pad=True):
    """
    Extract patches from WSIs at a given level and save them to disk.

    Parameters:
    - patch_size (int): Size of the patch to extract.
    - level (int): WSI level for patch extraction.
    - stride (int or None): Stride for patch extraction. Defaults to patch_size.
    - pad (bool): Whether to pad the image to fit patches exactly.
    """
    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Extracting patches at level {level}...")
    stride = stride or patch_size

    patch_sizes = {0: 1792, 1: 896, 2: 448, 3: 224}
    patch_size = patch_sizes.get(level, 224)

    sets = [
        {
            "name": "train",
            "wsi_dir": os.path.join(os.getcwd(), "data", "camelyon16", "train", "img"),
            "annot_dir": os.path.join(
                os.getcwd(), "data", "camelyon16", "train", "mask", "annotations"
            ),
            "patch_dir": os.path.join(
                os.getcwd(), "data", "camelyon16", "patches", f"level_{level}", "train"
            ),
        },
        {
            "name": "test",
            "wsi_dir": os.path.join(os.getcwd(), "data", "camelyon16", "test", "img"),
            "annot_dir": os.path.join(
                os.getcwd(), "data", "camelyon16", "test", "mask", "annotations"
            ),
            "patch_dir": os.path.join(
                os.getcwd(), "data", "camelyon16", "patches", f"level_{level}", "test"
            ),
        },
        {
            "name": "val",
            "wsi_dir": os.path.join(os.getcwd(), "data", "camelyon16", "val", "img"),
            "annot_dir": os.path.join(
                os.getcwd(), "data", "camelyon16", "val", "mask", "annotations"
            ),
            "patch_dir": os.path.join(
                os.getcwd(), "data", "camelyon16", "patches", f"level_{level}", "val"
            ),
        },
    ]

    for s in sets:
        wsi_dir = s["wsi_dir"]
        annot_dir = s["annot_dir"]
        level_dir = s["patch_dir"]
        os.makedirs(level_dir, exist_ok=True)
        if not os.path.exists(wsi_dir):
            print(
                f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} WSI directory {wsi_dir} does not exist, skipping."
            )
            continue
        for file in os.listdir(wsi_dir):
            if not file.endswith(".tif"):
                continue
            prefix = file.replace(".tif", "")

            # Check if patches for this image already exist
            patch_save_dir = os.path.join(level_dir, prefix)
            print(
                f"{bcolors.INFO}[INFO]{bcolors.ENDC} Checking patches for {file} in {patch_save_dir}..."
            )
            if os.path.exists(patch_save_dir) and len(os.listdir(patch_save_dir)) > 0:
                print(
                    f"{bcolors.INFO}[INFO]{bcolors.ENDC} Patches for {file} already extracted, skipping."
                )
                continue
            os.makedirs(patch_save_dir, exist_ok=True)

            wsi_path = os.path.join(wsi_dir, file)
            xml_name = file.replace(".tif", ".xml")
            xml_path = os.path.join(annot_dir, xml_name)
            try:
                slide = openslide.OpenSlide(wsi_path)
            except Exception as e:
                print(
                    f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Could not open {wsi_path}: {e}"
                )
                continue
            downsample = slide.level_downsamples[level]
            width, height = slide.level_dimensions[level]

            # Calculate padded size if needed
            if pad:
                pad_w = (patch_size - width % patch_size) % patch_size
                pad_h = (patch_size - height % patch_size) % patch_size
                padded_width = width + pad_w
                padded_height = height + pad_h
            else:
                padded_width = width
                padded_height = height

            # Load and render XML mask
            mask = None
            if os.path.exists(xml_path):
                try:
                    mask = parse_xml_mask(xml_path, (width, height), slide)
                    if pad and (pad_w > 0 or pad_h > 0):
                        mask = ImageOps.expand(mask, (0, 0, pad_w, pad_h), fill=0)
                except Exception as e:
                    print(
                        f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Failed to parse XML for {file}: {e}"
                    )
            else:
                print(
                    f"{bcolors.INFO}[INFO]{bcolors.ENDC} No annotation found for {file} in {xml_path}, treating as normal."
                )

            print(
                f"{bcolors.INFO}[INFO]{bcolors.ENDC} Processing {file} at level {level} (size: {width}x{height}, padded: {padded_width}x{padded_height})"
            )

            patch_count = 0
            for x in range(0, padded_width, stride):
                for y in range(0, padded_height, stride):
                    # Only process if the top-left corner is inside the original image
                    if x >= width or y >= height:
                        continue

                    patch_w = min(patch_size, width - x)
                    patch_h = min(patch_size, height - y)
                    if patch_w <= 0 or patch_h <= 0:
                        continue

                    region = slide.read_region(
                        (int(x * downsample), int(y * downsample)),
                        level,
                        (patch_w, patch_h),
                    ).convert("RGB")

                    # If patch is smaller than patch_size (at border), pad it to patch_size
                    if patch_w < patch_size or patch_h < patch_size:
                        padded_region = Image.new(
                            "RGB", (patch_size, patch_size), (255, 255, 255)
                        )
                        padded_region.paste(region, (0, 0))
                        region = padded_region

                    label = "unlabeled"
                    # Check if the patch overlaps with any positive (tumor) region in the generated binary mask
                    if mask:
                        mask_patch = mask.crop((x, y, x + patch_size, y + patch_size))
                        if np.any(np.array(mask_patch) > 0):
                            label = "tumor"
                        else:
                            label = "normal"
                    else:
                        label = "normal"

                    patch_array = np.array(region)
                    if np.mean(patch_array) > 240:  # too white (empty tissue)
                        continue

                    patch_name = f"{prefix}_x{x}_y{y}_{label}.png"
                    patch_path = os.path.join(patch_save_dir, patch_name)
                    if not os.path.exists(patch_path):
                        region.save(patch_path)
                    patch_count += 1

            print(
                f"{bcolors.INFO}[INFO]{bcolors.ENDC} Patch extraction complete for {file} at level {level}. Total patches: {patch_count}"
            )


def count_number_tumor_patches(level=3):
    """
    Count the number of tumor and normal patches in the patch directory for a given level across all slides.

    Parameters:
    - level (int): WSI level for patch extraction.
    """
    patch_dir = os.path.join(
        os.getcwd(), "data", "camelyon16", "patches", f"level_{level}"
    )
    if not os.path.exists(patch_dir):
        print(
            f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Patch directory '{patch_dir}' does not exist. Please run patch extraction first."
        )
        return

    total_tumor = 0
    total_normal = 0
    slides_with_no_tumor = []
    slides_with_tumor_in_normal = []

    empty_folders = [
        d
        for d in os.listdir(patch_dir)
        if os.path.isdir(os.path.join(patch_dir, d))
        and not os.listdir(os.path.join(patch_dir, d))
    ]
    # Print empty folders
    if empty_folders:
        print(
            f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} The following folders are empty: {', '.join(empty_folders)}"
        )
    print(f"Number of empty folders: {len(empty_folders)}")
    for slide_name in os.listdir(patch_dir):
        slide_path = os.path.join(patch_dir, slide_name)
        if os.path.isdir(slide_path):
            num_tumor = sum(
                1 for f in os.listdir(slide_path) if f.endswith("_tumor.png")
            )
            num_normal = sum(
                1 for f in os.listdir(slide_path) if f.endswith("_normal.png")
            )
            total_tumor += num_tumor
            total_normal += num_normal
            if num_tumor == 0:
                slides_with_no_tumor.append(slide_name)
            # Warn if a normal slide contains tumor patches
            if slide_name.startswith("normal_") and num_tumor > 0:
                slides_with_tumor_in_normal.append(slide_name)

    print(
        f"{bcolors.INFO}[INFO]{bcolors.ENDC} Total tumor patches at level {level}: {total_tumor}"
    )
    print(
        f"{bcolors.INFO}[INFO]{bcolors.ENDC} Total non-tumor patches at level {level}: {total_normal}"
    )
    print(
        f"{bcolors.INFO}[INFO]{bcolors.ENDC} Total slides with no tumor patches at level {level}: {len(slides_with_no_tumor)}"
    )
    print(
        f"{bcolors.INFO}[INFO]{bcolors.ENDC} Slides with no tumor patches: {', '.join(slides_with_no_tumor)}"
        if slides_with_no_tumor
        else f"{bcolors.INFO}All slides have tumor patches.{bcolors.ENDC}"
    )
    if slides_with_tumor_in_normal:
        print(
            f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} The following normal slides contain tumor patches: {', '.join(slides_with_tumor_in_normal)}"
        )


def extract_features(
    level=3,
    model_type="resnet18",
    simclr_trained_model=False,
):
    """
    Extract features from patches using a trained ResNet model and save them grouped by WSI for MIL.

    Parameters:
    - level (int): WSI level for patch extraction.
    - model_name (str): Name of the trained ResNet model to use for feature extraction.
    - simclr_trained_model (bool): Whether the model was trained with SimCLR.
    """
    model_path = os.path.join(os.getcwd(), "src", "models", model_type)
    if not os.path.exists(model_path):
        print(
            f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Model file '{model_path}' not found."
        )
        return

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    sets = ["train", "test", "val"]
    for split in sets:
        patch_dir = os.path.join(
            os.getcwd(), "data", "camelyon16", "patches", f"level_{level}", split
        )
        if not os.path.exists(patch_dir) or not os.listdir(patch_dir):
            print(
                f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Patch directory {patch_dir} is missing or empty. Skipping {split}."
            )
            continue

        print(
            f"{bcolors.INFO}[INFO]{bcolors.ENDC} Extracting features for {split} set..."
        )

        dataset = PatchDataset(patch_dir, transform=transform)
        loader = torch.utils.data.DataLoader(  # get_dataloaders is opnly for training data augmentation. in feature extraction we don't need it
            dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8
        )

        try:
            model = ResNetFeatureExtractor(
                trained_classifier_weights_path=model_path,
                simclr_trained=simclr_trained_model,
            ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        except NameError:
            print(
                f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} ResNetFeatureExtractor not defined."
            )
            return

        model.eval()

        wsi_features_dict = defaultdict(lambda: {"features": [], "patch_labels": []})
        wsi_overall_labels = {}

        with torch.no_grad():
            for batch_idx, (imgs, lbls, img_paths) in enumerate(
                tqdm(loader, desc=f"Extracting Features - {split}")
            ):
                feats = (
                    model(imgs.cuda() if torch.cuda.is_available() else imgs)
                    .cpu()
                    .numpy()
                )
                for i in range(imgs.size(0)):
                    patch_path = img_paths[i]
                    patch_label = lbls[i].item()
                    rel_path = os.path.relpath(patch_path, patch_dir)
                    wsi_name = rel_path.split(os.sep)[0]
                    wsi_features_dict[wsi_name]["features"].append(feats[i])
                    wsi_features_dict[wsi_name]["patch_labels"].append(patch_label)
                    if patch_label == 1:
                        wsi_overall_labels[wsi_name] = 1
                    elif wsi_name not in wsi_overall_labels:
                        wsi_overall_labels[wsi_name] = 0

        if not wsi_features_dict:
            print(
                f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} No features extracted for {split}."
            )
            continue

        # Save features to correct path
        features_save_dir = os.path.join(
            os.getcwd(),
            "data",
            "camelyon16",
            "features",
            f"level_{level}",
            model_type,
            split,
        )
        os.makedirs(features_save_dir, exist_ok=True)

        wsi_paths_list = []
        for wsi_name, data in wsi_features_dict.items():
            wsi_feature_array = np.array(data["features"])
            wsi_patch_labels_array = np.array(data["patch_labels"])
            wsi_overall_label = wsi_overall_labels[wsi_name]

            wsi_feature_path = os.path.join(
                features_save_dir, f"{wsi_name}_features.npy"
            )
            wsi_label_path = os.path.join(features_save_dir, f"{wsi_name}_label.npy")
            wsi_patch_labels_path = os.path.join(
                features_save_dir, f"{wsi_name}_patch_labels.npy"
            )

            if not (
                os.path.exists(wsi_feature_path)
                and os.path.exists(wsi_label_path)
                and os.path.exists(wsi_patch_labels_path)
            ):
                np.save(wsi_feature_path, wsi_feature_array)
                np.save(wsi_label_path, np.array(wsi_overall_label))
                np.save(wsi_patch_labels_path, wsi_patch_labels_array)
            else:
                print(
                    f"{bcolors.INFO}[INFO]{bcolors.ENDC} Feature files for {wsi_name} already exist, skipping."
                )

            wsi_paths_list.append(wsi_feature_path)

        feature_list_path = os.path.join(
            features_save_dir, f"wsi_feature_paths_{split}.txt"
        )
        with open(feature_list_path, "w") as f:
            for p in wsi_paths_list:
                f.write(f"{p}\n")

        print(
            f"{bcolors.INFO}[INFO]{bcolors.ENDC} Features for {split} saved to {features_save_dir}."
        )


def prepare_data():
    """
    Prepare data by extracting training and testing masks from zip files.
    """
    print(f"{bcolors.HEADER}{bcolors.BOLD}[HEADER]{bcolors.ENDC} Preparing data...")

    # Extract training masks
    train_zip = os.path.join(
        os.getcwd(), "data", "camelyon16", "train", "mask", "lesion_annotations.zip"
    )
    train_extract_to = os.path.join(
        os.getcwd(), "data", "camelyon16", "train", "mask", "annotations"
    )
    if not os.path.exists(train_zip):
        print(
            f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Training masks zip file not found. Please download the dataset first."
        )
    else:
        extract_zip(train_zip, train_extract_to)

    # Extract validation masks
    val_zip = os.path.join(
        os.getcwd(), "data", "camelyon16", "val", "mask", "lesion_annotations.zip"
    )
    test_extract_to = os.path.join(
        os.getcwd(), "data", "camelyon16", "val", "mask", "annotations"
    )
    if not os.path.exists(val_zip):
        print(
            f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Validation masks zip file not found. Please download the dataset first."
        )
    else:
        extract_zip(val_zip, test_extract_to)

    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Data preparation completed.")


def images_downloaded():
    """
    Check if training images have been downloaded.

    Returns:
    - bool: True if images are present, False otherwise.
    """
    img_dir = os.path.join(os.getcwd(), "data", "camelyon16", "train", "img")
    return (
        os.path.exists(img_dir)
        and len([f for f in os.listdir(img_dir) if f.endswith(".tif")]) > 0
    )


def patches_extracted(patch_level):
    """
    Check if patches have been extracted at the specified level.

    Parameters:
    - patch_level (int): WSI level for patch extraction.

    Returns:
    - bool: True if patches exist, False otherwise.
    """
    patch_dir = os.path.join(
        os.getcwd(), "data", "camelyon16", "patches", f"level_{patch_level}"
    )
    return os.path.exists(patch_dir) and any(os.listdir(patch_dir))


def features_extracted(patch_level):
    """
    Check if features have been extracted for the specified patch level.

    Parameters:
    - patch_level (int): WSI level for patch extraction.

    Returns:
    - bool: True if feature files exist, False otherwise.
    """
    return os.path.exists(f"patch_features_{patch_level}.npy") and os.path.exists(
        f"patch_labels_{patch_level}.npy"
    )


def create_validation_set():
    """
    Create a validation set by moving 20% of training slides and their annotations to a validation directory.
    """
    train_dir = os.path.join(os.getcwd(), "data", "camelyon16", "train", "img")
    if not os.path.exists(train_dir):
        print(
            f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Training directory '{train_dir}' does not exist. Please download the dataset first."
        )
        return

    val_dir = os.path.join(os.getcwd(), "data", "camelyon16", "val", "img")
    os.makedirs(val_dir, exist_ok=True)

    slides = [f for f in os.listdir(train_dir) if f.endswith(".tif")]
    if not slides:
        print(
            f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} No slides found in training directory '{train_dir}'. Please check if the dataset is downloaded."
        )
        return
    _, val_slides = train_test_split(slides, test_size=0.2, random_state=42)
    # Move files
    for slide in val_slides:
        src_path = os.path.join(train_dir, slide)
        dst_path = os.path.join(val_dir, slide)
        if os.path.exists(dst_path):
            print(
                f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Slide {slide} already exists in validation directory. Skipping."
            )
            continue
        os.rename(src_path, dst_path)
        print(
            f"{bcolors.INFO}[INFO]{bcolors.ENDC} Moved {slide} to validation directory."
        )

    annot_train_dir = os.path.join(
        os.getcwd(), "data", "camelyon16", "train", "mask", "annotations"
    )
    annot_val_dir = os.path.join(
        os.getcwd(), "data", "camelyon16", "val", "mask", "annotations"
    )
    os.makedirs(annot_val_dir, exist_ok=True)

    # Move annotations
    for slide in val_slides:
        wsi_name = os.path.splitext(slide)[0]
        xml_filename = f"{wsi_name}.xml"
        src_xml = os.path.join(annot_train_dir, xml_filename)
        dst_xml = os.path.join(annot_val_dir, xml_filename)
        if os.path.exists(src_xml):
            shutil.copy2(src_xml, dst_xml)
            print(
                f"{bcolors.INFO}[INFO]{bcolors.ENDC} Copied annotation {xml_filename} to validation annotations."
            )
        else:
            print(
                f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Annotation {xml_filename} not found in training annotations."
            )


def main():
    parser = argparse.ArgumentParser(description="Camelyon Dataset Processing")
    parser.add_argument(
        "--download", action="store_true", help="Download CAMELYON16 dataset"
    )
    parser.add_argument(
        "--remote", action="store_true", help="Execute on remote server"
    )
    parser.add_argument("-p", "--patch", action="store_true", help="Extract patches")
    parser.add_argument(
        "--patch_level",
        type=str,
        default="3",
        help="WSI level for patch extraction (0, 1, 2, 3, or 'all' for all levels)",
    )
    parser.add_argument(
        "--test_patch",
        type=str,
        default="val",
    )
    parser.add_argument("-prep", "--prepare", action="store_true", help="Prepare data")
    parser.add_argument(
        "-val", "--validation", action="store_true", help="Create validation set"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate Resnet model (sanity check for extracted patch features)",
    )
    parser.add_argument(
        "-train", "--train", action="store_true", help="Train Resnet model"
    )
    parser.add_argument(
        "-eval", "--evaluate", action="store_true", help="Evaluate Resnet model"
    )
    parser.add_argument(
        "--extract_features", action="store_true", help="Extract features from patches"
    )
    parser.add_argument(
        "--run_evaluation",
        action="store_true",
        help="Run CAMELYON16 evaluation script.",
    )
    parser.add_argument(
        "--balance_dataset",
        action="store_true",
        help="Balance dataset by downloading all tumor images and extracting patches from them.",
    )
    parser.add_argument(
        "--count_tumor_patches",
        action="store_true",
        help="Count number of tumor patches at a given level.",
    )
    parser.add_argument(
        "--slide",
        type=str,
        default=None,
        help="Extract patches from a single slide directory (e.g. tumor_109) at a given level",
    )
    parser.add_argument(
        "--move_files",
        action="store_true",
        help="Move patches to a new directory structure based on slide names",
    )
    parser.add_argument(
        "--train_strategy",
        action="store_true",
        help="Train ResNet classifier with a specific strategy",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="self_supervised",
        choices=["balanced", "weighted_loss", "self_supervised", "baseline"],
        help="Training strategy for ResNet classifier",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="resnet18",
        help="Name of the ResNet model to use for feature extraction",
    )

    parser.add_argument(
        "--train_mil",
        action="store_true",
        help="Train MIL classifier with specified pooling method",
    )
    parser.add_argument(
        "--test_mil",
        action="store_true",
        help="Test MIL classifier on test dataset",
    )
    # Check for unknown arguments
    known_args = {action.dest for action in parser._actions}
    input_args = {
        arg.lstrip("-").replace("-", "_") for arg in sys.argv[1:] if arg.startswith("-")
    }
    unknown_args = input_args - known_args
    if unknown_args:
        print(
            f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Unknown command line arguments: {', '.join(unknown_args)}"
        )
        sys.exit(1)

    args = parser.parse_args()

    if args.download:
        download_dataset(args.remote)

    # Extract patches
    if args.patch:
        if not images_downloaded():
            print(
                f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Images must be downloaded before extracting patches."
            )
            return
        if args.patch_level == "all":
            for lvl in [0, 1, 2, 3]:
                extract_patches(level=lvl)
        else:
            extract_patches(level=int(args.patch_level))

    # Extract features
    if args.extract_features:
        # Check for patches at the requested level
        patch_levels = (
            [0, 1, 2, 3] if args.patch_level == "all" else [int(args.patch_level)]
        )
        for lvl in patch_levels:
            if not patches_extracted(lvl):
                print(
                    f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Patches must be extracted at level {lvl} before extracting features."
                )
                return
        extract_features(
            level=int(args.patch_level) if args.patch_level != "all" else 3,
            model_type=args.model_type,
        )  # default to level 3 if all

    # Train model
    if args.train:
        if not images_downloaded():
            print(
                f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Images must be downloaded before training."
            )
            return
        if not patches_extracted(patch_level=args.patch_level):
            print(
                f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Patches must be extracted before training."
            )
            return
        train_resnet_classifier(args.patch_level)

    if args.train_strategy:
        if not images_downloaded():
            print(
                f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Images must be downloaded before training."
            )
            return
        if not patches_extracted(patch_level=args.patch_level):
            print(
                f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Patches must be extracted before training."
            )
            return
        train_resnet_classifier(level=int(args.patch_level), strategy=args.strategy)

    if args.train_mil:
        if not features_extracted(patch_level=args.patch_level):
            print(
                f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Features must be extracted before training MIL classifier."
            )
            return
        train_mil_classifier(
            feature_level=int(args.patch_level),
            pooling="attention",
            model_type=args.model_type,
        )

    # Test MIL classifier
    if args.test_mil:
        if not features_extracted(patch_level=args.patch_level):
            print(
                f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Features must be extracted before testing MIL classifier."
            )
            return
        test_mil_classifier(
            feature_level=int(args.patch_level),
            pooling="attention",
            model_type=args.model_type,
        )

    if args.prepare:
        prepare_data()
    if args.balance_dataset:
        download_all_tumor_extract_patches()
    if args.count_tumor_patches:
        count_number_tumor_patches(level=3)

    if args.run_evaluation:
        """
        Calculate False Positives (FPs), True Positives (TPs), and generates a Free-Response Receiver Operating Characteristic (FROC) curve.
        """
        print(
            f"{bcolors.INFO}[INFO]{bcolors.ENDC} Running CAMELYON16 evaluation script."
        )
        mask_folder_for_eval = os.path.join(
            os.getcwd(), "data", "camelyon16", "val", "mask"
        )
        results_folder_for_eval = os.path.join(
            os.getcwd(), "models", "first_model", "model_predictions_csv"
        )

        if not os.path.exists(mask_folder_for_eval):
            print(
                f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Evaluation mask folder '{mask_folder_for_eval}' not found. Please generate TIFF masks from XML annotations first."
            )
        elif not os.path.exists(results_folder_for_eval):
            print(
                f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Model results folder '{results_folder_for_eval}' not found. Please run your detection model first."
            )
        else:
            result_file_list = [
                each
                for each in os.listdir(results_folder_for_eval)
                if each.endswith(".csv")
            ]

            EVALUATION_MASK_LEVEL = 5
            L0_RESOLUTION = 0.243

            FROC_data = np.empty((4, len(result_file_list)), dtype=object)
            FP_summary = np.empty((2, len(result_file_list)), dtype=object)
            detection_summary = np.empty((2, len(result_file_list)), dtype=object)

            caseNum = 0
            for case in result_file_list:
                print(f"Evaluating Performance on image: {case[0:-4]}")
                sys.stdout.flush()
                csvDIR = os.path.join(results_folder_for_eval, case)
                Probs, Xcorr, Ycorr = readCSVContent(
                    csvDIR
                )  # is this function is updated for Python 3?

                is_tumor = case[0:5].lower() == "tumor"  # Use .lower() for robustness
                if is_tumor:
                    maskDIR = (
                        os.path.join(mask_folder_for_eval, case[0:-4]) + "_Mask.tif"
                    )
                    if not os.path.exists(maskDIR):
                        print(
                            f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Mask TIFF '{maskDIR}' not found for tumor case. Skipping."
                        )
                        continue  # Skip to next case if mask is missing
                    evaluation_mask = computeEvaluationMask(
                        maskDIR, L0_RESOLUTION, EVALUATION_MASK_LEVEL
                    )  # Python 3?
                    ITC_labels = computeITCList(
                        evaluation_mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL
                    )
                else:
                    evaluation_mask = 0  # Or a blank mask for consistency
                    ITC_labels = []

                FROC_data[0][caseNum] = case
                FP_summary[0][caseNum] = case
                detection_summary[0][caseNum] = case

                # Update compute_FP_TP_Probs for Python 3 division (//)
                (
                    FROC_data[1][caseNum],
                    FROC_data[2][caseNum],
                    FROC_data[3][caseNum],
                    detection_summary[1][caseNum],
                    FP_summary[1][caseNum],
                ) = compute_FP_TP_Probs(
                    Ycorr,
                    Xcorr,
                    Probs,
                    is_tumor,
                    evaluation_mask,
                    ITC_labels,
                    EVALUATION_MASK_LEVEL,
                )
                caseNum += 1

            # Compute FROC curve
            if caseNum > 0:  # Only compute if there were cases processed
                total_FPs, total_sensitivity = computeFROC(
                    FROC_data
                )  # Update for Python 3

                # plot FROC curve
                plotFROC(total_FPs, total_sensitivity)  # Update for Python 3
            else:
                print(
                    f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} No cases processed for FROC evaluation."
                )


if __name__ == "__main__":
    main()
