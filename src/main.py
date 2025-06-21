import os
import sys
import argparse
import requests
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from torchvision import transforms
from models.resnet import ResNet18FeatureExtractor
""""
os.add_dll_directory(
    r"C:\Program Files\OpenSlide\openslide-bin-4.0.0.8-windows-x64\bin"
)
"""
import openslide
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.resnet import ResNet18Classifier
from datasets.patch_dataset import PatchDataset
from utils.structure import group_patches_by_slide
from torch.optim import Adam
import numpy as np
import shutil
import torch.nn as nn

# TODO: add dll directory for OpenSlide avoiding giving specific path

import zipfile
from PIL import Image

# Base URL for the CAMELYON16 dataset
BASE_URL = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100439/"

# Patches size
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
    "test_masks": ["CAMELYON16/testing/lesion_annotations.zip"],
}


def download_file(url, destination_path):
    """
    Downloads a file from a URL to a destination path with a progress bar.
    """
    try:
        print(f"[INFO] Downloading: {url}")
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
        print(f"[INFO] Successfully downloaded {os.path.basename(destination_path)}.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        return False


def download_files(base_dir, file_groups, remote=False):
    """
    Download and manage the required dataset files (with strict size limits).
    """
    limits = {"train_normal": 35, "train_tumor": 35, "test_images": 10}

    for group_name, files in file_groups.items():
        group_dir = os.path.join(base_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)

        # Categorize files
        categorized_files = {"train_normal": [], "train_tumor": [], "test_images": []}
        for f in files:
            fname = os.path.basename(f)
            if fname.startswith("normal"):
                categorized_files["train_normal"].append(f)
            elif fname.startswith("tumor"):
                categorized_files["train_tumor"].append(f)
            elif fname.startswith("test"):
                categorized_files["test_images"].append(f)

        # Process each category
        for category, file_list in categorized_files.items():
            limit = limits[category]
            file_list = file_list[:limit]  # restrict to desired number

            existing = [
                f
                for f in os.listdir(group_dir)
                if f.endswith(".tif") and category.split("_")[1] in f
            ]
            for f in existing:
                try:
                    num = int(f.split("_")[1].split(".")[0])
                    if num > limit:
                        print(f"[INFO] Deleting excess file: {f}")
                        os.remove(os.path.join(group_dir, f))
                except Exception as e:
                    print(f"[WARNING] Failed to parse file number for {f}: {e}")

            # Only download missing files
            if not remote:
                file_list = file_list[:1]  # For local testing, only download one file
            for file_path in file_list:
                file_name = os.path.basename(file_path)
                destination_path = os.path.join(group_dir, file_name)
                if os.path.exists(destination_path):
                    continue
                url = BASE_URL + file_path
                download_file(url, destination_path)


def download_dataset(base_dir="./data", remote=False):
    """
    Downloads the CAMELYON16 dataset, including training, testing, and mask files.

    Parameters:
    - base_dir: str, directory to save the downloaded files.
    - remote: bool, if True, download all files; if False, download only one file for testing.
    """
    camelyon_dir = os.path.join(base_dir, "camelyon16")
    os.makedirs(camelyon_dir, exist_ok=True)

    # Define file groups for training, testing, and masks
    file_groups = {
        "train/img": CAMELYON16_FILES["train_normal"]
        + CAMELYON16_FILES["train_tumor"],  # Both normal and tumor images go to img
        "test/img": CAMELYON16_FILES["test_images"],
        "masks": CAMELYON16_FILES["train_masks"] + CAMELYON16_FILES["test_masks"],
    }

    download_files(camelyon_dir, file_groups, remote)


def extract_zip(zip_path, extract_to):
    """
    Extract masks to annotations.
    Parameters:
    - zip_path: str, path to the zip file to extract.
    - extract_to: str, directory to extract the contents to.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    else:
        print(f"[INFO] Directory {extract_to} already exists. Skipping extraction.")
        return
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[INFO] Extracted {zip_path} to {extract_to}")


def parse_xml_mask(xml_path, level_dims, downsample):
    """Convert XML annotation to binary mask."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    mask = Image.new("L", level_dims, 0)  # 'L' mode for 8-bit pixels, black/white
    draw = ImageDraw.Draw(mask)

    for annotation in root.findall(".//Annotation"):
        for region in annotation.findall("Region"):
            vertices = region.find("Vertices")
            coords = [
                (
                    int(float(vertex.get("X")) / downsample),
                    int(float(vertex.get("Y")) / downsample),
                )
                for vertex in vertices.findall("Vertex")
            ]
            draw.polygon(coords, outline=1, fill=1)

    return mask


def train_resnet_classifier():
    print("[INFO] Training ResNet18 classifier on extracted patches...")
    patch_dir = "./data/camelyon16/patches/level_4"

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = PatchDataset(patch_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Classifier().to(device)

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct = 0, 0
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

        acc = correct / len(dataset)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}"
        )

    torch.save(model.state_dict(), "resnet18_patch_classifier.pth")
    print("[INFO] ResNet18 classifier training complete and saved.")


def extract_patches(base_dir="data", patch_size=224, level=3, stride=None, pad=True):
    """
    Extract patches from WSIs at a specified level, apply mask overlays, and save tumor vs normal labels.
    Only extracts patches if they have not already been extracted for a given image.
    Parameters:
    - base_dir: str, base directory for the dataset.
    - patch_size: int, size of the patches to extract.
    - level: int, level of the WSI to extract patches from.
    - stride: int, stride for patch extraction.
    - pad: bool, if True, pad the image to cover all regions.
    """
    print(f"[INFO] Extracting patches at level {level}...")
    stride = stride or patch_size

    # Set patch size according to level
    patch_sizes = {0: 1792, 1: 896, 2: 448, 3: 224}
    patch_size = patch_sizes.get(level, 224)

    wsi_dir = os.path.join(os.getcwd(), base_dir, "camelyon16", "train", "img")
    annot_dir = os.path.join(
        os.getcwd(), base_dir, "camelyon16", "masks", "annotations"
    )
    level_dir = os.path.join(
        os.getcwd(), base_dir, "camelyon16", "patches", f"level_{level}"
    )
    os.makedirs(level_dir, exist_ok=True)

    for file in os.listdir(wsi_dir):
        if not file.endswith(".tif"):
            continue

        prefix = file.replace(".tif", "")
        # Check if patches for this image already exist
        already_extracted = False
        for label in ["normal", "tumor"]:
            patch_save_dir = os.path.join(level_dir, label)
            if os.path.exists(patch_save_dir):
                existing_patches = [
                    f for f in os.listdir(patch_save_dir) if f.startswith(prefix)
                ]
                if len(existing_patches) > 0:
                    already_extracted = True
                    break
        if already_extracted:
            print(f"[INFO] Patches for {file} already extracted, skipping.")
            continue

        wsi_path = os.path.join(wsi_dir, file)
        xml_name = file.replace(".tif", ".xml")
        xml_path = os.path.join(annot_dir, xml_name)
        print(f"[DEBUG] Processing file: {wsi_path} with XML: {xml_path}")
        try:
            slide = openslide.OpenSlide(wsi_path)
        except Exception as e:
            print(f"[ERROR] Could not open {wsi_path}: {e}")
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
                mask = parse_xml_mask(xml_path, (width, height), downsample)
                if pad and (pad_w > 0 or pad_h > 0):
                    mask = ImageOps.expand(mask, (0, 0, pad_w, pad_h), fill=0)
            except Exception as e:
                print(f"[WARNING] Failed to parse XML for {file}: {e}")
        else:
            print(f"[INFO] No annotation found for {file}, treating as normal.")

        print(f"[INFO] Processing {file} at level {level} (size: {width}x{height}, padded: {padded_width}x{padded_height})")

        patch_count = 0
        for x in range(0, padded_width, stride):
            for y in range(0, padded_height, stride):
                # If patch goes beyond original image, pad with white
                region = slide.read_region(
                    (int(x * downsample), int(y * downsample)),
                    level,
                    (patch_size, patch_size),
                ).convert("RGB")
                if x + patch_size > width or y + patch_size > height:
                    region = Image.new("RGB", (patch_size, patch_size), (255, 255, 255)).copy()
                    region_part = slide.read_region(
                        (int(x * downsample), int(y * downsample)),
                        level,
                        (
                            min(patch_size, width - x),
                            min(patch_size, height - y),
                        ),
                    ).convert("RGB")
                    region.paste(region_part, (0, 0))

                label = "normal"
                if mask:
                    mask_patch = mask.crop((x, y, x + patch_size, y + patch_size))
                    if np.any(np.array(mask_patch) > 0):
                        label = "tumor"

                patch_array = np.array(region)
                if np.mean(patch_array) > 240:  # too white (empty tissue)
                    continue

                patch_save_dir = os.path.join(level_dir, label)
                os.makedirs(patch_save_dir, exist_ok=True)
                patch_name = f"{prefix}_x{x}_y{y}.png"
                region.save(os.path.join(patch_save_dir, patch_name))
                patch_count += 1
                if patch_count % 100 == 0:
                    print(f"Extracted patches {patch_count} for {file}")

        print(
            f"[INFO] Patch extraction complete for {file} at level {level}. Total patches: {patch_count}"
        )


def extract_features(level=4):
    """
    Extract features from the patches using a ResNet18 model.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    patch_dir = os.path.join(
        os.getcwd(), "data", "camelyon16", "patches", f"level_{level}"
    )
    dataset = PatchDataset(patch_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    print(
        f"[INFO] Extracting features from patches at level {level} with patch directory: {patch_dir}, which exists: {os.path.exists(patch_dir)}"
    )
    print(
        "[INFo] WSI images:",
        os.listdir(patch_dir) if os.path.exists(patch_dir) else "Not found",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18FeatureExtractor().to(device)
    model.eval()

    features = []
    labels = []
    paths = []

    with torch.no_grad():
        for imgs, lbls, img_paths in loader:
            print("Batch size:", imgs.shape)
            feats = model(imgs.to(device))
            features.append(feats.cpu())
            labels.extend(lbls)
            paths.extend(img_paths)
    if not features:
        print(
            "[ERROR] No features were extracted. Check your patch directory and dataset."
        )
        return
    features = torch.cat(features, dim=0)  # (num_patches, 512)

    np.save("patch_features.npy", features.numpy())
    np.save("patch_labels.npy", np.array(labels))
    with open("patch_paths.txt", "w") as f:
        for p in paths:
            f.write(f"{p}\n")


def create_validation_set(base_dir="./data"):
    """
    Create validation set: exactly 10 files (5 normal + 5 tumor).
    """
    src_dir = os.path.join(base_dir, "camelyon16/train/img")
    dst_dir = os.path.join(base_dir, "camelyon16/val/img")
    os.makedirs(dst_dir, exist_ok=True)

    normal_files = sorted([f for f in os.listdir(src_dir) if f.startswith("normal")])[
        :5
    ]
    tumor_files = sorted([f for f in os.listdir(src_dir) if f.startswith("tumor")])[:5]

    for f in normal_files + tumor_files:
        shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))
    print(
        f"[INFO] Validation set created with {len(normal_files)} normal and {len(tumor_files)} tumor files."
    )


def check_structure():
    """
    Check if the directory structure is correct.
    """
    expected_structure = [
        "data/camelyon16/train/img",
        "data/camelyon16/val/img",
        "data/camelyon16/test/img",
        "data/camelyon16/masks/annotations",
        "data/camelyon16/patches/level_3/normal_002",
    ]

    for path in expected_structure:
        if not os.path.exists(path):
            print(f"[ERROR] Missing expected directory: {path}")
            if path.endswith("normal_002"):
                group_patches_by_slide(
                    patch_root=os.path.join(
                        os.getcwd(), "data", "camelyon16", "patches", "level_3"
                    )
                )
            return False
    print("[INFO] Directory structure is correct.")
    return True


def prepare_data():  # TODO: WIP
    """
    Prepare data for training (e.g., preprocessing or augmentation).
    """
    print("[INFO] Preparing data...")

    # Create a validation set from the training data
    create_validation_set(base_dir="./data")

    # Extract masks
    if not os.path.exists(
        os.path.join(
            os.getcwd(), "..", "data", "camelyon16", "masks", "lesion_annotations.zip"
        )
    ):
        print("[ERROR] Masks zip file not found. Please download the dataset first.")
        return
    zip_path = os.path.join(
        os.getcwd(), "..", "data", "camelyon16", "masks", "lesion_annotations.zip"
    )
    extract_to = os.path.join(
        os.getcwd(), "..", "data", "camelyon16", "masks", "annotations"
    )
    extract_zip(zip_path, extract_to)

    print("[INFO] Data preparation completed.")


def main():
    parser = argparse.ArgumentParser(description="Camelyon Dataset Processing")
    parser.add_argument(
        "--download", action="store_true", help="Download CAMELYON16 dataset"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./data",
        help="Base directory for downloaded files",
    )
    parser.add_argument(
        "--remote", action="store_true", help="Execute on remote server"
    )
    parser.add_argument(
        "-p", "--patch", action="store_true", help="Extract patches"
    )
    parser.add_argument(
        "--patch_level",
        type=str,
        default="3",
        help="WSI level for patch extraction (0, 1, 2, 3, or 'all' for all levels)",
    )
    parser.add_argument("-prep", "--prepare", action="store_true", help="Prepare data")
    parser.add_argument(
        "-val", "--validation", action="store_true", help="Create validation set"
    )
    parser.add_argument(
        "-train", "--train", action="store_true", help="Train U-Net model"
    )
    parser.add_argument("-test", "--test", action="store_true", help="Test U-Net model")
    parser.add_argument(
        "--extract_features", action="store_true", help="Extract features from patches"
    )
    parser.add_argument(
        "--check_structure", action="store_true", help="Check directory structure"
    )

    args = parser.parse_args()

    if args.download:
        download_dataset(args.base_dir, args.remote)
    if args.patch:
        if args.patch_level == "all":
            for lvl in [0, 1, 2, 3]:
                extract_patches(level=lvl)
        else:
            extract_patches(level=int(args.patch_level))
    if args.prepare:
        prepare_data()
    if args.validation:
        create_validation_set(args.base_dir)
    if args.train:
        train_resnet_classifier()
    if args.extract_features:
        extract_features()
    if args.check_structure:
        check_structure()


if __name__ == "__main__":
    main()
