import os
import sys
import argparse
import requests
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import shutil
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageOps
from lxml import etree
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

""""
os.add_dll_directory(
    r"C:\Program Files\OpenSlide\openslide-bin-4.0.0.8-windows-x64\bin"
)
"""
import openslide
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.resnet import ResNet18Classifier, ResNet18FeatureExtractor, UnifiedResNet
from datasets.patch_dataset import PatchDataset
from utils.evaluation_FROC import computeEvaluationMask, computeITCList, readCSVContent, compute_FP_TP_Probs, computeFROC, plotFROC
from models.simclr import pretrain_simclr, get_simclr_transform
import zipfile

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    DEBUG = '\033[96m'
    INFO = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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
    "test_masks": ["CAMELYON16/testing/lesion_annotations.zip", "CAMELYON16/testing/evaluation/evaluation_python.zip"],
}

DOWNLOADED_FILES = {
    "train_normal": 
        [f"normal_{i:03d}" for i in range(1, 112)],
    "train_tumor":
        [f"tumor_{i:03d}" for i in range(1, 112)],
    "test_images":
        [f"test_{i:03d}" for i in range(1, 51)],
}


def download_file(url, destination_path):
    """
    Downloads a file from a URL to a destination path with a progress bar.
    """
    try:
        print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Downloading: {url} into {destination_path}")
        os.makedirs(os.path.dirname(destination_path), exist_ok=True) # Ensure destination dir exists
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
        print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Successfully downloaded {os.path.basename(destination_path)}.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Failed to download {url}: {e}")
        return False
    except Exception as e:
        print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} An unexpected error occurred: {e}")
        return False

def download_dataset(remote=False):
    """
    Downloads the CAMELYON16 dataset, including training, testing, and mask files.

    Parameters:
    - remote: bool, if True, download all files; if False, download only one file for testing.
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
    limits = {"train_normal": 35, "train_tumor": 35, "test_images": 10}

    for file_type, target_dir in download_map.items():
        files_to_download = CAMELYON16_FILES[file_type]

        # Apply limits based on file type
        if file_type in limits:
            files_to_download = files_to_download[:limits[file_type]]
        
        # In non-remote mode, only download one image file per category
        if not remote and file_type in ["train_normal", "train_tumor", "test_images"]:
            files_to_download = files_to_download[:1]
        
        for remote_file_path in files_to_download:
            file_name = os.path.basename(remote_file_path)
            
            # Skip evaluation_python.zip if remote=True, as per original logic's intent (though it was inverted)
            if "evaluation_python" in file_name and remote:
                print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Skipping download of {file_name} in remote mode.")
                continue

            # Check if the file exists in train/img, val/img, or test/img
            train_img_path = os.path.join(train_img_dir, file_name)
            val_img_path = os.path.join(val_img_dir, file_name)
            test_img_path = os.path.join(test_img_dir, file_name)
            destination_path = os.path.join(target_dir, file_name)

            if any(os.path.exists(p) for p in [train_img_path, val_img_path, test_img_path]):
                print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Skipping: {file_name} already exists in train/img, val/img, or test/img.")
                continue
            # check if lesion_annotations.zip already exist in the mask directories train_mask_dir or test_mask_dir
            if file_type in ["train_masks", "test_masks"] and os.path.exists(destination_path):
                print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Skipping: {file_name} already exists in {target_dir}.")
                continue

            url = BASE_URL + remote_file_path
            download_file(url, destination_path)

def move_files():
    base_dir = os.path.join(os.getcwd(), "data", "camelyon16", "patches", "level_3")

    # Iterate through each subfolder in level_3
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        # Only process directories
        if not os.path.isdir(folder_path):
            continue

        tumor_subdir = os.path.join(folder_path, "tumor")
        
        if os.path.isdir(tumor_subdir):
            print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Processing {tumor_subdir}...")
            # Move all .png files to the parent directory
            for file_name in os.listdir(tumor_subdir):
                if file_name.endswith(".png"):
                    src_file = os.path.join(tumor_subdir, file_name)
                    dst_file = os.path.join(folder_path, file_name)
                    shutil.move(src_file, dst_file)
            
            # Remove the now-empty 'tumor' subfolder
            try:
                os.rmdir(tumor_subdir)
                print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Deleted empty directory: {tumor_subdir}")
            except OSError as e:
                print(f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Could not delete {tumor_subdir}: {e}")
        else:
            print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} No 'tumor' subdirectory in {folder_path}, skipping.")

def extract_zip(zip_path, extract_to):
    """
    Extract masks to annotations.
    Parameters:
    - zip_path: str, path to the zip file to extract.
    - extract_to: str, directory to extract the contents to.
    """
    # Check if the path extract_to exists. If yes, check contains all elements from tumor_001.xml to tumor_050.xml.
    # If it does not contain them, delete the directory and extract again. If exists and contains all elements, skip extraction.
    expected_xmls = [f"tumor_{i:03d}.xml" for i in range(1, 51)]

    if os.path.exists(extract_to):
        existing_xmls = set(os.listdir(extract_to))
        if all(xml in existing_xmls for xml in expected_xmls):
            print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Directory {extract_to} already exists and contains all expected XMLs. Skipping extraction.")
            return
        else:
            print(f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Directory {extract_to} exists but is missing some XMLs. Re-extracting...")
            shutil.rmtree(extract_to)
            os.makedirs(extract_to)
    else:
        os.makedirs(extract_to)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Extracted {zip_path} to {extract_to}")

def download_all_tumor_extract_patches(download = False):
    print(f"{bcolors.HEADER}{bcolors.BOLD}[HEADER]{bcolors.ENDC} Download all tumor images and extract tumor patches")
    camelyon_dir = os.path.join(os.getcwd(), "data", "camelyon16")
    train_img_dir = os.path.join(camelyon_dir, "train", "img")
    train_mask_dir = os.path.join(camelyon_dir, "train", "mask", "annotations")

    # Download all tumor images
    if download:
        print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Downloading all tumor images...")
        for i in range(36, 112): # from validation images to end
            file_name = f"tumor_{i:03d}.tif"
            if os.path.exists(os.path.join(train_img_dir, file_name)):
                print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Tumor image {file_name} already exists in {train_img_dir}. Skipping download.")
                continue
            url = BASE_URL + f"CAMELYON16/training/tumor/{file_name}"
            destination_path = os.path.join(train_img_dir, file_name)
            download_file(url, destination_path)

    # Extract only tumor patches from downloaded tumor images
    extract_patches(patch_size=224, level=3, stride=None, pad=True, only_tumor=True) 
    
def extract_patches_per_slide(slide_path="tumor_109", patch_size=224, level=3, stride=None, pad=True, only_tumor=False):
    """
    Extract patches from a single slide directory.
    Parameters:
    - slide_dir: str, path to the slide directory containing patches.
    - patch_size: int, size of the patches to extract.
    - level: int, level of the WSI to extract patches from.
    - stride: int, stride for patch extraction.
    - pad: bool, if True, pad the image to cover all regions.
    - only_tumor: bool, if True, only extract tumor patches from tumor images.
    """
    stride = stride or patch_size
    patch_sizes = {0: 1792, 1: 896, 2: 448, 3: 224}
    patch_size = patch_sizes.get(level, 224)
    slide_path = os.path.join(os.getcwd(), "data", "camelyon16", "train", "img", slide_path + ".tif")

    prefix = os.path.splitext(os.path.basename(slide_path))[0]
    wsi_dir = os.path.dirname(slide_path)
    # Guess annotation directory based on slide location
    if "test" in wsi_dir:
        annot_dir = os.path.join(os.path.dirname(os.path.dirname(wsi_dir)), "mask", "annotations")
    else:
        annot_dir = os.path.join(os.path.dirname(os.path.dirname(wsi_dir)), "mask", "annotations")
    xml_name = prefix + ".xml"
    xml_path = os.path.join(annot_dir, xml_name)

    level_dir = os.path.join(os.getcwd(), "data", "camelyon16", "patches", f"level_{level}")
    os.makedirs(level_dir, exist_ok=True)
    patch_save_dir = os.path.join(level_dir, prefix)
    # Only skip extraction if the directory exists, is not empty, and contains both _normal.png and _tumor.png files
    if any(f.endswith("_normal.png") for f in os.listdir(patch_save_dir)):
        print(f"DEBUG: we are on patch {slide_path} with normal")
    else:
        print(f"DEBUG: we are on patch {slide_path} without normal and we should check this")
    if (
        os.path.exists(patch_save_dir)
        and len(os.listdir(patch_save_dir)) > 0
        and any(f.endswith("_normal.png") for f in os.listdir(patch_save_dir))
        and any(f.endswith("_tumor.png") for f in os.listdir(patch_save_dir))
    ):
        print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Patches for {slide_path} already extracted, skipping.")
        return
    os.makedirs(patch_save_dir, exist_ok=True)

    print(f"{bcolors.DEBUG}[DEBUG]{bcolors.ENDC} Processing file: {slide_path} with XML: {xml_path}")
    try:
        slide = openslide.OpenSlide(slide_path)
    except Exception as e:
        print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Could not open {slide_path}: {e}")
        return
    width, height = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]

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
            mask = parse_xml_mask(xml_path, (width, height), slide, level)
            if pad and (pad_w > 0 or pad_h > 0):
                mask = ImageOps.expand(mask, (0, 0, pad_w, pad_h), fill=0)
        except Exception as e:
            print(f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Failed to parse XML for {slide_path}: {e}")
    else:
        print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} No annotation found for {slide_path}, treating as normal.")

    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Processing {slide_path} at level {level} (size: {width}x{height}, padded: {padded_width}x{padded_height})")

    patch_count = 0
    for x in range(0, padded_width, stride):
        for y in range(0, padded_height, stride):
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

            if patch_w < patch_size or patch_h < patch_size:
                padded_region = Image.new("RGB", (patch_size, patch_size), (255, 255, 255))
                padded_region.paste(region, (0, 0))
                region = padded_region

            label = "normal"
            if mask:
                mask_patch = mask.crop((x, y, x + patch_size, y + patch_size))
                if np.any(np.array(mask_patch) > 0):
                    label = "tumor"

            patch_array = np.array(region)
            if np.mean(patch_array) > 240:  # too white (empty tissue)
                continue

            if (only_tumor and label == "tumor") or not only_tumor:
                patch_save_dir_labeled = os.path.join(level_dir, prefix, label)
                os.makedirs(patch_save_dir_labeled, exist_ok=True)
                patch_name = f"{prefix}_x{x}_y{y}_{label}.png"
                region.save(os.path.join(patch_save_dir_labeled, patch_name))
                patch_count += 1

    print(
        f"{bcolors.INFO}[INFO]{bcolors.ENDC} Patch extraction complete for {slide_path} at level {level}. Total patches: {patch_count}"
    )

def parse_xml_mask(xml_path, level_dims, slide, level):
    """
    Convert XML annotation to binary mask.
    Parameters:
    - xml_path: str, path to the XML file containing annotations.
    - level_dims: tuple, dimensions of the WSI at the specified level (width, height).
    - slide: OpenSlide object for the WSI.
    - level: int, target level for mask.
    """
    try:
        tree = etree.parse(xml_path)
    except etree.XMLSyntaxError as e:
        print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Error parsing XML file {xml_path}: {e}")
        return None

    # Compute scaling factors based on actual dimensions
    base_dims = slide.level_dimensions[0]
    scale_x = level_dims[0] / base_dims[0]
    scale_y = level_dims[1] / base_dims[1]

    mask = Image.new("L", level_dims, 0)
    draw = ImageDraw.Draw(mask)

    for coordinates_node in tree.xpath("//Annotation/Coordinates | //Annotations/Annotation/Coordinates"):
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
                print(f"{bcolors.WARNING}Warning: Could not parse coordinate (X,Y) from XML for {xml_path}: {e}{bcolors.ENDC}")
                continue
        if coords:
            draw.polygon(coords, outline=255, fill=255)
    return mask

def get_dataloaders(patch_dir, transform, test_ratio=0.2, batch_size=32, balanced=False):
    slide_dirs = [d for d in os.listdir(patch_dir) if os.path.isdir(os.path.join(patch_dir, d))]
    train_slides, val_slides = train_test_split(slide_dirs, test_size=test_ratio, random_state=42)

    if balanced:
        train_dataset = PatchDataset(patch_dir, slide_names=train_slides, transform=transform, balanced=True, max_samples=SAMPLES_PER_CLASS)
    else:
        train_dataset = PatchDataset(patch_dir, slide_names=train_slides, transform=transform)
    val_dataset = PatchDataset(patch_dir, slide_names=val_slides, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset, val_dataset

def train_resnet_classifier(level=3):
    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Training ResNet18 classifier...")
    patch_dir = os.path.join(os.getcwd(), "data", "camelyon16", "patches", f"level_{level}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(
        patch_dir, transform, test_ratio=0.2, batch_size=32
    )

    model = ResNet18Classifier().to(device)

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct = 0, 0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
        train_acc = correct / len(train_dataset)

        # Validation loop
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
        val_acc = val_correct / len(val_dataset)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "src/models/resnet18_patch_classifier.pth")
    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Training complete. Model saved resnet18_patch_classifier.pth.")

def train_resnet_classifier_strategic(level=3, strategy="balanced"):
    assert strategy in {"balanced", "weighted_loss", "self_supervised"}, "Invalid strategy option"

    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Training ResNet18 classifier using strategy: {strategy}...")
    patch_dir = os.path.join(os.getcwd(), "data", "camelyon16", "patches", f"level_{level}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if strategy == "self_supervised":
        if not os.path.exists("simclr_encoder.pth"):
            pretrain_simclr(patch_dir, epochs=10)

        model = ResNet18Classifier(pretrained_weights_path="simclr_encoder.pth").to(device)
        train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(patch_dir, transform)
        criterion = nn.CrossEntropyLoss()

    elif strategy == "balanced":
        train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(patch_dir, transform, balanced=True)
        model = ResNet18Classifier().to(device)
        criterion = nn.CrossEntropyLoss()

    elif strategy == "weighted_loss":
        train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(patch_dir, transform)
        class_counts = train_dataset.get_class_counts()
        total = sum(class_counts.values())
        weights = [total / class_counts[i] for i in range(len(class_counts))]
        weights = torch.FloatTensor(weights)
        model = ResNet18Classifier().to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = Adam(model.parameters(), lr=1e-4)

    # TRAINING LOOP (unchanged)
    for epoch in range(5):
        model.train()
        total_loss, correct = 0, 0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_dataset)

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_dataset)
        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), f"src/models/resnet18_patch_classifier_{strategy}.pth")
    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Training complete.")

def extract_patches(patch_size=224, level=3, stride=None, pad=True, only_tumor=False, test=False):
    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Extracting patches at level {level}...")
    stride = stride or patch_size

    # Set patch size according to level
    patch_sizes = {0: 1792, 1: 896, 2: 448, 3: 224}
    patch_size = patch_sizes.get(level, 224)

    wsi_dir = os.path.join(os.getcwd(), "data", "camelyon16", "train", "img")
    annot_dir_train = os.path.join(
        os.getcwd(), "data", "camelyon16", "train", "mask", "annotations"
    )
    annot_dir_test = os.path.join(
        os.getcwd(), "data", "camelyon16", "test", "mask", "annotations"
    )
    level_dir = os.path.join(
        os.getcwd(), "data", "camelyon16", "patches", f"level_{level}"
    )
    os.makedirs(level_dir, exist_ok=True)

    for file in os.listdir(wsi_dir):
        if not file.endswith(".tif"):
            continue
        prefix = file.replace(".tif", "")

        # Check if patches for this image already exist
        patch_save_dir = os.path.join(level_dir, prefix)
        if (
            os.path.exists(patch_save_dir)
            and len(os.listdir(patch_save_dir)) > 0
            and any(f.endswith("_normal.png") for f in os.listdir(patch_save_dir))
            and any(f.endswith("_tumor.png") for f in os.listdir(patch_save_dir))
        ):
            print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Patches for {file} already extracted, skipping.")
            continue
        os.makedirs(patch_save_dir, exist_ok=True)

        wsi_path = os.path.join(wsi_dir, file)
        xml_name = file.replace(".tif", ".xml")
        if file.startswith("test_"):
            xml_path = os.path.join(annot_dir_test, xml_name)
        elif file.startswith("normal_") or file.startswith("tumor_"):
            xml_path = os.path.join(annot_dir_train, xml_name)
        try:
            slide = openslide.OpenSlide(wsi_path)
        except Exception as e:
            print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Could not open {wsi_path}: {e}")
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
                print(f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Failed to parse XML for {file}: {e}")
        else:
            print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} No annotation found for {file}, treating as normal.")

        print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Processing {file} at level {level} (size: {width}x{height}, padded: {padded_width}x{padded_height})")

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
                    padded_region = Image.new("RGB", (patch_size, patch_size), (255, 255, 255))
                    padded_region.paste(region, (0, 0))
                    region = padded_region

                label = "unlabeled"  # Default label for normal patches
                # Check if the patch overlaps with any positimve (tumor) region in the generated binary mask
                if mask:
                    mask_patch = mask.crop((x, y, x + patch_size, y + patch_size))
                    if np.any(np.array(mask_patch) > 0):
                        label = "tumor"
                    else:
                        label = "normal"
                    
                else:
                    print(f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} No mask available for {prefix}, treating as normal.")
                    label = "normal"

                patch_array = np.array(region)
                if np.mean(patch_array) > 240:  # too white (empty tissue)
                    continue

                patch_name = f"{prefix}_x{x}_y{y}_{label}.png"
                # Only save the patch if it was not saved yet
                patch_path = os.path.join(patch_save_dir, patch_name)
                if not os.path.exists(patch_path):
                    region.save(patch_path)
                patch_count += 1


        print(
            f"{bcolors.INFO}[INFO]{bcolors.ENDC} Patch extraction complete for {file} at level {level}. Total patches: {patch_count}"
        )

def check_good_downloaded_files(level=3):
    camelyon_dir = os.path.join(os.getcwd(), "data", "camelyon16", "patches", f"level_{level}")
    to_redownload = []
    for folder_type, folders in DOWNLOADED_FILES.items():
        for remote_folder_path in folders:
            folder_name = os.path.basename(remote_folder_path)
            local_path = os.path.join(camelyon_dir, remote_folder_path)

            if not os.path.exists(local_path):
                print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Missing folder: {local_path}")
                to_redownload.append(remote_folder_path)
            else:
                # For every file in the folder, check if it exists and is valid
                for file_name in os.listdir(local_path):
                    # Try to open image files to check for corruption
                    if file_name.endswith(('.png')):
                        try:
                            with Image.open(local_path) as img:
                                img.load()  # Force loading all image data
                        except Exception as e:
                            print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} File is corrupt or incomplete: {file_name} ({e})")
                            to_redownload.append(remote_folder_path)
                    else:
                        print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} File exists and is valid: {file_name}")
    if to_redownload:
        redownload_file = os.path.join(camelyon_dir, "redownload.txt")
        with open(redownload_file, 'w') as f:
            for folder in to_redownload:
                f.write(folder + "\n")

def count_number_tumor_patches(level=3):
    """ 
    Count how many patches ending with _tumor.png and _normal.png are in the patch directory for a given level across all slides.
    """
    patch_dir = os.path.join(os.getcwd(), "data", "camelyon16", "patches", f"level_{level}")
    if not os.path.exists(patch_dir):
        print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Patch directory '{patch_dir}' does not exist. Please run patch extraction first.")
        return

    total_tumor = 0
    total_normal = 0
    slides_with_no_tumor = []

    for slide_name in os.listdir(patch_dir):
        slide_path = os.path.join(patch_dir, slide_name)
        if os.path.isdir(slide_path):
            # Count tumor and normal patches in this slide
            number_tumor_slide = sum(1 for f in os.listdir(slide_path) if f.endswith("_tumor.png"))
            total_tumor += number_tumor_slide
            if number_tumor_slide == 0:
                slides_with_no_tumor.append(slide_name)
            total_normal += sum(1 for f in os.listdir(slide_path) if f.endswith("_normal.png"))
        if (total_tumor != 0 or slide_path.endswith("_tumor.png")) and slide_name.startswith("normal_"):
            print(f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} {slide_name} finds a tumor, and it is normal")

    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Total tumor patches at level {level}: {total_tumor}")
    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Total non-tumor patches at level {level}: {total_normal}")
    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Total slides with no tumor patches at level {level}: {len(slides_with_no_tumor)}")
    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Slides with no tumor patches: {', '.join(slides_with_no_tumor)}" if slides_with_no_tumor else f"{bcolors.INFO}All slides have tumor patches.{bcolors.ENDC}")

def extract_features(level=3, model_path="resnet18_patch_classifier.pth"):
    """
    Extract features from the patches using a ResNet18 model.
    Parameters:
    - level: int, WSI level to extract patches from (0, 1, 2, 3).
    - model_path: str, path to the pre-trained ResNet18 model.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    model_path = os.path.join(os.getcwd(), "src", "models", model_path)
    patch_dir = os.path.join(
        os.getcwd(), "data", "camelyon16", "patches", f"level_{level}"
    )
    
    if not os.path.exists(patch_dir) or not os.listdir(patch_dir):
        print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Patch directory '{patch_dir}' does not exist or is empty. Please run patch extraction first.")
        return

    dataset = PatchDataset(patch_dir, transform=transform)
    # Use higher batch size and num_workers for feature extraction as it's typically I/O bound
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2) 
    
    print(
        f"{bcolors.INFO}[INFO]{bcolors.ENDC} Extracting features from patches at level {level} with patch directory: {patch_dir}, which exists: {os.path.exists(patch_dir)}"
    )
    print(
        f"{bcolors.INFO}[INFO]{bcolors.ENDC} Listing first 5 subdirectories in patch_dir: {os.listdir(patch_dir)[:5] if os.path.exists(patch_dir) else 'Not found'}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18FeatureExtractor().to(device)
    full_classifier_model = ResNet18Classifier().to(device)
    if os.path.exists(model_path):
        print(f"[INFO] Loading trained classifier weights from {model_path}")
        full_classifier_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"[WARNING] Trained classifier model not found at {model_path}. "
              "Extracting features with ImageNet pre-trained weights only. "
              "Consider running `train_resnet_classifier()` first.")

    # Load the state_dict and filter out the 'fc' layer weights
    pretrained_dict = full_classifier_model.state_dict()
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('model.fc')}

    # copy params from pretrained_dict to model_dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.eval() 

    features = []
    labels = []
    paths = []

    with torch.no_grad():
        for batch_idx, (imgs, lbls, img_paths) in enumerate(tqdm(loader, desc="Extracting Features")):
            # print("Batch size:", imgs.shape)
            feats = model(imgs.to(device))
            features.append(feats.cpu())
            labels.extend(lbls.tolist()) # Convert tensor to list for extend
            paths.extend(img_paths)
    
    if not features:
        print(
            f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} No features were extracted. Check your patch directory and dataset. "
            "It might be that PatchDataset found no images, or data loader was empty."
        )
        return
        
    features = torch.cat(features, dim=0)  # (num_patches, 512)

    # Save features, labels, and paths
    features_save_path = f"patch_features_{level}.npy"
    labels_save_path = f"patch_labels_{level}.npy"
    paths_save_path = f"patch_paths_{level}.txt"
    
    np.save(features_save_path, features.numpy())
    np.save(labels_save_path, np.array(labels))
    with open(paths_save_path, "w") as f:
        for p in paths:
            f.write(f"{p}\n")
    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Features saved to {features_save_path}, labels to {labels_save_path}, paths to {paths_save_path}")

# Feature extraction function
def extract_features_with_simclr(level=3, simclr_encoder_path="simclr_encoder.pth"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    patch_dir = os.path.join(os.getcwd(), "data", "camelyon16", "patches", f"level_{level}")
    if not os.path.exists(patch_dir) or not os.listdir(patch_dir):
        print(f"[ERROR] Patch directory {patch_dir} is missing or empty.")
        return

    dataset = PatchDataset(patch_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnifiedResNet(pretrained_weights_path=simclr_encoder_path, classifier=False).to(device)
    model.eval()

    features, labels, paths = [], [], []

    with torch.no_grad():
        for imgs, lbls, img_paths in tqdm(loader, desc="Extracting Features"):
            feats = model(imgs.to(device))
            features.append(feats.cpu())
            labels.extend(lbls.tolist())
            paths.extend(img_paths)

    features = torch.cat(features, dim=0)
    np.save(f"patch_features_{level}.npy", features.numpy())
    np.save(f"patch_labels_{level}.npy", np.array(labels))
    with open(f"patch_paths_{level}.txt", "w") as f:
        for p in paths:
            f.write(f"{p}\n")

    print(f"[INFO] Feature extraction complete: {features.shape[0]} patches saved.")

def create_validation_set(remote=False):
    print(f"{bcolors.HEADER}{bcolors.BOLD}[HEADER]{bcolors.ENDC} Create validation set")
    src_dir = os.path.join(os.getcwd(), "data", "camelyon16", "train", "img")
    dst_dir = os.path.join(os.getcwd(), "data", "camelyon16", "val", "img")
    os.makedirs(dst_dir, exist_ok=True)

    normal_files = sorted([f for f in os.listdir(src_dir) if f.startswith("normal")])[-5:]
    tumor_files = sorted([f for f in os.listdir(src_dir) if f.startswith("tumor")])[-5:]

    for f in normal_files + tumor_files:
        if remote:
            shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    # Move as well masks to validation set
    mask_src_dir = os.path.join(os.getcwd(), "data", "camelyon16", "train", "mask", "annotations")
    mask_dst_dir = os.path.join(os.getcwd(), "data", "camelyon16", "val", "mask", "annotations")
    os.makedirs(mask_dst_dir, exist_ok=True)
    for f in os.listdir(mask_src_dir):
        # Take the masks of the tumor_files on validation set
        wsi_name = f.split('.')[0]  # e.g. tumor_001.xml -> tumor_001
        if wsi_name in [f.split('.')[0] for f in tumor_files]:
            shutil.move(os.path.join(mask_src_dir, f), os.path.join(mask_dst_dir, f))

    print(
        f"{bcolors.INFO}[INFO]{bcolors.ENDC} Validation set created with {len(normal_files)} normal and {len(tumor_files)} tumor files."
    )

def prepare_data():
    print(f"{bcolors.HEADER}{bcolors.BOLD}[HEADER]{bcolors.ENDC} Preparing data...")

    # Extract training masks
    train_zip = os.path.join(
        os.getcwd(), "data", "camelyon16", "train", "mask", "lesion_annotations.zip"
    )
    train_extract_to = os.path.join(
        os.getcwd(), "data", "camelyon16", "train", "mask", "annotations"
    )
    if not os.path.exists(train_zip):
        print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Training masks zip file not found. Please download the dataset first.")
    else:
        extract_zip(train_zip, train_extract_to)

    # Extract testing masks
    test_zip = os.path.join(
        os.getcwd(), "data", "camelyon16", "test", "mask", "lesion_annotations.zip"
    )
    test_extract_to = os.path.join(
        os.getcwd(), "data", "camelyon16", "test", "mask", "annotations"
    )
    if not os.path.exists(test_zip):
        print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Testing masks zip file not found. Please download the dataset first.")
    else:
        extract_zip(test_zip, test_extract_to)

    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Data preparation completed.")

def images_downloaded():
    img_dir = os.path.join(os.getcwd(), "data", "camelyon16", "train", "img")
    return os.path.exists(img_dir) and len([f for f in os.listdir(img_dir) if f.endswith(".tif")]) > 0

def patches_extracted(patch_level):
    patch_dir = os.path.join(os.getcwd(), "data", "camelyon16", "patches", f"level_{patch_level}")
    return os.path.exists(patch_dir) and any(os.listdir(patch_dir))

def features_extracted(patch_level):
    return os.path.exists(f"patch_features_{patch_level}.npy") and os.path.exists(f"patch_labels_{patch_level}.npy")

def evaluate_resnet_classifier(patch_level=3):
    """ 
    Evaluate the ResNet18 classifier on validation patches.
    """
    model_path = os.path.join(os.getcwd(), "src", "models", "resnet18_patch_classifier.pth")
    if not os.path.exists(model_path):
        print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Model file '{model_path}' does not exist. Please train the model first.")
        return
    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Evaluating ResNet18 classifier...")

    patch_dir = os.path.join(os.getcwd(), "data", "camelyon16", "patches", f"level_{patch_level}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    _, val_loader, _, val_dataset = get_dataloaders(
        patch_dir, transform, test_ratio=0.2, batch_size=64
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Classifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Classifier accuracy on validation patches: {acc:.4f}")

def validate_resnet_classifier(model_path="resnet18_patch_classifier.pth"):
    """
    Sanity check for extracted patch features — no plotting, CLI only.
    """
    features_path = os.path.join(os.getcwd(), "patch_features_3.npy")
    labels_path = os.path.join(os.getcwd(), "patch_labels_3.npy")
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Features or labels not found. Please run feature extraction first.")
        return
    features = np.load(features_path)     # shape: (N, 512)
    labels = np.load(labels_path)         # shape: (N,)

    print(f"[INFO] Feature shape: {features.shape}")
    print(f"[INFO] Labels shape: {labels.shape}")
    print(f"[INFO] Label distribution (0=normal, 1=tumor): {np.bincount(labels)}")

    # --------------------------------------
    # 1. PCA - print explained variance ratio
    # --------------------------------------
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    print(f"[INFO] PCA explained variance ratio (2 components): {pca.explained_variance_ratio_}")

    # Print means of each class in PCA space
    for cls in [0, 1]:
        mean_coords = features_pca[labels == cls].mean(axis=0)
        print(f"[INFO] PCA mean for class {cls}: {mean_coords}")

    # --------------------------------------
    # 2. t-SNE - print mean coordinates to see separation
    # --------------------------------------
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    features_tsne = tsne.fit_transform(features)
    for cls in [0, 1]:
        mean_coords = features_tsne[labels == cls].mean(axis=0)
        print(f"[INFO] t-SNE mean for class {cls}: {mean_coords}")

    # --------------------------------------
    # 3. Logistic Regression
    # --------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"[INFO] Logistic Regression Accuracy: {acc:.4f}")
    print("[INFO] Confusion Matrix:")
    print(cm)


def main():
    parser = argparse.ArgumentParser(description="Camelyon Dataset Processing")
    parser.add_argument("--download", action="store_true", help="Download CAMELYON16 dataset")
    parser.add_argument("--remote", action="store_true", help="Execute on remote server")
    parser.add_argument("-p", "--patch", action="store_true", help="Extract patches")
    parser.add_argument("--patch_level", type=str, default="3", help="WSI level for patch extraction (0, 1, 2, 3, or 'all' for all levels)")
    parser.add_argument("-prep", "--prepare", action="store_true", help="Prepare data")
    parser.add_argument("-val", "--validation", action="store_true", help="Create validation set")
    parser.add_argument("--validate", action="store_true", help="Validate Resnet model (sanity check for extracted patch features)")
    parser.add_argument("-train", "--train", action="store_true", help="Train Resnet model")
    parser.add_argument("-eval", "--evaluate", action="store_true", help="Evaluate Resnet model")
    parser.add_argument("--extract_features", action="store_true", help="Extract features from patches")
    parser.add_argument("--run_evaluation", action="store_true", help="Run CAMELYON16 evaluation script.")
    parser.add_argument("--balance_dataset", action="store_true", help="Balance dataset by downloading all tumor images and extracting patches from them.")
    parser.add_argument("--count_tumor_patches", action="store_true", help="Count number of tumor patches at a given level.")
    parser.add_argument("--patch_one_slide", type=str, default=None, help="Extract patches from a single slide directory (e.g. tumor_109)")
    parser.add_argument("--slide", type=str, default=None, help="Extract patches from a single slide directory (e.g. tumor_109) at a given level")
    parser.add_argument("--move_files", action="store_true", help="Move patches to a new directory structure based on slide names")
    parser.add_argument("--train_strategy", action="store_true", help="Train ResNet classifier with a specific strategy")
    parser.add_argument("--check_good_downloaded_files", action="store_true", help="Check if downloaded files are good (not corrupted)")
    parser.add_argument("--strategy", type=str, default="self_supervised", choices=["balanced", "weighted_loss", "self_supervised"], help="Training strategy for ResNet classifier")
    # Check for unknown arguments
    known_args = {action.dest for action in parser._actions}
    input_args = {arg.lstrip('-').replace('-', '_') for arg in sys.argv[1:] if arg.startswith('-')}
    unknown_args = input_args - known_args
    if unknown_args:
        print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Unknown command line arguments: {', '.join(unknown_args)}")
        sys.exit(1)
    
    args = parser.parse_args()

    # Download images
    if args.check_good_downloaded_files:
        print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Checking downloaded files for corruption...")
        check_good_downloaded_files(level=int(args.patch_level) if args.patch_level.isdigit() else 3)
        return
    if args.download:
        download_dataset(args.remote)

    if args.move_files:
        move_files()

    # Extract patches
    if args.patch:
        if not images_downloaded():
            print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Images must be downloaded before extracting patches.")
            return
        if args.patch_level == "all":
            for lvl in [0, 1, 2, 3]:
                extract_patches(level=lvl)
        else:
            extract_patches(level=int(args.patch_level))

    # Extract features
    if args.extract_features:
        # Check for patches at the requested level
        patch_levels = [0, 1, 2, 3] if args.patch_level == "all" else [int(args.patch_level)]
        for lvl in patch_levels:
            if not patches_extracted(lvl):
                print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Patches must be extracted at level {lvl} before extracting features.")
                return
        extract_features(level=int(args.patch_level) if args.patch_level != "all" else 3)  # default to level 3 if all

    # Train model
    if args.train:
        if not images_downloaded():
            print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Images must be downloaded before training.")
            return
        if not patches_extracted(patch_level=args.patch_level):
            print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Patches must be extracted before training.")
            return
        train_resnet_classifier(args.patch_level)

    if args.train_strategy:
        if not images_downloaded():
            print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Images must be downloaded before training.")
            return
        if not patches_extracted(patch_level=args.patch_level):
            print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Patches must be extracted before training.")
            return
        train_resnet_classifier_strategic(level=int(args.patch_level), strategy=args.strategy)

    if args.prepare:
        prepare_data()
    if args.validation:
        create_validation_set(args.remote)
    if args.validate:
        validate_resnet_classifier()
    if args.evaluate:
        evaluate_resnet_classifier(args.patch_level)
    if args.balance_dataset:
        download_all_tumor_extract_patches()
    if args.count_tumor_patches:
        count_number_tumor_patches(level=3)
    if args.patch_one_slide:
        extract_patches_per_slide(slide_path=args.patch_one_slide, level=int(args.patch_level))

    if args.run_evaluation:
        """
        Calculate False Positives (FPs), True Positives (TPs), and generates a Free-Response Receiver Operating Characteristic (FROC) curve. 
        """
        print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Running CAMELYON16 evaluation script.")
        mask_folder_for_eval = os.path.join(os.getcwd(), "data", "camelyon16", "test", "mask")
        results_folder_for_eval = os.path.join(os.getcwd(), "models", "first_model", "model_predictions_csv") 
        
        if not os.path.exists(mask_folder_for_eval):
            print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Evaluation mask folder '{mask_folder_for_eval}' not found. Please generate TIFF masks from XML annotations first.")
        elif not os.path.exists(results_folder_for_eval):
             print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Model results folder '{results_folder_for_eval}' not found. Please run your detection model first.")
        else:
            result_file_list = [each for each in os.listdir(results_folder_for_eval) if each.endswith('.csv')]
            
            EVALUATION_MASK_LEVEL = 5 
            L0_RESOLUTION = 0.243 
            
            FROC_data = np.empty((4, len(result_file_list)), dtype=object)
            FP_summary = np.empty((2, len(result_file_list)), dtype=object)
            detection_summary = np.empty((2, len(result_file_list)), dtype=object)
            
            caseNum = 0    
            for case in result_file_list:
                print(f'Evaluating Performance on image: {case[0:-4]}')
                sys.stdout.flush()
                csvDIR = os.path.join(results_folder_for_eval, case)
                Probs, Xcorr, Ycorr = readCSVContent(csvDIR) # is this function is updated for Python 3?
                            
                is_tumor = case[0:5].lower() == 'tumor' # Use .lower() for robustness    
                if (is_tumor):
                    maskDIR = os.path.join(mask_folder_for_eval, case[0:-4]) + '_Mask.tif'
                    if not os.path.exists(maskDIR):
                        print(f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Mask TIFF '{maskDIR}' not found for tumor case. Skipping.")
                        continue # Skip to next case if mask is missing
                    evaluation_mask = computeEvaluationMask(maskDIR, L0_RESOLUTION, EVALUATION_MASK_LEVEL) # Python 3?
                    ITC_labels = computeITCList(evaluation_mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL) 
                else:
                    evaluation_mask = 0 # Or a blank mask for consistency
                    ITC_labels = []
                        
                FROC_data[0][caseNum] = case
                FP_summary[0][caseNum] = case
                detection_summary[0][caseNum] = case
                
                # Update compute_FP_TP_Probs for Python 3 division (//)
                FROC_data[1][caseNum], FROC_data[2][caseNum], FROC_data[3][caseNum], detection_summary[1][caseNum], FP_summary[1][caseNum] = \
                    compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, ITC_labels, EVALUATION_MASK_LEVEL)
                caseNum += 1
            
            # Compute FROC curve 
            if caseNum > 0: # Only compute if there were cases processed
                total_FPs, total_sensitivity = computeFROC(FROC_data) # Update for Python 3
                
                # plot FROC curve
                plotFROC(total_FPs, total_sensitivity) # Update for Python 3
            else:
                print(f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} No cases processed for FROC evaluation.")

if __name__ == "__main__":
    main()
