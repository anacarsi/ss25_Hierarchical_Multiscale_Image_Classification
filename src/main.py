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
"""
os.add_dll_directory(
    r"C:\Program Files\OpenSlide\openslide-bin-4.0.0.8-windows-x64\bin"
)
"""
import openslide
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.resnet import ResNet18Classifier, ResNet18FeatureExtractor
from datasets.patch_dataset import PatchDataset
from utils.evaluation_FROC import computeEvaluationMask, computeITCList, readCSVContent, compute_FP_TP_Probs, computeFROC, plotFROC
from utils.structure import group_patches_by_slide
import zipfile
from PIL import Image

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


def download_file(url, destination_path):
    """
    Downloads a file from a URL to a destination path with a progress bar.
    """
    try:
        print(f"[INFO] Downloading: {url} into {destination_path}")
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
        print(f"[INFO] Successfully downloaded {os.path.basename(destination_path)}.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
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
                print(f"[INFO] Skipping download of {file_name} in remote mode.")
                continue

            destination_path = os.path.join(target_dir, file_name)

            if os.path.exists(destination_path):
                print(f"[INFO] Skipping: {destination_path} already exists.")
                continue
            
            url = BASE_URL + remote_file_path
            download_file(url, destination_path)


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
            print(f"[INFO] Directory {extract_to} already exists and contains all expected XMLs. Skipping extraction.")
            return
        else:
            print(f"[WARNING] Directory {extract_to} exists but is missing some XMLs. Re-extracting...")
            shutil.rmtree(extract_to)
            os.makedirs(extract_to)
    else:
        os.makedirs(extract_to)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[INFO] Extracted {zip_path} to {extract_to}")


def parse_xml_mask(xml_path, level_dims, downsample):
    """
    Convert XML annotation to binary mask.
    Parameters:
    - xml_path: str, path to the XML file containing annotations.
    - level_dims: tuple, dimensions of the WSI at the specified level (width, height).
    - downsample: float, downsample factor for the WSI level.
    """
    try:
        tree = etree.parse(xml_path)
    except etree.XMLSyntaxError as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        return None

    mask = Image.new("L", level_dims, 0)
    draw = ImageDraw.Draw(mask)

    for coordinates_node in tree.xpath("//Annotation/Coordinates | //Annotations/Annotation/Coordinates"):
        coords = []
        for coord_node in coordinates_node.findall("Coordinate"):
            try:
                x = float(coord_node.get("X"))
                y = float(coord_node.get("Y"))
                # Scale coordinates to the target level
                scaled_x = int(x / downsample)
                scaled_y = int(y / downsample)
                coords.append((scaled_x, scaled_y))
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not parse coordinate (X,Y) from XML for {xml_path}: {e}")
                continue
        if coords:
            # Draw with 255 for white on a black background
            draw.polygon(coords, outline=255, fill=255)
    return mask


def train_resnet_classifier(level=3):
    """ 
    Train a ResNet18 classifier on the extracted patches.
    """
    print("[INFO] Training ResNet18 classifier on extracted patches...")
    patch_dir = os.path.join(os.getcwd(), "..", "data", "camelyon16", "patches", f"level_{level}")

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

    torch.save(model.state_dict(), "models/resnet18_patch_classifier.pth")
    print("[INFO] ResNet18 classifier training complete and saved.")

def extract_patches(patch_size=224, level=3, stride=None, pad=True):
    """
    Extract patches from WSIs at a specified level, apply mask overlays, and save tumor vs normal labels.
    Only extracts patches if they have not already been extracted for a given image.
    Parameters:
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
        if os.path.exists(patch_save_dir) and len(os.listdir(patch_save_dir)) > 0:
            print(f"[INFO] Patches for {file} already extracted, skipping.")
            continue
        os.makedirs(patch_save_dir, exist_ok=True)

        wsi_path = os.path.join(wsi_dir, file)
        xml_name = file.replace(".tif", ".xml")
        if file.startswith("test_"):
            xml_path = os.path.join(annot_dir_test, xml_name)
        elif file.startswith("normal_") or file.startswith("tumor_"):
            xml_path = os.path.join(annot_dir_train, xml_name)
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

                label = "normal"
                # Check if the patch overlaps with any positimve (tumor) region in the generated binary mask
                if mask:
                    mask_patch = mask.crop((x, y, x + patch_size, y + patch_size))
                    if np.any(np.array(mask_patch) > 0):
                        label = "tumor"

                patch_array = np.array(region)
                if np.mean(patch_array) > 240:  # too white (empty tissue)
                    continue

                patch_save_dir = os.path.join(level_dir, prefix)
                os.makedirs(patch_save_dir, exist_ok=True)
                patch_name = f"{prefix}_x{x}_y{y}_{label}.png"
                region.save(os.path.join(patch_save_dir, patch_name))
                patch_count += 1
                if patch_count % 100 == 0:
                    print(f"Extracted patches {patch_count} for {file}")

        print(
            f"[INFO] Patch extraction complete for {file} at level {level}. Total patches: {patch_count}"
        )


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
    model_path = os.path.join(os.getcwd(), "models", model_path)
    patch_dir = os.path.join(
        os.getcwd(), "..", "data", "camelyon16", "patches", f"level_{level}"
    )
    
    if not os.path.exists(patch_dir) or not os.listdir(patch_dir):
        print(f"[ERROR] Patch directory '{patch_dir}' does not exist or is empty. Please run patch extraction first.")
        return

    dataset = PatchDataset(patch_dir, transform=transform)
    # Use higher batch size and num_workers for feature extraction as it's typically I/O bound
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=os.cpu_count() or 1) 
    
    print(
        f"[INFO] Extracting features from patches at level {level} with patch directory: {patch_dir}, which exists: {os.path.exists(patch_dir)}"
    )
    print(
        "[INFO] Listing first 5 subdirectories in patch_dir:",
        os.listdir(patch_dir)[:5] if os.path.exists(patch_dir) else "Not found",
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

    model = ResNet18FeatureExtractor().to(device)
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
            "[ERROR] No features were extracted. Check your patch directory and dataset. "
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
    print(f"[INFO] Features saved to {features_save_path}, labels to {labels_save_path}, paths to {paths_save_path}")


def create_validation_set(remote=False):
    """
    Create validation set: exactly 10 files (5 normal + 5 tumor).
    """
    src_dir = os.path.join(os.getcwd(), "data", "camelyon16", "train", "img")
    dst_dir = os.path.join(os.getcwd(), "data", "camelyon16", "val", "img")
    os.makedirs(dst_dir, exist_ok=True)

    normal_files = sorted([f for f in os.listdir(src_dir) if f.startswith("normal")])[-5:]
    tumor_files = sorted([f for f in os.listdir(src_dir) if f.startswith("tumor")])[-5:]

    for f in normal_files + tumor_files:
        if remote:
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
        "data/camelyon16/train/mask",
        "data/camelyon16/test/mask",
        "data/camelyon16/patches/level_3/normal_002", # to check expected patch struct
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


def prepare_data():
    print("[INFO] Preparing data...")

    # Extract training masks
    train_zip = os.path.join(
        os.getcwd(), "data", "camelyon16", "train", "mask", "lesion_annotations.zip"
    )
    train_extract_to = os.path.join(
        os.getcwd(), "data", "camelyon16", "train", "mask", "annotations"
    )
    if not os.path.exists(train_zip):
        print("[ERROR] Training masks zip file not found. Please download the dataset first.")
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
        print("[ERROR] Testing masks zip file not found. Please download the dataset first.")
    else:
        extract_zip(test_zip, test_extract_to)

    print("[INFO] Data preparation completed.")


def images_downloaded():
    img_dir = os.path.join(os.getcwd(), "data", "camelyon16", "train", "img")
    return os.path.exists(img_dir) and len([f for f in os.listdir(img_dir) if f.endswith(".tif")]) > 0

def patches_extracted(patch_level):
    patch_dir = os.path.join(os.getcwd(), "data", "camelyon16", "patches", f"level_{patch_level}")
    return os.path.exists(patch_dir) and any(os.listdir(patch_dir))

def features_extracted(patch_level):
    return os.path.exists(f"patch_features_{patch_level}.npy") and os.path.exists(f"patch_labels_{patch_level}.npy")

def main():
    parser = argparse.ArgumentParser(description="Camelyon Dataset Processing")
    parser.add_argument("--download", action="store_true", help="Download CAMELYON16 dataset")
    parser.add_argument("--remote", action="store_true", help="Execute on remote server")
    parser.add_argument("-p", "--patch", action="store_true", help="Extract patches")
    parser.add_argument("--patch_level", type=str, default="3", help="WSI level for patch extraction (0, 1, 2, 3, or 'all' for all levels)")
    parser.add_argument("-prep", "--prepare", action="store_true", help="Prepare data")
    parser.add_argument("-val", "--validation", action="store_true", help="Create validation set")
    parser.add_argument("-train", "--train", action="store_true", help="Train U-Net model")
    parser.add_argument("-test", "--test", action="store_true", help="Test U-Net model")
    parser.add_argument("--extract_features", action="store_true", help="Extract features from patches")
    parser.add_argument("--check_structure", action="store_true", help="Check directory structure")
    parser.add_argument("--run_evaluation", action="store_true", help="Run CAMELYON16 evaluation script.")

    # Check for unknown arguments
    known_args = {action.dest for action in parser._actions}
    input_args = {arg.lstrip('-').replace('-', '_') for arg in sys.argv[1:] if arg.startswith('-')}
    unknown_args = input_args - known_args
    if unknown_args:
        print(f"[ERROR] Unknown command line arguments: {', '.join(unknown_args)}")
        sys.exit(1)
    
    args = parser.parse_args()

    # Download images
    if args.download:
        download_dataset(args.remote)

    # Extract patches
    if args.patch:
        if not images_downloaded():
            print("[ERROR] Images must be downloaded before extracting patches.")
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
                print(f"[ERROR] Patches must be extracted at level {lvl} before extracting features.")
                return
        extract_features(level=int(args.patch_level) if args.patch_level != "all" else 3)  # default to level 3 if all

    # Train model
    if args.train:
        if not images_downloaded():
            print("[ERROR] Images must be downloaded before training.")
            return
        if not patches_extracted(patch_level=args.train_patch_level):
            print("[ERROR] Patches must be extracted before training.")
            return
        if not features_extracted():
            print("[ERROR] Features must be extracted before training.")
            return
        train_resnet_classifier(args.patch_level)

    if args.prepare:
        prepare_data()
    if args.validation:
        create_validation_set(args.remote)
    if args.test:
        pass
    if args.check_structure:
        check_structure()

    if args.run_evaluation:
        """
        Calculate False Positives (FPs), True Positives (TPs), and generates a Free-Response Receiver Operating Characteristic (FROC) curve. 
        """
        print("[INFO] Running CAMELYON16 evaluation script.")
        mask_folder_for_eval = os.path.join(os.getcwd(), "data", "camelyon16", "test", "mask")
        results_folder_for_eval = os.path.join(os.getcwd(), "models", "first_model", "model_predictions_csv") 
        
        if not os.path.exists(mask_folder_for_eval):
            print(f"[ERROR] Evaluation mask folder '{mask_folder_for_eval}' not found. Please generate TIFF masks from XML annotations first.")
        elif not os.path.exists(results_folder_for_eval):
             print(f"[ERROR] Model results folder '{results_folder_for_eval}' not found. Please run your detection model first.")
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
                        print(f"[WARNING] Mask TIFF '{maskDIR}' not found for tumor case. Skipping.")
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
                print("[WARNING] No cases processed for FROC evaluation.")

if __name__ == "__main__":
    main()
