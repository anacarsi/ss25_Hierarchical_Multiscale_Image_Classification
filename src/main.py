import os
import sys
import argparse
import requests
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# TODO: add dll directory for OpenSlide avoiding giving specific path
# os.add_dll_directory(r"C:\Program Files\OpenSlide\openslide-bin-4.0.0.8-windows-x64\bin")
from src.preprocessing.camelyon16_mil_dataset import Camelyon16MILDataset
from src.train import train_model
from src.eval import evaluate_model
from src.config import Config
from src.preprocessing.pre_WSI import val_wsi
from src.models.unet.UNet import UNet
import random
import shutil
import openslide
import zipfile

# Base URL for the CAMELYON16 dataset
BASE_URL = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100439/"

# File paths for CAMELYON16
CAMELYON16_FILES = {
    "train_normal": [f"CAMELYON16/training/normal/normal_{i:03d}.tif" for i in range(1, 112)],
    "train_tumor": [f"CAMELYON16/training/tumor/tumor_{i:03d}.tif" for i in range(1, 112)],
    "test_images": [f"CAMELYON16/testing/images/test_{i:03d}.tif" for i in range(1, 51)],
    "train_masks": ["CAMELYON16/training/lesion_annotations.zip"],
    "test_masks": ["CAMELYON16/testing/lesion_annotations.zip"]
}

def download_file(url, destination_path):
    """
    Downloads a file from a URL to a destination path with a progress bar.
    """
    try:
        print(f"[INFO] Downloading: {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(destination_path, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, unit_divisor=1024,
                desc=f"Downloading {os.path.basename(destination_path)}"
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
    Generalized function to download files into specified directories.

    Parameters:
    - base_dir: str, base directory for downloaded files.
    - file_groups: dict, mapping of subdirectories to file lists.
    - remote: bool, if True, download all files; if False, download only one file for testing.
    """
    for group_name, files in file_groups.items():
        group_dir = os.path.join(base_dir, group_name)  # Ensure correct subdirectory
        os.makedirs(group_dir, exist_ok=True)
        print(f"[INFO] Downloading files to {group_dir}...")

        if not remote:
            files = files[:9]  # For local testing, limit to the first file

        for file_path in files:
            destination_path = os.path.join(group_dir, os.path.basename(file_path))
            if os.path.exists(destination_path):
                print(f"[INFO] File already exists: {destination_path}")
                continue  # Skip downloading if the file already exists

            url = BASE_URL + file_path
            print(f"[INFO] Downloading: {url}")
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
        "train/img": CAMELYON16_FILES["train_normal"] + CAMELYON16_FILES["train_tumor"],  # Both normal and tumor images go to img
        "test/img": CAMELYON16_FILES["test_images"],
        "masks": CAMELYON16_FILES["train_masks"] + CAMELYON16_FILES["test_masks"]
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
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[INFO] Extracted {zip_path} to {extract_to}")


def extract_patches(type: str, ):
    """
    Extract patches from the downloaded WSI images.
    """
    print("[INFO] Extracting patches...")
    patch_size = 512
    batch_size = 12
    image_path = "../data/test_040.tif" 
    image = openslide.OpenSlide(image_path)
    val_wsi(image, patch_size, batch_size)
    print("[INFO] Patch extraction completed.")

def create_validation_set(base_dir="./data", val_split=0.2):
    """
    Create a validation set by splitting a portion of the training set.

    Parameters:
    - base_dir: str, base directory for the dataset.
    - val_split: float, fraction of the training set to use as validation (e.g., 0.2 for 20%).
    """
    train_img_dir = os.path.join(base_dir, "camelyon16/train/img")
    val_img_dir = os.path.join(base_dir, "camelyon16/val/img")
    os.makedirs(val_img_dir, exist_ok=True)

    # Get all training files
    train_files = [f for f in os.listdir(train_img_dir) if f.endswith(".tif")]

    # Shuffle and split files for validation
    random.shuffle(train_files)
    val_count = int(len(train_files) * val_split)
    val_files = train_files[:val_count]

    print(f"[INFO] Moving {val_count} files to validation set...")

    # Move files to validation directory
    for file_name in val_files:
        src_path = os.path.join(train_img_dir, file_name)
        dest_path = os.path.join(val_img_dir, file_name)
        shutil.move(src_path, dest_path)

    print("[INFO] Validation set creation completed.")

def prepare_data(): # TODO: WIP
    """
    Prepare data for training (e.g., preprocessing or augmentation).
    """
    print("[INFO] Preparing data...")

    # Create a validation set from the training data
    create_validation_set(base_dir="./data", val_split=0.2)

    # Extract masks
    zip_path = os.path.join(os.getcwd(), "..", "data", "camelyon16", "masks", "lesion_annotations.zip")
    extract_to = os.path.join(os.getcwd(), "..", "data", "camelyon16", "masks", "annotations")
    extract_zip(zip_path, extract_to)

    print("[INFO] Data preparation completed.")

def train_unet():
    """
    Train the U-Net model on the Camelyon dataset.
    """
    print("[INFO] Training U-Net model...")
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = Camelyon16MILDataset(config.train_data_path)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = Camelyon16MILDataset(config.val_data_path)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = UNet(3, 3).to(device)
    train_model(model, train_loader, device, config)
    print("[INFO] U-Net training completed.")

def test_unet():
    """
    Test the U-Net model on the test set.
    """
    print("[INFO] Testing U-Net model...")
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = Camelyon16MILDataset(config.test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = UNet(3, 3).to(device)
    model.load_state_dict(torch.load("UNet.pth"))
    evaluate_model(model, test_loader, device, config)
    print("[INFO] U-Net testing completed.")

def main():
    parser = argparse.ArgumentParser(description="Camelyon Dataset Processing")
    parser.add_argument("--download", action="store_true", help="Download CAMELYON16 dataset")
    parser.add_argument("--base_dir", type=str, default="./data", help="Base directory for downloaded files")
    parser.add_argument("--remote", action="store_true", help="Execute on remote server")
    parser.add_argument("-p", "--patch", action="store_true", help="Extract patches")
    parser.add_argument("-prep", "--prepare", action="store_true", help="Prepare data")
    parser.add_argument("-val", "--validation", action="store_true", help="Create validation set")
    parser.add_argument("-train", "--train", action="store_true", help="Train U-Net model")
    parser.add_argument("-test", "--test", action="store_true", help="Test U-Net model")
    args = parser.parse_args()

    if args.download:
        download_dataset(args.base_dir, args.remote)
    if args.patch:
        extract_patches()
    if args.prepare:
        prepare_data()
    if args.validation:
        create_validation_set(args.base_dir)
    if args.train:
        train_unet()
    if args.test:
        test_unet()

if __name__ == "__main__":
    main()