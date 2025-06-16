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
import openslide

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

def download_dataset(base_dir="./data", remote=False):
    """
    Downloads the CAMELYON16 dataset, including training, testing, and mask files.

    Parameters:
    - base_dir: str, directory to save the downloaded files.
    - remote: bool, whether to execute on a remote server (download all).
    """
    os.makedirs(os.path.join(base_dir, "camelyon16"), exist_ok=True)
    train_dir = os.path.join(base_dir, "camelyon16/train")
    val_dir = os.path.join(base_dir, "camelyon16/val")
    test_dir = os.path.join(base_dir, "camelyon16/test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Download training images
    print("[INFO] Downloading training images...")
    for category, files in {"normal": CAMELYON16_FILES["train_normal"], "tumor": CAMELYON16_FILES["train_tumor"]}.items():
        category_dir = os.path.join(train_dir, "img" if category == "normal" else "mask")
        os.makedirs(category_dir, exist_ok=True)
        if not remote:
            files = files[:1]  # For local testing, limit to the first file
        for file_path in files:
            destination_path = os.path.join(category_dir, os.path.basename(file_path))
            if not os.path.exists(destination_path):
                url = BASE_URL + file_path
                download_file(url, destination_path)
            else:
                print(f"[INFO] File already exists: {destination_path}")

    # Download testing images
    print("[INFO] Downloading testing images...")
    test_img_dir = os.path.join(test_dir, "img")
    os.makedirs(test_img_dir, exist_ok=True)
    for file_path in CAMELYON16_FILES["test_images"]:
        destination_path = os.path.join(test_img_dir, os.path.basename(file_path))
        if not os.path.exists(destination_path):
            url = BASE_URL + file_path
            download_file(url, destination_path)
        else:
            print(f"[INFO] File already exists: {destination_path}")

    # Download masks
    print("[INFO] Downloading masks...")
    for mask_type, files in {"train_masks": CAMELYON16_FILES["train_masks"], "test_masks": CAMELYON16_FILES["test_masks"]}.items():
        mask_dir = os.path.join(base_dir, "Camelyon16/masks")
        os.makedirs(mask_dir, exist_ok=True)
        for file_path in files:
            destination_path = os.path.join(mask_dir, os.path.basename(file_path))
            if not os.path.exists(destination_path):
                url = BASE_URL + file_path
                download_file(url, destination_path)
            else:
                print(f"[INFO] File already exists: {destination_path}")

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

def prepare_data(): # TODO: WIP
    """
    Prepare data for training (e.g., preprocessing or augmentation).
    """
    print("[INFO] Preparing data...")
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
    parser.add_argument("-train", "--train", action="store_true", help="Train U-Net model")
    parser.add_argument("-test", "--test", action="store_true", help="Test U-Net model")
    args = parser.parse_args()

    if args.download:
        download_dataset(args.base_dir, args.remote)
    if args.patch:
        extract_patches()
    if args.prepare:
        prepare_data()
    if args.train:
        train_unet()
    if args.test:
        test_unet()

if __name__ == "__main__":
    main()