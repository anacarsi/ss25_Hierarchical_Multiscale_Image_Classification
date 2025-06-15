import os
import sys
import argparse
import requests
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.add_dll_directory(r"C:\Program Files\OpenSlide\openslide-bin-4.0.0.8-windows-x64\bin")
from src.preprocessing.camelyon16_mil_dataset import Camelyon16MILDataset
from src.train import train_model
from src.eval import evaluate_model
from src.config import Config
from src.preprocessing.pre_WSI import val_wsi
from src.models.unet.UNet import UNet
import openslide
# Base URL for the GigaDB dataset
BASE_URL = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100439/"

# File paths for CAMELYON16 and CAMELYON17
CAMELYON16_FILES = {
    "normal": [f"CAMELYON16/training/normal/normal_{i:03d}.tif" for i in range(1, 112)],
    "tumor": [f"CAMELYON16/training/tumor/tumor_{i:03d}.tif" for i in range(1, 112)],
}

CAMELYON17_FILES = {
    "center_0": [f"CAMELYON17/training/center_0/patient_{i:03d}.zip" for i in range(0, 50)],
}

def download_file(url, destination_path):
    """
    Downloads a file from a URL to a destination path with a progress bar.
    """
    try:
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

def download_dataset(data_type, subset, base_dir="../data", remote=False):
    """
    Downloads the specified subset of the CAMELYON dataset.
    Args:
        data_type: "CAMELYON16" or "CAMELYON17"
        subset: "normal", "tumor", or "center_0"
        base_dir: Directory to save the downloaded files.
        remote: If True, download all files; if False, download only one file for testing.
    """
    os.makedirs(base_dir, exist_ok=True)
    if data_type == "CAMELYON16":
        files = CAMELYON16_FILES.get(subset, [])
    elif data_type == "CAMELYON17":
        files = CAMELYON17_FILES.get(subset, [])
    else:
        print(f"[ERROR] Invalid data type: {data_type}")
        return

    if not remote:
        # Download only the first file for local testing
        files = files[:1]

    if base_dir.endswith('/'):
            base_dir = base_dir[:-1]
        
    if not os.path.exists(base_dir):
        print(f"[ERROR] Base directory {base_dir} does not exist. Please create it first.")
        return
    for file_path in files:
        url = BASE_URL + file_path
        destination_path = os.path.join(base_dir, os.path.basename(file_path))
        print(f"[INFO] Downloading {file_path}...")
        if not download_file(url, destination_path):
            print(f"[ERROR] Failed to download {file_path}. Skipping.")

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

def prepare_data():
    """
    Prepare data for training (e.g., preprocessing or augmentation).
    """
    print("[INFO] Preparing data...")
    # Add data preparation logic here
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
    parser = argparse.ArgumentParser(description="Camelyon Dataset Processing and U-Net Training")
    parser.add_argument("--remote", action="store_true", help="Execute on server (full dataset)")
    parser.add_argument("-d", "--download", action="store_true", help="Download dataset")
    parser.add_argument("-p", "--patch", action="store_true", help="Extract patches")
    parser.add_argument("-prep", "--prepare", action="store_true", help="Prepare data")
    parser.add_argument("-train", "--train", action="store_true", help="Train U-Net model")
    parser.add_argument("-test", "--test", action="store_true", help="Test U-Net model")
    parser.add_argument("--type", type=str, choices=["CAMELYON16", "CAMELYON17"], required=True, help="Dataset type")
    parser.add_argument("--subset", type=str, choices=["normal", "tumor", "center_0"], required=True, help="Subset to download") # TODO: from the time being, only one center
    parser.add_argument("--base_dir", type=str, default="../data", help="Base directory for downloaded files")
    args = parser.parse_args()

    if args.download:
        download_dataset(args.type, args.subset, args.base_dir, remote=args.remote)
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