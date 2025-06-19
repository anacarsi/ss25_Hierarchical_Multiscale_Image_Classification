from torch.utils.data import Dataset
from PIL import Image
import os

class PatchDataset(Dataset):
    def __init__(self, patch_dir, transform=None):
        self.transform = transform
        self.samples = []

        # Walk through all subdirectories and collect image paths
        for root, _, files in os.walk(patch_dir):
            for fname in files:
                if fname.endswith(".png"):
                    fpath = os.path.join(root, fname)
                    label = self.get_label_from_path(fpath)
                    self.samples.append((fpath, label))

        print(f"[DEBUG] Found {len(self.samples)} patches in {patch_dir}")

    def get_label_from_path(self, path):
        if "tumor" in path.lower():
            return 1
        else:
            return 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, path
