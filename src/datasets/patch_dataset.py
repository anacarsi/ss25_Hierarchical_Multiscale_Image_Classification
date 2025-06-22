from torch.utils.data import Dataset
from PIL import Image
import os

class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.patches = []
        self.labels = [] # 0 for normal, 1 for tumor
        self.paths = []

        # Load patch paths and labels
        for prefix_dir in os.listdir(root_dir):
            full_prefix_dir = os.path.join(root_dir, prefix_dir)
            if os.path.isdir(full_prefix_dir):
                for patch_name in os.listdir(full_prefix_dir):
                    if patch_name.endswith(".png"):
                        label = 1 if "_tumor.png" in patch_name else 0 # Assuming naming convention
                        self.patches.append(os.path.join(full_prefix_dir, patch_name))
                        self.labels.append(label)
                        self.paths.append(os.path.join(prefix_dir, patch_name)) # Store relative path for features

        print(f"Loaded {len(self.patches)} patches from {root_dir}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_path = self.patches[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        path = self.paths[idx]

        if self.transform:
            image = self.transform(image)
        return image, label, path