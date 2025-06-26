import os
from torch.utils.data import Dataset
from PIL import Image

import os
import glob
from torch.utils.data import Dataset
from PIL import Image

import glob

class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
        self.labels = []

        for path in self.image_paths:
            filename = os.path.basename(path)
            if "_tumor" in filename:
                self.labels.append(1)
            elif "_normal" in filename:
                self.labels.append(0)
            else:
                print(f"Warning: No label in {filename}, assuming normal.")
                self.labels.append(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, img_path
