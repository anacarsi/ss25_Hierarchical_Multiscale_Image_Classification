from torch.utils.data import Dataset
from PIL import Image
import os

class PatchDataset(Dataset):
    def __init__(self, patch_dir, transform=None):
        self.patch_paths = []
        for label in ['normal', 'tumor']:
            label_dir = os.path.join(patch_dir, label)
            if os.path.exists(label_dir):
                self.patch_paths += [(os.path.join(label_dir, f), label)
                                     for f in os.listdir(label_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        img_path, label = self.patch_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # Convert label to index: 'tumor' -> 1, 'normal' -> 0
        label_idx = 0 if label == 'normal' else 1
        return img, label_idx, img_path