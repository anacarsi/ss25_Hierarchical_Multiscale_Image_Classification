""" 
CAMELYON was the first challenge using WSIs in computational pathology.
Aims to help pathologists identify breast cancer metastases in sentinel lymph nodes. 
Lymph node metastases are extremely important to find, as they indicate that the cancer is no longer localized and systemic treatment might be warranted.
"""

import os
from torch.utils.data import Dataset
from PIL import Image

class Camelyon16MILDataset(Dataset):
    def __init__(self, patch_dir, labels, transform=None):
        self.patch_dir = patch_dir
        self.labels = labels  # dict: {slide_id: label}
        self.transform = transform
        self.slide_ids = list(labels.keys())

    def _create_bags(self):
        bags = []
        # Logic to load WSI files and create bags of patches
        # Each bag corresponds to a Whole Slide Image and contains its patches
        # Example: bags.append((patches, label))
        return bags

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx] 
        patch_paths = [os.path.join(self.patch_dir, slide_id, f) 
                       for f in os.listdir(os.path.join(self.patch_dir, slide_id))]
        patches = [Image.open(p) for p in patch_paths]
        if self.transform:
            patches = [self.transform(p) for p in patches]
        label = self.labels[slide_id]
        return patches, label

    def get_labels(self):
        return [self.labels[slide_id] for slide_id in self.slide_ids]