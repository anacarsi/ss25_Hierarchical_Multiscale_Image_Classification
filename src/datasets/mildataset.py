import numpy as np
import os
import torch
from torch.utils.data import Dataset

class WSIMILDDataset(Dataset):
    def __init__(self, features_path, labels_path, paths_path):
        self.features = np.load(features_path, allow_pickle=True)
        self.labels = np.load(labels_path, allow_pickle=True)
        with open(paths_path, 'r') as f:
            self.paths = [line.strip() for line in f]

        self.wsi_data = self._group_patches_by_wsi()

    def _group_patches_by_wsi(self):
        wsi_dict = {}
        for i, path in enumerate(self.paths):
            # Extract WSI name from patch path (e.g., "slide_001_x10_y10_normal.png" -> "slide_001")
            # Assuming your patch naming convention like "{prefix}_x{x}_y{y}_{label}.png"
            wsi_name = '_'.join(os.path.basename(path).split('_')[:-2]) # Removes _x{x}_y{y}_{label}.png

            if wsi_name not in wsi_dict:
                wsi_dict[wsi_name] = {'features': [], 'patch_labels': [], 'wsi_label': 0}

            wsi_dict[wsi_name]['features'].append(self.features[i])
            wsi_dict[wsi_name]['patch_labels'].append(self.labels[i])

            # Determine WSI-level label: if any patch is tumor, the WSI is tumor
            if self.labels[i] == 1:
                wsi_dict[wsi_name]['wsi_label'] = 1

        # Convert lists to tensors for each WSI
        for wsi_name in wsi_dict:
            wsi_dict[wsi_name]['features'] = torch.tensor(np.array(wsi_dict[wsi_name]['features']), dtype=torch.float32)
            wsi_dict[wsi_name]['patch_labels'] = torch.tensor(np.array(wsi_dict[wsi_name]['patch_labels']), dtype=torch.long)
            wsi_dict[wsi_name]['wsi_label'] = torch.tensor(wsi_dict[wsi_name]['wsi_label'], dtype=torch.long)

        return list(wsi_dict.values())

    def __len__(self):
        return len(self.wsi_data)

    def __getitem__(self, idx):
        # Returns a dictionary for each WSI, containing its features (bag) and the WSI-level label
        return self.wsi_data[idx]['features'], self.wsi_data[idx]['wsi_label']