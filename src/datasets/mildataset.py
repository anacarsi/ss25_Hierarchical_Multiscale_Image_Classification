import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    DEBUG = "\033[96m"
    INFO = "\033[95m"  # pink
    WARNING = "\033[93m"  # yellow
    ERROR = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class WSIMILDDataset(Dataset):
    def __init__(self, features_path, labels_path, paths_path):
        self.features = np.load(features_path, allow_pickle=True)
        self.labels = np.load(labels_path, allow_pickle=True)
        with open(paths_path, "r") as f:
            self.paths = [line.strip() for line in f]

        self.wsi_data = self._group_patches_by_wsi()

    def _group_patches_by_wsi(self):
        wsi_dict = {}
        for i, path in enumerate(self.paths):
            # patch naming convention like "{prefix}_x{x}_y{y}_{label}.png"
            wsi_name = "_".join(
                os.path.basename(path).split("_")[:-2]
            )  # Removes _x{x}_y{y}_{label}.png

            if wsi_name not in wsi_dict:
                wsi_dict[wsi_name] = {
                    "features": [],
                    "patch_labels": [],
                    "wsi_label": 0,
                }

            wsi_dict[wsi_name]["features"].append(self.features[i])
            wsi_dict[wsi_name]["patch_labels"].append(self.labels[i])

            # Determine WSI-level label: if any patch is tumor, the WSI is tumor
            if self.labels[i] == 1:
                wsi_dict[wsi_name]["wsi_label"] = 1

        # Convert lists to tensors for each WSI
        for wsi_name in wsi_dict:
            wsi_dict[wsi_name]["features"] = torch.tensor(
                np.array(wsi_dict[wsi_name]["features"]), dtype=torch.float32
            )
            wsi_dict[wsi_name]["patch_labels"] = torch.tensor(
                np.array(wsi_dict[wsi_name]["patch_labels"]), dtype=torch.long
            )
            wsi_dict[wsi_name]["wsi_label"] = torch.tensor(
                wsi_dict[wsi_name]["wsi_label"], dtype=torch.long
            )

        return list(wsi_dict.values())

    def __len__(self):
        return len(self.wsi_data)

    def __getitem__(self, idx):
        # Returns a dictionary for each WSI, containing its features (bag) and the WSI-level label
        return self.wsi_data[idx]["features"], self.wsi_data[idx]["wsi_label"]


class WSIMILDDataset(Dataset):
    def __init__(self, feature_base_dir):
        """
        Initializes the dataset for MIL.
        Parameters:
            feature_base_dir (str): Base directory where WSI features (npy files) are saved.
                                    This directory contains subdirectories for each WSI's features,
                                    e.g., features_base_dir/slide_001_features.npy, slide_001_label.npy etc.
        """
        self.feature_base_dir = feature_base_dir
        self.wsi_data_info = (
            []
        )  # List of dictionaries: {'feature_path': ..., 'label_path': ...}

        # Find all WSI feature files in the directory
        wsi_names = set()
        for filename in os.listdir(feature_base_dir):
            if filename.endswith("_features.npy"):
                wsi_name = filename.replace("_features.npy", "")
                wsi_names.add(wsi_name)

        for wsi_name in sorted(list(wsi_names)):  # Sort for consistent order
            feature_path = os.path.join(feature_base_dir, f"{wsi_name}_features.npy")
            label_path = os.path.join(feature_base_dir, f"{wsi_name}_label.npy")

            if os.path.exists(feature_path) and os.path.exists(label_path):
                self.wsi_data_info.append(
                    {
                        "wsi_name": wsi_name,
                        "feature_path": feature_path,
                        "label_path": label_path,
                    }
                )
            else:
                print(
                    f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Missing feature or label file for WSI: {wsi_name}. Skipping."
                )

        if not self.wsi_data_info:
            raise ValueError(
                f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} No valid WSI feature files found in {feature_base_dir}"
            )

        print(
            f"{bcolors.INFO}[INFO]{bcolors.ENDC} WSIMILDDataset initialized with {len(self.wsi_data_info)} WSIs."
        )

    def __len__(self):
        return len(self.wsi_data_info)

    def __getitem__(self, idx):
        wsi_info = self.wsi_data_info[idx]

        features = np.load(wsi_info["feature_path"], allow_pickle=True)
        wsi_label = np.load(
            wsi_info["label_path"], allow_pickle=True
        ).item()  # .item() to get scalar

        # Convert to torch tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(wsi_label, dtype=torch.long)

        return features_tensor, label_tensor


def get_mil_dataloaders(feature_base_dir, batch_size=1, test_ratio=0.2):
    full_dataset = WSIMILDDataset(feature_base_dir)

    wsi_indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        wsi_indices, test_size=test_ratio, random_state=42
    )

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    return train_loader, val_loader
