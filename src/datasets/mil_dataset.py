import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import glob


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
    def __init__(self, feature_dir):
        """
        Generic Dataset for MIL with separated directories.
        Assumes *_features.npy and *_label.npy pairs.
        """
        self.feature_dir = feature_dir
        self.wsi_data_info = []

        wsi_names = set()
        for filename in os.listdir(feature_dir):
            if filename.endswith("_features.npy"):
                wsi_name = filename.replace("_features.npy", "")
                wsi_names.add(wsi_name)

        for wsi_name in sorted(wsi_names):
            feature_path = os.path.join(feature_dir, f"{wsi_name}_features.npy")
            label_path = os.path.join(feature_dir, f"{wsi_name}_label.npy")

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
                    f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Missing file(s) for {wsi_name}"
                )

        if not self.wsi_data_info:
            raise ValueError(
                f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} No valid data in {feature_dir}"
            )

        print(
            f"{bcolors.INFO}[INFO]{bcolors.ENDC} WSIMILDDataset loaded {len(self.wsi_data_info)} WSIs from {feature_dir}"
        )

    def __len__(self):
        return len(self.wsi_data_info)

    def __getitem__(self, idx):
        info = self.wsi_data_info[idx]
        features = np.load(info["feature_path"], allow_pickle=True)
        label = np.load(info["label_path"], allow_pickle=True).item()

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return features_tensor, label_tensor, info["wsi_name"]


class WSIMILTestDataset(Dataset):
    def __init__(self, feature_dir):
        """
        Separate test dataset class (if you want to track WSI names during inference).
        """
        self.feature_dir = feature_dir
        self.wsi_data_info = []

        for feature_path in sorted(
            glob.glob(os.path.join(feature_dir, "*_features.npy"))
        ):
            wsi_name = os.path.basename(feature_path).replace("_features.npy", "")
            label_path = os.path.join(feature_dir, f"{wsi_name}_label.npy")

            if not os.path.exists(label_path):
                print(
                    f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Skipping {wsi_name}, missing label."
                )
                continue

            self.wsi_data_info.append(
                {
                    "wsi_name": wsi_name,
                    "feature_path": feature_path,
                    "label_path": label_path,
                }
            )

        if not self.wsi_data_info:
            raise ValueError(
                f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} No test data in {feature_dir}"
            )

        print(
            f"{bcolors.INFO}[INFO]{bcolors.ENDC} WSIMILTestDataset loaded {len(self.wsi_data_info)} WSIs."
        )

    def __len__(self):
        return len(self.wsi_data_info)

    def __getitem__(self, idx):
        info = self.wsi_data_info[idx]
        features = np.load(info["feature_path"], allow_pickle=True)
        label = np.load(info["label_path"], allow_pickle=True).item()

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return features_tensor, label_tensor, info["wsi_name"]


def get_mil_dataloaders(
    feature_base_dir_train, feature_base_dir_val, feature_base_dir_test, batch_size=1
):
    train_dataset = WSIMILDDataset(feature_base_dir_train)
    val_dataset = WSIMILDDataset(feature_base_dir_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    test_loader = None
    if feature_base_dir_test is not None:
        test_dataset = WSIMILTestDataset(feature_base_dir_test)
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, pin_memory=True
        )
    print(
        f"{bcolors.INFO}[INFO]{bcolors.ENDC} len: {len(train_loader.dataset)}, {len(val_loader.dataset)}, {len(test_loader.dataset) if test_loader else 0}"
    )

    return train_loader, val_loader, test_loader
