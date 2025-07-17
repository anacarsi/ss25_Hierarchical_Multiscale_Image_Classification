import os
import glob
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict
import random
from collections import Counter  # For counting class distributions


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


class PatchDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        tumor_transform=None,
        normal_transform=None,
        balanced=False,
        max_samples=None,
        slide_names=None,
    ):
        self.tumor_transform = (
            tumor_transform if tumor_transform is not None else transform
        )
        self.normal_transform = (
            normal_transform if normal_transform is not None else transform
        )
        self.transform = transform  # for backward compatibility
        self.image_paths = []
        self.labels = []
        self.label_map = {"_normal": 0, "_tumor": 1}
        # print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Initializing PatchDataset from {root_dir}")
        # Collect samples by class
        class_to_paths = defaultdict(list)
        for path in glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True):
            # print(f"{bcolors.DEBUG}[DEBUG]{bcolors.ENDC} Found path: {path}")
            if slide_names is not None:
                # Extract slide_dir as the immediate parent folder of the patch file
                slide_dir = os.path.basename(os.path.dirname(path))
                # print(f"{bcolors.DEBUG}[DEBUG]{bcolors.ENDC} Checking slide directory: {path}")
                if slide_dir not in slide_names:
                    # print(f"{bcolors.DEBUG}[DEBUG]{bcolors.ENDC} Skipping path {slide_dir} not in slide_names.")
                    continue
            filename = os.path.basename(path)
            # print(f"{bcolors.DEBUG}[DEBUG]{bcolors.ENDC} Processing file: {filename}")
            if "_tumor" in filename:
                class_to_paths[1].append(path)
                # print(f"{bcolors.DEBUG}[DEBUG]{bcolors.ENDC} Adding path to tumor class: {filename}")
            elif "_normal" in filename:
                class_to_paths[0].append(path)
                # print(f"{bcolors.DEBUG}[DEBUG]{bcolors.ENDC} Adding path to normal class: {filename}")
            else:
                print(
                    f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Could not determine label from filename: {filename}"
                )

        # Balance the dataset
        if balanced:
            min_count = (
                min(len(paths) for paths in class_to_paths.values())
                if class_to_paths
                else 0
            )
            if min_count == 0:
                print(
                    f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} No patches found for balancing."
                )
            for label, paths in class_to_paths.items():
                if max_samples:
                    count = min(min_count, max_samples)
                else:
                    count = min_count
                sampled = random.sample(paths, min(count, len(paths)))
                self.image_paths.extend(sampled)
                self.labels.extend([label] * len(sampled))
        else:
            # Check class_tp_paths not empty
            if not class_to_paths:
                print(
                    f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} No patches found in {root_dir}."
                )
            for label, paths in class_to_paths.items():
                print(
                    f"{bcolors.DEBUG}[DEBUG]{bcolors.ENDC} Adding {len(paths)} paths for label {label}"
                )
                if max_samples:
                    paths = random.sample(paths, min(len(paths), max_samples))
                self.image_paths.extend(paths)
                self.labels.extend([label] * len(paths))

        # Shuffle dataset
        if self.image_paths:
            combined = list(zip(self.image_paths, self.labels))
            random.shuffle(combined)
            self.image_paths, self.labels = zip(*combined)
            self.image_paths = list(self.image_paths)
            self.labels = list(self.labels)

        label_counts = Counter(self.labels)
        print(
            f"{bcolors.INFO}[INFO]{bcolors.ENDC} PatchDataset initialized: {len(self.labels)} total patches."
        )
        print(
            f"{bcolors.INFO}[INFO]{bcolors.ENDC} Tumor patches: {label_counts.get(1, 0)} | Normal patches: {label_counts.get(0, 0)}"
        )
        print(
            f"{bcolors.INFO}[INFO]{bcolors.ENDC} Label distribution: {dict(label_counts)}"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        # 0 = normal, 1 = tumor
        if label == 1 and self.tumor_transform:
            image = self.tumor_transform(image)
        elif label == 0 and self.normal_transform:
            image = self.normal_transform(image)
        elif self.transform:
            image = self.transform(image)
        return image, label, img_path

    def get_class_counts(self):
        from collections import Counter

        return dict(Counter(self.labels))
