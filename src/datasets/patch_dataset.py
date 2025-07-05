import os
import glob
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict
import random

class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None, tumor_transform=None, normal_transform=None, balanced=False, max_samples=None, slide_names=None):
        self.tumor_transform = tumor_transform if tumor_transform is not None else transform
        self.normal_transform = normal_transform if normal_transform is not None else transform
        self.transform = transform  # for backward compatibility
        self.image_paths = []
        self.labels = []
        self.label_map = {"_normal": 0, "_tumor": 1}

        # Collect samples by class
        class_to_paths = defaultdict(list)
        for path in glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True):
            # If slide_names is provided, only include patches from those slides
            if slide_names is not None:
                # slide_name is the immediate subdirectory under root_dir
                rel_path = os.path.relpath(path, root_dir)
                slide_dir = rel_path.split(os.sep)[0]
                if slide_dir not in slide_names:
                    continue
            filename = os.path.basename(path)
            if "_tumor" in filename:
                class_to_paths[1].append(path)
            elif "_normal" in filename:
                class_to_paths[0].append(path)
            else:
                print(f"[WARNING] Could not determine label from filename: {filename}")

        # Balance the dataset
        if balanced:
            min_count = min(len(paths) for paths in class_to_paths.values())
            for label, paths in class_to_paths.items():
                if max_samples:
                    count = min(min_count, max_samples)
                else:
                    count = min_count
                sampled = random.sample(paths, min(count, len(paths)))
                self.image_paths.extend(sampled)
                self.labels.extend([label] * len(sampled))
        else:
            for label, paths in class_to_paths.items():
                if max_samples:
                    paths = random.sample(paths, min(len(paths), max_samples))
                self.image_paths.extend(paths)
                self.labels.extend([label] * len(paths))

        # Shuffle dataset
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)
        self.image_paths = list(self.image_paths)
        self.labels = list(self.labels)

        # Print summary of dataset
        from collections import Counter
        label_counts = Counter(self.labels)
        print(f"[INFO] PatchDataset initialized: {len(self.labels)} total patches.")
        print(f"[INFO] Tumor patches: {label_counts.get(1, 0)} | Normal patches: {label_counts.get(0, 0)}")
        print(f"[INFO] Label distribution: {dict(label_counts)}")

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
