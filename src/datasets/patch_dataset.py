import os
import glob
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict
import random

class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None, balanced=False, max_samples=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {"_normal": 0, "_tumor": 1}

        # Collect samples by class
        class_to_paths = defaultdict(list)
        for path in glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True):
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

    def get_class_counts(self):
        from collections import Counter
        return dict(Counter(self.labels))
