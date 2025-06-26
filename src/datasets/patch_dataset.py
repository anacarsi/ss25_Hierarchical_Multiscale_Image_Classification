import os
from torch.utils.data import Dataset
from PIL import Image

class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None, slide_names=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for slide_name in os.listdir(root_dir):
            if slide_names and slide_name not in slide_names:
                continue

            slide_path = os.path.join(root_dir, slide_name)
            if os.path.isdir(slide_path):
                for patch_file in os.listdir(slide_path):
                    if patch_file.endswith(".png"):
                        self.image_paths.append(os.path.join(slide_path, patch_file))
                        if "_tumor" in patch_file:
                            self.labels.append(1)
                        elif "_normal" in patch_file:
                            self.labels.append(0)
                        else:
                            print(f"Warning: Unlabeled patch {patch_file}, assuming normal.")
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
