# Restnet18 to convert each patch in a feature vector
# Reference: https://discuss.pytorch.org/t/use-resnet18-as-feature-extractor/8267

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, weight_path="./resnet18_patch_classifier.pth"):
        super().__init__()
        resnet = models.resnet18(weights=None)  # Start with uninitialized weights
        if weight_path and os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location="cpu")
            # Remove the classifier layer weights
            state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
            resnet.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded fine-tuned weights from {weight_path}")
        else:
            print("[WARNING] Using ImageNet weights (not fine-tuned)")

        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


class ResNet18Classifier(nn.Module):
    """
    ResNet18 model for binary classification of patches.
    """

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(
            pretrained=True
        )  # use pretrained weights on imagenet
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)  # binary classification

    def forward(self, x):
        """
        Forward pass through the ResNet18 model to extract 512 (or more) dimensional feature vector.
        Parameters:
            x (torch.Tensor): Input tensor of shape (B, 3, 224, 224) where B is batch size.
        Returns:
            torch.Tensor: Output tensor of shape (B, 2) for binary classification.
        """
        return self.model(x)


def extract_features_for_slide(slide_dir, model, transform, device, save_path):
    patch_files = sorted([f for f in os.listdir(slide_dir) if f.endswith(".png")])
    features = []

    for fname in patch_files:
        img_path = os.path.join(slide_dir, fname)
        image = Image.open(img_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(tensor)
        features.append(feat.cpu())

    if features:
        slide_tensor = torch.cat(features, dim=0)  # Shape: (N_patches, 512)
        torch.save(slide_tensor, save_path)
        print(f"[INFO] Saved {slide_tensor.shape[0]} patch features to {save_path}")
    else:
        print(f"[WARNING] No patches found in {slide_dir}")


def run_feature_extraction(
    patch_root="data/camelyon16/patches/level_0", save_root="data/features"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18FeatureExtractor().to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    os.makedirs(save_root, exist_ok=True)
    slide_dirs = [
        d for d in os.listdir(patch_root) if os.path.isdir(os.path.join(patch_root, d))
    ]

    for slide_id in slide_dirs:
        slide_path = os.path.join(patch_root, slide_id)
        save_path = os.path.join(save_root, f"{slide_id}.pt")
        extract_features_for_slide(slide_path, model, transform, device, save_path)
