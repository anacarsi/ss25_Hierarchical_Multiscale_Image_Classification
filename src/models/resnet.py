# Restnet18 to convert each patch in a feature vector
# Reference: https://discuss.pytorch.org/t/use-resnet18-as-feature-extractor/8267

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


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


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, weight_path="resnet18_patch_classifier__level3.pth"):
        super().__init__()
        resnet = models.resnet18(weights=None)  # Start with uninitialized weights
        weight_path = os.path.join(os.getcwd(), "src", "models", weight_path)
        if weight_path and os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location="cpu")
            # Remove the classifier layer weights
            state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
            resnet.load_state_dict(state_dict, strict=False)

        else:
            print("[WARNING] Using ImageNet weights (not fine-tuned)")

        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # we're only using the feature extractor part


class UnifiedResNet(nn.Module):
    def __init__(self, pretrained_weights_path=None, classifier=False):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Identity()
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            state_dict = torch.load(pretrained_weights_path, map_location="cpu")
            state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
            self.model.load_state_dict(state_dict, strict=False)
        if classifier:
            self.model.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.model(x)


class ResNet18Classifier(nn.Module):
    """
    ResNet18 model for binary classification of patches.
    """

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
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


# ------------------- ResNet18 Classifier w Pretrained SIMCLR -------------------
class ResNet18ClassifierSIMCLR(nn.Module):
    def __init__(self, pretrained_weights_path=None, num_classes=2):
        super().__init__()
        self.encoder = models.resnet18(pretrained=False)
        self.encoder.fc = nn.Identity()  # SimCLR doesn't have a classifier head

        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            state_dict = torch.load(pretrained_weights_path, map_location="cpu")

            if pretrained_weights_path and os.path.exists(pretrained_weights_path):
                state_dict = torch.load(pretrained_weights_path, map_location="cpu")

                # Remove "encoder." prefix, skip "projector." keys
                new_state_dict = {
                    k.replace("encoder.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("encoder.")
                }

                missing, unexpected = self.encoder.load_state_dict(new_state_dict, strict=False)
                print(f"[INFO] Loaded weights with {len(missing)} missing and {len(unexpected)} unexpected keys.")


            missing, unexpected = self.encoder.load_state_dict(new_state_dict, strict=False)
            print(f"[INFO] Loaded weights with {len(missing)} missing and {len(unexpected)} unexpected keys.")

        # Final classification head
        self.encoder.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.encoder(x)

