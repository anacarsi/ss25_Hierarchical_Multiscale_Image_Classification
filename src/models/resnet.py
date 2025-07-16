# Restnet18 to convert each patch in a feature vector
# Reference: https://discuss.pytorch.org/t/use-resnet18-as-feature-extractor/8267

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, ResNet50_Weights
import os


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    DEBUG = "\033[96m"
    INFO = "\033[95m"
    WARNING = "\033[93m"
    ERROR = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class ResNetFeatureExtractor(nn.Module):
    """
    Feature extractor using ResNet18 or ResNet50 backbone.
    Parameters:
    - model_type (str): 'resnet18' or 'resnet50'.
    - trained_classifier_weights_path (str or None): Path to trained classifier weights to load.
    - simclr_trained (bool): Whether the model was trained with SimCLR.
    """

    def __init__(
        self,
        model_type="resnet18",  # 'resnet18' or 'resnet50'
        trained_classifier_weights_path=None,
        simclr_trained=False,
    ):
        super().__init__()
        assert model_type in ["resnet18", "resnet50"], "Only resnet18/50 supported"
        self.model_type = model_type
        if model_type == "resnet18":
            base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.feature_dim = 512
        elif model_type == "resnet50":
            base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.feature_dim = 2048
        else:
            raise ValueError("Invalid model type")

        self.features = nn.Sequential(*list(base_model.children())[:-1])

        if trained_classifier_weights_path and os.path.exists(
            trained_classifier_weights_path
        ):
            state_dict = torch.load(trained_classifier_weights_path, map_location="cpu")
            try:
                self.features.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"[WARNING] Could not load weights into feature extractor: {e}")

    def forward(self, x):
        """
        Forward pass to extract features from input images.
        Parameters:
        - x (torch.Tensor): Input tensor of shape (B, 3, 224, 224).
        Returns:
        - torch.Tensor: Output tensor of shape (B, feature_dim).
        """
        x = self.features(x)
        return x.view(x.size(0), -1)  # (batch_size, feature_dim)


class ResNetClassifier(nn.Module):
    """
    ResNet18 or ResNet50 model for binary classification of patches.
    Parameters:
    - model_type (str): 'resnet18' or 'resnet50'.
    """

    def __init__(self, model_type="resnet18"):
        super().__init__()
        if model_type == "resnet18":
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.feature_dim = 512
        elif model_type == "resnet50":
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.feature_dim = 2048
        else:
            raise ValueError("Invalid model type")
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)  # binary classification

    def forward(self, x):
        """
        Forward pass through the ResNet model for binary classification.
        Parameters:
        - x (torch.Tensor): Input tensor of shape (B, 3, 224, 224).
        Returns:
        - torch.Tensor: Output tensor of shape (B, 2) for binary classification.
        """
        return self.model(x)


class ResNet18ClassifierSIMCLR(nn.Module):
    """
    ResNet18 classifier for use with SimCLR pretraining.
    Parameters:
    - pretrained_weights_path (str or None): Path to SimCLR-pretrained weights.
    - num_classes (int): Number of output classes (default 2).
    - freeze_encoder (bool): Whether to freeze encoder layers (for phase 1 fine-tuning).
    """

    def __init__(
        self, pretrained_weights_path=None, num_classes=2, freeze_encoder=True
    ):
        super().__init__()
        self.encoder = models.resnet18(pretrained=False)
        self.encoder.fc = nn.Identity()  # SimCLR doesn't have a classifier head

        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            state_dict = torch.load(pretrained_weights_path, map_location="cpu")
            new_state_dict = {
                k.replace("encoder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("encoder.")
            }
            missing, unexpected = self.encoder.load_state_dict(
                new_state_dict, strict=False
            )
            print(
                f"[INFO] Loaded weights with {len(missing)} missing and {len(unexpected)} unexpected keys from SIMCLR."
            )
        else:
            print(
                "[WARNING] No pre-trained SimCLR weights found. Starting from scratch or ImageNet if not 'pretrained=False'."
            )
            self.encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.encoder.fc = nn.Identity()

        # Freeze encoder layers if specified for phase 1 fine-tuning
        if freeze_encoder:
            print(
                f"{bcolors.INFO}[INFO]{bcolors.ENDC} Freezing encoder layers for fine-tuning phase."
            )
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            print(
                f"{bcolors.INFO}[INFO]{bcolors.ENDC} Encoder layers are trainable (unfrozen)."
            )
            for param in self.encoder.parameters():
                param.requires_grad = True

        self.encoder.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.6),  # TODO: check this dropout rate
            nn.Linear(256, num_classes),
        )
        for param in self.encoder.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Forward pass through the SimCLR-based ResNet18 classifier.
        Parameters:
        - x (torch.Tensor): Input tensor of shape (B, 3, 224, 224).
        Returns:
        - torch.Tensor: Output tensor of shape (B, num_classes).
        """
        return self.encoder(x)
