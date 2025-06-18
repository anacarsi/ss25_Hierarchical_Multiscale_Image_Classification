# Restnet18 to convert each patch in a feature vector
# Reference: https://discuss.pytorch.org/t/use-resnet18-as-feature-extractor/8267

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Output: (batch, 512, 1, 1)

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # Flatten to (batch, 512)
