import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(CNNEncoder, self).__init__()
        # Use a pre-trained ResNet model as the backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        # Add a new fully connected layer for feature extraction
        self.fc = nn.Linear(self.backbone[-1].in_features, 512)

    def forward(self, x):
        # Forward pass through the backbone
        with torch.no_grad():
            features = self.backbone(x)
        # Flatten the features
        features = features.view(features.size(0), -1)
        # Pass through the new fully connected layer
        features = self.fc(features)
        return features

    def get_feature_dimension(self):
        return 512  # Output dimension of the feature vector