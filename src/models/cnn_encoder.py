import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(CNNEncoder, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        self.fc = nn.Linear(self.backbone[-1].in_features, 512)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

    def get_feature_dimension(self):
        return 512  