import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module


class MeanPooling:
    def __init__(self):
        pass

    def forward(self, features):
        """
        Apply mean pooling to the extracted features.

        Args:
            features (torch.Tensor): A tensor of shape (batch_size, num_patches, feature_dim).

        Returns:
            torch.Tensor: Pooled features of shape (batch_size, feature_dim).
        """
        return features.mean(dim=1)


class AttentionPooling:
    def __init__(self, feature_dim):
        self.attention_weights = torch.nn.Parameter(torch.randn(feature_dim))

    def forward(self, features):
        """
        Apply attention-based pooling to the extracted features.

        Args:
            features (torch.Tensor): A tensor of shape (batch_size, num_patches, feature_dim).

        Returns:
            torch.Tensor: Pooled features of shape (batch_size, feature_dim).
        """
        attention_scores = torch.matmul(
            features, self.attention_weights.unsqueeze(0).unsqueeze(2)
        )
        attention_weights = torch.softmax(attention_scores, dim=1)
        pooled_features = (features * attention_weights).sum(dim=1)
        return pooled_features


def get_pooling_method(method_name, feature_dim):
    if method_name == "mean":
        return MeanPooling()
    elif method_name == "attention":
        return AttentionPooling(feature_dim)
    else:
        raise ValueError(f"Unknown pooling method: {method_name}")
