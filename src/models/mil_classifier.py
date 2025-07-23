import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Removed predict and uncertainty_estimation as they use np.exp and are for post-training analysis
# For inference, we use model(bag)[0].softmax(dim=-1)
class MILAttentionPooling(nn.Module):
    def __init__(self, in_dim, attn_dim=128):
        super().__init__()
        self.attn_V = nn.Linear(in_dim, attn_dim)
        self.attn_U = nn.Linear(attn_dim, 1)

    def forward(self, x):
        A = torch.tanh(self.attn_V(x))
        A = self.attn_U(A)
        A = torch.softmax(A, dim=0)
        M = torch.sum(A * x, dim=0)
        return M, A


class MILAttentionPooling(nn.Module):
    """
    Attention-based pooling as in Ilse et al. (ABMIL)
    """

    def __init__(self, in_dim, attn_dim=128):
        super().__init__()
        self.attn_V = nn.Linear(in_dim, attn_dim)
        self.attn_U = nn.Linear(attn_dim, 1)

    def forward(self, x):
        # x: (num_instances, feature_dim)
        A = torch.tanh(self.attn_V(x))  # (N, attn_dim)
        A = self.attn_U(A)  # (N, 1)
        A = torch.softmax(A, dim=0)  # (N, 1)
        M = torch.sum(A * x, dim=0)  # (feature_dim,)
        return M, A  # Return pooled feature and attention weights


class GatedAttentionPooling(nn.Module):
    def __init__(self, in_dim, attn_dim=128):
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim)
        self.U = nn.Linear(in_dim, attn_dim)
        self.attn = nn.Linear(attn_dim, 1)

    def forward(self, x):
        # Gated mechanism: tanh(Vx) * sigmoid(Ux)
        A = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        A = self.attn(A)
        A = A - A.max(dim=0, keepdim=True)[0]
        A = self.dropout(A)
        A = torch.softmax(A, dim=0)
        M = torch.sum(A * x, dim=0)
        return M, A


class MILClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes=2, pooling="attention"):
        super().__init__()
        self.pooling = pooling
        if pooling == "attention":
            self.aggregator = MILAttentionPooling(feature_dim)
        elif pooling == "gated_attention":
            self.aggregator = GatedAttentionPooling(feature_dim)
        elif pooling == "mean":
            self.aggregator = lambda x: (x.mean(dim=0), None)
        elif pooling == "max":
            self.aggregator = lambda x: (x.max(dim=0)[0], None)
        else:
            raise ValueError("Unknown pooling: choose from 'attention', 'mean', 'max'")
        """self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            # nn.LayerNorm(128),  # added layer norm, gotta try
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )"""
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, bag):
        """
        bag: Tensor of shape (num_patches, feature_dim)
        Returns: logits (num_classes), attention_weights (or None)
        """
        pooled, attn = self.aggregator(bag)
        logits = self.classifier(pooled)
        return logits, attn

    def predict(self, bags):
        logits = self.forward(bags)
        probabilities = self.softmax(logits)
        return probabilities

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
