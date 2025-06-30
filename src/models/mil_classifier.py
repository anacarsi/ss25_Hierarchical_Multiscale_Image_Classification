import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class MILAttentionPooling(nn.Module):
    """Attention-based pooling as in Ilse et al. (ABMIL)"""
    def __init__(self, in_dim, attn_dim=128):
        super().__init__()
        self.attn_V = nn.Linear(in_dim, attn_dim)
        self.attn_U = nn.Linear(attn_dim, 1)

    def forward(self, x):
        # x: (num_instances, feature_dim)
        A = torch.tanh(self.attn_V(x))  # (N, attn_dim)
        A = self.attn_U(A)              # (N, 1)
        A = torch.softmax(A, dim=0)     # (N, 1)
        M = torch.sum(A * x, dim=0)     # (feature_dim,)
        return M, A                     # Return pooled feature and attention weights

class MILClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes=2, pooling='attention'):
        super().__init__()
        self.pooling = pooling
        if pooling == 'attention':
            self.aggregator = MILAttentionPooling(feature_dim)
        elif pooling == 'mean':
            self.aggregator = lambda x: (x.mean(dim=0), None)
        elif pooling == 'max':
            self.aggregator = lambda x: (x.max(dim=0)[0], None)
        else:
            raise ValueError("Unknown pooling: choose from 'attention', 'mean', 'max'")
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
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

    def uncertainty_estimation(self, logits):
        # TODOOOOOOOOOO
        pass
