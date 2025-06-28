import numpy as np

class MILClassifier:
    def __init__(self, encoder, pooling, num_classes):
        self.encoder = encoder
        self.pooling = pooling
        self.num_classes = num_classes

    def forward(self, bags):
        # bags: list of patches for each bag
        encoded_bags = [self.encoder(patches) for patches in bags]
        pooled_features = self.pooling(encoded_bags)
        logits = self.classify(pooled_features)
        return logits

    def classify(self, pooled_features):
        # fully connected layer for classification
        return self.fc_layer(pooled_features)

    def fc_layer(self, pooled_features):
        
        pass

    def predict(self, bags):
        logits = self.forward(bags)
        probabilities = self.softmax(logits)
        return probabilities

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def uncertainty_estimation(self, logits):
        # uncertainty estimation techniques here
        pass
