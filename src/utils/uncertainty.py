import torch
import torch.nn.functional as F


def softmax_thresholding(logits, threshold=0.5):
    """
    Applies softmax to the logits and thresholds the probabilities.

    Parameters:
        logits (torch.Tensor): The raw output logits from the model.
        threshold (float): The threshold for classification.

    Returns:
        torch.Tensor: Binary classification results based on the threshold.
    """
    probabilities = F.softmax(logits, dim=1)
    predictions = (probabilities[:, 1] > threshold).float()
    return predictions, probabilities


def monte_carlo_dropout(model, x, n_samples=100):
    """
    Estimates uncertainty using Monte Carlo Dropout.

    Parameters:
        model (torch.nn.Module): The model with dropout layers.
        x (torch.Tensor): Input data for which to estimate uncertainty.
        n_samples (int): Number of forward passes to perform.

    Returns:
        torch.Tensor: Mean predictions from the model.
        torch.Tensor: Uncertainty estimates (variance).
    """
    model.train()  # Set the model to training mode to enable dropout
    predictions = []

    for _ in range(n_samples):
        with torch.no_grad():
            preds = model(x)
            predictions.append(preds.unsqueeze(0))

    predictions = torch.cat(predictions, dim=0)
    mean_predictions = predictions.mean(dim=0)
    uncertainty = predictions.var(dim=0)

    return mean_predictions, uncertainty
