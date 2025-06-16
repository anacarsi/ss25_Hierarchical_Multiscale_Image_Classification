import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.preprocessing.camelyon16_mil_dataset import Camelyon16MILDataset
from src.models.mil_classifier import MILClassifier
# from src.utils.metrics import calculate_metrics
# from src.utils.uncertainty import estimate_uncertainty
from src.config import Config

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on the given dataset.

    Parameters:
    - model: nn.Module, the trained model to evaluate.
    - dataloader: DataLoader, the data loader for the evaluation dataset.
    - device: torch.device, the device to run the evaluation on.

    Returns:
    - all_predictions: list, predicted labels for the dataset.
    - all_labels: list, ground truth labels for the dataset.
    - uncertainties: list, uncertainty estimates for the predictions.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    uncertainties = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch['images'].to(device), batch['labels'].to(device)
            outputs, _ = model(images)
            predictions = torch.argmax(outputs, dim=1)
            # uncertainty = estimate_uncertainty(outputs)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # uncertainties.extend(uncertainty.cpu().numpy())

    return all_predictions, all_labels

def main():
    """
    Main function to evaluate the model and calculate metrics.

    Parameters:
    - None

    Returns:
    - None: Prints evaluation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()

    test_dataset = Camelyon16MILDataset(config.test_data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = MILClassifier(config).to(device)
    model.load_state_dict(torch.load(config.model_path))

    predictions, labels, uncertainties = evaluate_model(model, test_dataloader, device)

    # metrics = calculate_metrics(predictions, labels)
    # print("Evaluation Metrics:", metrics)
