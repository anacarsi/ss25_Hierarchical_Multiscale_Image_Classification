import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset.camelyon16_mil_dataset import Camelyon16MILDataset
from src.models.mil_classifier import MILClassifier
from src.utils.metrics import calculate_metrics
from src.utils.uncertainty import estimate_uncertainty
from src.config import Config

def evaluate_model(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    uncertainties = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch['images'].to(device), batch['labels'].to(device)
            outputs, _ = model(images)
            predictions = torch.argmax(outputs, dim=1)
            uncertainty = estimate_uncertainty(outputs)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            uncertainties.extend(uncertainty.cpu().numpy())

    return all_predictions, all_labels, uncertainties

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()

    # Load dataset
    test_dataset = Camelyon16MILDataset(config.test_data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model
    model = MILClassifier(config).to(device)
    model.load_state_dict(torch.load(config.model_path))

    # Evaluate the model
    predictions, labels, uncertainties = evaluate_model(model, test_dataloader, device)

    # Calculate metrics
    metrics = calculate_metrics(predictions, labels)
    print("Evaluation Metrics:", metrics)

if __name__ == "__main__":
    main()