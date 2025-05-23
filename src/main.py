import os
import torch
from torch.utils.data import DataLoader
from src.dataset.camelyon16_mil_dataset import Camelyon16MILDataset
from src.models.cnn_encoder import CNNEncoder
from src.models.mil_classifier import MILClassifier
from src.models.mil_pooling import MeanPooling, AttentionPooling
from src.train import train_model
from src.eval import evaluate_model
from src.config import Config

def main():
    # Load configuration
    config = Config()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset
    train_dataset = Camelyon16MILDataset(config.train_data_path)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = Camelyon16MILDataset(config.val_data_path)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model components
    cnn_encoder = CNNEncoder().to(device)
    pooling_method = MeanPooling() if config.pooling_method == 'mean' else AttentionPooling()
    mil_classifier = MILClassifier(cnn_encoder, pooling_method).to(device)

    # Train the model
    train_model(mil_classifier, train_loader, device, config)

    # Evaluate the model
    evaluate_model(mil_classifier, val_loader, device, config)

if __name__ == "__main__":
    main()