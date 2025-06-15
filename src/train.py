import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.preprocessing.camelyon16_mil_dataset import Camelyon16MILDataset
from src.models.cnn_encoder import CNNEncoder
from src.models.mil_classifier import MILClassifier
from src.models.mil_pooling import MeanPooling, AttentionPooling
from src.config import Config

def train_model():
    """
    Train the MILClassifier model on the Camelyon dataset.

    Parameters:
    - None

    Returns:
    - None: Saves model checkpoints during training.
    """
    train_dataset = Camelyon16MILDataset(data_dir=Config.DATA_DIR, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    encoder = CNNEncoder()
    pooling = MeanPooling() if Config.USE_MEAN_POOLING else AttentionPooling()
    classifier = MILClassifier(encoder, pooling)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=Config.LEARNING_RATE)

    classifier.train()
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        for bags, labels in train_loader:
            optimizer.zero_grad()
            outputs = classifier(bags)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{Config.EPOCHS}], Loss: {total_loss/len(train_loader):.4f}')

        # Save model checkpoint
        if (epoch + 1) % Config.SAVE_INTERVAL == 0:
            torch.save(classifier.state_dict(), os.path.join(Config.MODEL_DIR, f'model_epoch_{epoch+1}.pth'))

    print("Training complete.")

if __name__ == "__main__":
    train_model()