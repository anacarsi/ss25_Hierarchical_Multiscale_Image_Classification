import torch.nn as nn
import torch
from .datasets.patch_dataset import PatchDataset
from torch.utils.data import DataLoader
from .models.resnet import ResNet18Classifier


def train_resnet_classifier(data_dir, epochs=5, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PatchDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = ResNet18Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Acc: {100 * correct / total:.2f}%"
        )

    torch.save(model.state_dict(), "resnet18_patch_classifier.pth")
    print("[INFO] Training complete. Model saved.")
