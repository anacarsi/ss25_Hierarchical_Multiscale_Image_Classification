import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.transforms.functional import center_crop
from .BaseModel import BaseModel


class UNet(BaseModel):
    """
    U-Net model implementation for image segmentation and classification.

    Parameters:
    - batch_size: int, batch size for training.
    - learning_rate: float, learning rate for the optimizer.
    - num_epochs: int, number of epochs for training.
    - dataset_name: str, name of the dataset.

    Returns:
    - None: Initializes the U-Net model.
    """

    def __init__(
        self,
        batch_size=128,
        learning_rate=0.001,
        num_epochs=50,
        dataset_name="imagenet",
    ):
        super().__init__(
            batch_size, learning_rate, num_epochs, dataset_name=dataset_name
        )
        self.dataset_name = dataset_name
        self.build_model()
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def build_model(self):
        """
        Build the U-Net model architecture.

        Parameters:
        - None

        Returns:
        - None: Constructs the encoder, bottleneck, decoder, and final layers.
        """

        class UNetModel(nn.Module):
            def __init__(self):
                super(UNetModel, self).__init__()
                # Encoder
                self.enc1 = self.conv_block(3, 64)
                self.enc2 = self.conv_block(64, 128)
                self.enc3 = self.conv_block(128, 256)
                self.enc4 = self.conv_block(256, 512)

                # Bottleneck
                self.bottleneck = self.conv_block(512, 1024)

                # Decoder
                self.dec4 = self.upconv_block(1024 + 512, 512)
                self.dec3 = self.upconv_block(512 + 256, 256)
                self.dec2 = self.upconv_block(256 + 128, 128)
                self.dec1 = self.upconv_block(128 + 64, 64)

                # Final output for classification
                self.global_pool = nn.AdaptiveAvgPool2d(
                    1
                )  # Global average pooling ----- or before bottleneck?
                self.fc = nn.Linear(64, 200)  # Fully connected layer for 200 classes

            def conv_block(self, in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )

            def upconv_block(self, in_channels, out_channels):
                return nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels, out_channels, kernel_size=2, stride=2
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                # Encoder
                enc1 = self.enc1(x)
                enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
                enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
                enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))

                # Bottleneck
                bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))

                # Decoder
                dec4 = self.crop_and_concat(bottleneck, enc4)
                dec4 = self.dec4(dec4)
                dec3 = self.crop_and_concat(dec4, enc3)
                dec3 = self.dec3(dec3)
                dec2 = self.crop_and_concat(dec3, enc2)
                dec2 = self.dec2(dec2)
                dec1 = self.crop_and_concat(dec2, enc1)
                dec1 = self.dec1(dec1)

                # Global average pooling and classification, not sure...
                pooled = self.global_pool(dec1)
                pooled = pooled.view(pooled.size(0), -1)  # Flatten
                return self.fc(pooled)

            def crop_and_concat(self, upsampled, bypass):
                """
                Crop the bypass tensor to match the spatial dimensions of the upsampled tensor,
                and concatenate them along the channel dimension.
                """
                _, _, h, w = upsampled.size()
                bypass = center_crop(
                    bypass, (h, w)
                )  # Use torchvision.transforms.functional.center_crop
                return torch.cat((upsampled, bypass), dim=1)

        self.net = UNetModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def get_transforms(self, is_train=True) -> transforms.Compose:
        """
        Get the data transformations for the Tiny ImageNet dataset.

        Parameters:
        - is_train: bool, whether the transformations are for training or testing.

        Returns:
        - transforms.Compose: Composed transformations for the dataset.
        """
        return transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def train(self):
        """
        Train the U-Net model.

        Parameters:
        - None

        Returns:
        - None: Updates model weights during training.
        """
        self.net.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for i, data in enumerate(self.trainloader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(self.trainloader)
            epoch_acc = 100 * correct / total
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)
            print(
                f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%"
            )

        print("Training completed!")

    def test(self):
        """
        Test the U-Net model.

        Parameters:
        - None

        Returns:
        - avg_loss: float, average loss on the test set.
        - accuracy: float, accuracy on the test set.
        """
        self.net.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = test_loss / len(self.testloader)
        print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy
