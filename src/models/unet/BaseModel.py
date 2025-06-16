# OLD, to work with Imagenet Dataset
# Abstract Class for NN Model (Factory Design Pattern)

from abc import ABC, abstractmethod
import torch
import torchvision
from torchvision import transforms
import os
import logging
from torch.utils.data import random_split


class BaseModel(ABC):
    def __init__(
        self,
        batch_size=256,
        learning_rate=0.001,
        num_epochs=20,
        dataset_path=None,  # Default to project directory
        dataset_name="cifar10",
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Set dataset_path to default if not provided
        if dataset_path is None:
            self.dataset_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "data")
            )
            print(f"[SAFE LENS] Dataset path: {self.dataset_path}")
        else:
            self.dataset_path = os.path.abspath(dataset_path)

        self.dataset_name = dataset_name
        self._prepare_data_cross_validation()

    def data_augmentation(self, is_train=True):
        """
        Apply data augmentation techniques to the dataset imagenet. To prevent overfitting.

        Parameters:
        - is_train (bool): If True, applies augmentations for training data.

        Returns:
        - torchvision.transforms.Compose: Transformations applied to the dataset.
        """
        input_size = 64
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images
                    transforms.RandomRotation(15),  # Rotate images by up to 15 degrees
                    transforms.RandomResizedCrop(
                        input_size, scale=(0.8, 1.0)
                    ),  # Random crop
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),  # Color augmentation
                    transforms.ToTensor(),  # Convert to tensor
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                    ),  # Normalize
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        (input_size, input_size)
                    ),  # Resize for validation/testing
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

        return transform

    def get_transforms(self, is_train=True):
        if self.dataset_name == "cifar10":
            if is_train:  # Training transformations (with augmentation)
                return transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )
            else:  # Validation/Test transformations (No augmentation)
                return transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )

        elif self.dataset_name == "imagenet":
            return self.data_augmentation(is_train=is_train)
        else:
            raise ValueError(
                "Unsupported dataset type. Choose 'cifar10' or 'imagenet'."
            )

    def _prepare_data_cross_validation(self, poisoned_data=None):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            poisoned_data: Optional poisoned dataset to use instead of clean data
        """
        if self.dataset_name == "cifar10":
            # Load CIFAR-10 dataset
            if poisoned_data is None:
                full_trainset = torchvision.datasets.CIFAR10(
                    root=self.dataset_path + "/cifar",
                    train=True,
                    download=True,
                    transform=self.get_transforms(is_train=True),
                )
            else:
                full_trainset = poisoned_data

            # Split into train, validation, and test sets
            total_size = len(full_trainset)
            train_size = int(0.7 * total_size)
            remaining_size = total_size - train_size
            val_size = int(remaining_size / 2)
            test_size = remaining_size - val_size

            self.trainset, self.valset, self.testset = random_split(
                full_trainset, [train_size, val_size, test_size]
            )

        elif self.dataset_name == "imagenet":
            imagenet_path = os.path.join(self.dataset_path, "imagenet")
            if not os.path.exists(imagenet_path):
                raise FileNotFoundError(
                    f"ImageNet directory not found at '{imagenet_path}'.\n"
                    "Please download Tiny-ImageNet and place it in 'data/imagenet'."
                )

            self.logger.info("[SAFE LENS] Loading Tiny-ImageNet dataset...")

            # Load full training set
            full_trainset = torchvision.datasets.ImageFolder(
                root=os.path.join(imagenet_path, "train"),
                transform=self.get_transforms(is_train=True),
            )

            # Split training set into train and test sets (80% train, 20% test)
            total_size = len(full_trainset)
            train_size = int(0.7 * total_size)
            remaining_size = total_size - train_size
            val_size = int(remaining_size / 2)
            test_size = remaining_size - val_size

            self.trainset, self.valset, self.testset = random_split(
                full_trainset, [train_size, val_size, test_size]
            )

            # For ImageNet, we'll use the same transform for all sets
            transform = self.get_transforms(is_train=False)

            # Wrap the subsets to ensure they use the correct transform
            self.trainset = torch.utils.data.Subset(
                torchvision.datasets.ImageFolder(
                    root=os.path.join(imagenet_path, "train"),
                    transform=self.get_transforms(is_train=True),
                ),
                self.trainset.indices,
            )

            self.valset = torch.utils.data.Subset(
                torchvision.datasets.ImageFolder(
                    root=os.path.join(imagenet_path, "train"), transform=transform
                ),
                self.valset.indices,
            )

            self.testset = torch.utils.data.Subset(
                torchvision.datasets.ImageFolder(
                    root=os.path.join(imagenet_path, "train"), transform=transform
                ),
                self.testset.indices,
            )

            self.logger.info("[SAFE LENS] Tiny-ImageNet dataset loaded successfully.")
            print(f"Class to index map: {full_trainset.class_to_idx}")

        # Create DataLoaders
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,  # Reduced from 8
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )
        self.valloader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.batch_size * 2,  # Larger batch size for validation
            shuffle=False,
            num_workers=4,  # Reduced from 8
            pin_memory=True,
            persistent_workers=True,
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.batch_size * 2,  # Larger batch size for testing
            shuffle=False,
            num_workers=4,  # Reduced from 8
            pin_memory=True,
            persistent_workers=True,
        )

    @abstractmethod
    def build_model(self):
        pass

    def export_onnx(self, file_name="model.onnx"):
        if self.dataset_name == "cifar10":
            dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
        elif self.dataset_name == "imagenet":
            dummy_input = torch.randn(1, 3, 64, 64).to(self.device).to(self.device)

        torch.onnx.export(self.net, dummy_input, file_name, opset_version=11)
        (
            print(f"Model exported to {file_name}")
            if not self.logger
            else self.logger.info(f"Model exported to {file_name}")
        )
