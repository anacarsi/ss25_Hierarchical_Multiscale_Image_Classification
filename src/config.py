# Configuration settings for the WSI MIL CAMELYON16 project

import os

class Config:
    # Dataset paths
    DATASET_PATH = os.path.join('data', 'camelyon16')
    PATCHES_PATH = os.path.join('data', 'preprocessing', 'patches')

    # Model parameters
    INPUT_SIZE = (224, 224)  # Input size for the CNN encoder
    NUM_CLASSES = 2  # Number of classes for classification (e.g., tumor vs. non-tumor)

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4

    # Uncertainty estimation parameters
    SOFTMAX_THRESHOLD = 0.7  # Threshold for softmax confidence
    MONTE_CARLO_SAMPLES = 100  # Number of samples for Monte Carlo Dropout

    # Device configuration
    DEVICE = 'cuda' if os.cuda.is_available() else 'cpu'

    # Logging and saving
    LOG_DIR = 'logs'
    CHECKPOINT_DIR = 'checkpoints'
    MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, 'mil_model.pth')

    @staticmethod
    def print_config():
        print("Configuration:")
        for key, value in vars(Config).items():
            if not key.startswith('__'):
                print(f"{key}: {value}")