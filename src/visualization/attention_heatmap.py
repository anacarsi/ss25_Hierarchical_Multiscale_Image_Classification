import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms


def visualize_attention_heatmap(image, attention_weights, save_path=None):
    """
    Visualizes the attention heatmap over the input image.

    Parameters:
    - image: The original input image (C, H, W) format.
    - attention_weights: The attention weights (H', W') from the model.
    - save_path: Optional path to save the heatmap image.

    Returns:
    - None
    """
    attention_weights = torch.nn.functional.softmax(
        attention_weights.view(-1), dim=0
    ).view(attention_weights.shape)
    attention_weights = attention_weights.detach().cpu().numpy()

    # Resize attention weights to match the original image size
    attention_weights_resized = np.clip(attention_weights, 0, 1)

    # Create a heatmap
    heatmap = plt.cm.jet(attention_weights_resized)[
        ..., :3
    ]  # Get RGB from the colormap
    heatmap = np.uint8(255 * heatmap)  # Scale to [0, 255]

    # Overlay the heatmap on the original image
    overlay = np.clip(
        0.5 * image.permute(1, 2, 0).numpy() + 0.5 * heatmap, 0, 255
    ).astype(np.uint8)

    # Plot the original image and the overlay
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Attention Heatmap")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path)
    plt.show()
