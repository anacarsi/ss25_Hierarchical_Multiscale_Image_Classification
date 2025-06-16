import os
import numpy as np
import cv2
from skimage import io

def extract_patches_from_wsi(wsi_path, patch_size=(224, 224), stride=112, output_dir='patches'):
    """
    Extract patches from a Whole Slide Image (WSI).

    Parameters:
    - wsi_path: str, path to the WSI file.
    - patch_size: tuple, size of the patches to extract (height, width).
    - stride: int, the step size for moving the patch extraction window.
    - output_dir: str, directory to save the extracted patches.
    """
    os.makedirs(output_dir, exist_ok=True)

    wsi_image = io.imread(wsi_path)
    h, w, _ = wsi_image.shape

    patch_count = 0
    for y in range(0, h - patch_size[0] + 1, stride):
        for x in range(0, w - patch_size[1] + 1, stride):
            patch = wsi_image[y:y + patch_size[0], x:x + patch_size[1]]
            patch_filename = os.path.join(output_dir, f'patch_{patch_count}.png')
            cv2.imwrite(patch_filename, patch)
            patch_count += 1

    print(f'Extracted {patch_count} patches from {wsi_path} and saved to {output_dir}.')
