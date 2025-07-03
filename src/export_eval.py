import os
import numpy as np
from PIL import Image, ImageDraw
from lxml import etree
import matplotlib.pyplot as plt
"""os.add_dll_directory(
    r"C:\Program Files\OpenSlide\openslide-bin-4.0.0.8-windows-x64\bin"
)"""
from openslide import OpenSlide

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    DEBUG = '\033[96m'
    INFO = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def parse_xml_mask(xml_path, level_dims, slide, level):
    """
    Convert XML annotation to binary mask.
    Parameters:
    - xml_path: str, path to the XML file containing annotations.
    - level_dims: tuple, dimensions of the WSI at the specified level (width, height).
    - slide: OpenSlide object for the WSI.
    - level: int, target level for mask.
    """
    try:
        tree = etree.parse(xml_path)
    except etree.XMLSyntaxError as e:
        print(f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} Error parsing XML file {xml_path}: {e}")
        return None

    # Compute scaling factors based on actual dimensions
    base_dims = slide.level_dimensions[0]
    scale_x = level_dims[0] / base_dims[0]
    scale_y = level_dims[1] / base_dims[1]

    mask = Image.new("L", level_dims, 0)
    draw = ImageDraw.Draw(mask)

    for coordinates_node in tree.xpath("//Annotation/Coordinates | //Annotations/Annotation/Coordinates"):
        coords = []
        for coord_node in coordinates_node.findall("Coordinate"):
            try:
                x = float(coord_node.get("X"))
                y = float(coord_node.get("Y"))
                # Scale coordinates to the target level
                scaled_x = int(x * scale_x)
                scaled_y = int(y * scale_y)
                coords.append((scaled_x, scaled_y))
            except (ValueError, TypeError) as e:
                print(f"{bcolors.WARNING}Warning: Could not parse coordinate (X,Y) from XML for {xml_path}: {e}{bcolors.ENDC}")
                continue
        if coords:
            draw.polygon(coords, outline=255, fill=255)
    return mask

def visualize_and_save_wsi(wsi_name, output_dir, level=3, x=1000, y=1500):
    """
    Visualize and save WSI mask, patch, and mask patch as PNGs.
    Args:
        wsi_name (str): Name of the WSI file (e.g., 'tumor_076.tif')
        output_dir (str): Directory to save PNGs
        level (int): OpenSlide level to use
        x (int): X coordinate for patch extraction
        y (int): Y coordinate for patch extraction
    """
    os.makedirs(output_dir, exist_ok=True)
    cwd = os.getcwd()
    wsi_dir = os.path.join(cwd, "..", "data", "camelyon16", "train", "img")
    annot_dir_train = os.path.join(cwd, "..", "data", "camelyon16", "train", "mask")
    level_dir = os.path.join(cwd, "..", "data", "camelyon16", "patches", f"level_{level}")

    print("WSI directory exists:", wsi_dir, os.path.exists(wsi_dir))
    print("Annotation directory (train) exists:", annot_dir_train, os.path.exists(annot_dir_train))
    print("Level directory exists:", level_dir, os.path.exists(level_dir))

    xml_path = os.path.join(annot_dir_train, wsi_name.replace(".tif", ".xml"))
    wsi_path = os.path.join(wsi_dir, wsi_name)
    print("XML path exists:", os.path.exists(xml_path))
    print("WSI path exists:", os.path.exists(wsi_path))
    if not os.path.exists(xml_path) or not os.path.exists(wsi_path):
        print(f"[ERROR] Missing WSI or XML for {wsi_name}")
        return

    slide = OpenSlide(wsi_path)
    level_dims = slide.level_dimensions[level]
    mask = parse_xml_mask(xml_path, level_dims, slide, level)

    # Save mask as PNG
    mask_png_path = os.path.join(output_dir, f"{wsi_name.replace('.tif', '')}_mask_level{level}.png")
    mask.save(mask_png_path)
    print(f"Saved mask to {mask_png_path}")

    # Plot and save mask visualization
    plt.figure()
    plt.imshow(mask)
    plt.title("Parsed XML Mask")
    mask_fig_path = os.path.join(output_dir, f"{wsi_name.replace('.tif', '')}_mask_level{level}_viz.png")
    plt.savefig(mask_fig_path)
    plt.close()
    print(f"Saved mask visualization to {mask_fig_path}")

    # Extract patch and mask patch
    patch = slide.read_region((int(x * slide.level_downsamples[level]), int(y * slide.level_downsamples[level])), level, (224, 224)).convert("RGB")
    patch_png_path = os.path.join(output_dir, f"{wsi_name.replace('.tif', '')}_patch_{x}_{y}_level{level}.png")
    patch.save(patch_png_path)
    print(f"Saved patch to {patch_png_path}")

    mask_patch = mask.crop((x, y, x + 224, y + 224))
    mask_patch_png_path = os.path.join(output_dir, f"{wsi_name.replace('.tif', '')}_maskpatch_{x}_{y}_level{level}.png")
    mask_patch.save(mask_patch_png_path)
    print(f"Saved mask patch to {mask_patch_png_path}")

    # Plot and save side-by-side visualization
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(patch)
    ax[0].set_title("Image Patch")
    ax[1].imshow(mask_patch, cmap="gray")
    ax[1].set_title("Mask Patch")
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    sidebyside_path = os.path.join(output_dir, f"{wsi_name.replace('.tif', '')}_patch_maskpatch_{x}_{y}_level{level}.png")
    plt.savefig(sidebyside_path)
    plt.close()
    print(f"Saved patch/mask side-by-side to {sidebyside_path}")

if __name__ == "__main__":
    visualize_and_save_wsi(
        wsi_name="tumor_076.tif",
        output_dir="wsi_visualizations",
        level=3,
        x=1000,
        y=1500
    )

