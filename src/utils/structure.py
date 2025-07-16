import os
import shutil
import glob


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    DEBUG = "\033[96m"
    INFO = "\033[95m"  # pink
    WARNING = "\033[93m"  # yellow
    ERROR = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def group_patches_by_slide(patch_root="data/camelyon16/patches/level_0"):
    """
    Groups patches by slide ID, moving them into directories named after the slide ID.
    """
    print(f"[INFO] Grouping patches in {patch_root}")
    for label in ["normal", "tumor"]:
        label_dir = os.path.join(patch_root, label)
        if not os.path.isdir(label_dir):
            print(f"[INFO] Label directory {label_dir} is not a directory. Skipping.")
            continue
        for fname in os.listdir(label_dir):
            print(f"[INFO] Processing file: {fname}")
            if not fname.endswith(".png"):
                print(f"[INFO] Skipping non-patch file: {fname}")
                continue
            slide_id = fname.split("_x")[0]  # e.g. "tumor_001"
            slide_dir = os.path.join(patch_root, slide_id)
            os.makedirs(slide_dir, exist_ok=True)

            src = os.path.join(label_dir, fname)
            dst = os.path.join(slide_dir, fname)
            shutil.move(src, dst)

    print(f"[INFO] Grouping complete.")


def get_latest_mil_model_path():
    """
    Returns the path to the most recently created model file in models_dir
    whose filename starts with 'mil' and ends with '.pth'.
    """
    pattern = os.path.join(os.getcwd(), "src", "models", "mil*.pth")
    model_files = glob.glob(pattern)
    if not model_files:
        print(
            f"{bcolors.ERROR}[ERROR]{bcolors.ENDC} No MIL model files found in {pattern}."
        )
        return None
    latest_model = max(model_files, key=os.path.getctime)
    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Latest MIL model found: {latest_model}")
    return latest_model
