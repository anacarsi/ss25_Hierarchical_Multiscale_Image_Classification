import os
import shutil


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


patch_directory = os.path.join(
    os.getcwd(), "..", "data", "camelyon16", "patches", "level_4"
)
group_patches_by_slide(patch_root=patch_directory)
