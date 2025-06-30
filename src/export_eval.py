import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
import numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw

# =============================
#   Feature Analysis Pipeline
# =============================

def load_features_and_labels(feature_path, label_path):
    """Load extracted patch features and labels from .npy files."""
    features = np.load(feature_path)
    labels = np.load(label_path)
    print(f"\n[INFO] Feature shape: {features.shape}")
    print(f"[INFO] Labels shape: {labels.shape}")
    print(f"[INFO] Label distribution (0=normal, 1=tumor): {np.bincount(labels)}")
    return features, labels

# -----------------------------
def plot_pca(features, labels, out_path="pca_patch_features.png"):
    """Plot and save PCA visualization of patch features."""
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=features_pca[:,0], y=features_pca[:,1], hue=labels, palette='Set1')
    plt.title("PCA: Patch Features (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(["Normal", "Tumor"])
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved PCA plot to {out_path}")

# -----------------------------
def plot_tsne(features, labels, out_path="tsne_patch_features.png"):
    """Plot and save t-SNE visualization of patch features."""
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    features_tsne = tsne.fit_transform(features)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=features_tsne[:,0], y=features_tsne[:,1], hue=labels, palette='Set1')
    plt.title("t-SNE: Patch Features (2D)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(["Normal", "Tumor"])
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved t-SNE plot to {out_path}")

# -----------------------------
def plot_logreg_confusion(features, labels, out_path="logreg_confusion_matrix.png"):
    """Train logistic regression and save confusion matrix plot."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"[INFO] Logistic Regression Accuracy: {acc:.4f}")
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Tumor"], yticklabels=["Normal", "Tumor"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved confusion matrix to {out_path}")

# -----------------------------
def find_unlabeled_patches(patch_dir):
    """Find all unlabeled patches and group by slide."""
    unlabeled_patches = defaultdict(list)
    for slide_folder in os.listdir(patch_dir):
        slide_path = os.path.join(patch_dir, slide_folder)
        if not os.path.isdir(slide_path):
            continue
        unlabeled_dir = os.path.join(slide_path, "unlabeled")
        if os.path.isdir(unlabeled_dir):
            for fname in os.listdir(unlabeled_dir):
                if fname.endswith(".png"):
                    parts = fname.split("_")
                    try:
                        x = int([p for p in parts if p.startswith("x")][0][1:])
                        y = int([p for p in parts if p.startswith("y")][0][1:])
                        unlabeled_patches[slide_folder].append((x, y, fname))
                    except Exception as e:
                        print(f"[WARN] Could not parse coordinates from {fname}: {e}")
    return unlabeled_patches

# -----------------------------
def overlay_unlabeled_on_wsi(slide_name, unlabeled_patches, wsi_dir, patch_size=224, max_patches=50):
    """Overlay rectangles for unlabeled patches on the WSI and save the result."""
    wsi_path = os.path.join(wsi_dir, f"{slide_name}.tif")
    if not os.path.exists(wsi_path):
        print(f"[WARN] WSI not found: {wsi_path}")
        return
    try:
        wsi = Image.open(wsi_path)
    except Exception as e:
        print(f"[WARN] Could not open WSI: {e}")
        return
    draw = ImageDraw.Draw(wsi)
    patches = unlabeled_patches.get(slide_name, [])
    for i, (x, y, fname) in enumerate(patches):
        if i >= max_patches:
            break
        draw.rectangle([x, y, x+patch_size, y+patch_size], outline="red", width=3)
    plt.figure(figsize=(10,10))
    plt.imshow(wsi)
    plt.title(f"Unlabeled patches (red) on {slide_name}")
    plt.axis("off")
    plt.tight_layout()
    outname = f"unlabeled_overlay_{slide_name}.png"
    plt.savefig(outname)
    plt.close()
    print(f"[INFO] Saved overlay: {outname}")

# -----------------------------
def main():
    """
    Main entry point for feature analysis and visualization.
    All plots are saved to disk.
    """
    # ---- Load Data ----
    features, labels = load_features_and_labels("patch_features_3.npy", "patch_labels_3.npy")

    # ---- Visualizations ----
    plot_pca(features, labels)
    plot_tsne(features, labels)
    plot_logreg_confusion(features, labels)

    # ---- Unlabeled Patch Overlay ----
    patch_dir = os.path.join(os.getcwd(), "..", "data", "camelyon16", "train", "patches")
    wsi_dir = os.path.join(os.getcwd(), "..", "data", "camelyon16", "train", "img")
    unlabeled_patches = find_unlabeled_patches(patch_dir)
    slide_to_check = list(unlabeled_patches.keys())[0] if unlabeled_patches else None
    if slide_to_check:
        overlay_unlabeled_on_wsi(slide_to_check, unlabeled_patches, wsi_dir)
    else:
        print("[INFO] No unlabeled patches found.")

if __name__ == "__main__":
    main()
