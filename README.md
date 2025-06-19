# ss25_Hierarchical_Multiscale_Image_Classification
HiPAC — Hierarchical Patch-based Adaptive Classifier

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ss25_Hierarchical_Multiscale_Image_Classification.git
    cd ss25_Hierarchical_Multiscale_Image_Classification
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### 1. Download Dataset

- To download the CAMELYON16 dataset:
    ```sh
    python src/main.py --download --remote
    ```
    - `--remote`: Downloads all files (set to `False` for testing to download only one file).

### 2. Extract Patches

- Extract patches from WSIs:
    ```sh
    python src/main.py --patch
    ```

### 3. Prepare Data

- Preprocess or augment data:
    ```sh
    python src/main.py --prep
    ```

### 4. Create a validation set
    ```sh
    python src/main.py --val
    ```

### 5. Extract feature vectors from patches using ResNet18.
    ```sh
    python src/main.py --extract_features
    ```
### 6. Check Structure
- Check if the directory structure is correct. If not, creates the correct one.
    ```sh
    python src/main.py --check_structure
    ```
### 7. Train
- Train a ResNet18 classifier on extracted patches.
    ```sh
    python src/main.py --train
    ```

## Configuration

- Modify [`src/config.py`](src/config.py) to adjust paths, hyperparameters, and experiment settings.

## Citation

If you use this codebase, please cite the repository and the CAMELYON16 dataset.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
