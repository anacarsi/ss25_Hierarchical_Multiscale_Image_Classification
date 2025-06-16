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
    python src/main.py --download --base_dir ./data --remote
    ```
    - `--remote`: Downloads all files (set to `False` for testing to download only one file).

### 2. Extract Patches

- Extract patches from WSIs: (WIP)
    ```sh
    python src/main.py --patch
    ```

### 3. Prepare Data

- Preprocess or augment data: (WIP)
    ```sh
    python src/main.py --prepare
    ```

### 4. Train Model

- Train the U-Net model: (WIP)
    ```sh
    python src/main.py --train
    ```

### 5. Test Model

- Test the trained U-Net model:(WIP)
    ```sh
    python src/main.py --test
    ```

## Configuration

- Modify [`src/config.py`](src/config.py) to adjust paths, hyperparameters, and experiment settings.

## Citation

If you use this codebase, please cite the repository and the CAMELYON16 dataset.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
