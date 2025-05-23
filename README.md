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

### 1. Data Preparation

- Place the CAMELYON16 dataset in the `data/camelyon16/` directory.
- Extract patches from WSIs using:
    ```sh
    python src/preprocessing/patch_extraction.py
    ```

### 2. Training

- Configure experiment settings in [`experiments/experiment_configs.yaml`](experiments/experiment_configs.yaml).
- Start training:
    ```sh
    python src/train.py
    ```

### 3. Evaluation

- Evaluate a trained model:
    ```sh
    python src/eval.py
    ```

### 4. Visualization

- Generate attention heatmaps using [`src/visualization/attention_heatmap.py`](src/visualization/attention_heatmap.py).

## Configuration

- Modify [`src/config.py`](src/config.py) or [`experiments/experiment_configs.yaml`](experiments/experiment_configs.yaml) to adjust paths, hyperparameters, and experiment settings.

## Citation

If you use this codebase, please cite the repository and the CAMELYON16 dataset.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
