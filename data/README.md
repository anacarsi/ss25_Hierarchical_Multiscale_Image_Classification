# CAMELYON16 Dataset README

# CAMELYON16 Dataset

The CAMELYON16 dataset is a publicly available dataset designed for the evaluation of algorithms for the detection of metastatic breast cancer in whole slide images (WSIs). It consists of a large collection of histopathological images that have been annotated for the presence of cancerous tissue.

## Dataset Structure

The dataset is organized into two main parts:

1. **Training Set**: Contains WSIs with annotated regions of interest (ROI) that indicate the presence of metastases.
2. **Test Set**: Contains WSIs that are used for evaluating the performance of the trained models.

Each WSI is provided in a format suitable for analysis, and the dataset includes accompanying metadata that describes the annotations.

## Accessing the Dataset

To access the CAMELYON16 dataset, you can download it from the official website or repository where it is hosted. Ensure that you comply with any usage restrictions or licensing agreements associated with the dataset.

## Usage

In this project, the CAMELYON16 dataset is utilized for training a Multiple Instance Learning (MIL) model. The following steps outline how to use the dataset within this framework:

1. **Patch Extraction**: Use the `patch_extraction.py` script located in the `data/preprocessing` directory to extract patches from the WSIs. This script will generate smaller image patches that can be fed into the MIL model.

2. **Dataset Loading**: The `camelyon16_mil_dataset.py` file in the `src/dataset` directory defines a dataset class that handles loading the extracted patches and creating bags for the MIL framework.

3. **Model Training**: The training process can be initiated by running the `main.py` script in the `src` directory. This script will handle the training of the MIL model using the extracted patches.

4. **Evaluation**: After training, the model can be evaluated using the `eval.py` script, which computes various performance metrics based on the test set.

## References

For more information about the CAMELYON16 dataset and its usage, please refer to the original publication:

- "The CAMELYON16 Challenge: A Multi-Institutional Evaluation of Algorithms for Detection of Lymph Node Metastases in Breast Cancer" (https://arxiv.org/abs/1612.01674)

## License

Please refer to the dataset's official website for licensing information and usage restrictions.