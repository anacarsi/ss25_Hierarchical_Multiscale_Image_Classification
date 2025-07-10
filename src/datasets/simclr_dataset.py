from torch.utils.data import Dataset
import torch


class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img = self.base_dataset[idx][0]  # use only the image
        img_i = self.transform(img)
        img_j = self.transform(img)
        return img_i, img_j
