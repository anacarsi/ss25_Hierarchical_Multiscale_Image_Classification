from torch.utils.data import Dataset

class SimCLRDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _, _ = self.base_dataset[idx]
        return self.transform(img), self.transform(img)