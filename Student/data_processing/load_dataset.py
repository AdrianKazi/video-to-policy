import os
import torch
from torch.utils.data import Dataset
from config.paths import DATASETS_DIR


class DownloadVideoDataset(Dataset):

    def __init__(self, dataset_path):
        print(f"[LOADING] {dataset_path}")
        self.samples = torch.load(dataset_path)
        print(f"[LOADED] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    train_path = os.path.join(DATASETS_DIR, "train.pt")
    test_path = os.path.join(DATASETS_DIR, "test.pt")

    train_dataset = DownloadVideoDataset(train_path)
    test_dataset = DownloadVideoDataset(test_path)

    x_seq = train_dataset[0]

    print("\n[CHECK]")
    print(f"x_seq shape: {x_seq.shape}")

    print("\n[INFO]")
    print(f"Train dataset size: {len(train_dataset)} sequences")
    print(f"Test dataset size:  {len(test_dataset)} sequences")