import torch
import lightning.pytorch as pl
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as transform_lib


class HuggingFaceDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name: str = "cifar10", batch_size: int = 32, num_workers: int = 4, shuffle: bool = True):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.num_classes = 10

    def setup(self, stage: str):
        if self.dataset_name == "cifar10":
            self.train_set = CIFAR10HF(split="train")
            self.test_set = CIFAR10HF(split="test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self):
        # TODO: split train data for validation
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        raise NotImplementedError("This function is not implemented.")


class CIFAR10HF(Dataset):
    def __init__(self, split: str) -> None:
        super().__init__()
        self.dataset = load_dataset("cifar10").with_format("torch")[split]

        self.transforms = transform_lib.Compose([
            transform_lib.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index):
        img = self.dataset[index]["img"].permute(2, 0, 1).to(torch.float32)
        img = self.transforms(img)
        return img, self.dataset[index]["label"]
