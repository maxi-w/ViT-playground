import unittest
import torch

from src.datasets import CIFAR10HF


class TestDataModule(unittest.TestCase):
    def test_loading(self):
        data = CIFAR10HF(split="train")
        datapoint = data.__getitem__(0)
        self.assertEqual(len(datapoint), 2)
        self.assertEqual(datapoint[0].shape, torch.Size([3, 32, 32]))
        self.assertEqual(datapoint[1].shape, torch.Size([]))


if __name__ == '__main__':
    unittest.main()
