import unittest
import torch

from src.vit import ViT


class TestModel(unittest.TestCase):
    def test_forward(self):
        vit = ViT(
            image_size=32,
            patch_size=4,
            num_classes=10,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1
        )

        inputs = torch.randn(1, 3, 32, 32)
        predictions = vit(inputs)
        self.assertEqual(predictions.shape, torch.Size([1, 10]))


if __name__ == '__main__':
    unittest.main()
