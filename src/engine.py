import time

import lightning.pytorch as pl
import torch
import torchmetrics
from torch import nn

from src.vit import ViT


class LightningViT(pl.LightningModule):
    def __init__(self, image_size: int = 32, patch_size: int = 4, num_classes: int = 10, dim: int = 512,
                 depth: int = 6, heads: int = 8, mlp_dim: int = 512, dropout: float = 0.1,
                 emb_dropout: float = 0.1, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
        self.loss = nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=1)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        start_time = time.time()
        pred = self.forward(x)
        loss = self.loss(pred, y)
        self.log('train_step_duration', time.time() - start_time)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred = self.forward(x)
        loss = self.loss(pred, y)
        self.log('val_loss', loss)
        accuracy = self.accuracy(pred, y)
        self.log('accuracy', accuracy)
        return loss

    @staticmethod
    def build_model_from_args(args):
        return LightningViT(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout,
            emb_dropout=args.emb_dropout
        )
