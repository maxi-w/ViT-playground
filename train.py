from argparse import ArgumentParser
from lightning.pytorch import Trainer
from src.engine import LightningViT
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from src.datasets import HuggingFaceDataModule


def main(hp):
    data = HuggingFaceDataModule(dataset_name="cifar10", batch_size=hp.batch_size, num_workers=hp.num_workers, shuffle=hp.shuffle_data)
    setattr(hp, "num_classes", data.num_classes)

    model = LightningViT.build_model_from_args(hp)

    trainer = Trainer(
        accelerator=hp.accelerator,
        devices=hp.devices,
        callbacks=[
            ModelCheckpoint(monitor="val_loss", save_last=True),
            EarlyStopping(monitor="val_loss", patience=10),
            LearningRateMonitor(logging_interval='step')
        ],
        max_epochs=-1,
    )

    trainer.fit(model, datamodule=data, ckpt_path=hp.resume_from_ckpt)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--num-workers", type=int, default="12")
    parser.add_argument("--shuffle-data", type=bool, default=True)
    parser.add_argument("--resume-from-ckpt", default=None)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--emb-dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    main(args)
