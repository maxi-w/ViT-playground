from argparse import ArgumentParser
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning import Trainer
from src.engine import LightningViT
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor


def main(hp):
    model = LightningViT()

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

    cifar_data = CIFAR10DataModule(data_dir="./data", num_workers=12, shuffle=True)

    trainer.fit(model, datamodule=cifar_data, ckpt_path=hp.resume_from_ckpt)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--resume-from-ckpt", default=None)
    args = parser.parse_args()

    main(args)
