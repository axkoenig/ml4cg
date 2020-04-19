__author__ = 'Alexander Koenig, Li Nguyen'

from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils

from pytorch_lightning import loggers
from pytorch_lightning import Trainer
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

# normalization constants
MEAN = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
STD = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)

class LitAutoencoder(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.encoder = nn.Sequential(
            # input (nc) x 128 x 128
            nn.Conv2d(hparams.nc, hparams.nfe, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),
            # output (nfe) x 32 x 32

            nn.Conv2d(hparams.nfe, hparams.nfe * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 2),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),
            # output (nfe*2) x 8 x 8

            nn.Conv2d(hparams.nfe * 2, hparams.nfe * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 4),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),
            # output (nfe*4) x 2 x 2

            nn.Flatten()
            # output (nfe*8) x 1 x 1
        )

        self.decoder = nn.Sequential(
            # input (nfe*8) x 1 x 1
            View((-1, hparams.nfd * 4, 2, 2)),
            # output (nfe*4) x 2 x 2

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(hparams.nfd * 4, hparams.nfd * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 2),
            nn.LeakyReLU(True),
            # output (nfd*2) x 8 x 8

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(hparams.nfd * 2, hparams.nfd, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd),
            nn.LeakyReLU(True),
            # output (nfd*2) x 32 x 32

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(hparams.nfd, hparams.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # output (nc) x 128 x 128
        )
            
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def prepare_data(self):
        
        transform = transforms.Compose([transforms.Resize(self.hparams.image_size), 
                                        transforms.CenterCrop(self.hparams.image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(MEAN.tolist(), STD.tolist()),
                                        ])
        
        dataset = ImageFolder(root=self.hparams.data_root, transform=transform)

        # train, val and test split taken from "list_eval_partition.txt" of original celebA paper
        end_train_idx = 162770
        end_val_idx = 182637
        end_test_idx = len(dataset)

        self.train_dataset = Subset(dataset, range(0, end_train_idx))
        self.val_dataset = Subset(dataset, range(end_train_idx+1, end_val_idx)) 
        self.test_dataset = Subset(dataset, range(end_val_idx+1, end_test_idx))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2))

    def save_images(self, x, output, n, name):
        """Saves a plot of n images from input and output batch
        """

        if self.hparams.batch_size < n:
            raise IndexError("You are trying to plot more images than your batch contains!")

        # denormalize images
        denormalization = transforms.Normalize((-MEAN / STD).tolist(), (1.0 / STD).tolist())
        x = [denormalization(i) for i in x[:n]]
        output = [denormalization(i) for i in output[:n]]

        # make grids and save to logger
        grid_top = vutils.make_grid(x, nrow=n)
        grid_bottom = vutils.make_grid(output, nrow=n)
        grid = torch.cat((grid_top, grid_bottom), 1)
        self.logger.experiment.add_image(name, grid)

    def training_step(self, batch, batch_idx):
        
        x, _ = batch
        output = self(x)
        loss = F.mse_loss(output, x)

        n = 16
        
        # save n input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(x, output, n, "train_input_output")
        
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        output = self(x)
        loss = F.mse_loss(output, x)
        logs = {"val_loss": loss}
        return {"val_loss": loss, "log": logs}

    def validation_epoch_end(self, outputs):
        return self._shared_avg_eval(outputs, "val")
    
    def test_step(self, batch, batch_idx):
        x, _ = batch
        output = self(x)
        loss = F.mse_loss(output, x)

        n = 16
        
        # save n input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(x, output, n, "test_input_output")
        
        logs = {"test_loss": loss}
        return {"test_loss": loss, "log": logs}

    def test_epoch_end(self, outputs):
        return self._shared_avg_eval(outputs, "test")

    def _shared_avg_eval(self, outputs, prefix):
        avg_loss = torch.stack([x[f'{prefix}_loss'] for x in outputs]).mean()
        logs = {f'avg_{prefix}_loss': avg_loss}
        return {f'avg_{prefix}_loss': avg_loss, 'log': logs}

def main(hparams):
    logger = loggers.TensorBoardLogger(hparams.log_dir, name=f"max_grid_bs{hparams.batch_size}_lr{hparams.lr}")

    model = LitAutoencoder(hparams)
    trainer = Trainer(logger=logger, gpus=hparams.gpus, max_epochs=hparams.max_epochs)
    trainer.fit(model)

    trainer.test(model)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/specific/netapp5_3/rent_public/dcor-01-2021/ronmokady/workshop20/team6/ml4cg/data", help="Data root directory")
    parser.add_argument("--log_dir", type=str, default="/specific/netapp5_3/rent_public/dcor-01-2021/ronmokady/workshop20/team6/ml4cg/assignment/logs", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--image_size", type=int, default=128, help="Spatial size of training images")
    parser.add_argument("--max_epochs", type=int, default=8, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size during training")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images")
    parser.add_argument("--nz", type=int, default=256, help="Size of latent vector z")
    parser.add_argument("--nfe", type=int, default=16, help="Size of feature maps in encoder")
    parser.add_argument("--nfd", type=int, default=16, help="Size of feature maps in decoder")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs. Use 0 for CPU mode")

    args = parser.parse_args()
    main(args)
