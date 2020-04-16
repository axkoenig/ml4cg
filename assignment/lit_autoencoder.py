__author__ = 'Alexander Koenig and Li Nguyen'

import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from argparse import ArgumentParser

class LitAutoencoder(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.encoder = nn.Sequential(
            # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
            # out_dims = (in_dims - kernel_size + 2*padding) / 2 + 1 
            # Layer 1: Input is (nc) x 128 x 128
            nn.Conv2d(hparams.nc, hparams.nfe, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe),
            nn.ReLU(True),

            # Layer 2: State size is (nfe) x 64 x 64
            nn.Conv2d(hparams.nfe, hparams.nfe * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 2),
            nn.ReLU(True),

            # Layer 3: State size is (nfe*2) x 32 x 32
            nn.Conv2d(hparams.nfe * 2, hparams.nfe * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 4),
            nn.ReLU(True),

            # Layer 4: State size is (nfe*4) x 16 x 16
            nn.Conv2d(hparams.nfe * 4, hparams.nfe * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 8),
            nn.ReLU(True),

            # Layer 5: State size is (nfe*8) x 8 x 8
            nn.Conv2d(hparams.nfe * 8, hparams.nfe * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 16),
            nn.ReLU(True),
            
            # Layer 6: State size is (nfe*16) x 4 x 4
            nn.Conv2d(hparams.nfe * 16, hparams.nz, 4, 1, 0, bias=False),
            nn.ReLU(True)

            # Output size is (nz) x 1 x 1
        )

        self.decoder = nn.Sequential(             
            # Layer 1: Input is (nz) x 1 x 1
            nn.ConvTranspose2d(hparams.nz, hparams.nfd * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hparams.nfd * 16),
            nn.LeakyReLU(True),

            # Layer 2: State size is (nfd*16) x 4 x 4
            nn.ConvTranspose2d(hparams.nfd * 16, hparams.nfd * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 8),
            nn.LeakyReLU(True),

            # Layer 3: State size is (nfd*8) x 8 x 8
            nn.ConvTranspose2d(hparams.nfd * 8, hparams.nfd * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 4),
            nn.LeakyReLU(True),

            # Layer 4: State size is (nfd*4) x 16 x 16
            nn.ConvTranspose2d(hparams.nfd * 4, hparams.nfd * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 2),
            nn.LeakyReLU(True),

            # Layer 5: State size is (nfd*2) x 32 x 32
            nn.ConvTranspose2d(hparams.nfd * 2, hparams.nfd, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd),
            nn.LeakyReLU(True),

            # Layer 6: State size is (nfd) x 64 x 64
            nn.ConvTranspose2d(hparams.nfd, hparams.nc, 4, 2, 1, bias=False),
            nn.Tanh()

            # Output size is (nc) x 128 x 128
        )
            
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train_dataloader(self):
        transform = transforms.Compose([transforms.Resize(self.hparams.image_size), 
                                       transforms.CenterCrop(self.hparams.image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = dset.ImageFolder(root=self.hparams.data_root, transform=transform)

        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2))

    def training_step(self, batch, batch_idx):
        x, _ = batch
        output = self(x)
        loss = F.mse_loss(output, x)

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

def main(hparams):
    model = LitAutoencoder(hparams)
    trainer = Trainer(gpus=hparams.gpus)
    trainer.fit(model)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/home/dcor/ronmokady/workshop20/team6/ml4cg/data")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--image_size", type=int, default=128, help="Spatial size of training images")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of workers for dataloader")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images")
    parser.add_argument("--nz", type=int, default=256, help="Size of latent vector z")
    parser.add_argument("--nfe", type=int, default=64, help="Size of feature maps in encoder")
    parser.add_argument("--nfd", type=int, default=64, help="Size of feature maps in decoder")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs. Use 0 for CPU mode")

    args = parser.parse_args()
    main(args)