__author__ = 'Alexander Koenig, Li Nguyen'

from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from pytorch_lightning import Trainer, loggers
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchsummary import summary

from modules import EncoderA, EncoderB, Generator

# normalization constants
MEAN = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
STD = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

class Net(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.encoder_a = EncoderA(hparams)
        self.encoder_b = EncoderB(hparams)
        self.generator = Generator(hparams)
        self.mlp = MLP(64,
                       get_num_adain_params(self.generator),
                       256,
                       3,
                       norm='none',
                       activ='relu')

    def forward(self, x1, x2):
        """Forward pass of network

        Note that for brevity we use a slightly different notation than in the
        presentation. The mixed images m1 and m2 correspond to xhat_a1_b2 and 
        xhat_a2_b1 in the presentation, respectively. The reconstructed images
        r1 and r2 correspond to xhat_1 and xhat_2, respectively. 

        Args:
            x1 (tensor): first input image
            x2 (tensor): second input image

        Returns:
            tuple: both reconstructed images
        """
        # disassembly of original images
        x1_a = self.encoder_a(x1)
        x1_b = self.encoder_b(x1)
        x2_a = self.encoder_a(x2)
        x2_b = self.encoder_b(x2)

        # generation of mixed images
        m1 = self.generator(x1_a, x2_b)
        m2 = self.generator(x2_a, x1_b)

        # disassembly of mixed images 
        m1_a = self.encoder_a(m1)
        m1_b = self.encoder_b(m1)
        m2_a = self.encoder_a(m2)
        m2_b = self.encoder_b(m2)

        # generation of reconstructed images
        r1 = self.generator(m1_a, m2_b)
        r2 = self.generator(m1_b, m2_a)

        return r1, r2

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

    def save_images(self, x, output, name, n=8):
        """Saves a plot of n images from input and output batch
        x         inputs batch
        output    output batch
        name      name of plot
        n         number of pictures to compare
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
        return None

    def validation_step(self, batch, batch_idx):
        return None

    def validation_epoch_end(self, outputs):
        return None
    
    def test_step(self, batch, batch_idx):
        return None

    def test_epoch_end(self, outputs):
        return None
        
def main(hparams):
    logger = loggers.TensorBoardLogger(hparams.log_dir, name="naive_1")

    model = Net(hparams)

    # print detailed summary with estimated network size
    # summary(model, [(hparams.nc, hparams.image_size, hparams.image_size), 
    #                 (hparams.nc, hparams.image_size, hparams.image_size)], device="cpu")
    
    trainer = Trainer(logger=logger, gpus=hparams.gpus, max_epochs=hparams.max_epochs)
    trainer.fit(model)
    trainer.test(model)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/specific/netapp5_3/rent_public/dcor-01-2021/ronmokady/workshop20/team6/ml4cg/data", help="Data root directory")
    parser.add_argument("--log_dir", type=str, default="/specific/netapp5_3/rent_public/dcor-01-2021/ronmokady/workshop20/team6/ml4cg/project/logs", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--image_size", type=int, default=128, help="Spatial size of training images")
    parser.add_argument("--max_epochs", type=int, default=8, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size during training")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs. Use 0 for CPU mode")

    args = parser.parse_args()
    main(args)
