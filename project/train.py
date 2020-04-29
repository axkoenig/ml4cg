__author__ = "Alexander Koenig, Li Nguyen"

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

from modules import Encoder, Modulation, Generator

# normalization constants
MEAN = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
STD = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.encoder_a = Encoder(hparams)
        self.encoder_b = Encoder(hparams)
        self.modulation = Modulation(hparams)
        self.generator = Generator(hparams)

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
            dict: codes, mixed and reconstruced images
        """
        # disassembly of original images
        x1_a = self.encoder_a(x1)
        x1_b = self.encoder_b(x1)
        x2_a = self.encoder_a(x2)
        x2_b = self.encoder_b(x2)

        # calculate modulation parameters
        x1_a_ada = self.modulation(x1_a)
        x2_a_ada = self.modulation(x2_a)

        # generate mixed images
        m1 = self.generator(x2_b, x1_a_ada)
        m2 = self.generator(x1_b, x2_a_ada)

        # disassembly of mixed images
        m1_a = self.encoder_a(m1)
        m1_b = self.encoder_b(m1)
        m2_a = self.encoder_a(m2)
        m2_b = self.encoder_b(m2)

        # calculate modulation parameters
        m1_a_ada = self.modulation(m1_a)
        m2_a_ada = self.modulation(m2_a)

        # generate reconstructed images
        r1 = self.generator(m2_b, m1_a_ada)
        r2 = self.generator(m1_b, m2_a_ada)

        return {
            "x1_a": x1_a,
            "x1_b": x1_b,
            "x2_a": x2_a,
            "x2_b": x2_b,
            "m1": m1,
            "m2": m2,
            "m1_a": m1_a,
            "m1_b": m1_b,
            "m2_a": m2_a,
            "m2_b": m2_b,
            "r1": r1,
            "r2": r2,
        }

    def prepare_data(self):

        transform = transforms.Compose([transforms.Resize(self.hparams.img_size), 
                                        transforms.CenterCrop(self.hparams.img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(MEAN.tolist(), STD.tolist()),
                                        ])
        
        dataset = ImageFolder(root=self.hparams.data_root, transform=transform)

        # train, val and test split taken from "list_eval_partition.txt" of original celebA paper
        end_train_idx = 162770
        end_val_idx = 182637
        end_test_idx = len(dataset)

        self.train_dataset = Subset(dataset, range(0, end_train_idx))
        self.val_dataset = Subset(dataset, range(end_train_idx + 1, end_val_idx))
        self.test_dataset = Subset(dataset, range(end_val_idx + 1, end_test_idx))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers,)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers,)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers,)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2),)

    def plot(self, input_batches, mixed_batches, reconstr_batches, prefix, n=2):
        """Plots n triplets of ((x1, x2), (m1, m2), (r1, r2)) 

        Args:
            input_batches (tuple): Two batches of input images
            mixed_batches (tuple): Two batches of mixed images
            reconstr_batches (tuple): Two batches of reconstructed images
            prefix (str): Prefix for plot name
            n (int, optional): How many triplets to plot. Defaults to 2.

        Raises:
            IndexError: If n exceeds batch size
        """

        if input_batches[0].shape[0] < n:
            raise IndexError("You are attempting to plot more images than your batch contains!")
    
        # denormalize images
        denormalization = transforms.Normalize((-MEAN / STD).tolist(), (1.0 / STD).tolist())
        x1 = [denormalization(i) for i in input_batches[0][:n]]
        x2 = [denormalization(i) for i in input_batches[1][:n]]
        m1 = [denormalization(i) for i in mixed_batches[0][:n]]
        m2 = [denormalization(i) for i in mixed_batches[1][:n]]
        r1 = [denormalization(i) for i in reconstr_batches[0][:n]]
        r2 = [denormalization(i) for i in reconstr_batches[1][:n]]

        # create empty plot and send to device
        plot = torch.tensor([], device=x1[0].device)

        for i in range(n):
            grid_top = vutils.make_grid([x1[i], x2[i]], 2)
            grid_mid = vutils.make_grid([m1[i], m2[i]], 2)
            grid_bot = vutils.make_grid([r1[i], r2[i]], 2)
            grid_cat = torch.cat((grid_top, grid_mid, grid_bot), 1)
            plot = torch.cat((plot, grid_cat), 2)
            
            # add offset between image triplets
            if i > 0 and i < n:
                border_width = 4
                border = torch.zeros(plot.shape[0], plot.shape[1], border_width, device=x1[0].device)
                plot = torch.cat((plot, border), 2)

        name = f"{prefix}_input_mixed_reconstr_images"
        self.logger.experiment.add_image(name, plot)

    def training_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, prefix="val", plot=True)

    def validation_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, prefix="test", plot=True)

    def test_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, "test")

    def _shared_eval(self, batch, batch_idx, prefix="", plot=False):
        # retrieve batch and split in half
        imgs, _ = batch
        split_idx = imgs.shape[0] // 2
        x1 = imgs[:split_idx]
        x2 = imgs[split_idx:]

        # forward pass
        out = self(x1, x2)
        reconstr_loss = F.l1_loss(x1, out["r1"]) + F.l1_loss(x2, out["r2"])
        cycle_loss_a = F.mse_loss(out["x1_a"], out["m1_a"]) + F.mse_loss(out["x2_a"], out["m2_a"])
        cycle_loss_b = F.mse_loss(out["x1_b"], out["m2_b"]) + F.mse_loss(out["x2_b"], out["m1_b"])
        loss = (self.hparams.alpha * reconstr_loss 
                + self.hparams.gamma * cycle_loss_a 
                + self.hparams.delta * cycle_loss_b)

        # plot input, mixed and reconstructed images at beginning of epoch
        if plot and batch_idx == 0:
            self.plot((x1, x2), (out["m1"], out["m2"]), (out["r1"], out["r2"]), prefix)

        # add underscore to prefix
        if prefix:
            prefix = prefix + "_"

        logs = {f"{prefix}loss": loss}
        return {f"{prefix}loss": loss, "log": logs}

    def _shared_eval_epoch_end(self, outputs, prefix):
        avg_loss = torch.stack([x[f"{prefix}_loss"] for x in outputs]).mean()
        logs = {f"avg_{prefix}_loss": avg_loss}
        return {f"avg_{prefix}_loss": avg_loss, "log": logs}


def main(hparams):
    logger = loggers.TensorBoardLogger(hparams.log_dir, name="naive_1")

    model = Net(hparams)

    # print detailed summary with estimated network size
    summary(model, input_size=[(hparams.nc, hparams.img_size, hparams.img_size), (hparams.nc, hparams.img_size, hparams.img_size),], device="cpu")

    if hparams.batch_size < 2:
        raise IndexError("Batch size must be at least 2 because we need 2 input images.")
    if hparams.batch_size % 2 != 0:
        raise IndexError("Batch size must be divisble by 2 because we feed pairs of images to the network.")

    trainer = Trainer(logger=logger, gpus=hparams.gpus, max_epochs=hparams.max_epochs)
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/specific/netapp5_3/rent_public/dcor-01-2021/ronmokady/workshop20/team6/ml4cg/data", help="Data root directory")
    parser.add_argument("--log_dir", type=str, default="/specific/netapp5_3/rent_public/dcor-01-2021/ronmokady/workshop20/team6/ml4cg/project/logs", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--img_size", type=int, default=128, help="Spatial size of training images")
    parser.add_argument("--max_epochs", type=int, default=8, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size during training")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs. Use 0 for CPU mode")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images")
    parser.add_argument("--nfe", type=int, default=32, help="Number of feature maps in encoders")
    parser.add_argument("--nz", type=int, default=256, help="Size of latent codes after encoders")
    parser.add_argument("--n_adain", type=int, default=4, help="Number of AdaIn layers in generator")
    parser.add_argument("--dim_adain", type=int, default=256, help="Dimension of AdaIn layer in generator")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight of reconstruction loss")
    parser.add_argument("--gamma", type=float, default=0.5, help="Weight of cycle loss for features a")
    parser.add_argument("--delta", type=float, default=0.5, help="Weight of cycle loss for features b")

    ### NOTES
    # we use same class and content code size, whereas LORD used content_dim=128, class_dim=256

    args = parser.parse_args()
    main(args)
