__author__ = "Alexander Koenig, Li Nguyen"

import gc

import numpy as np
import pytorch_lightning as pl
import torch as torch
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms as transforms
import torchvision.utils as vutils
import wandb
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader, Subset
from torchvision import models
from torchvision.datasets import ImageFolder

from args import parse_args
from networks import cyclegan, g2g, resnet, vgg

# normalization constants for FUNIT
MEAN_FUNIT = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
STD_FUNIT = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

# normalization constants for VGG16
MEAN_IMGNET = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
STD_IMGNET = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams = hparams

        self.gen = g2g.Generator(
            self.hparams.nf,
            self.hparams.nf_mlp,
            self.hparams.down_class,
            self.hparams.down_content,
            self.hparams.n_mlp_blks,
            self.hparams.n_res_blks,
            self.hparams.latent_dim,
            self.hparams.face_detector_pth,
        )
        self.dis = cyclegan.define_D(
            self.hparams.nc,
            self.hparams.nfd,
            self.hparams.dis_arch,
            self.hparams.n_layers_D,
            self.hparams.norm,
            self.hparams.init_type,
            self.hparams.init_gain,
        )
        self.vgg = vgg.Vgg16()
        self.gan_criterion = cyclegan.GANLoss(self.hparams.gan_mode)
        self.mixed_imgs = None

        self.funit_denorm = transforms.Normalize((-MEAN_FUNIT / STD_FUNIT).tolist(), (1.0 / STD_FUNIT).tolist())
        self.vgg_norm = transforms.Normalize(MEAN_IMGNET.tolist(), STD_IMGNET.tolist())

    def forward(self, x1, x2):
        """Forward pass of network
        Args:
            x1 (tensor): first input image
            x2 (tensor): second input image
        Returns:
            dict: codes, mixed and reconstruced images
        """
        return self.gen(x1, x2)

    def setup(self, mode):
        transform = transforms.Compose(
            [
                transforms.Resize(self.hparams.img_size),
                transforms.CenterCrop(self.hparams.img_size),
                transforms.ToTensor(),
                transforms.Normalize(MEAN_FUNIT.tolist(), STD_FUNIT.tolist()),
            ]
        )

        dataset = ImageFolder(root=self.hparams.data_root, transform=transform)

        # train, val and test split taken from "list_eval_partition.txt" of original celebA paper
        end_train_idx = 162770
        end_val_idx = 182637
        end_test_idx = len(dataset)

        self.train_dataset = Subset(dataset, range(0, end_train_idx))
        self.val_dataset = Subset(dataset, range(end_train_idx + 1, end_val_idx))
        self.test_dataset = Subset(dataset, range(end_val_idx + 1, end_test_idx))

        # define at which indices to plot during training
        num_train_batches = len(self.train_dataset) // self.hparams.batch_size
        self.train_plot_indices = np.linspace(0, num_train_batches, self.hparams.num_plots_per_epoch, dtype=int)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, drop_last=True,
        )

    def configure_optimizers(self):
        gen_opt = Adam(self.gen.parameters(), lr=self.hparams.lr_gen, betas=(self.hparams.beta1, self.hparams.beta2))
        dis_opt = Adam(self.dis.parameters(), lr=self.hparams.lr_dis, betas=(self.hparams.beta1, self.hparams.beta2))
        return [gen_opt, dis_opt], []

    def plot(self, input_batches, mixed_batches, reconstr_batches, prefix, caption=""):
        """Plots n triplets of ((x1, x2), (m1, m2), (r1, r2)) 
        Args:
            input_batches (tuple): Two batches of input images
            mixed_batches (tuple): Two batches of mixed images
            reconstr_batches (tuple): Two batches of reconstructed images
            prefix (str): Prefix for plot name
        Raises:
            IndexError: If n exceeds batch size
        """

        n = self.hparams.num_plot_triplets
        m = input_batches[0].shape[0]
        if m < n:
            raise IndexError(
                f"You are attempting to plot too many images. For --num_plot_triplets={n} your batch size must be at least {2*n}!"
            )

        # denormalize images
        denormalization = transforms.Normalize((-MEAN_FUNIT / STD_FUNIT).tolist(), (1.0 / STD_FUNIT).tolist())
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
            if n > 1 and i < n - 1:
                border_width = 6
                border = torch.zeros(plot.shape[0], plot.shape[1], border_width, device=x1[0].device)
                plot = torch.cat((plot, border), 2)

        name = f"{prefix}/input_mixed_reconstr_images"
        self.logger.experiment.log({name: [wandb.Image(plot, caption=caption)]})

    def scale_for_vgg(self, imgs):
        """Scales a batch of images from FUNIT normalization to ImageNet normalization
        """

        scaled_imgs = imgs.clone()
        num_imgs = scaled_imgs.shape[0]

        # denormalize images to transform FUNIT range [-1,1] to [0,1]
        for i in range(num_imgs):
            scaled_imgs[i] = self.funit_denorm(imgs[i])

        # normalize with ImageNet mean and std
        for i in range(num_imgs):
            scaled_imgs[i] = self.vgg_norm(scaled_imgs[i])

        return scaled_imgs

    def calc_g_loss(self, x1, x2, out, prefix):

        ### RECONSTRUCTION LOSS ###

        # long reconstruction loss
        orig_features_1 = self.vgg(self.scale_for_vgg(x1))[1]
        orig_features_2 = self.vgg(self.scale_for_vgg(x2))[1]
        recon_l_features_1 = self.vgg(self.scale_for_vgg(out["r1"]))[1]
        recon_l_features_2 = self.vgg(self.scale_for_vgg(out["r2"]))[1]
        vgg_loss_l = self.hparams.alpha_l * (
            F.l1_loss(orig_features_1, recon_l_features_1) + F.l1_loss(orig_features_2, recon_l_features_2)
        )

        # short reconstruction loss
        recon_s_features_1 = self.vgg(self.scale_for_vgg(out["x1_hat"]))[1]
        recon_s_features_2 = self.vgg(self.scale_for_vgg(out["x2_hat"]))[1]
        vgg_loss_s = self.hparams.alpha_s * (
            F.l1_loss(orig_features_1, recon_s_features_1) + F.l1_loss(orig_features_2, recon_s_features_2)
        )

        ### CYCLE CONSISTENCY LOSSES ###
        cycle_loss_c = self.hparams.gamma_c * (
            F.mse_loss(out["x1_c"], out["m2_c"]) + F.mse_loss(out["x2_c"], out["m1_c"])
        )
        cycle_loss_id = self.hparams.gamma_id * (
            F.mse_loss(out["x1_id"], out["m2_id"]) + F.mse_loss(out["x2_id"], out["m1_id"])
        )

        ### ADVERSARIAL LOSS ###
        self.mixed_imgs = torch.cat((out["m1"], out["m2"]), 0)
        adv_g_loss = self.hparams.delta * self.gan_criterion(self.dis(self.mixed_imgs), True)

        ### OVERALL GENERATOR LOSS ###
        loss = vgg_loss_l + vgg_loss_s + cycle_loss_c + cycle_loss_id + adv_g_loss
        log = {
            f"{prefix}/vgg_loss_l": vgg_loss_l,
            f"{prefix}/vgg_loss_s": vgg_loss_s,
            f"{prefix}/cycle_loss_c": cycle_loss_c,
            f"{prefix}/cycle_loss_id": cycle_loss_id,
            f"{prefix}/adv_g_loss": adv_g_loss,
        }

        return loss, log

    def split_batch(self, batch):
        # retrieve batch and split in half
        imgs, _ = batch
        split_idx = imgs.shape[0] // 2
        x1 = imgs[:split_idx]
        x2 = imgs[split_idx:]
        return x1, x2, imgs

    def training_step(self, batch, batch_idx, optimizer_idx):

        x1, x2, imgs = self.split_batch(batch)

        # GENERATOR STEP
        if optimizer_idx == 0:

            out = self.gen(x1, x2)
            loss, log = self.calc_g_loss(x1, x2, out, prefix="train")

            if batch_idx in self.train_plot_indices:
                caption = f"batch_idx: {batch_idx} | cur_epoch: {self.current_epoch}"
                self.plot((x1, x2), (out["m1"], out["m2"]), (out["r1"], out["r2"]), "train", caption)

            log.update({"train/g_loss": loss})
            return {"loss": loss, "progress_bar": log, "log": log}

        # DISCRIMINATOR STEP
        if optimizer_idx == 1:

            real_loss = self.gan_criterion(self.dis(imgs), True)
            fake_loss = self.gan_criterion(self.dis(self.mixed_imgs.detach()), False)
            d_loss = self.hparams.zeta * (real_loss + fake_loss) / 2

            log = {"train/d_loss": d_loss}
            return {"loss": d_loss, "progress_bar": log, "log": log}

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, prefix="val")

    def validation_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, prefix="test")

    def test_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, "test")

    def _shared_eval(self, batch, batch_idx, prefix):

        x1, x2, _ = self.split_batch(batch)
        out = self(x1, x2)
        loss, log = self.calc_g_loss(x1, x2, out, prefix=prefix)

        # plot at beginning of epoch
        if batch_idx == 0:
            self.plot((x1, x2), (out["m1"], out["m2"]), (out["r1"], out["r2"]), prefix)

        log.update({f"{prefix}/loss": loss})
        return {f"{prefix}_loss": loss, "log": log}

    def _shared_eval_epoch_end(self, outputs, prefix):
        avg_loss = torch.stack([x[f"{prefix}_loss"] for x in outputs]).mean()
        log = {f"{prefix}/avg_loss": avg_loss}
        return {f"avg_{prefix}_loss": avg_loss, "log": log}


def main(hparams):
    # clean up
    gc.collect()
    torch.cuda.empty_cache()

    logger = loggers.WandbLogger(name=hparams.log_name, project="ml4cg")

    model = Net(hparams)

    checkpoint_callback = ModelCheckpoint(
        filepath="checkpoints/{epoch}",
        save_top_k=20,
        verbose=True,
        monitor="val_loss",
        save_weights_only=True,
        period=1,
        mode="min",
        prefix="",
    )

    trainer = Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        gpus=hparams.gpus,
        max_epochs=hparams.max_epochs,
        num_sanity_val_steps=hparams.num_sanity_val_steps,
    )

    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":

    args = parse_args()
    main(args)
