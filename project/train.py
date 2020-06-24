__author__ = "Alexander Koenig, Li Nguyen"

from collections import OrderedDict

import pytorch_lightning as pl
import torch as torch
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms as transforms
import torchvision.utils as vutils
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision import models
from torchvision.datasets import ImageFolder

from args import parse_args
from networks import cyclegan, funit, vgg, resnet
from networks.resnet import init_id_encoder
# from networks.vgg import Vgg16

# normalization constants
MEAN = torch.tensor([131.0912, 103.8827, 91.4953], dtype=torch.float32)
STD = torch.tensor([1, 1, 1], dtype=torch.float32)


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams = hparams
        
        self.gen = funit.Generator(
            self.hparams.nf,
            self.hparams.nf_mlp,
            self.hparams.down_class,
            self.hparams.down_content,
            self.hparams.n_mlp_blks,
            self.hparams.n_res_blks,
            self.hparams.latent_dim)
        self.dis = cyclegan.define_D(
            self.hparams.nc,
            self.hparams.nfd,
            self.hparams.dis_arch,
            self.hparams.n_layers_D,
            self.hparams.norm,
            self.hparams.init_type,
            self.hparams.init_gain,
        )
        # self.vgg = Vgg16()
        self.id_enc = init_id_encoder(self.hparams.face_detector_pth)
        self.gan_criterion = cyclegan.GANLoss(self.hparams.gan_mode)
        self.mixed_imgs = None

    def forward(self, x1, x2):
        """Forward pass of network
        Args:
            x1 (tensor): first input image
            x2 (tensor): second input image
        Returns:
            generated images
        """
        return self.gen(x1, x2)

    def prepare_data(self):

        transform = transforms.Compose(
            [
                transforms.Resize(self.hparams.img_size),
                transforms.CenterCrop(self.hparams.img_size),
                transforms.ToTensor(),
                transforms.Normalize(MEAN.tolist(), STD.tolist()),
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
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )

    def configure_optimizers(self):
        gen_opt = Adam(self.gen.parameters(), lr=self.hparams.lr_gen, betas=(self.hparams.beta1, self.hparams.beta2))
        dis_opt = Adam(self.dis.parameters(), lr=self.hparams.lr_dis, betas=(self.hparams.beta1, self.hparams.beta2))
        return [gen_opt, dis_opt], []

    def plot(self, input_batches, mixed_batches, reconstr_batches, prefix, n=4):
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
            raise IndexError(
                "You are attempting to plot more images than your batch contains!"
            )

        # denormalize images
        denormalization = transforms.Normalize(
            (-MEAN / STD).tolist(), (1.0 / STD).tolist()
        )
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
                border = torch.zeros(
                    plot.shape[0], plot.shape[1], border_width, device=x1[0].device
                )
                plot = torch.cat((plot, border), 2)

        name = f"{prefix}/input_mixed_reconstr_images"
        self.logger.experiment.add_image(name, plot)

    def id_loss_weight(self, n_epochs, n_epochs_increase, delta_max, delta_min = 0.0):
        """
        Parameters:
            - (int) n_epochs: We keep the same weight for the first <n_epochs> epochs
            - (int) n_epochs_increase: and successively increase the rate by 0.1 over the next <n_epochs_increase> epochs
            - (float) delta_max: Upper threshold which delta should reach by successive increase after <n_epochs_increase> epochs
            - (float) delta_min: Initial weight for delta kept for the first <n_epochs>
        Returns:
            - (float) delta: the weight for the id loss in the current epoch
        """
        if self.current_epoch > (n_epochs + n_epochs_increase):
            delta = delta_max
        else:
            delta = delta_min + max(0, self.current_epoch - n_epochs) * (delta_max / float(n_epochs_increase))
        return delta

    def calc_g_loss(self, x1, x2, out, prefix):

        ### RECONSTRUCTION LOSS ###
        reconstr_loss = F.l1_loss(x1, out["r1"]) + F.l1_loss(x2, out["r2"])        

        ### CYCLE CONSISTENCY LOSSES ###
        
        cycle_loss_a = F.mse_loss(out["x1_a"], out["m1_a"]) + F.mse_loss(out["x2_a"], out["m2_a"])
        cycle_loss_b = F.mse_loss(out["x1_b"], out["m2_b"]) + F.mse_loss(out["x2_b"], out["m1_b"])
        cycle_loss = cycle_loss_a + cycle_loss_b

        ### IDENTITY LOSSES ###
        
        # get identity encodings 
        orig_id_features_1, _ = self.id_enc(x1)
        orig_id_features_2, _ = self.id_enc(x2)
        mixed_id_features_1, _ = self.id_enc(out["m1"])
        mixed_id_features_2, _ = self.id_enc(out["m2"])

        id_loss = F.l1_loss(orig_id_features_1, mixed_id_features_2) + F.l1_loss(orig_id_features_2, mixed_id_features_1)

        ### ADVERSARIAL LOSS ###
        
        adv_g_loss = self.gan_criterion(self.dis(self.mixed_imgs), True)

        ### OVERALL GENERATOR LOSS ###
        delta = self.id_loss_weight(self.hparams.n_epochs, self.hparams.n_epochs_increase, self.hparams.delta_max, self.hparams.delta_min)
        loss = self.hparams.alpha * reconstr_loss + self.hparams.gamma * cycle_loss + delta * id_loss + self.hparams.lambda_g * adv_g_loss
        log = {f"{prefix}/vgg_loss": reconstr_loss, f"{prefix}/cycle_loss": cycle_loss, f"{prefix}/id_loss": id_loss, f"{prefix}/adv_g_loss": adv_g_loss, f"{prefix}/delta": delta}

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
            self.mixed_imgs = torch.cat((out["m1"], out["m2"]), 0)
            loss, log = self.calc_g_loss(x1, x2, out, prefix="train")

            # plot at beginning of epoch
            if batch_idx == 0:
                self.plot((x1, x2), (out["m1"], out["m2"]), (out["r1"], out["r2"]), "train")

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
    logger = loggers.TensorBoardLogger(hparams.log_dir, name=hparams.log_name)

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
        nb_sanity_val_steps=hparams.nb_sanity_val_steps,
        distributed_backend="ddp",
    )

    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":

    args = parse_args()
    main(args)
