__author__ = "Alexander Koenig, Li Nguyen"

from argparse import ArgumentParser
import pytorch_lightning as pl
import torch as torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import models
from pytorch_lightning import Trainer, loggers
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchsummary import summary
from collections import OrderedDict

from pytorch_lightning.callbacks import ModelCheckpoint

# imports from other files
import networks
from GANLoss import GANLoss
from vgg import Vgg16

# normalization constants
MEAN = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
STD = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

class Net(pl.LightningModule):

    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams = hparams
        self.criterionGAN = GANLoss(self.hparams.gan_mode) # define GAN loss
        self.gen = networks.Generator(hparams)
        self.dis = networks.define_D(self.hparams.nc, self.hparams.nfd, self.hparams.dis_arch,
                                            self.hparams.n_layers_D, self.hparams.norm, self.hparams.init_type, self.hparams.init_gain, self.hparams.gpu_ids)

        self.vgg = Vgg16()
        #if hparams.gpus > 0:
        #    self.vgg.cuda()            

        # cache for generated images
        self.generated_imgs = None

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
            shuffle=False,
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )

    def configure_optimizers(self):
        lr_g = self.hparams.lr_gen
        lr_d = self.hparams.lr_dis
        b1 = self.hparams.beta1
        b2 = self.hparams.beta2
        gen_opt = Adam(self.gen.parameters(), lr=lr_g, betas=(b1, b2))
        dis_opt = Adam(self.dis.parameters(), lr=lr_d, betas=(b1, b2))
        return [gen_opt, dis_opt], []

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
            if n > 1 and i < n-1:
                border_width = 6
                border = torch.zeros(plot.shape[0], plot.shape[1], border_width, device=x1[0].device)
                plot = torch.cat((plot, border), 2)

        name = f"{prefix}_input_mixed_reconstr_images"
        self.logger.experiment.add_image(name, plot)
   
    def training_step(self, batch, batch_idx, optimizer_idx):

        # retrieve batch and split in half, first half represents domain A other half represents domain B
        imgs, _ = batch
        split_idx = imgs.shape[0] // 2
        x1 = imgs[:split_idx]
        x2 = imgs[split_idx:]

        # train generator
        if optimizer_idx == 0:

            # forward pass: generate images by passing through generator
            out = self.gen(x1, x2)

            # save all generated images to be classified by discriminator in array (initializing the generated_imgs object)                                
            self.generated_imgs = torch.cat((out['m1'], out['m2']), 0)

            """
            Explanation of the number of generated images:
            We have a default batch size of 16, we divide this by 2, 
            to get 8 images belonging to domain A, and 8 for domain B.
            These will then be combined pairwise, which results in 8 mixed images 
            for each feature as the separate feature, i.e. 16 mixed images.
            The same applies for the reconstructed images again: 16 images
            This results in 32 images in total.
            Therefore, generated_imgs should have shape ([32, 3, 128, 128]) if both mixed and reconstructed images are used for the gan loss.
            If using only mixed images, generated_imgs should have shape ([16, 3, 128, 128]) respectively.
            """
            # reconstruction loss using VGG perceptual loss: get vgg features
            orig_features_1 = self.vgg(x1)
            orig_features_2 = self.vgg(x2)

            # x2_features = vgg(x2)
            recon_features_1 = self.vgg(out['r1'])
            recon_features_2 = self.vgg(out['r2'])

            # comparing features at second layer of VGG
            orig1 = orig_features_1[1]
            orig2 = orig_features_2[1]
            recon1 = recon_features_1[1]
            recon2 = recon_features_2[1]

            # calculate reconstruction loss (h_relu_2_2)
            vgg_loss_a = F.mse_loss(orig1, recon1)
            vgg_loss_b = F.mse_loss(orig2, recon2)

            # cycle consistency loss
            cycle_loss_a = F.mse_loss(out["x1_a"], out["m1_a"]) + F.mse_loss(out["x2_a"], out["m2_a"])
            cycle_loss_b = F.mse_loss(out["x1_b"], out["m2_b"]) + F.mse_loss(out["x2_b"], out["m1_b"])

            g_loss = self.criterionGAN(self.dis(self.generated_imgs), True)

            # over all loss for generator
            loss = self.hparams.alpha * (vgg_loss_a + vgg_loss_b) + self.hparams.gamma * (cycle_loss_a + cycle_loss_b) + self.hparams.lambda_g * g_loss

            # plot input, mixed and reconstructed images at beginning of epoch
            #if batch_idx == 0:
            #    self.plot((x1, x2), (out["m1"], out["m2"]), (out["r1"], out["r2"]), prefix='')

            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator: Measure discriminator's ability to classify real from generated samples
        if optimizer_idx == 1:

            real_loss = self.criterionGAN(self.dis(imgs), True)

            fake_loss = self.criterionGAN(self.dis(self.generated_imgs.detach()), False)

            # we take the objective times 0.5 which leads to the discriminator learning at half speed as compared to the generator
            d_loss = ((real_loss + fake_loss) / 2) * 0.5

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, prefix="val", plot=True)

    def validation_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, prefix="test", plot=True)

    def test_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, "test")

    def _shared_eval(self, batch, batch_idx, prefix="", plot=False):
        """
        This method is used for validation and test step, since the quality can not only be induced by the loss.
        Instead, qualitative and quantitative methods have to be used. 
        Therefore, the GAN loss is not inluded in the validation and test step.
        """

        # retrieve batch and split in half
        imgs, _ = batch
        split_idx = imgs.shape[0] // 2
        x1 = imgs[:split_idx]
        x2 = imgs[split_idx:]

        # forward pass
        out = self(x1, x2)

        # reconstruction loss using VGG perceptual loss: get vgg features
        orig_features_1 = self.vgg(x1)
        orig_features_2 = self.vgg(x2)

        # x2_features = vgg(x2)
        recon_features_1 = self.vgg(out['r1'])
        recon_features_2 = self.vgg(out['r2'])

        # compare features in all 4 layers of vgg
        # loss = 0.0
        # for i in range(4):
        #     orig1 = orig_features_1[i]      
        #     orig2 = orig_features_2[i]      
        #     recon1 = recon_features_1[i]
        #     recon2 = recon_features_2[i]
        #     loss += F.mse_loss(orig1, recon1)
        #     loss += F.mse_loss(orig2, recon2)

        orig1 = orig_features_1[1]
        orig2 = orig_features_2[1]
        recon1 = recon_features_1[1]
        recon2 = recon_features_2[1]

        # calculate content loss (h_relu_2_2)
        vgg_loss_a = F.mse_loss(orig1, recon1)
        vgg_loss_b = F.mse_loss(orig2, recon2)

        # cycle consistency loss
        cycle_loss_a = F.mse_loss(out["x1_a"], out["m1_a"]) + F.mse_loss(out["x2_a"], out["m2_a"])
        cycle_loss_b = F.mse_loss(out["x1_b"], out["m2_b"]) + F.mse_loss(out["x2_b"], out["m1_b"])

        # overall loss
        loss = self.hparams.alpha * (vgg_loss_a + vgg_loss_b) + self.hparams.gamma * (cycle_loss_a + cycle_loss_b)

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
    logger = loggers.TensorBoardLogger(hparams.log_dir, name=f"{hparams.gamma}-CycleGAN_G2G")

    model = Net(hparams)

    ### TESTING ONLY ###
    # checkpoint_callback = ModelCheckpoint(
    #     filepath='checkpoints/{epoch}-{val_loss:.2f}',
    #     save_top_k=200,
    #     verbose=True,
    #     monitor='val_loss',
    #     save_weights_only=True,
    #     period=1,
    #     mode='min',
    #     prefix=''
    # )

    trainer = Trainer(logger=logger, resume_from_checkpoint="checkpoints200/epoch=40-val_loss=0.00.ckpt", gpus=hparams.gpus, max_epochs=hparams.max_epochs, distributed_backend='ddp')
    # trainer.fit(model)
    trainer.test(model)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../../../data/celebA", help="Data root directory")
    parser.add_argument("--log_dir", type=str, default="200_GAN_logs_test", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--img_size", type=int, default=128, help="Spatial size of training images")
    parser.add_argument("--max_epochs", type=int, default=200, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size during training")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--gpus", type=int, default=4, help="Number of GPUs. Use 0 for CPU mode")

    parser.add_argument("--nf", type=int, default=64, help="Number of feature maps in encoders")
    parser.add_argument("--nf_mlp", type=int, default=256, help="Number of feature maps for MLP module, i.e. dimension of FC layers")
    parser.add_argument("--down_class", type=int, default=4, help="How often image is downsampled by half of its size in class encoder")
    parser.add_argument("--down_content", type=int, default=3, help="How often image is downsampled by half of its size in content encoder")
    parser.add_argument("--n_mlp_blks", type=int, default=3, help="Number of FC layers in MLP module")
    parser.add_argument("--n_res_blks", type=int, default=2, help="number of ResBlks in content encoder")
    parser.add_argument("--latent_dim", type=int, default=1024, help="Size of latent class code")

    parser.add_argument("--alpha", type=float, default=1.0, help="Weight of vgg perceptual loss")
    parser.add_argument("--gamma", type=float, default=10.0, help="Weight of cycle consistency losses")

    # hyper parameters for adversarial training
    parser.add_argument("--lr_gen", type=float, default=0.0002, help="Learning rate of generator network")
    parser.add_argument("--lr_dis", type=float, default=0.0002, help="Learning rate of discriminator network")
    parser.add_argument("--nc", type=int, default=3, help="The number of channels in input images")
    parser.add_argument("--nfd", type=int, default=64, help="The number of filters in the first conv layer of the discriminator")
    parser.add_argument("--dis_arch", type=str, default='basic', help="The architecture's name: basic | n_layers | pixel")
    parser.add_argument("--n_layers_D", type=int, default=3, help="The number of conv layers in the discriminator; effective when netD=='n_layers'")
    parser.add_argument("--norm", type=str, default='instance', help="The type of normalization layers used in the network, either BN or IN.")
    parser.add_argument("--init_type", type=str, default='normal', help="The name of the initialization method for network weights")
    parser.add_argument("--init_gain", type=float, default=0.02, help="Scaling factor for normal, xavier and orthogonal")
    parser.add_argument("--gan_mode", type=str, default='lsgan', help="The type of GAN objective. It currently supports vanilla, lsgan, and wgangp.")
    parser.add_argument("--gpu_ids", type=list, default=[2,3,4,5], help="Which GPUs the network runs on: e.g., 0,1,2")
    parser.add_argument("--lambda_g", type=float, default=1.0, help="Weight of generator loss")

    args = parser.parse_args()

    if args.batch_size < 2:
        raise IndexError("Batch size must be at least 2 because we need 2 input images.")
    if args.batch_size % 2 != 0:
        raise IndexError("Batch size must be divisble by 2 because we feed pairs of images to the network.")

    main(args)
