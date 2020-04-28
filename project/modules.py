from argparse import ArgumentParser

import torch
import torch.nn as nn

from blocks import *

class EncoderA(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # Input (3 * 128 * 128), Output (512 * 2 * 2)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class EncoderB(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # Input (3 * 128 * 128), Output (512 * 2 * 2)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):        
        return self.net(x)

# TODO: Input [(512 * 2 * 2), (512 * 2 * 2)], Output (3 * 128 * 128)
class Generator(nn.Module):
    """Implementation of Generator from https://github.com/NVlabs/FUNIT/blob/4cd5f22cc330bbf1804db7d5364e870e412c324a/blocks.py 
    """
    def __init__(self, ups=3, n_res=2, dim=512, out_dim=3, res_norm="adain", activ="relu", pad_type="reflect"):
        super().__init__()

        self.net = []
        self.net += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        for i in range(ups):
            self.net += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2
        self.net += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dim, n_blk, norm, activ):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(in_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, out_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

# only for debugging purposes
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_size", type=int, default=128, help="Spatial size of training images")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size during training")
    hparams = parser.parse_args()

    # create two latent vectors
    img_1 = torch.rand((hparams.batch_size, 3, 128, 128))
    img_2 = torch.rand((hparams.batch_size, 3, 128, 128))
    
    enc_a = EncoderA(hparams)
    enc_b = EncoderB(hparams)
    gen = Generator()

    code_a = enc_a(img_1)
    code_b = enc_a(img_2)
    mixed = gen(code_a, code_b)

