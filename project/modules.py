from argparse import ArgumentParser

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.net = nn.Sequential(
            # input (nc) x 128 x 128
            nn.Conv2d(hparams.nc, hparams.nfe, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe),
            nn.LeakyReLU(0.2, inplace=True),

            # input (nfe) x 64 x 64
            nn.Conv2d(hparams.nfe, hparams.nfe * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # input (nfe*2) x 32 x 32
            nn.Conv2d(hparams.nfe * 2, hparams.nfe * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # input (nfe*4) x 16 x 16
            nn.Conv2d(hparams.nfe * 4, hparams.nfe * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # input (nfe*8) x 8 x 8
            nn.Conv2d(hparams.nfe * 8, hparams.nfe * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # input (nfe*16) x 4 x 4
            nn.Conv2d(hparams.nfe * 16, hparams.nz, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hparams.nz),
            nn.LeakyReLU(0.2, inplace=True),
            # output (nz) x 1 x 1
        )

    def forward(self, x):
        return self.net(x)

# TODO: Input [(512 * 2 * 2), (512 * 2 * 2)], Output (3 * 128 * 128)

