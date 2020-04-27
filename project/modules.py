import torch
import torch.nn as nn

class EncoderA(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # Input (3 * 128 * 128), Output (512 * 2 * 2)
        self.full = nn.Sequential(
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

    def forward(self, net):
        net = self.full(net)
        print(net.shape) # gives us required shape 2x2x512
        return net

class EncoderB(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # Input (3 * 128 * 128), Output (512 * 2 * 2)
        self.full = nn.Sequential(
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

    def forward(self, net):
        net = self.full(net)
        print(net.shape) # gives us required shape 2x2x512
        return net

class Generator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # TODO: Input [(512 * 2 * 2), (512 * 2 * 2)], Output (3 * 128 * 128)
        self.net = nn.Sequential()

    def forward(self, x, y):
        return self.net(x, y)