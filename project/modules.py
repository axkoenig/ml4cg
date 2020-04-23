import torch
import torch.nn as nn

class EncoderA(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # TODO: Input (3 * 128 * 128), Output (512 * 2 * 2)
        self.net = nn.Sequential()

    def forward(self, x):
        return self.net(x)

class EncoderB(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # TODO: Input (3 * 128 * 128), Output (512 * 2 * 2)
        self.net = nn.Sequential()

    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # TODO: Input [(512 * 2 * 2), (512 * 2 * 2)], Output (3 * 128 * 128)
        self.net = nn.Sequential()

    def forward(self, x, y):
        return self.net(x, y)