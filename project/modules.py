import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
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
        x = self.net(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return x


class Generator(nn.Module):
    """Implementation adapted from https://github.com/avivga/lord-pytorch/blob/master/model/modules.py
    """

    def __init__(self, hparams):
        super().__init__()

        self.initial_img_size = hparams.img_size // (2 ** hparams.n_adain)
        self.dim_adain = hparams.dim_adain

        self.fc_layers = nn.Sequential(
            nn.Linear(
                in_features=hparams.nz, out_features=self.initial_img_size ** 2 * (hparams.dim_adain // 8),
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=self.initial_img_size ** 2 * (hparams.dim_adain // 8),
                out_features=self.initial_img_size ** 2 * (hparams.dim_adain // 4),
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=self.initial_img_size ** 2 * (hparams.dim_adain // 4),
                out_features=self.initial_img_size ** 2 * hparams.dim_adain,
            ),
            nn.LeakyReLU(),
        )

        self.adain_conv_layers = nn.ModuleList()
        for i in range(hparams.n_adain):
            self.adain_conv_layers += [
                nn.Upsample(scale_factor=(2, 2)),
                nn.Conv2d(
                    in_channels=hparams.dim_adain, out_channels=hparams.dim_adain, padding=1, kernel_size=3,
                ),
                nn.LeakyReLU(),
                AdaptiveInstanceNorm2d(idx=i),
            ]

        self.adain_conv_layers = nn.Sequential(*self.adain_conv_layers)

        self.last_conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=hparams.dim_adain, out_channels=64, padding=2, kernel_size=5),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=hparams.nc, padding=3, kernel_size=7),
            nn.Sigmoid(),
        )

    def assign_adain_params(self, adain_params):
        for m in self.adain_conv_layers.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                m.bias = adain_params[:, m.idx, :, 0]
                m.weight = adain_params[:, m.idx, :, 1]

    def forward(self, content_code, class_adain_params):
        self.assign_adain_params(class_adain_params)

        x = self.fc_layers(content_code)
        x = x.reshape(-1, self.dim_adain, self.initial_img_size, self.initial_img_size)
        x = self.adain_conv_layers(x)
        x = self.last_conv_layers(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, hparams):
        super(Discriminator, self).__init__()
        self.hparams = hparams
        self.net = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(hparams.nc, hparams.nfd, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nfd) x 64 x 64
            nn.Conv2d(hparams.nfd, hparams.nfd * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hparams.nfd*2) x 32 x 32
            nn.Conv2d(hparams.nfd*2, hparams.nfd * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hparams.nfd*4) x 16 x 16
            nn.Conv2d(hparams.nfd * 4, hparams.nfd * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hparams.nfd*8) x 8 x 8
            nn.Conv2d(hparams.nfd * 8, hparams.nfd * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hparams.nfd*16) x 4 x 4
            nn.Conv2d(hparams.nfd * 16, 1, 4, 1, 0, bias=False),
            # final probability of input image being fake or real (binary classification)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), 1)
        return x

class Modulation(nn.Module):
    """Implementation adapted from https://github.com/avivga/lord-pytorch/blob/master/model/modules.py
    """

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.adain_per_layer = nn.ModuleList(
            [
                nn.Linear(in_features=hparams.nz, out_features=hparams.dim_adain * 2)
                for _ in range(hparams.n_adain)
            ]
        )

    def forward(self, x):
        adain_all = torch.cat([f(x) for f in self.adain_per_layer], dim=-1)
        adain_params = adain_all.reshape(-1, self.hparams.n_adain, self.hparams.dim_adain, 2)

        return adain_params


class AdaptiveInstanceNorm2d(nn.Module):
    """Implementation adapted from https://github.com/avivga/lord-pytorch/blob/master/model/modules.py
    """

    def __init__(self, idx):
        super().__init__()
        self.weight = None
        self.bias = None
        self.idx = idx

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]

        x_reshaped = x.contiguous().view(1, b * c, *x.shape[2:])
        weight = self.weight.contiguous().view(-1)
        bias = self.bias.contiguous().view(-1)

        out = F.batch_norm(
            x_reshaped, running_mean=None, running_var=None, weight=weight, bias=bias, training=True,
        )

        out = out.view(b, c, *x.shape[2:])
        return out
