"""
Below code is adapted from blocks.py and networks.py from https://github.com/NVlabs/FUNIT
"""


"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn.functional as F
from torch import nn


"""
Code below this line including classes and functions:
    - ResBlocks
    - ResBlock
    - LinearBlock
    - Conv2dBlock
    - AdaptiveInstanceNorm2d
taken from blocks.py from https://github.com/NVlabs/FUNIT.
"""

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, activation, pad_type):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [
                ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)
            ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm="in", activation="relu", pad_type="zero"):
        super(ResBlock, self).__init__()
        model = []
        # Conv2dBlock params: self, in_dim, out_dim, ks, st, padding=0, norm='none', activation='relu', pad_type='zero', use_bias=True, activation_first=False
        model += [
            Conv2dBlock(
                dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type
            )
        ]
        model += [
            Conv2dBlock(
                dim, dim, 3, 1, 1, norm=norm, activation="none", pad_type=pad_type
            )
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm="none", activation="relu"):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)

        # initialize normalization
        norm_dim = out_dim
        if norm == "bn":
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        ks,
        st,
        padding=0,
        norm="none",
        activation="relu",
        pad_type="zero",
        use_bias=True,
        activation_first=False,
    ):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first

        # initialize padding
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == "bn":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == "adain":
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    """
    Params:
        - num_features: number of feature maps, e.g. for decoder 512
        - eps: epsilon hyperparameter
        - momentum:
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"

"""
Code below this line including classes and functions from :
    - Generator (modified forward pass)
    - ClassModelEncoder
    - ContentEncoder
    - Decoder
    - MLP
    - assign_adain_params
    - get_num_adain_params
taken from networks.py from https://github.com/NVlabs/FUNIT.
"""

class Generator(nn.Module):
    def __init__(self, hparams):
        super(Generator, self).__init__()
        self.hparams = hparams
        """
        Params:
            - nf: number of feature maps of encoder and decoder
            - nf_mlp: number of feature maps for MLP module, i.e. dimension of FC layers (256)
            - down_class: how often image is downsampled by half of its size in class encoder
            - down_content: how often image is downsampled by half of its size in content encoder
            - n_mlp_blks: Number of FC layers in MLP module, in this case 3
            - n_res_blks: number of ResBlks in content encoder, i.e. 2
            - latent_dim: latent dimension of class code, i.e. 1024
        """
        nf = self.hparams.nf
        nf_mlp = self.hparams.nf_mlp
        down_class = self.hparams.down_class
        down_content = self.hparams.down_content
        n_mlp_blks = self.hparams.n_mlp_blks
        n_res_blks = self.hparams.n_res_blks
        latent_dim = self.hparams.latent_dim

        self.enc_class_model = ClassModelEncoder(
            down_class, 3, nf, latent_dim, norm="none", activ="relu", pad_type="reflect"
        )

        self.enc_content = ContentEncoder(
            down_content, n_res_blks, 3, nf, "in", activ="relu", pad_type="reflect"
        )

        self.dec = Decoder(
            down_content,
            n_res_blks,
            self.enc_content.output_dim,
            3,
            res_norm="adain",
            activ="relu",
            pad_type="reflect",
        )

        self.mlp = MLP(
            latent_dim,
            get_num_adain_params(self.dec),
            nf_mlp,
            n_mlp_blks,
            norm="none",
            activ="relu",
        )

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
        # disassembly of original images: get class and content code for each image x1 and x2
        x1_a, x1_b = self.encode(x1)
        x2_a, x2_b = self.encode(x2)

        # generate mixed images
        m1 = self.decode(x1_a, x2_b)
        m2 = self.decode(x2_a, x1_b)

        # disassembly of mixed images
        m1_a, m1_b = self.encode(m1)
        m2_a, m2_b = self.encode(m2)

        # generate reconstructed images
        r1 = self.decode(m1_a, m2_b)
        r2 = self.decode(m2_a, m1_b)

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

    def encode(self, x):
        # feed original images x1 and x2 into class and content encoder
        content_code = self.enc_content(x)
        class_code = self.enc_class_model(x)
        return content_code, class_code

    def decode(self, content_code, class_code):
        # decode content and style codes to an image
        adain_params = self.mlp(class_code)
        assign_adain_params(adain_params, self.dec)
        images = self.dec(content_code)
        return images


class ClassModelEncoder(nn.Module):
    """
    Params:
        - downs: how often image is downsampled by half of its size
        - ind_im: input dimensions of image, in this case 128 x 128 x 3
        - dim: number of feature maps encoder, we call it nfe and for our case 64
        - latent_dim: latent dimension of class code is 1 x 1 x latent_dim
        - norm: normalization, either BN, IN, AdaIN or none
        - activ: activation function, either relu, lrelu, tanh or none
        - pad_type: reflect, replicate or zero
    """

    def __init__(self, downs, ind_im, dim, latent_dim, norm, activ, pad_type):
        super(ClassModelEncoder, self).__init__()
        self.model = []
        self.model += [
            Conv2dBlock(
                ind_im, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type
            )
        ]
        for i in range(2):
            self.model += [
                Conv2dBlock(
                    dim,
                    2 * dim,
                    4,
                    2,
                    1,
                    norm=norm,
                    activation=activ,
                    pad_type=pad_type,
                )
            ]
            dim *= 2
        for i in range(downs - 2):
            self.model += [
                Conv2dBlock(
                    dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type
                )
            ]

        self.model += [nn.AdaptiveAvgPool2d(1)]
        self.model += [nn.Conv2d(dim, latent_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class ContentEncoder(nn.Module):
    """
    Params:
        - downs: how often image is downsampled by half of its size
        - n_res: number of residual blocks, in this case 2
        - input_dim: input dimension of image channels, i.e. 3 channels for 128 x 128 x 3
        - dim: number of feature maps encoder, we call it nfe and for our case 64
        - norm: normalization, either BN, IN, AdaIN or none
        - activ: activation function, either relu, lrelu, tanh or none
        - pad_type: reflect, replicate or zero
    """

    def __init__(self, downs, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [
            Conv2dBlock(
                input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type
            )
        ]
        for i in range(downs):
            self.model += [
                Conv2dBlock(
                    dim,
                    2 * dim,
                    4,
                    2,
                    1,
                    norm=norm,
                    activation=activ,
                    pad_type=pad_type,
                )
            ]
            dim *= 2
        self.model += [
            ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)
        ]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    """
    Params:
        - ups: how often image is upsampled by half of its size (using nearest neighbor)
        - n_res: number of residual blocks, in this case 2
        - dim: number of feature maps encoder, we call it nfe and for our case 64
        - out_dim: output dimension of image channels, i.e. 3 for 128 x 128 x 3
        - res_norm: normalization, here using AdaIn to incorporate class code
        - activ: activation function, either relu, lrelu, tanh or none
        - pad_type: reflect, replicate or zero
    """

    def __init__(self, ups, n_res, dim, out_dim, res_norm, activ, pad_type):
        super(Decoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        for i in range(ups):
            self.model += [
                nn.Upsample(scale_factor=2),
                Conv2dBlock(
                    dim,
                    dim // 2,
                    5,
                    1,
                    2,
                    norm="in",
                    activation=activ,
                    pad_type=pad_type,
                ),
            ]
            dim //= 2
        self.model += [
            Conv2dBlock(
                dim, out_dim, 7, 1, 3, norm="none", activation="tanh", pad_type=pad_type
            )
        ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    """
    Params:
        - in_dim: input dimension into MLP module is dimension of class code, i.e. 1 x 1 x 1024
        - out_dim: output dimension of MLP module, i.e. 256
        - dim: dimension of FC layers, i.e. 256
        - n_blk: how many FC layers
        - norm: normalization, either BN, IN, AdaIN or none
        - activ: activation function, either relu, lrelu, tanh or none
    """

    def __init__(self, in_dim, out_dim, dim, n_blk, norm, activ):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(in_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, out_dim, norm="none", activation="none")]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model via MLP module
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, : m.num_features]
            std = adain_params[:, m.num_features : 2 * m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[:, 2 * m.num_features :]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2 * m.num_features
    return num_adain_params
