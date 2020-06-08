"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np

import torch
from torch import tensor 
from torch import nn
from torch import autograd
import functools
from torch.nn import init
from torch.optim import lr_scheduler

# file imports
from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock

##### incorporates learned params in MLP module over AdaIN
def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    init_weights(net, init_type, init_gain=init_gain)
    return net


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator
    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a discriminator
    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70?~W70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.
        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


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

        # nf = hparams['nf']
        # nf_mlp = hparams['nf_mlp']

        self.enc_class_model = ClassModelEncoder(down_class,
                                                 3,
                                                 nf,
                                                 latent_dim,
                                                 norm='none',
                                                 activ='relu',
                                                 pad_type='reflect')

        self.enc_content = ContentEncoder(down_content,
                                          n_res_blks,
                                          3,
                                          nf,
                                          'in',
                                          activ='relu',
                                          pad_type='reflect')

        self.dec = Decoder(down_content,
                           n_res_blks,
                           self.enc_content.output_dim,
                           3,
                           res_norm='adain',
                           activ='relu',
                           pad_type='reflect')

        self.mlp = MLP(latent_dim,
                       get_num_adain_params(self.dec),
                       nf_mlp,
                       n_mlp_blks,
                       norm='none',
                       activ='relu')

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
        self.model += [Conv2dBlock(ind_im, dim, 7, 1, 3,
                                   norm=norm,
                                   activation=activ,
                                   pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1,
                                       norm=norm,
                                       activation=activ,
                                       pad_type=pad_type)]
            dim *= 2
        for i in range(downs - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1,
                                       norm=norm,
                                       activation=activ,
                                       pad_type=pad_type)]

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
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3,
                                   norm=norm,
                                   activation=activ,
                                   pad_type=pad_type)]
        for i in range(downs):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1,
                                       norm=norm,
                                       activation=activ,
                                       pad_type=pad_type)]
            dim *= 2
        self.model += [ResBlocks(n_res, dim,
                                 norm=norm,
                                 activation=activ,
                                 pad_type=pad_type)]
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
        self.model += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]
        for i in range(ups):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
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
        self.model += [LinearBlock(dim, out_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
