import torch
import torchvision.transforms as transforms
from torch import nn

from .funit import MLP, ClassModelEncoder, ContentEncoder, Decoder, assign_adain_params, get_num_adain_params
from .resnet import init_id_encoder

"""
This file defines the G2G architecture
Some of the below code comes from https://github.com/NVlabs/FUNIT/blob/master/networks.py 
"""

# normalization constants for ID Encoder
MEAN_ID = torch.tensor([131.0912, 103.8827, 91.4953], dtype=torch.float32)
STD_ID = torch.tensor([1, 1, 1], dtype=torch.float32)

# normalization constants for FUNIT
MEAN_FUNIT = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
STD_FUNIT = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)


class Generator(nn.Module):
    def __init__(self, nf, nf_mlp, down_class, down_content, n_mlp_blks, n_res_blks, latent_dim, face_detector_pth):
        super(Generator, self).__init__()
        """
        Params:
            - nf: number of feature maps of encoder and decoder
            - nf_mlp: number of feature maps for MLP module, i.e. dimension of FC layers (256)
            - down_class: how often image is downsampled by half of its size in class encoder
            - down_content: how often image is downsampled by half of its size in content encoder
            - n_mlp_blks: Number of FC layers in MLP module, in this case 3
            - n_res_blks: number of ResBlks in content encoder, i.e. 2
            - latent_dim: latent dimension of class code, i.e. 2048
        """

        self.enc_id = init_id_encoder(face_detector_pth)

        self.enc_content = ContentEncoder(down_content, n_res_blks, 3, nf, "in", activ="relu", pad_type="reflect")

        self.dec = Decoder(
            down_content,
            n_res_blks,
            self.enc_content.output_dim,
            3,
            res_norm="adain",
            activ="relu",
            pad_type="reflect",
        )

        self.mlp = MLP(latent_dim, get_num_adain_params(self.dec), nf_mlp, n_mlp_blks, norm="none", activ="relu",)

        # pretrained id encoder requires different normalization
        self.id_norm = transforms.Normalize(MEAN_ID.tolist(), STD_ID.tolist())
        self.funit_denorm = transforms.Normalize((-MEAN_FUNIT / STD_FUNIT).tolist(), (1.0 / STD_FUNIT).tolist())

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
        # disassembly of original images: get content and identity code for each image x1 and x2
        x1_c, x1_id = self.encode(x1)
        x2_c, x2_id = self.encode(x2)

        # generate mixed images
        m1 = self.decode(x1_c, x2_id)
        m2 = self.decode(x2_c, x1_id)

        # reconstruct input images
        x1_hat = self.decode(x1_c, x1_id)
        x2_hat = self.decode(x2_c, x2_id)

        # disassembly of mixed images
        m1_c, m1_id = self.encode(m1)
        m2_c, m2_id = self.encode(m2)

        # generate reconstructed images
        r1 = self.decode(m1_c, m2_id)
        r2 = self.decode(m2_c, m1_id)

        return {
            "x1_c": x1_c,
            "x1_id": x1_id,
            "x2_c": x2_c,
            "x2_id": x2_id,
            "m1": m1,
            "m2": m2,
            "x1_hat": x1_hat,
            "x2_hat": x2_hat,
            "m1_c": m1_c,
            "m1_id": m1_id,
            "m2_c": m2_c,
            "m2_id": m2_id,
            "r1": r1,
            "r2": r2,
        }

    def scale_for_id_encoder(self, imgs):
        """Scales a batch of images from FUNIT normalization to VGGFace2 normalization
        """

        scaled_imgs = imgs.clone()
        num_imgs = scaled_imgs.shape[0]

        # denormalize images to transform FUNIT range [-1,1] to [0,1]
        for i in range(num_imgs):
            scaled_imgs[i] = self.funit_denorm(imgs[i])

        # scale to range [0,255]
        scaled_imgs *= 255

        # normalize with VGGFace2 mean and std
        for i in range(num_imgs):
            scaled_imgs[i] = self.id_norm(scaled_imgs[i])

        return scaled_imgs

    def encode(self, x):
        content_code = self.enc_content(x)
        id_code, _ = self.enc_id(self.scale_for_id_encoder(x))
        return content_code, id_code

    def decode(self, content_code, id_code):
        adain_params = self.mlp(id_code)
        assign_adain_params(adain_params, self.dec)
        images = self.dec(content_code)
        return images
