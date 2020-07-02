from torch import nn

from .funit import ClassModelEncoder, ContentEncoder, Decoder, MLP, get_num_adain_params, assign_adain_params
from .resnet import init_id_encoder

"""
This file defines the G2G architecture
Some of the below code comes from https://github.com/NVlabs/FUNIT/blob/master/networks.py 
"""

class Generator(nn.Module):
    def __init__(self, nf, nf_mlp, down_content, n_mlp_blks, n_res_blks, latent_dim, face_detector_pth):
        super(Generator, self).__init__()
        """
        Params:
            - nf: number of feature maps of encoder and decoder
            - nf_mlp: number of feature maps for MLP module, i.e. dimension of FC layers (256)
            - down_class: how often image is downsampled by half of its size in class encoder
            - down_content: how often image is downsampled by half of its size in content encoder
            - n_mlp_blks: Number of FC layers in MLP module, in this case 3
            - n_res_blks: number of ResBlks in content encoder, i.e. 2
            - latent_dim: latent dimension of class code, i.e. 1x1x2048
        """

        self.enc_class_model = init_id_encoder(face_detector_pth)

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
