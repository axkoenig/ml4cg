import torch as torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from args import parse_args
from networks.resnet import init_id_encoder

# normalization constants for ID Encoder
MEAN_ID = torch.tensor([131.0912, 103.8827, 91.4953], dtype=torch.float32)
STD_ID = torch.tensor([1, 1, 1], dtype=torch.float32)


def scale_for_id_encoder(imgs):
    scaled_imgs = imgs.clone()
    num_imgs = scaled_imgs.shape[0]
    id_norm = transforms.Normalize(MEAN_ID.tolist(), STD_ID.tolist())

    # scale to range [0,255]
    scaled_imgs *= 255

    # normalize with VGGFace2 mean and std
    for i in range(num_imgs):
        scaled_imgs[i] = id_norm(scaled_imgs[i])

    return scaled_imgs


def main(hparams):
    transform = transforms.Compose(
        [
            transforms.Resize(hparams.img_size),
            transforms.CenterCrop(hparams.img_size),
            transforms.ToTensor(),
        ]
    )
    
    id_enc = init_id_encoder(hparams.face_detector_pth)

    img1 = Image.open("/specific/netapp5_3/rent_public/dcor-01-2021/ronmokady/workshop20/team6/data/celebA/images/000001.jpg")
    img2 = Image.open("/specific/netapp5_3/rent_public/dcor-01-2021/ronmokady/workshop20/team6/data/celebA/images/000402.jpg")
    img1 = transform(img1).unsqueeze(0)
    img2 = transform(img2).unsqueeze(0)

    enc1, _ = id_enc(scale_for_id_encoder(img1))
    enc2, _ = id_enc(scale_for_id_encoder(img2))

    loss = F.mse_loss(enc1, enc2)
    print(f"loss is {loss}")

if __name__ == "__main__":

    args = parse_args()
    main(args)
