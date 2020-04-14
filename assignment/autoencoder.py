__author__ = 'Alexander Koenig, Li Nguyen'

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Logger setup
logFormatter = logging.Formatter("%(asctime)s - %(name)s - [%(levelname)s]  %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger = logging.getLogger("AUTOENCODER")
logger.addHandler(consoleHandler)
logger.setLevel(logging.DEBUG)
logger.debug("Starting program")

# Root directory for dataset
dataroot = "/home/dcor/ronmokady/workshop20"
logger.debug(f"Dataroot is {dataroot}")

workers = 8        # Number of workers for dataloader
batch_size = 8     # Batch size during training (assignment requirement > 2)
image_size = 128   # Spatial size of training images (assignment requirement 128x128)
nc = 3             # Number of channels in the training images (RGB)
nz = 256           # Size of latent vector z (assigment requirement 256)
nfe = 64           # Size of feature maps in encoder
nfd = 64           # Size of feature maps in decoder
num_epochs = 5     # Number of training epochs
lr = 0.0002        # Learning rate for optimizers
beta1 = 0.5        # Beta1 hyperparameter for Adam optimizer
beta2 = 0.999      # Beta2 hyperparameter for Adam optimizer
ngpu = 2           # Number of GPUs available. Use 0 for CPU mode

# Resize and center-crop images, normalize all channels
logger.debug(f"Loading dataset with {workers} workers")
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (
    torch.cuda.is_available() and ngpu > 0) else "cpu")

logger.debug(f"Device you are running is \"{device}\"")

# TODO Train / Val split
# TODO Plot some training images (do this with Tensorboard)
# TODO Save models
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))         
 
# Reminder: Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)

class Autoencoder(nn.Module):
    def __init__(self, ngpu):
        super(Autoencoder, self).__init__()
        self.ngpu = ngpu
        self.encoder = nn.Sequential(
            # Layer 1: Input is (nc) x 128 x 128
            nn.Conv2d(nc, nfe, 4, bias=False),
            nn.BatchNorm2d(nfe),
            nn.ReLU(True),

            # Layer 2: State size is (nfe) x 64 x 64
            nn.Conv2d(nfe, nfe * 2, 4, bias=False),
            nn.BatchNorm2d(nfe * 2),
            nn.ReLU(True),

            # Layer 3: State size is (nfe*2) x 32 x 32
            nn.Conv2d(nfe * 2, nfe * 4, 4, bias=False),
            nn.BatchNorm2d(nfe * 4),
            nn.ReLU(True),

            # Layer 4: State size is (nfe*4) x 16 x 16
            nn.Conv2d(nfe * 4, nfe * 8, 4, bias=False),
            nn.BatchNorm2d(nfe * 8),
            nn.ReLU(True),

            # Layer 5: State size is (nfe*8) x 8 x 8
            nn.Conv2d(nfe * 8, nfe * 16, 4, bias=False),
            nn.BatchNorm2d(nfe * 16),
            nn.ReLU(True),
            
            # Layer 6: State size is (nfe*16) x 4 x 4
            nn.Conv2d(nfe * 16, nz, 4, bias=False),
            nn.ReLU(True)

            # Output size is (nz) x 1 x 1
        )

        self.decoder = nn.Sequential(             
            # Layer 1: Input is (nz) x 1 x 1
            nn.ConvTranspose2d(nz, nfd * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nfd * 16),
            nn.LeakyReLU(True),

            # Layer 2: State size is (nfd*16) x 4 x 4
            nn.ConvTranspose2d(nfd * 16, nfd * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfd * 8),
            nn.LeakyReLU(True),

            # Layer 3: State size is (nfd*8) x 8 x 8
            nn.ConvTranspose2d(nfd * 8, nfd * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfd * 4),
            nn.LeakyReLU(True),

            # Layer 4: State size is (nfd*4) x 16 x 16
            nn.ConvTranspose2d(nfd * 4, nfd * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfd * 2),
            nn.LeakyReLU(True),

            # Layer 5: State size is (nfd*2) x 32 x 32
            nn.ConvTranspose2d(nfd * 2, nfd, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfd),
            nn.LeakyReLU(True),

            # Layer 6: State size is (nfd) x 64 x 64
            nn.ConvTranspose2d(nfd, nc, 4, 2, 1, bias=False),
            nn.Tanh()

            # Output size is (nc) x 128 x 128
        )
            
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# custom weight initialization called on autoencoder
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

autoencoder = Autoencoder(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    logger.debug(f"Preparing model for {ngpu} GPUs")
    autoencoder = nn.DataParallel(autoencoder, list(range(ngpu)))

logger.debug(f"Autoencoder architecture is\n{autoencoder}")

# Randomly initialize all weights to mean=0 and stdev=0.2
logger.debug("Initializing weights")
autoencoder.apply(weights_init)

# Initialize MSE Loss and Adam optimizer 
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=lr, betas=(beta1, beta2))

logger.debug("Starting training loop")
train_losses = []

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):

        # retrieve batch
        img, _ = data
        img = img.to(device)
        
        # forward pass
        output = autoencoder(img)
        loss = criterion(output, img)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.detach())

        # Output training stats
        if i % 50 == 0:
            logger.debug(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\t Training Loss: {loss.item()}")