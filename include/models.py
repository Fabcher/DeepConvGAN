# DCGAN implementation as in the paper   ( Radford et al.)

import torch
from torch import nn
import numpy as np



def dcgan_upsample_block(in_channels, out_channels, normalize=True, activation=None):
    layers = [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU() if activation is None else activation)
    return layers


class GenerativeConvNet(nn.Module):

    def __init__(self,latent_space_dim=100):
        super().__init__()

        input_channels = 1024
        self.init_volume_shape = (input_channels, 4, 4)  #  Kernel size

        self.linear = nn.Linear(latent_space_dim, input_channels * np.prod(self.init_volume_shape[1:]))

        self.net = nn.Sequential(
            *dcgan_upsample_block(input_channels, 512),  # As defined in the paper
            *dcgan_upsample_block(512 , 256),
            *dcgan_upsample_block(256 , 128),
            *dcgan_upsample_block(128, 3, normalize=False, activation=nn.Tanh())
        )

    def forward(self, latent_vector_batch):
     # Starting from the latent space it generates an image
        latent_vector_batch_projected = self.linear(latent_vector_batch)
        latent_vector_batch_projected_reshaped = latent_vector_batch_projected.view(latent_vector_batch_projected.shape[0], *self.init_volume_shape)

        return self.net(latent_vector_batch_projected_reshaped)


def dcgan_downsample_block(in_channels, out_channels, normalize=True, activation=None, padding=1):
    layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=padding, bias=False)]
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2) if activation is None else activation)
    return layers


class DiscriminativeConvNet(nn.Module):

    def __init__(self):
        super().__init__()

     # It goes from the image all the way to a single output
        self.net = nn.Sequential(
            *dcgan_downsample_block(3  ,  128, normalize=False),
            *dcgan_downsample_block(128,  256),
            *dcgan_downsample_block(256,  512),
            *dcgan_downsample_block(512, 1024),
            *dcgan_downsample_block(1024,   1, normalize=False, activation=nn.Sigmoid(), padding=0),
        )

    def forward(self, img_batch):
        return self.net(img_batch)


