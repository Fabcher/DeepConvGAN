import os, sys
import numpy as np
import torch
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

#-----------------------------------------------------------


def image_transform(set_size=64):
    transform = transforms.Compose([
        transforms.Resize(set_size),  #FAB
        transforms.ToTensor(),
        transforms.Normalize((.5,), (.5,))
    ])
    return transform


def image_loader(set_path, batch_size,set_size=64):  #FAB
    if not os.path.exists(set_path):  
        print("Data folder not found! Stopping program.")  #FAB
        sys.exit()  #FAB

    dataset = ImageFolder(set_path, transform=image_transform(set_size))  #FAB
    print("Found %i images" % len(dataset) )  #FAB
    ret_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return ret_data_loader


def get_gaussian_latent_batch(latent_space_dim, batch_size, device):  #FAB
    return torch.randn((batch_size, latent_space_dim), device=device)  #FAB



def get_training_state(generator_net, gan_type_name):
    training_state = {
        "state_dict": generator_net.state_dict(),
        "gan_type": gan_type_name
    }
    return training_state

