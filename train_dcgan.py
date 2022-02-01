#  Deep Convolutional Generative Adversarial Networks  ( Radford et al.)

import os
import argparse
import time


import numpy as np
import torch
from torch import nn
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import  matplotlib.pyplot as plt

from include.utils import  image_loader, get_gaussian_latent_batch, get_training_state
from include.models import  DiscriminativeConvNet, GenerativeConvNet

#--------------------------------------------------------------

# Parameters:

data_dir = 'data'    
latent_space_dim = 100
CHECKPOINTS_PATH = "checkpoints"
batch_size = 64
BINARIES_PATH = "binaries"
debug_path = 'trainig_trace' 
num_epochs = 100


def train_dcgan(training_config):
    writer = SummaryWriter()  # (tensorboard) writer will output to ./runs/ directory by default

 # Initialize data loader
    my_data_loader = image_loader(data_dir,batch_size=training_config['batch_size'])  #FAB

 # Define Discriminator and Generator (place them on GPU if present)
    discriminator_net = DiscriminativeConvNet().train().to(device)  #FAB
    generator_net     = GenerativeConvNet().train().to(device)  #FAB

 # Define optimizers which will tweak their weights
    discriminator_opt = Adam(discriminator_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    generator_opt     = Adam(generator_net.parameters(), lr=0.0002, betas=(0.5, 0.999))

 # Define the losses (binary cross-entropy)
    nn_loss = nn.BCELoss()

    real_images_vec = torch.ones( (training_config['batch_size'], 1, 1, 1), device=device)
    fake_images_vec = torch.zeros((training_config['batch_size'], 1, 1, 1), device=device)

 # For logging purposes
    ref_batch_size = 25
    ref_noise_batch = get_gaussian_latent_batch(latent_space_dim,ref_batch_size, device)  # Track G's quality during training #FAB
    discriminator_loss_values = []
    generator_loss_values = []
    save_cnt = 0      

    print("Starting training...");
     
 # Start training
    for epoch in range(training_config['num_epochs']):
        ts = time.time()  # start measuring time
        for batch_idx, (real_images, _) in enumerate(my_data_loader):

            
            real_images = real_images.to(device)  # Load images into GPU (if present)

         # Train discriminator

            discriminator_opt.zero_grad()  # Always do this in pytorch

            real_discriminator_loss = nn_loss(discriminator_net(real_images), real_images_vec)

            fake_images = generator_net(get_gaussian_latent_batch(latent_space_dim, training_config['batch_size'], device))  #FAB
            fake_images_predictions = discriminator_net(fake_images.detach())

            fake_discriminator_loss = nn_loss(fake_images_predictions, fake_images_vec)

            discriminator_loss = real_discriminator_loss + fake_discriminator_loss

            discriminator_loss.backward()  # this will populate .grad vars in the discriminator net
            discriminator_opt.step()  # perform D weights update according to optimizer's strategy

         # Train generator

            generator_opt.zero_grad()

            generated_images_predictions = discriminator_net(generator_net(get_gaussian_latent_batch(latent_space_dim, training_config['batch_size'], device)))  #FAB

            generator_loss = nn_loss(generated_images_predictions, real_images_vec)

            generator_loss.backward()  # this will populate .grad vars in the G net (also in D but we won't use those)
            generator_opt.step()  # perform G weights update according to optimizer's strategy


         # Save losses
            generator_loss_values.append(generator_loss.item())
            discriminator_loss_values.append(discriminator_loss.item())

            if training_config['enable_tensorboard']:
                writer.add_scalars('losses/g-and-d', {'g': generator_loss.item(), 'd': discriminator_loss.item()}, len(my_data_loader) * epoch + batch_idx + 1)
                # Save intermediate images to tensorboard as well 
                if training_config['debug_imagery_log_freq'] is not None and batch_idx % training_config['debug_imagery_log_freq'] == 0:
                    with torch.no_grad():
                        log_generated_images = generator_net(ref_noise_batch)
                        log_generated_images_resized = nn.Upsample(scale_factor=2, mode='nearest')(log_generated_images)
                        intermediate_imagery_grid = make_grid(log_generated_images_resized, nrow=int(np.sqrt(ref_batch_size)), normalize=True)
                        writer.add_image('intermediate generated imagery', intermediate_imagery_grid, len(my_data_loader) * epoch + batch_idx + 1)

            if training_config['console_log_freq'] is not None and (batch_idx+1) % training_config['console_log_freq'] == 0:
                print(f'Epoch={epoch + 1}/{num_epochs} | time elapsed= {(time.time() - ts):.2f}s | batch= {batch_idx + 1}/{len(my_data_loader)} | G_loss={generator_loss.item():1.3f} | D_loss={discriminator_loss.item():1.3f}' )
                ts = time.time()  # start measuring time


     # Save intermediate generator images to folder
        with torch.no_grad():
            log_generated_images = generator_net(ref_noise_batch)
            log_generated_images_resized = nn.Upsample(scale_factor=2, mode='nearest')(log_generated_images)
            save_image(log_generated_images_resized, os.path.join(training_config['debug_path'], f'Gen{str(save_cnt).zfill(4)}.png'), nrow=int(np.sqrt(ref_batch_size)), normalize=True)
            save_cnt += 1
            image = log_generated_images_resized[0].cpu().transpose(0,2).transpose(0,1)
            plt.close('all')
            plt.matshow(image)
            plt.draw()
            plt.pause(0.1)
            
 # Save the latest generator in the binaries directory
    torch.save(get_training_state(generator_net, "DCGAN"), os.path.join(BINARIES_PATH, "DCGAN"))  #FAB


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU
    print("Using %s " % device)

    os.makedirs(debug_path, exist_ok=True)
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
    os.makedirs(BINARIES_PATH, exist_ok=True)  
 # Parse command line 

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, help="height of content and style images", default=num_epochs)
    parser.add_argument("--batch_size", type=int, help="height of content and style images", default=batch_size)  #FAB
    parser.add_argument("--enable_tensorboard", type=bool, help="enable tensorboard logging (D and G loss)", default=True)
    parser.add_argument("--debug_imagery_log_freq", type=int, help="log generator images during training (batch) freq", default=5000)  #FAB
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq", default=15)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq", default=5)
    parser.add_argument("--data_dir", type=str, help="data directory", default=data_dir)  #FAB
    args = parser.parse_args()

 # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    training_config['debug_path'] = debug_path

    num_epochs = training_config['num_epochs']

 # train GAN model
    train_dcgan(training_config)

