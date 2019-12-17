import argparse
import os
import time
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

from gans import Generator, Discriminator, init_weights


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    cudnn.benchmark = True

def get_dataloader(batch_size, num_workers):

    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # we can use an image folder dataset
    dataset = datasets.ImageFolder(root='data/celeba', transform=tfms)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        pin_memory=device.type == 'cuda',
        num_workers=num_workers if num_workers is not None \
            else torch.multiprocessing.cpu_count())

    return dataloader


def get_noise(batch_size, dim):
    """Defines the prior probability of z.
    Returns:
        (torch.Tensor): a Gaussian random tensor with mean 0 and variance 1.
    """
    noise = torch.randn(batch_size, dim, 1, 1)
    return noise.to(device)


class DataSupplier():
    """Class used to provide batches of real and fake images for training GANs."""
    REAL_LABEL = 1
    FAKE_LABEL = 0

    def __init__(self, dataloader, net_G):
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.net_G = net_G
        self.latent_dim = net_G.latent_dim
        self.iterator = iter(dataloader)

    def get_batch_real(self):
        """Returns a batch of real images from the dataloader and training targets
        (iterates infinitely on the dataloader).
        Returns:
            torch.Tensor: tensor data
            torch.Tensor: tensor target vector
        """
        try:
            data_real, _ = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            data_real, _ = next(self.iterator)

        target_real = torch.full((data_real.size(0),), self.REAL_LABEL)
        return data_real.to(device), target_real.to(device)

    def get_batch_fake(self, train_G=False):
        """Returns a batch of generated images and training targets.
        Returns:
            torch.Tensor: tensor data
            torch.Tensor: tensor target vector
        """
        z = get_noise(self.batch_size, self.latent_dim)
        data_fake = self.net_G(z)
        if not train_G:
            target_fake = torch.full((data_fake.size(0),), self.FAKE_LABEL)
        else:
            # if we train the generator G, then set training targets to real to
            # to fool the discriminator D.
            target_fake = torch.full((data_fake.size(0),), self.REAL_LABEL)
        return data_fake, target_fake


def main(args):

    dataloader = get_dataloader(args.batch_size, args.workers)

    # # plot some training images
    # real_batch = next(iter(loader))
    # plt.figure(figsize=(10,10))
    # plt.axis('off')
    # plt.title('Training Images Sample')
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2,
    #                                          normalize=True), (1, 2, 0)))
    # plt.show()
    print(args)

    net_G = Generator(args.latent_dim, args.num_feature_maps_G)
    net_D = Discriminator(args.num_feature_maps_D)
    # initialize the weights of the networks
    net_G.apply(init_weights)
    net_D.apply(init_weights)

    # create the criterion function for the discriminator:
    # binary-cross entropy loss
    criterion = nn.BCELoss()
    # create optimizers
    optimizer_G = optim.Adam(net_G.parameters(), lr=args.lr_G,
                             betas=(args.beta1_G, 0.999))
    optimizer_D = optim.Adam(net_D.parameters(), lr=args.lr_D,
                             betas=(args.beta1_D, 0.999))

    supplier = DataSupplier(dataloader, net_G)
    data_real, target_real = supplier.get_batch_real()
    print(f"Real batch: {tuple(data_real.size())} -> {tuple(target_real.size())}")
    data_fake, target_fake = supplier.get_batch_fake()
    print(f"Fake batch (for training D): {tuple(data_fake.size())} -> "
          f"{tuple(target_fake.size())}")
    data_fake, target_fake = supplier.get_batch_fake(True)
    print(f"Fake batch (for training G): {tuple(data_fake.size())} -> "
          f"{tuple(target_fake.size())}")

    # create a random noise vector, will be used during training for visualization
    FIXED_NOISE = get_noise(196, args.latent_dim)


if __name__ == '__main__':

    args = argparse.Namespace()
    # number of workders for dataloader (/!\ set to 4 when you're done
    # debugging))
    args.workers = 0
    # size of the latent vector z, the generator input
    args.latent_dim = 100
    # base size of feature maps in discriminator / generator
    args.num_feature_maps_D = 32
    args.num_feature_maps_G = 32
    # learning rate for the discriminator / generator
    args.lr_D = 0.0002
    args.lr_G = 0.0002
    # momentum beta1 for the discriminator / generattor
    args.beta1_D = 0.5
    args.beta1_G = 0.5
    # number of images per batch
    args.batch_size = 256
    # number of sub-steps of discriminator / generator optim. at each step
    args.num_updates_D = 1
    args.num_updates_G = 1
    # number of global steps in the training loop
    args.steps = 8000
    # number of epochs; leave None fi you set the number of steps (i.e. batch updates)
    args.epochs = None

    if args.epochs is None:
        args.epochs = (args.steps * args.batch_size) / (args.num_updates_D * 202000)
    else:
        args.steps = int(args.epochs * args.num_updates_D * 202000 / args.batch_size)

    np.random.seed(42)  # random seed for reproducibility
    torch.manual_seed(42)
    main(args)
