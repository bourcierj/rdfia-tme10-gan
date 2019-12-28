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
from torch.utils.tensorboard import SummaryWriter

from gans import Generator, Discriminator, init_weights
from train_utils import *

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


def train(net_G, net_D, optimizer_G, optimizer_D, criterion, data_supplier, steps, num_updates_D,
          num_updates_G, writer=None):
    """Full training loop."""

    print("Training on", 'GPU' if device.type == 'cuda' else 'CPU')
    gens_list = []
    G_losses = []
    D_losses = []
    # the minimum generator G loss
    min_G_loss = float('inf')
    step = 1
    # create a random noise vector, will be used during training for visualization
    FIXED_NOISE = get_noise(196, args.latent_dim)
    tic = time.time()  # start time
    # global counter value to record for G / D in tensorboard
    global_cnt_G = 1
    global_cnt_D = 1

    for step in range(1, steps+1):

        for _ in range(num_updates_D):
            # update the discriminator network D:
            # maximize log(D(x)) + log(1 - D(G(z)))
            net_D.zero_grad()
            # get batches
            data_real, target_real = data_supplier.get_batch_real()
            data_fake, target_fake = data_supplier.get_batch_fake(False)
            # forward pass
            out_real = net_D(data_real).view(-1)
            out_fake = net_D(data_fake).view(-1)
            # sum of criterions on real and fake samples
            loss_D = criterion(out_real, target_real) + criterion(out_fake, target_fake)
            # backward pass and parameters update
            loss_D.backward()
            optimizer_D.step()
            optimizer_D.zero_grad()

            # compute and save metrics, log to tensorboard
            avg_output_real = out_real.mean().item()
            avg_output_fake = out_fake.mean().item()
            D_losses.append(loss_D.item())
            if writer:
                writer.add_scalar("Loss_D", loss_D.item(), global_cnt_G)
                writer.add_scalar("Mean_Real_D(x)", avg_output_real, global_cnt_G)
                writer.add_scalar("Mean_Fake_D(G(z))", avg_output_fake, global_cnt_G)
            global_cnt_G += 1

        for _ in range(num_updates_G):
            # update the generator network G:
            # maximize log(D(G(z)))
            net_G.zero_grad()
            # get batches
            data_fake, target_fake = data_supplier.get_batch_fake(True)
            # forward pass
            out_fake = net_D(data_fake).view(-1)
            # criterion
            loss_G = criterion(out_fake, target_fake)
            # backward pass and parameters update
            loss_G.backward()
            optimizer_G.step()
            optimizer_G.zero_grad()

            # compute and save metrics, log to tensorboard
            G_losses.append(loss_G.item())
            if writer:
                writer.add_scalar("Loss_G", loss_G.item(), global_cnt_D)
            global_cnt_D += 1

        if (step-1) % 25 == 0:
            # log training metrics
            print("[{:5d}/{:5d}]\tLoss_D: {:.4f}\tLoss_G: {:.4f}\tD(x): {:.4f}\tD(G(z)): {:.4f}"
                  .format(step, steps, loss_D.item(), loss_G.item(), avg_output_real,
                          avg_output_fake))
        if (step-1) % 100 == 0:
            # generate images from the fixed noise
            with torch.no_grad():
                data_fake = net_G(FIXED_NOISE).detach().cpu()
            grid = vutils.make_grid(data_fake, padding=2, normalize=True, nrow=14)
            gens_list.append(grid)

            if writer:
                writer.add_image('Generated', grid, step)
            # plt.figure(figsize=(8,8))
            # plt.imshow(np.transpose(gens_list[-1], (1, 2, 0)))
            # plt.axis('off')
            # plt.show()

    print("\n======> Done. Total time {:d}s\t".format(time.time() - tic))
    return G_losses, D_losses, gens_list


def train_from_checkpoint(checkpoint, **kwargs):
    """Train from an existing checkpoint."""
    net_G, net_D = checkpoint.net_G, checkpoint.net_D
    optimizer_G, optimizer_D = checkpoint.optimizer_G, checkpoint.optimizer_D
    return train(net_G, net_D, optimizer_G, optimizer_D, **kwargs)


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
    # data_real, target_real = supplier.get_batch_real()
    # print(f"Real batch: {tuple(data_real.size())} -> {tuple(target_real.size())}")
    # data_fake, target_fake = supplier.get_batch_fake()
    # print(f"Fake batch (for training D): {tuple(data_fake.size())} -> "
    #       f"{tuple(target_fake.size())}")
    # data_fake, target_fake = supplier.get_batch_fake(True)
    # print(f"Fake batch (for training G): {tuple(data_fake.size())} -> "
    #       f"{tuple(target_fake.size())}")

    # experiment name for tensorboard
    hparams = get_hparams_dict(args,
                               ignore_keys={'no_tensorboard', 'workers', 'epochs'})
    expe_name = get_experiment_name(prefix='__CelebA__', hparams=hparams)

    if args.no_tensorboard:
        writer = None
    else:
        writer = SummaryWriter(comment=expe_name, flush_secs=10)
        # log sample data and net graph in tensorboard
        #@todo

    train(net_G, net_D, optimizer_G, optimizer_D, criterion, supplier, args.steps,
          args.num_updates_D, args.num_updates_G, writer)


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
    # if False, log to tensorboard
    args.no_tensorboard = False

    np.random.seed(42)  # random seed for reproducibility
    torch.manual_seed(42)
    main(args)
