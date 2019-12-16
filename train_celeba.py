import argparse
import os
import time
from pathlib import Path
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as T
import torchvision.utils as vutils

# from utils import *


def main(args):

    print(args)
    return


if __name__ == '__main__':

    args = argparse.Namespace()
    # number of workders for dataloader (/!\ set to 4 when you're done
    # debugging))
    args.workers = 0
    # size of the latent vector z, the generator input
    args.latent_dim = 100
    # base size of feature maps in discriminator / generator
    args.num_feature_maps_D = 32
    args.num_features_maps_G = 32
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
