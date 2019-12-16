
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminator network.
    The architecture is inspired from DCGANs (Radford et al 2016).
    """
    def __init__(self, num_channels_in=32):
        super(Discriminator, self).__init__()
        ndf = num_channels_in
        self.model = nn.Sequential(
            nn.Conv2d(3, ndf, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, 2*ndf, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2*ndf),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2*ndf, 4*ndf, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4*ndf),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4*ndf, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


class Generator(nn.Module):
    """
    Generator network.
    The architecture is inspired from DCGANs (Radford et al 2016)
    """
    def __init__(self, latent_dim=100, num_channels_out=32):
        super(Generator, self).__init__()
        ngf = num_channels_out
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 4*ngf, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4*ngf),
            nn.ReLU(),
            nn.ConvTranspose2d(4*ngf, 2*ngf, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2*ngf),
            nn.ReLU(),
            nn.ConvTranspose2d(2*ngf, ngf, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            nn.ConvTranspose2d(ngf, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)


def init_weights(m):
    """Initializes module's weights."""
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':

    latent_dim = 100
    num_feature_maps_G = 32
    num_feature_maps_D = 32

    net_G = Generator(latent_dim, num_feature_maps_G)
    net_D = Discriminator(num_feature_maps_D)
    # initialize the weights of the networks
    net_G.apply(init_weights)
    net_D.apply(init_weights)

    # Test the models to check if they have the right shape
    z = torch.zeros(16, latent_dim, 1, 1)
    x = torch.zeros(16, 3, 32, 32)
    assert(tuple(net_G(z).size()) == (16, 3, 32, 32))
    assert(tuple(net_D(x).size()) == (16, 1, 1, 1))

    assert(tuple(net_D(net_G(z)).size()) == (16, 1, 1, 1))
