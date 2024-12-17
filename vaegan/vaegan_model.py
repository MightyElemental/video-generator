import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, image_size, latent_dim, feature_map_multiplier=64):
        super(Encoder, self).__init__()
        assert image_size % 16 == 0, "Image size must be a multiple of 16"

        self.latent_dim = latent_dim
        self.feature_map_multiplier = feature_map_multiplier

        # Calculate the number of downsampling steps
        num_downsamples = int(torch.log2(torch.tensor(image_size))) - 3  # Subtracting 3 because we'll end up with 4x4 feature maps

        layers = []
        in_channels = 3
        out_channels = feature_map_multiplier

        # Initial convolution
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        in_channels = out_channels

        # Add downsampling layers
        for _ in range(1, num_downsamples):
            out_channels *= 2
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        # Final dense layers for mu and logvar
        self.fc_mu = nn.Linear(in_channels * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(in_channels * 4 * 4, latent_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, image_size, latent_dim, feature_map_multiplier=64):
        super(Decoder, self).__init__()
        assert image_size % 16 == 0, "Image size must be a multiple of 16"

        self.latent_dim = latent_dim
        self.feature_map_multiplier = feature_map_multiplier

        # Calculate the number of upsampling steps
        num_upsamples = int(torch.log2(torch.tensor(image_size))) - 3

        in_channels = feature_map_multiplier * (2 ** (num_upsamples - 1))

        # Initial dense layer
        self.fc = nn.Linear(latent_dim, in_channels * 4 * 4)

        layers = []

        # Add upsampling layers
        for _ in range(num_upsamples - 1):
            out_channels = in_channels // 2
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        # Final upsampling layer to get back to image with 3 channels
        layers.append(nn.ConvTranspose2d(in_channels, 3, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.Tanh())  # Output values between -1 and 1

        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        batch_size = z.size(0)
        x = self.fc(z)
        x = x.view(batch_size, -1, 4, 4)
        x = self.deconv(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, image_size, feature_map_multiplier=64):
        super(Discriminator, self).__init__()
        assert image_size % 16 == 0, "Image size must be a multiple of 16"

        self.feature_map_multiplier = feature_map_multiplier

        # Calculate the number of downsampling steps
        num_downsamples = int(torch.log2(torch.tensor(image_size))) - 3

        layers = []
        in_channels = 3
        out_channels = feature_map_multiplier

        # Initial convolution
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        in_channels = out_channels

        # Add downsampling layers
        for _ in range(1, num_downsamples):
            out_channels *= 2
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels

        # Final output layer
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=0, bias=False))
        layers.append(nn.Sigmoid())

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 1)
        return x

