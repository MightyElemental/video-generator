import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, img_size, latent_dim):
        super().__init__()
        layers = []
        in_channels = 3  # starting with 3 channels for RGB
        out_channels = 32  # arbitrary number of channels to start with
        kernel_size = 4
        stride = 2
        padding = 1
        self.img_size = img_size

        # We'll keep convoluting down until we reach a small enough feature map, before flattening and adding dense layers.
        while img_size > 4:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels *= 2
            img_size = (img_size - kernel_size + 2 * padding) // stride + 1

        self.conv_layers = nn.Sequential(*layers)
        # After conv_layers, flatten to dense latent_mean and latent_log_var
        self.flatten_size = in_channels * img_size * img_size
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, img_size, latent_dim, final_img_channels=3):
        super().__init__()
        layers = []
        kernel_size = 4
        stride = 2
        padding = 1
        # Start increasing the channels from a lower level to the final
        in_channels = 128  # This needs to match the encoder's end of conv layers
        
        # Here we calculate the intermediate size after flattening for fully connected conversion
        init_img_size = 4  # typically chosen, but needs to be consistent with encoder's ending feature map
        self.fc = nn.Linear(latent_dim, in_channels * init_img_size * init_img_size)

        # while img_size large enough, keep increasing height and width using transposed convs
        while init_img_size < img_size:
            layers.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size, stride, padding))
            layers.append(nn.ReLU())
            in_channels //= 2
            init_img_size = init_img_size * stride - 2 * padding + kernel_size

        # Always make the last layer of decoder return an image-like structure
        layers.append(nn.ConvTranspose2d(in_channels * 2, final_img_channels, kernel_size, stride, padding))
        layers.append(nn.Sigmoid())  # Output between 0 and 1
        self.deconv_layers = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), -1, 4, 4)  # Resize to fit the first conv transpose
        x = self.deconv_layers(x)
        return x

class VAE(nn.Module):
    def __init__(self, img_size: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = Encoder(img_size, latent_dim)
        self.decoder = Decoder(img_size, latent_dim)
        # Initialize weights for the final layers
        self._init_weights()

    def _init_weights(self):
        # Initialize specific layers as mentioned
        nn.init.uniform_(self.encoder.fc_mu.weight, -0.05, 0.05)
        nn.init.uniform_(self.encoder.fc_mu.bias, -0.05, 0.05)
        nn.init.uniform_(self.encoder.fc_logvar.weight, -0.05, 0.05)
        nn.init.uniform_(self.encoder.fc_logvar.bias, -0.05, 0.05)

        # Similar custom initialization can be applied to specific decoder layers if necessary
        for layer in self.decoder.deconv_layers:
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.uniform_(layer.weight, -0.05, 0.05)
                if layer.bias is not None:
                    nn.init.uniform_(layer.bias, -0.05, 0.05)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar