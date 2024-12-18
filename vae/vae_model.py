import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, img_size: int, latent_dim: int, multiplier: int = 1):
        super().__init__()
        layers = []
        in_channels = 3  # starting with 3 channels for RGB
        out_channels = 4 * multiplier # arbitrary number of channels to start with
        kernel_size = 4
        stride = 2
        padding = 1
        self.img_size = img_size

        print(f" encoder img size = {img_size}, {in_channels}")

        # We'll keep convoluting down until we reach a small enough feature map,
        # before flattening and adding dense layers.
        while img_size > 4:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels *= 2
            img_size = (img_size - kernel_size + 2 * padding) // stride + 1
            print(f" encoder img size = {img_size}, {out_channels // 2}")

        self.conv_layers = nn.Sequential(*layers)
        # After conv_layers, flatten to dense latent_mean and latent_log_var
        self.flatten_size = in_channels * img_size * img_size
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv_layers(x)
        # for layer in self.conv_layers:
        #     x = layer(x)
        #     if isinstance(layer, nn.Conv2d):
        #         print("encode", x.shape)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self,
                img_size: int,
                latent_dim: int,
                final_img_channels: int = 3,
                multiplier: int = 1):
        super().__init__()
        layers = []
        kernel_size = 4
        stride = 2
        padding = 1
        # Start increasing the channels from a lower level to the final
        in_channels = img_size//2 * multiplier  # This needs to match the encoder's end of conv layers

        # Here we calculate the intermediate size after flattening for fully connected conversion
        init_img_size = 4  # typically chosen, but needs to be consistent with encoder's ending feature map
        self.fc = nn.Linear(latent_dim, in_channels * init_img_size * init_img_size)

        print(f"dencoder img size = {init_img_size}, {in_channels}")

        # while img_size large enough, keep increasing height and width using transposed convs
        while init_img_size < img_size//2:
            layers.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size, stride, padding))
            layers.append(nn.BatchNorm2d(in_channels // 2))
            layers.append(nn.ReLU())
            in_channels //= 2
            init_img_size = (init_img_size-1) * stride - 2 * padding + kernel_size
            print(f"dencoder img size = {init_img_size}, {in_channels}")

        # Always make the last layer of decoder return an image-like structure
        layers.append(nn.ConvTranspose2d(in_channels, final_img_channels, kernel_size, stride, padding))
        layers.append(nn.Tanh())  # Output between -1 and 1
        self.deconv_layers = nn.Sequential(*layers)

    def forward(self, z):
        # print(z.shape)
        x = self.fc(z)
        # print(x.shape)
        x = x.view(x.size(0), -1, 4, 4)  # Resize to fit the first conv transpose
        # print(x.shape)
        x = self.deconv_layers(x)
        # for layer in self.deconv_layers:
        #     x = layer(x)
        #     if isinstance(layer, nn.ConvTranspose2d):
        #         print(x.shape)
        return x

class VAE(nn.Module):
    def __init__(self, img_size: int, latent_dim: int = 32, multiplier: int = 1):
        super().__init__()
        self.encoder = Encoder(img_size, latent_dim, multiplier=multiplier)
        self.decoder = Decoder(img_size, latent_dim, multiplier=multiplier)
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