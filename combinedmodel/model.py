import math
from typing import Optional, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from tqdm import tqdm


# ==========================
# Variational Autoencoder (VAE)
# ==========================
class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_channels = 3  # starting with 3 channels for RGB

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512] # for a 64x64 image

        self.hidden_dims = hidden_dims

        # Halves the image size for each layer
        # The final layer should be 2x2
        for dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                )
            )
            in_channels = dim

        self.enc_layers = nn.Sequential(*layers)
        # After conv_layers, flatten to dense latent_mean and latent_log_var
        self.flatten_size = hidden_dims[-1] * 2 * 2
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # print("encoder:",x.shape)
        x = self.enc_layers(x)
        # print(x.shape)
        x = x.view(x.size(0), self.flatten_size)
        # print(x.shape)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        # print(mu.shape)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512] # for a 64x64 image

        hidden_dims.reverse()

        self.hidden_dims = hidden_dims

        # Convert the latent dimension
        self.fc = nn.Linear(latent_dim, hidden_dims[0] * 2 * 2)

        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=hidden_dims[i],
                        out_channels=hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                )
            )

        # Always make the last layer of decoder return an image-like structure
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=hidden_dims[-1],
                    out_channels=3,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.Tanh() # Output between -1 and 1
            )
        )

        self.deconv_layers = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        # (batch, seq_len, latent_dim)
        B, S, _ = z.shape
        # print("decoder:", z.shape)
        x = self.fc(z)
        # print(x.shape)
        x = x.view(B*S, -1, 2, 2)  # Resize to fit the first conv transpose
        # print(x.shape)
        x = self.deconv_layers(x) # ((batch*seq_len), 3, img_size, img_size)
        # print(x.shape)
        x = x.view(B, S, 3, x.size(-2), x.size(-1)) # (batch, seq_len, 3, img_size, img_size)
        # print(x.shape)
        return x

class VAE(nn.Module):
    def __init__(
        self,
        latent_dim: int = 300,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(latent_dim, hidden_dims, dropout)
        self.decoder = Decoder(latent_dim, hidden_dims, dropout)

    def encode_reparam(self, x: Tensor) -> Tensor:
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
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

# ==========================
# Transformer Vector Generator Integrated with VAE
# ==========================
class TransformerVAEModel(nn.Module):
    def __init__(
        self,
        embed_dim=300,         # Dimension of text embeddings and VAE latent vectors
        nhead=6,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_length=64,
        max_output_length=500,  # Adjust as needed
        latent_dim=300,         # Must match embed_dim for seamless integration
        hidden_vae_dims: Optional[list[int]] = None,
    ):
        super(TransformerVAEModel, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        self.max_output_length = max_output_length

        # Initialize VAE
        self.vae = VAE(
            latent_dim=latent_dim,
            hidden_dims=hidden_vae_dims,
            dropout=dropout
        )

        # Transformer model
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Positional encodings
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.pos_decoder = PositionalEncoding(embed_dim, dropout)

        # Learned start token
        self.start_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def generate_square_subsequent_mask(self, sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return self.transformer.generate_square_subsequent_mask(sz)

    def generate_src_key_padding_mask(self, src: Tensor) -> Tensor:
        """
        Generate a source key padding mask where positions with all zero embeddings are considered padding.

        Args:
            src (torch.Tensor): Tensor of shape (batch_size, src_seq_length, embed_dim)

        Returns:
            torch.Tensor: Boolean mask of shape (batch_size, src_seq_length),
                        where True indicates a padding position.
        """
        # Check if all elements in the embedding dimension are zero
        # This results in a mask of shape (batch_size, src_seq_length)
        src_key_padding_mask = torch.all(src == 0, dim=-1)
        return src_key_padding_mask

    def _train_step(self, batch_size: int, memory: Tensor, tgt_images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            batch_size: the size of the batch
            memory: (batch_size, memory_seq_length, embed_dim)
            tgt_images: (batch_size, tgt_seq_length, C, H, W)
        Returns:
            loss: scalar tensor representing the loss
        """
        # Encode target images to latent vectors
        flattened = tgt_images.view(-1, tgt_images.size(-3), tgt_images.size(-2), tgt_images.size(-1))  # (batch_size * tgt_seq_length, 3, H, W)
        mu, logvar = self.vae.encode(flattened)
        tgt_latents = self.vae.reparameterize(mu, logvar)  # (batch_size * tgt_seq_length, latent_dim)
        tgt_latents = tgt_latents.view(batch_size, -1, self.embed_dim)  # (batch_size, tgt_seq_length, latent_dim)

        # Prepend start token
        start_tokens = self.start_token.expand(batch_size, 1, -1)  # (batch_size, 1, embed_dim)
        tgt_input = torch.cat([start_tokens, tgt_latents[:, :-1, :]], dim=1)  # (batch_size, tgt_seq_length, embed_dim)

        # Apply positional encoding
        tgt_embed = self.pos_decoder(tgt_input)

        # Generate masks
        tgt_mask = self.generate_square_subsequent_mask(tgt_input.size(1)).to(tgt_input.device)

        # Decode
        out = self.transformer.decoder(tgt_embed, memory, tgt_mask=tgt_mask)  # (batch_size, tgt_seq_length, embed_dim)

        # Reconstruct images from latents
        reconstructed_images = self.vae.decode(out)  # (batch_size, tgt_seq_length, 3, H, W)

        return reconstructed_images, mu, logvar

    def _infer(self, batch_size: int, memory: Tensor) -> Tensor:
        """
        Args:
            batch_size: the size of the batch
            memory: (batch_size, memory_seq_length, embed_dim)
        Returns:
            generated_images: (batch_size, tgt_seq_length, C, H, W)
        """

        # the list of generated images
        generated_output = []

        # Initialize with start token
        current_input = self.start_token.expand(batch_size, 1, -1).to(memory.device)  # (batch_size, 1, embed_dim)

        for _ in tqdm(range(self.max_output_length)):
            tgt_embed = self.pos_decoder(current_input)  # (batch_size, current_length, embed_dim)

            # Decode
            out = self.transformer.decoder(tgt_embed, memory)  # (batch_size, current_length, embed_dim)

            next_input = out[:, -1:, :]  # (batch_size, 1, embed_dim)
            # Convert latent vector into image
            next_input = self.vae.decode(next_input) # (batch_size, 1, 3, H, W)

            # Stash the generated image
            generated_output.append(next_input) # (batch_size, 1, 3, H, W)

            # Convert output to input vector
            flattened = next_input.view(-1, 3, next_input.size(-2), next_input.size(-1))  # (batch_size * current_length, 3, H, W)
            next_input = self.vae.encode_reparam(flattened) # (batch_size * current_length, embed_dim)
            next_input = next_input.view(batch_size, -1, self.embed_dim)  # (batch_size, current_length, latent_dim)

            # Append the generated latent for the next step
            current_input = torch.cat([current_input, next_input], dim=1)  # Increment sequence length

        # Concatenate all generated latents
        generated_output = torch.cat(generated_output, dim=1)  # (batch_size, max_output_length, 3, H, W)

        return generated_output

    def forward(self, src: Tensor, tgt_images: Optional[Tensor]=None) -> Tuple[Tensor, Tensor, Tensor] | Tensor:
        """
        Args:
            src: (batch_size, src_seq_length, embed_dim)
            tgt_images: (batch_size, tgt_seq_length, 3, H, W)
        Returns:
            If tgt_images is provided, returns the training loss.
            If not, returns generated_images.
        """
        batch_size, _, _ = src.size()

        src_key_padding_mask = self.generate_src_key_padding_mask(src)

        # Apply positional encoding to the source
        src = self.pos_encoder(src)

        # Encode the source sequence
        memory = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)  # (batch_size, src_seq_length, embed_dim)

        if tgt_images is not None:
            # Training phase
            return self._train_step(batch_size, memory, tgt_images)
        else:
            # Inference phase
            return self._infer(batch_size, memory)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)
