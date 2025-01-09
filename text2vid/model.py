import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

class TransformerVectorGenerator(nn.Module):
    def __init__(
        self,
        embed_dim=300,       # Dimension of text embeddings
        vector_dim=200,      # Dimension of output vectors
        nhead=6,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_length=64,
        max_output_length=500,  # Adjust as needed
    ):
        super(TransformerVectorGenerator, self).__init__()
        self.embed_dim = embed_dim
        self.vector_dim = vector_dim
        self.max_seq_length = max_seq_length
        self.max_output_length = max_output_length

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

        # Linear layers to map between vector_dim and embed_dim
        self.vector_to_embed = nn.Linear(vector_dim, embed_dim)
        self.embed_to_vector = nn.Linear(embed_dim, vector_dim)

        # Stop token prediction
        self.stop_token = nn.Linear(embed_dim, 1)  # Outputs a logit for stop

        # Positional encodings
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_seq_length)
        self.pos_decoder = PositionalEncoding(embed_dim, dropout, max_output_length)

        # Start token
        self.start_token = torch.zeros(1, 1, vector_dim)

    def generate_square_subsequent_mask(self, sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return self.transformer.generate_square_subsequent_mask(sz)

    def _train_step(
        self,
        batch_size: int,
        memory: Tensor,
        tgt: Tensor
        ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch_size: the size of the batch
            memory: (batch_size, memory_seq_length, vector_dim)
            tgt: (batch_size, tgt_seq_length, vector_dim)
        Returns:
            output_vectors: (batch_size, tgt_seq_length, vector_dim)
            stop_logits: (batch_size, tgt_seq_length)
        """
        start_token_batch = self.start_token.to(memory.device).expand(batch_size, 1, -1)  # (batch_size, 1, vector_dim)
        tgt = torch.cat([start_token_batch, tgt], dim=1)  # Prepend START token

        # During training, use the target vectors as input to the decoder
        tgt_embed = self.vector_to_embed(tgt)  # (batch_size, tgt_seq_length, embed_dim)
        tgt_embed = self.pos_decoder(tgt_embed)

        # Generate masks
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        out = self.transformer.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
        out = out[:, :-1, :] # Account for START token

        output_vectors = self.embed_to_vector(out)  # (batch_size, tgt_seq_length, vector_dim)
        stop_logits = self.stop_token(out).squeeze(-1)  # (batch_size, tgt_seq_length)
        return output_vectors, stop_logits

    def _infer(
        self,
        batch_size: int,
        memory: Tensor
        ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch_size: the size of the batch
            memory: (batch_size, memory_seq_length, vector_dim)
        Returns:
            output_vectors: (batch_size, tgt_seq_length, vector_dim)
            stop_logits: (batch_size, tgt_seq_length)
        """
        generated_vectors = []
        stop_flags = []

        # Initialize the future target embeddings
        start = self.start_token.to(memory.device).expand(batch_size, 1, -1).to(memory.device)
        generated_vectors.append(start)

        for _ in range(self.max_output_length):
            tgt_embed = torch.cat(generated_vectors, dim=1) # Feed previous tokens back into the model
            tgt_embed = self.vector_to_embed(tgt_embed)
            tgt_embed = self.pos_decoder(tgt_embed)

            # Decode step-by-step
            out = self.transformer.decoder(tgt_embed, memory)

            out_vector = self.embed_to_vector(out[:, -1:, :])
            stop_logit = self.stop_token(out[:, -1:, :]).squeeze(-1)

            generated_vectors.append(out_vector)
            stop_flags.append(stop_logit)

            # Break if all inputs have stopped
            stop_probs = torch.sigmoid(stop_logit)
            if (stop_probs > 0.5).all():
                break

        output_vectors = torch.cat(generated_vectors[1:], dim=1)
        stop_logits = torch.cat(stop_flags, dim=1)
        return output_vectors, stop_logits

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

    def forward(self, src, src_mask=None, tgt=None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            src: (batch_size, src_seq_length, embed_dim)
            tgt: (batch_size, tgt_seq_length, vector_dim)
        Returns:
            output_vectors: (batch_size, tgt_seq_length, vector_dim)
            stop_logits: (batch_size, tgt_seq_length)
        """
        batch_size, _, _ = src.size()

        src_key_padding_mask = self.generate_src_key_padding_mask(src)

        # Apply positional encoding to the source
        src = self.pos_encoder(src)

        # Encode the source sequence (process the encoder once)
        memory = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)

        if tgt is not None:
            # During training, the target sequence is provided and masked
            return self._train_step(batch_size, memory, tgt)
        else:
            # During inference, generate the target sequence step by step
            # Only the memory (which contains the text encoding) is required
            # The inference step generates new data based on a start token
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
