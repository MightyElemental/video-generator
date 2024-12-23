import os
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TransformerVectorGenerator
from tqdm import tqdm
import numpy as np

from video_cap_dataset import VideoCaptionDataset, collate_fn
from utils import load_latest_checkpoint

def train(args):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("==> Loading dataset")

    # Initialize dataset
    train_dataset = VideoCaptionDataset(
        args.data_path,
        glove_dim=300,
        max_output_length=args.max_output_length
    )
    val_dataset = VideoCaptionDataset(
        args.val_path,
        glove_dim=300,
        max_output_length=args.max_output_length
    )

    # Initialize dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    print("==> Initializing model")

    # Initialize model
    model = TransformerVectorGenerator(
        embed_dim=train_dataset.embed_dim,
        vector_dim=200,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_length=train_dataset.max_src_length,
        max_output_length=args.max_output_length
    )
    model = model.to(device)

    print("==> Initializing optimizer and scheduler")

    # Define optimizer and loss functions
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Define a learning rate scheduler that reduces LR on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=args.scheduler_patience,
    )

    # Define loss functions with weights
    criterion_vector = nn.MSELoss()
    criterion_stop = nn.BCEWithLogitsLoss()
    loss_weights = {'vector': args.vector_loss_weight, 'stop': args.stop_loss_weight}

    start_epoch = 0
    if not args.ignore_checkpoint:
        print("==> Loading checkpoint")
        start_epoch = load_latest_checkpoint(model, optimizer, device, args.checkpoint_dir)

    print("==> Starting training")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # == Training phase ==
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")

        for batch in progress_bar:
            src = batch['src'].to(device)             # (batch_size, src_seq_length, embed_dim)
            tgt = batch['tgt'].to(device)             # (batch_size, tgt_seq_length, vector_dim)

            optimizer.zero_grad()

            # Forward pass
            output_vectors, stop_logits = model(src, tgt=tgt)
            # output_vectors: (batch_size, tgt_seq_length, vector_dim)
            # stop_logits: (batch_size, tgt_seq_length)

            # Compute loss
            loss_vector = criterion_vector(output_vectors, tgt)

            # Create stop targets: 0 for all except the last vector which is 1
            stop_targets = torch.zeros_like(stop_logits).to(device)
            stop_targets[:, :-1] = 0
            stop_targets[:, -1] = 1  # Only the last vector should predict to stop
            loss_stop = criterion_stop(stop_logits, stop_targets)

            # Weighted combination of losses
            loss = loss_weights['vector'] * loss_vector + loss_weights['stop'] * loss_stop

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} Training Loss: {avg_train_loss:.4f}")

        # == Validation phase ==
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Validation]")
            for batch in progress_bar:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)

                output_vectors, stop_logits = model(src, tgt=tgt)

                loss_vector = criterion_vector(output_vectors, tgt)

                stop_targets = torch.zeros_like(stop_logits).to(device)
                stop_targets[:, :-1] = 0
                stop_targets[:, -1] = 1
                loss_stop = criterion_stop(stop_logits, stop_targets)

                loss = loss_weights['vector'] * loss_vector + loss_weights['stop'] * loss_stop

                val_loss += loss.item()
                progress_bar.set_postfix({'Val Loss': loss.item()})

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f}")

        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)

        # Save the model checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.checkpoint_dir, f'model_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, checkpoint_path)
        print(f"==> Saved checkpoint: {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Transformer Vector Generator")

    # Data parameters
    parser.add_argument('--data_path', type=str, required=True, help='Path to the pickle data file')
    parser.add_argument('--val_path', type=str, required=True, help='Path to the pickle data file for validation')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of DataLoader workers')
    parser.add_argument('--nhead', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')
    parser.add_argument('--max_output_length', type=int, default=300, help='Maximum length of output vectors')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=4638511, help='Random seed')
    parser.add_argument('--ignore_checkpoint', action='store_true', default=False,
                        help='Ignore loading the latest checkpoint and start training from scratch')

    # Scheduler parameters
    parser.add_argument('--scheduler_patience', type=int, default=5, help='Number of epochs with no improvement after which learning rate will be reduced.')

    # Loss weights
    parser.add_argument('--vector_loss_weight', type=float, default=1.0, help='Weight for the vector loss')
    parser.add_argument('--stop_loss_weight', type=float, default=0.1, help='Weight for the stop token loss')

    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()
