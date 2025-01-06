import os
import random
import argparse
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from model import TransformerVAEModel
from tqdm import tqdm
import numpy as np

from video_cap_dataset import VideoCaptionDataset, collate_fn  # Ensure this is updated as per the new dataset class
from utils import *

def loss_function(
    recon_x: Tensor,
    true_x: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 1.0
) -> Tensor:
    """
    Computes the VAE loss as reconstruction loss + KL divergence.
    """
    recon_loss = F.mse_loss(recon_x, true_x, reduction='sum') / true_x.size(0)  # Mean over batch
    # KL divergence between the learned latent distribution and standard normal
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / true_x.size(0)
    return recon_loss + beta * kl_loss

def train(args):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except RuntimeError:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    print("==> Loading dataset")

    # Select every other frame
    frame_selection_fn = lambda lis: lis[::2]

    # Initialize dataset
    train_dataset = VideoCaptionDataset(
        json_file=args.data_path,
        root_dir=args.root_dir,
        cache_path=args.cache_path,
        glove_dim=300,
        max_output_length=args.max_output_length,
        image_size=args.image_size,
        frame_selection_fn=frame_selection_fn,
    )
    val_dataset = VideoCaptionDataset(
        json_file=args.val_path,
        root_dir=args.root_dir_val,
        cache_path=args.cache_path,
        glove_dim=300,
        max_output_length=args.max_output_length,
        image_size=args.image_size,
        frame_selection_fn=frame_selection_fn,
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
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    print("==> Initializing model")

    # Initialize model
    model = TransformerVAEModel(
        embed_dim=train_dataset.embed_dim,
        latent_dim=300,  # Must match embed_dim
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_length=train_dataset.max_src_length,
        max_output_length=args.max_output_length,
        hidden_vae_dims=[32, 64, 128, 256, 512]
    )
    model = model.to(device)

    print("==> Initializing optimizer and scheduler")

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Optionally, separate learning rates for VAE and Transformer
    if args.use_separate_lrs:
        transformer_params = []
        vae_params = []
        for name, param in model.named_parameters():
            if 'vae.encoder' in name or 'vae.decoder' in name:
                vae_params.append(param)
            else:
                transformer_params.append(param)
        optimizer = torch.optim.Adam([
            {'params': transformer_params, 'lr': args.lr},
            {'params': vae_params, 'lr': args.lr * args.vae_lr_multiplier}
        ], weight_decay=args.weight_decay)
        print(f"Using separate learning rates: Transformer LR={args.lr}, VAE LR={args.lr * args.vae_lr_multiplier}")

    # Define a learning rate scheduler that reduces LR on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=args.scheduler_patience,
    )

    start_epoch = 0
    if not args.ignore_checkpoint:
        print("==> Loading checkpoint")
        start_epoch = load_latest_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dir=args.checkpoint_dir
        )
        print(f"Resuming from epoch {start_epoch}")

    print("==> Initializing TensorBoard SummaryWriter")
    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, 'tensorboard_logs'))

    print("==> Starting training")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # == Training phase ==
        total_batch = 0
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs} [Train]",
            smoothing=0.01,
        )

        for batch in progress_bar:
            total_batch += 1

            src = batch['src'].to(device)                  # (batch_size, max_src_length, embed_dim)
            tgt_images = batch['tgt'].to(device)           # (batch_size, max_tgt_length, C, H, W)

            optimizer.zero_grad()

            # Forward pass
            reconstructed_images, mu, logvar = model(src, tgt_images=tgt_images)

            loss = loss_function(reconstructed_images, tgt_images, mu, logvar)

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix_str(f'Loss: {loss.item():.4f}')

            # Log batch loss to TensorBoard
            global_step = (epoch - 1) * len(train_loader) + total_batch
            writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)

            # Save sample every 250 batches
            if total_batch % 250 == 0:
                save_batch_sample(
                    checkpoint_dir=args.checkpoint_dir,
                    epoch=epoch,
                    total_batch=total_batch,
                    batch=batch,
                    reconstructed_images=reconstructed_images,
                )
                # Also log the reconstructed images to TensorBoard
                # Select the first sample in the batch for visualization
                img_grid = make_image_grid(reconstructed_images[0], nrow=15)
                writer.add_image('Train/Reconstructed_Images', img_grid, global_step)
                img_grid = make_image_grid(tgt_images[0], nrow=15)
                writer.add_image('Train/Target_Images', img_grid, global_step)

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} Training Loss: {avg_train_loss:.4f}")
        writer.add_scalar('Train/Epoch_Loss', avg_train_loss, epoch)

        # == Validation phase ==
        model.eval()
        val_loss = 0.0
        reconstructed_images = None
        with torch.no_grad():
            progress_bar = tqdm(
                val_loader,
                desc=f"Epoch {epoch}/{args.epochs} [Validation]",
                smoothing=0.01,
            )
            for batch in progress_bar:
                src = batch['src'].to(device)
                tgt_images = batch['tgt'].to(device)

                reconstructed_images, mu, logvar = model(src, tgt_images=tgt_images)

                loss = loss_function(reconstructed_images, tgt_images, mu, logvar)

                val_loss += loss.item()
                progress_bar.set_postfix_str(f'Val Loss: {loss.item():.4f}')

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f}")
        writer.add_scalar('Validation/Loss', avg_val_loss, epoch)

        # Log validation reconstructed images
        if reconstructed_images is not None:
            # Log the first batch's first reconstructed image
            img_grid = make_image_grid(reconstructed_images[0], nrow=15)
            writer.add_image('Validation/Reconstructed_Images', img_grid, epoch)

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

    print("==> Training complete. Closing TensorBoard SummaryWriter.")
    writer.close()

def main():
    parser = argparse.ArgumentParser(description="Train Transformer VAE Model for Image Sequence Generation")

    # Data parameters
    parser.add_argument('--data_path', type=str, required=True, help='Path to the JSON data file for training')
    parser.add_argument('--val_path', type=str, required=True, help='Path to the JSON data file for validation')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing image sequence folders')
    parser.add_argument('--root_dir_val', type=str, required=True, help='Root directory containing the validation image sequence folders')
    parser.add_argument('--cache_path', type=str, help='Root directory of the cache folder')
    parser.add_argument('--image_size', type=int, default=64, help='The target image size to generate')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for Transformer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of DataLoader workers')
    parser.add_argument('--nhead', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')
    parser.add_argument('--max_output_length', type=int, default=150, help='Maximum length of output image sequences')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--ignore_checkpoint', action='store_true', default=False,
                        help='Ignore loading the latest checkpoint and start training from scratch')

    # Scheduler parameters
    parser.add_argument('--scheduler_patience', type=int, default=5, help='Number of epochs with no improvement after which learning rate will be reduced.')

    # VAE parameters
    parser.add_argument('--use_separate_lrs', action='store_true', default=False,
                        help='Use separate learning rates for Transformer and VAE parameters')
    parser.add_argument('--vae_lr_multiplier', type=float, default=1.0, help='Multiplier for VAE learning rate if separate LRs are used')

    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()
