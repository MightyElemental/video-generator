import os
import random
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import TransformerVAEModel
from tqdm import tqdm
import numpy as np
import torchvision.utils as vutils

from video_cap_dataset import VideoCaptionDataset, collate_fn  # Ensure this is updated as per the new dataset class
from utils import load_latest_checkpoint  # May not be needed since VAE and Transformer are trained together

def loss_function(recon_x, x, mu, logvar):
    """
    Computes the VAE loss as reconstruction loss + KL divergence.
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)  # Mean over batch
    # KL divergence between the learned latent distribution and standard normal
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_loss

def save_batch_sample(
    checkpoint_dir: str,
    epoch: int,
    total_batch: int,
    batch,
    reconstructed_images: torch.Tensor,
):
    # Create samples directory
    samples_dir = os.path.join(checkpoint_dir, 'samples')  # Directory to save samples
    os.makedirs(samples_dir, exist_ok=True)

    # Create folder named "{epoch}-{total_batch}"
    sample_folder = os.path.join(samples_dir, f"{epoch:03d}-{total_batch:05d}")
    os.makedirs(sample_folder, exist_ok=True)

    tgt_images = batch["tgt"]
    prompts = batch["prompt"]
    videoID = batch["videoID"]

    # Save the first image sequence in the batch
    first_sequence = tgt_images[0]  # (max_tgt_length, C, H, W)
    created_seq    = reconstructed_images[0].to(first_sequence.device)
    for i, (img, generated) in enumerate(zip(first_sequence, created_seq)):
        img_path = os.path.join(sample_folder, f"image_{i+1:04d}.png")
        vutils.save_image([img, generated], img_path, normalize=True)

    # Save the prompt
    if prompts:
        prompt = prompts[0]
    else:
        prompt = 'no_prompt_available'

    prompt_path = os.path.join(sample_folder, "prompt.txt")
    with open(prompt_path, 'w', encoding="utf-8") as f:
        f.write(prompt)
        f.write("\n")
        f.write(videoID[0])

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

    # select every other frame
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

    # Define loss function
    criterion = torch.nn.MSELoss()

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
        start_epoch = load_latest_checkpoint(model, optimizer, device, args.checkpoint_dir)
        print(f"Resuming from epoch {start_epoch}")

    print("==> Starting training")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # == Training phase ==
        total_batch = 0
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs} [Train]",
            smoothing=1,
        )

        for batch in progress_bar:
            total_batch += 1

            src = batch['src'].to(device)                  # (batch_size, max_src_length, embed_dim)
            tgt_images = batch['tgt'].to(device)           # (batch_size, max_tgt_length, C, H, W)

            optimizer.zero_grad()

            # Forward pass
            reconstructed_images = model(src, tgt_images=tgt_images)

            loss = criterion(reconstructed_images, tgt_images)

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix_str(f'Loss: {loss.item():.4f}')

            # Save sample every 500 batches
            if total_batch % 50 == 0:
                save_batch_sample(
                    checkpoint_dir=args.checkpoint_dir,
                    epoch=epoch,
                    total_batch=total_batch,
                    batch=batch,
                    reconstructed_images=reconstructed_images,
                )

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} Training Loss: {avg_train_loss:.4f}")

        # == Validation phase ==
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Validation]")
            for batch in progress_bar:
                src = batch['src'].to(device)
                tgt_images = batch['tgt'].to(device)

                loss = model(src, tgt_images=tgt_images)

                val_loss += loss.item()
                progress_bar.set_postfix_str(f'Val Loss: {loss.item():.4f}')

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
