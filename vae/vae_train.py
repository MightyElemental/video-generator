import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from tqdm import tqdm

# Import the VAE model
from vae_model import VAE

# Constants
IMG_SIZE = 256
LATENT_DIM = 64
WORKERS = 10
BATCH_SIZE = 16  # Adjust based on GPU memory

def save_original_and_reconstructed(original_images, reconstructed_images, output_dir, filename='comparison.png', nrow = 8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Concatenate original and reconstructed images along the batch dimension
    comparison = torch.cat((original_images, reconstructed_images), dim=0)

    # Save the concatenated images
    vutils.save_image(comparison, os.path.join(output_dir, filename), nrow=nrow, normalize=True)

def load_latest_checkpoint(model, optimizer, checkpoint_dir='checkpoints/'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        return 0  # No checkpoints exist

    # Find the latest checkpoint
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return 0  # No checkpoints available

    latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[2].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint '{latest_checkpoint}' (epoch {epoch})")
    return epoch

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(IMG_SIZE,IMG_SIZE), scale=(0.75,1)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    #v2.ColorJitter(brightness=0.5, contrast=0.5, hue=0.3),
    transforms.ToTensor(),
])

# Dataset preparation
DATA_DIR = 'data/train'  # Adjust path as needed
dataset = ImageFolder(root=DATA_DIR, transform=transform)

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=WORKERS,
    pin_memory=True,
    drop_last=True,
)

# Initialize the VAE model
model = VAE(img_size=IMG_SIZE, latent_dim=LATENT_DIM).to(device)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4)
reconstruction_loss_fn = nn.MSELoss(reduction='sum')

# Load the latest model checkpoint (if available)
start_epoch = load_latest_checkpoint(model, optimizer)

# Loss function definition
def loss_function(recon_x, x, mu, logvar):
    # print(f"recon_x={recon_x.shape}, x={x.shape}")
    recon_loss = reconstruction_loss_fn(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = (recon_loss + kl_loss) / x.size(0)
    return total_loss

# TODO: Add to pickle
randomized_vectors1 = torch.randn((4, LATENT_DIM)).to(device)
randomized_vectors2 = torch.randn((4, LATENT_DIM)).to(device)

# Training loop
EPOCHS = 200
for epoch in range(start_epoch + 1, EPOCHS + 1):
    model.train()
    train_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}/{EPOCHS}', leave=False, unit="batches")
    for batch_idx, (data, _) in enumerate(progress_bar):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        progress_bar.set_postfix_str(f"loss={loss.item():03.1f}")

    avg_loss = train_loss / len(dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')

    # Save model checkpoint
    checkpoint_path = os.path.join('checkpoints/', f'vae_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    # Save reconstructions
    model.eval()
    with torch.no_grad():
        sample_data = data[:8].to(device)
        recon_data, _, _ = model(sample_data)
        # Code to save or display images
        # For example, using torchvision.utils.save_image()
        save_original_and_reconstructed(
            sample_data,
            recon_data,
            'output_images',
            f'comparison{epoch:03d}.png',
            nrow=sample_data.shape[0]
        )

        save_original_and_reconstructed(
            model.decode(randomized_vectors1),
            model.decode(randomized_vectors2),
            'output_images/randomized',
            f'randomized{epoch:03d}.png',
            nrow=4
        )
