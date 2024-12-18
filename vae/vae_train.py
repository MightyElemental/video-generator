import os
from random import shuffle
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from tqdm import tqdm
from utils import load_latest_checkpoint

# Import the VAE model
from vae_model import VAE

# Constants
IMG_SIZE = 256
LATENT_DIM = 200
WORKERS = 12
BATCH_SIZE = 16  # Adjust based on GPU memory
DATASET_FRACTION = 0.05 # The percentage of the dataset to use
CHANNEL_MULTIPLIER = 3

def save_original_and_reconstructed(original_images, reconstructed_images, output_dir, filename='comparison.png', nrow = 8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Concatenate original and reconstructed images along the batch dimension
    comparison = torch.cat((original_images, reconstructed_images), dim=0)

    # Save the concatenated images
    vutils.save_image(comparison, os.path.join(output_dir, filename), nrow=nrow, normalize=True)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

normalize = v2.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

inv_normalize = v2.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255]
)

# Data transformations
transform = v2.Compose([
    #v2.RandomResizedCrop(size=(IMG_SIZE,IMG_SIZE), scale=(0.75,1)),
    v2.CenterCrop(size=(IMG_SIZE,IMG_SIZE)),
    #v2.RandomHorizontalFlip(p=0.3),
    #v2.RandomVerticalFlip(p=0.3),
    #v2.ColorJitter(brightness=0.5, contrast=0.5, hue=0.3),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    normalize,
])



# Dataset preparation
DATA_DIR = 'data/train'
TEST_DIR = 'data/test'
dataset = ImageFolder(root=DATA_DIR, transform=transform)
testset = ImageFolder(root=TEST_DIR, transform=transform)

# Create subset of the data
if DATASET_FRACTION < 1:
    idxs = list(range(len(dataset)))
    shuffle(idxs)
    dataset = Subset(dataset, idxs[:int(len(dataset)*DATASET_FRACTION)] )


# Add another dataset for more training data
# imgnet = Imagenette(root="data/imgnet", download=False, transform=transform)
# dataset = torch.utils.data.ConcatDataset([dataset, imgnet])

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=WORKERS,
    pin_memory=True,
    drop_last=True,
)

# TestLoader
testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=WORKERS,
    pin_memory=True,
    drop_last=True,
)

# Initialize the VAE model
model = VAE(img_size=IMG_SIZE, latent_dim=LATENT_DIM, multiplier=CHANNEL_MULTIPLIER).to(device)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4)
reconstruction_loss_fn = nn.MSELoss(reduction='sum')

# Add an optimizer learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5
)

# Load the latest model checkpoint (if available)
start_epoch = load_latest_checkpoint(model, optimizer)

# Loss function definition
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = reconstruction_loss_fn(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = (recon_loss + beta * kl_loss) / x.size(0)
    return total_loss

def validation(model, test_dataloader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(
            test_dataloader,
            total=len(test_dataloader),
            leave=False,
            unit="batch",
            desc='Validation',
            smoothing=0.7,
        )
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    test_loss /= len(test_dataloader)
    last_data = data
    return test_loss, last_data

# TODO: Add to pickle
randomized_vectors1 = torch.randn((4, LATENT_DIM)).to(device)
randomized_vectors2 = torch.randn((4, LATENT_DIM)).to(device)

standard_sample_data = None
standard_val_data = None

# Training loop
EPOCHS = 100
for epoch in range(start_epoch + 1, EPOCHS + 1):
    model.train()
    train_loss = 0
    progress_bar = tqdm(
        dataloader,
        desc=f'Epoch {epoch}/{EPOCHS}',
        leave=False,
        unit="batch",
        smoothing=0.7,
    )
    for batch_idx, (data, _) in enumerate(progress_bar):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        progress_bar.set_postfix_str(f"loss={loss.item():.1e}")

    progress_bar.close()

    avg_loss, val_data = validation(model, testloader)
    scheduler.step(avg_loss)
    print(f'====> Epoch: {epoch} - Average loss: {avg_loss:.4f} - Learning rate: {optimizer.param_groups[0]["lr"]:.1e}')

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
        if standard_sample_data is None:
            standard_sample_data = data[:8].to(device)
        if standard_val_data is None:
            standard_val_data = val_data[:8].to(device)
        sample_data = standard_sample_data
        recon_data, _, _ = model(sample_data)
        val_recon_data, _, _ = model(standard_val_data)
        # Code to save or display images
        # For example, using torchvision.utils.save_image()
        save_original_and_reconstructed(
            sample_data,
            recon_data,
            'output_images/training',
            f'comparison{epoch:03d}.png',
            nrow=sample_data.shape[0]
        )

        save_original_and_reconstructed(
            standard_val_data,
            val_recon_data,
            'output_images/validation',
            f'comparison{epoch:03d}.png',
            nrow=standard_val_data.shape[0]
        )

        save_original_and_reconstructed(
            model.decode(randomized_vectors1),
            model.decode(randomized_vectors2),
            'output_images/randomized',
            f'randomized{epoch:03d}.png',
            nrow=4
        )
