import os
import subprocess
from tempfile import TemporaryDirectory
from typing import Optional, Tuple
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torchvision.utils as vutils
from torchvision.utils import make_grid

FFMPEG_PATH = "/usr/bin/ffmpeg"

def load_latest_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[Optimizer],
    scheduler: Optional[LRScheduler],
    checkpoint_dir: str='checkpoints/'
) -> int:
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        return 0  # No checkpoints exist

    # Find the latest checkpoint
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return 0  # No checkpoints available

    # Extract epoch numbers and find the maximum
    max_epoch = max(checkpoints, key=_extract_epoch)
    if max_epoch == -1:
        return 0  # Invalid checkpoint filenames

    latest_checkpoint = max_epoch
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint '{latest_checkpoint}' (epoch {epoch})")
    return epoch

def _extract_epoch(f):
    try:
        return int(f.split('_')[2].split('.')[0])
    except (IndexError, ValueError):
        return -1

def save_images_to_folder(
    generated_images: torch.Tensor,
    folder: str,
):
    """
    Saves generated images to a temporary directory.

    Args:
        generated_images (torch.Tensor): Tensor of generated images with shape (1, sequence_length, 3, H, W).

    Returns:
        str: Path to the temporary directory containing the saved images.
    """

    # Denormalize images if necessary (assuming images are in range [-1, 1])
    images = generated_images.squeeze(0)  # Shape: (sequence_length, 3, H, W)
    images = (images + 1) / 2  # Scale to [0, 1]

    for idx, img_tensor in enumerate(images, start=1):
        img_path = os.path.join(folder, f"image_{idx:04d}.png")
        vutils.save_image(img_tensor, img_path, normalize=False)

def imgs_to_video(
    img_path: str,
    out_path: str,
    file_name: Optional[str] = None,
    resize: Tuple[int, int] = (1024, 512),
    fps: int | str = 15
):
    # Change to the specified working directory
    path = os.path.join(img_path, "image_%04d.png")
    if file_name:
        output = os.path.join(out_path, f"{file_name}.mp4")
    else:
        output = out_path

    if not os.path.isfile(FFMPEG_PATH):
        raise FileNotFoundError(f"ffmpeg not found at '{FFMPEG_PATH}'. Please install ffmpeg or update the FFMPEG_PATH.")

    # Define the ffmpeg command
    command = [
        FFMPEG_PATH,
        '-y', # overwrite existing file
        '-framerate', str(fps),
        '-i', path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-s', f'{resize[0]}x{resize[1]}',
        output
    ]

    # Run the command
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print("An error occurred while creating the video:", e.stderr.decode())

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
    with TemporaryDirectory() as temp_dir:
        for i, (img, generated) in enumerate(zip(first_sequence, created_seq)):
            img_path = os.path.join(temp_dir, f"image_{i+1:04d}.png")
            vutils.save_image([img, generated], img_path, normalize=True)

        imgs_to_video(temp_dir, sample_folder, videoID[0])

    # Save the prompt
    prompt = prompts[0] if prompts else 'no_prompt_available'

    prompt_path = os.path.join(sample_folder, "prompt.txt")
    with open(prompt_path, 'w', encoding="utf-8") as f:
        f.write(prompt)


def make_image_grid(tensors, nrow=4):
    """
    Helper function to make a grid of images for TensorBoard visualization.
    Assumes tensors are in (batch_size, C, H, W) format and in [0,1] range.
    """
    grid = make_grid(tensors.detach(), nrow=nrow, normalize=True)
    return grid
