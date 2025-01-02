import os
import subprocess
from tempfile import TemporaryDirectory
import torch
import torchvision.utils as vutils

ffmpeg_path = "/usr/bin/ffmpeg"

def load_latest_checkpoint(
    model: torch.nn.Module,
    optimizer,
    scheduler,
    device: torch.device,
    checkpoint_dir: str='checkpoints/'
    ):
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
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
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

def imgs_to_video(img_path: str, out_path: str, file_name: str):
    # Change to the specified working directory
    path = os.path.join(img_path, "image_%04d.png")
    output = os.path.join(out_path, f"{file_name}.mp4")

    # Define the ffmpeg command
    command = [
        ffmpeg_path,
        '-framerate', '15',
        '-i', path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-s', '1024x512',
        output
    ]

    # Run the command
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print("An error occurred:", e)

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
