import os
import torch

def load_latest_checkpoint(
    model: torch.nn.Module,
    optimizer,
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
    max_epoch = max(checkpoints, key=extract_epoch)
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
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint '{latest_checkpoint}' (epoch {epoch})")
    return epoch

def extract_epoch(f):
    try:
        return int(f.split('_')[2].split('.')[0])
    except (IndexError, ValueError):
        return -1
