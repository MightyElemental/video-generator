import os
import torch

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
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint '{latest_checkpoint}' (epoch {epoch})")
    return epoch