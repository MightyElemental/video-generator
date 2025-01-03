# generate.py

import os
import torch
import torch.nn as nn
from typing import Optional
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
from PIL import Image
from tempfile import TemporaryDirectory

from combinedmodel.model import TransformerVAEModel
from combinedmodel.utils import load_latest_checkpoint, save_images_to_folder, imgs_to_video

# Define the path to ffmpeg. Modify this if ffmpeg is located elsewhere.
FFMPEG_PATH = "/usr/bin/ffmpeg"

def load_model(
    model_checkpoint: Optional[str] = None,
    checkpoint_dir: str = 'combinedmodel/checkpoints/',
    device: Optional[torch.device] = None
) -> TransformerVAEModel:
    """
    Loads the TransformerVAEModel from a checkpoint.

    Args:
        model_checkpoint (Optional[str]): Path to the model checkpoint. If None, the latest checkpoint in
                                         checkpoint_dir is loaded.
        checkpoint_dir (str): Directory where checkpoints are stored.
        device (Optional[torch.device]): Device to load the model on. If None, automatically selects CUDA or CPU.

    Returns:
        TransformerVAEModel: The loaded model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model with the same hyperparameters as used during training
    model = TransformerVAEModel(
        embed_dim=300,            # Must match train_dataset.embed_dim
        latent_dim=300,           # Must match train.py
        nhead=6,                  # Must match args.nhead in training
        num_encoder_layers=6,     # Must match args.num_encoder_layers in training
        num_decoder_layers=6,     # Must match args.num_decoder_layers in training
        dim_feedforward=2048,     # Must match args.dim_feedforward in training
        dropout=0.1,              # Must match args.dropout in training
        max_seq_length=64,        # Must match train_dataset.max_src_length in training
        max_output_length=300,    # Can be set to desired sequence_length
        hidden_vae_dims=[32, 64, 128, 256, 512]  # Must match hidden_vae_dims in training
    )
    
    model.to(device)

    if model_checkpoint is None:
        # Load the latest checkpoint from checkpoint_dir
        epoch = load_latest_checkpoint(model, optimizer=None, scheduler=None, device=device, checkpoint_dir=checkpoint_dir)
        print(f"Loaded model from the latest checkpoint at epoch {epoch}.")
    else:
        # Load the specified checkpoint
        if not os.path.isfile(model_checkpoint):
            raise FileNotFoundError(f"Checkpoint file '{model_checkpoint}' does not exist.")
        checkpoint = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint '{model_checkpoint}'.")
    
    model.eval()
    return model

def prepare_input_text(
    input_text: str,
    max_src_length: int = 64,
    embed_dim: int = 300,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Tokenizes and embeds the input text to create the source tensor.

    Args:
        input_text (str): The text prompt to be used in the transformer encoder.
        max_src_length (int): Maximum number of tokens for the input text.
        embed_dim (int): Dimension of the GloVe vectors.
        device (Optional[torch.device]): Device to create the tensor on. If None, automatically selects CUDA or CPU.

    Returns:
        torch.Tensor: Embedded and padded source tensor of shape (1, max_src_length, embed_dim).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = get_tokenizer('basic_english')
    vocab = GloVe(name='840B', dim=embed_dim)
    
    tokens = tokenizer(input_text)
    tokens = tokens[:max_src_length]
    
    token_vectors = vocab.get_vecs_by_tokens(tokens)
    
    src = torch.zeros(max_src_length, embed_dim)
    src[:token_vectors.size(0), :] = token_vectors[:max_src_length]
    
    src = src.unsqueeze(0).to(device)  # Shape: (1, max_src_length, embed_dim)
    return src

def generate_images(
    model: TransformerVAEModel,
    src: torch.Tensor,
    device: Optional[torch.device] = None,
    sequence_length: int = 300
) -> torch.Tensor:
    """
    Generates a sequence of images from the input source tensor using the model.

    Args:
        model (TransformerVAEModel): The loaded TransformerVAEModel.
        src (torch.Tensor): The source tensor of shape (1, max_src_length, embed_dim).
        device (Optional[torch.device]): Device to perform inference on. If None, automatically selects CUDA or CPU.
        sequence_length (int): The desired length of the video.

    Returns:
        torch.Tensor: Generated images tensor of shape (1, sequence_length, 3, H, W).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.max_output_length = sequence_length
    model.to(device)
    src = src.to(device)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            generated_images = model(src)  # Shape: (1, sequence_length, 3, H, W)
    
    return generated_images

def text_to_video(
    input_text: str,
    model_checkpoint: Optional[str] = None,
    output_video_path: str = './output_video.mp4',
    sequence_length: int = 300,
    fps: int = 15
) -> None:
    """
    Generates a video sequence from a text prompt and saves it to the specified path.

    Args:
        input_text (str): The text prompt to be used in the transformer encoder.
        model_checkpoint (Optional[str]): The path to the model checkpoint to use.
                                          If None, the latest checkpoint in './checkpoints' is used.
        output_video_path (str): The path where the output video will be saved.
        sequence_length (int): The desired length of the video (number of frames). Default is 300.
        fps (int): Frames per second for the output video. Default is 15.

    Raises:
        Exception: If any step in the generation process fails.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    model = load_model(model_checkpoint=model_checkpoint, device=device)
    
    print("Preparing input text...")
    src = prepare_input_text(input_text, max_src_length=64, embed_dim=300, device=device)
    
    print("Generating images...")
    generated_images = generate_images(model, src, device=device, sequence_length=sequence_length)
    
    with TemporaryDirectory() as tmpdir:
        print("Saving images to temporary directory...")
        save_images_to_folder(generated_images, tmpdir)

        print("Creating video from images...")
        imgs_to_video(tmpdir, output_video_path, resize=(512, 512), fps=fps)
    
if __name__ == '__main__':
    import argparse

    def main():
        parser = argparse.ArgumentParser(description="Generate a video sequence from a text prompt using a Transformer VAE model.")
        
        parser.add_argument('--input_text', type=str, required=True, help='The text prompt to generate the video.')
        parser.add_argument('--model_checkpoint', type=str, default=None, help='Path to the model checkpoint. If not provided, the latest checkpoint in ./checkpoints is used.')
        parser.add_argument('--output_video_path', type=str, default='./output_video.mp4', help='Path to save the generated video.')
        parser.add_argument('--sequence_length', type=int, default=150, help='Number of frames in the generated video.')
        parser.add_argument('--fps', type=int, default=15, help='Frames per second for the output video.')
        
        args = parser.parse_args()
        
        try:
            text_to_video(
                input_text=args.input_text,
                model_checkpoint=args.model_checkpoint,
                output_video_path=args.output_video_path,
                sequence_length=args.sequence_length,
                fps=args.fps
            )
        except Exception as e:
            print(f"An error occurred during video generation: {e}")
            exit(1)
    
    main()