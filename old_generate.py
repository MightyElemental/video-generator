import os
import tempfile
import subprocess
import torch
import torchvision.utils as vutils
from torchvision.transforms import v2
from torchtext.vocab import Vectors, GloVe
from torchtext.data.utils import get_tokenizer
from text2vid.model import TransformerVectorGenerator
from vae.vae_model import VAE

def load_transformer_model(checkpoint_path: str, device: torch.device, **kwargs) -> TransformerVectorGenerator:
    """
    Loads the TransformerVectorGenerator model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the transformer model checkpoint.
        device (torch.device): Device to load the model onto.
        **kwargs: Additional keyword arguments for TransformerVectorGenerator.

    Returns:
        TransformerVectorGenerator: The loaded transformer model in evaluation mode.
    """
    model = TransformerVectorGenerator(**kwargs).to(device)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_vae_model(checkpoint_path: str, device: torch.device, img_size: int, latent_dim: int, multiplier: int =1) -> VAE:
    """
    Loads the VAE model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the VAE model checkpoint.
        device (torch.device): Device to load the model onto.
        img_size (int): Image size used in the VAE.
        latent_dim (int): Dimension of the latent space.
        multiplier (int, optional): Multiplier for the VAE channels. Defaults to 1.

    Returns:
        VAE: The loaded VAE model in evaluation mode.
    """
    vae = VAE(img_size=img_size, latent_dim=latent_dim, multiplier=multiplier).to(device)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )["model_state_dict"]
    vae.load_state_dict(state_dict)
    vae.eval()
    return vae

def tokenize_and_embed(
    text: str,
    tokenizer,
    embedder: Vectors,
    device: torch.device,
    max_length: int = 64
    ) -> torch.Tensor:
    """
    Tokenizes and embeds the input text.

    Args:
        text (str): The input text to tokenize and embed.
        tokenizer (function): tokenizer function.
        embedder (Vectors): Pre-trained embeddings.
        device (torch.device): Device to perform computations on.
        max_length (int): Maximum token length. Defaults to 64.

    Returns:
        torch.Tensor: The embedded text tensor of shape (1, <=max_length, embed_dim).
    """
    tokens: list[str] = tokenizer(text)
    tokens_tensor = embedder.get_vecs_by_tokens(tokens[:max_length]).to(device)

    return tokens_tensor.unsqueeze(0)

def generate_vectors(transformer_model: TransformerVectorGenerator, src: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates output vectors and stop logits using the transformer model.

    Args:
        transformer_model (TransformerVectorGenerator): The transformer model.
        src (torch.Tensor): Source tensor of shape (1, max_length, embed_dim).
        device (torch.device): Device to perform computations on.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Output vectors of shape (seq_length, vector_dim) and stop logits of shape (seq_length,).
    """
    with torch.no_grad():
        output_vectors, stop_logits = transformer_model(src.to(device), tgt=None)
        # TODO: Trim last vector (stop vector)
        # output_vectors: (1, seq_length, vector_dim)
        # stop_logits: (1, seq_length)
    return output_vectors.squeeze(0), stop_logits.squeeze(0)

def truncate_vectors(vectors: torch.Tensor, stop_logits: torch.Tensor) -> torch.Tensor:
    """
    Truncates the vectors sequence up to the first stop token.

    Args:
        vectors (torch.Tensor): Output vectors of shape (seq_length, vector_dim).
        stop_logits (torch.Tensor): Stop logits of shape (seq_length,).

    Returns:
        torch.Tensor: Truncated vectors of shape (truncated_seq_length, vector_dim).
    """
    stop_probs = torch.sigmoid(stop_logits)  # (seq_length,)
    stop_indices = (stop_probs > 0.5).nonzero(as_tuple=True)[0]

    if stop_indices.numel() > 0:
        first_stop_index = stop_indices[0].item()
        truncated_vectors = vectors[:first_stop_index + 1]  # Include the stop vector
    else:
        truncated_vectors = vectors  # No stop token found

    return truncated_vectors

def generate_images(
    vae_model: VAE,
    vectors: torch.Tensor,
    device: torch.device,
    image_dir: str,
    batch_size: int = 32
    ) -> list:
    """
    Generates images from vectors using the VAE's decoder.

    Args:
        vae_model (VAE): The VAE model.
        vectors (torch.Tensor): Truncated vectors of shape (seq_length, vector_dim).
        device (torch.device): Device to perform computations on.
        image_dir (str): Directory to save the generated images.
        batch_size (int): Number of vectors to process in a batch.

    Returns:
        list: List of image file paths.
    """
    os.makedirs(image_dir, exist_ok=True)
    images = []

    transform = v2.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.255]
    )

    with torch.no_grad():
        for start in range(0, vectors.size(0), batch_size):
            end = start + batch_size
            batch_vectors = vectors[start:end].to(device)  # (batch_size, vector_dim)
            reconstructed = vae_model.decode(batch_vectors)  # (batch_size, channels, H, W)

            # Apply transforms
            transformed_images = transform(reconstructed)

            for i, image_array in enumerate(transformed_images):
                image_path = os.path.join(image_dir, f'image_{start + i:05d}.png')
                vutils.save_image(image_array, image_path, normalize=True)
                images.append(image_path)

    return images

def compile_images_to_video(image_dir: str, output_video_path: str, fps: int = 30):
    """
    Compiles a sequence of images into an MP4 video using ffmpeg.

    Args:
        image_dir (str): Directory containing the image sequence.
        output_video_path (str): Path to save the compiled video.
        fps (int, optional): Frames per second for the video. Defaults to 24.
    """
    # Ensure ffmpeg is installed
    try:
        subprocess.check_output(['ffmpeg', '-version'])
    except subprocess.CalledProcessError:
        raise RuntimeError('ffmpeg is not installed. Please install it and try again.')

    input_pattern = os.path.join(image_dir, 'image_%05d.png')
    # TODO: Add libx264
    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-framerate', str(fps),
        '-i', input_pattern,
        '-pix_fmt', 'yuv420p',
        output_video_path
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg failed to generate video") from e

def text_to_video(input_text: str,
                  transformer_checkpoint: str,
                  vae_checkpoint: str,
                  output_video_path: str,
                  embed_dim: int = 300,
                  max_src_length: int = 64,
                  max_output_length: int = 500,
                  transformer_params: dict | None = None,
                  vae_params: dict | None = None,
                  fps: int = 30):
    """
    Generates a video from input text by using a transformer model to generate a sequence of vectors,
    converting each vector to an image using a VAE decoder, and compiling the images into an MP4 video.

    Args:
        input_text (str): The text input to generate the video from.
        transformer_checkpoint (str): Path to the transformer model checkpoint.
        vae_checkpoint (str): Path to the VAE model checkpoint.
        output_video_path (str): Path to save the generated MP4 video.
        embed_dim (int, optional): Dimension of the embeddings. Defaults to 300.
        max_src_length (int, optional): Maximum token length for the input text. Defaults to 64.
        max_output_length (int, optional): Maximum number of vectors to generate. Defaults to 500.
        transformer_params (dict, optional): Additional parameters for the transformer model. Defaults to None.
        vae_params (dict, optional): Additional parameters for the VAE model. Defaults to None.
        fps (int, optional): Frames per second for the output video. Defaults to 24.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("==> Loading tokenizer / embedder")

    # Load tokenizer and embedder
    tokenizer = get_tokenizer('basic_english')
    embedder = GloVe(name='840B', dim=embed_dim)

    # Load transformer model
    if transformer_params is None:
        # These should match the training configuration
        transformer_params = {
            'embed_dim': embed_dim,
            'vector_dim': 200,
            'nhead': 6,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'dim_feedforward': 2048,
            'max_seq_length': max_src_length,
            'max_output_length': max_output_length
        }

    transformer_model = load_transformer_model(
        checkpoint_path=transformer_checkpoint,
        device=device,
        **transformer_params
    )

    # Load VAE model
    if vae_params is None:
        # These should match the training configuration
        vae_params = {
            'img_size': 256,
            'latent_dim': 200,
            'multiplier': 3
        }

    print("==> Loading VAE model")

    vae_model = load_vae_model(
        checkpoint_path=vae_checkpoint,
        device=device,
        **vae_params
    )

    print("==> Loading Transformer model")

    # Tokenize and embed input text
    src = tokenize_and_embed(
        text=input_text,
        tokenizer=tokenizer,
        embedder=embedder,
        device=device,
        max_length=max_src_length
    )  # (1, max_length, embed_dim)

    print("==> Generating frame vectors")

    # Generate vectors using transformer
    output_vectors, stop_logits = generate_vectors(
        transformer_model,
        src,
        device
    )  # (seq_length, vector_dim), (seq_length, )

    # Truncate vectors at the first stop token
    truncated_vectors = truncate_vectors(
        output_vectors,
        stop_logits
    )  # (truncated_seq_length, vector_dim)

    # Generate images from vectors
    with tempfile.TemporaryDirectory() as tmpdir:
        print("==> Generating frame images")
        generate_images(
            vae_model,
            truncated_vectors,
            device,
            tmpdir
        )

        print("==> Compiling video")

        # Compile images to video
        compile_images_to_video(
            image_dir=tmpdir,
            output_video_path=output_video_path,
            fps=fps
        )

    print(f"==> Video successfully saved to {output_video_path}")
