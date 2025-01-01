import os
import json
from pickle import HIGHEST_PROTOCOL
from typing import Optional, Callable, List
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image

from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer

class VideoCaptionDataset(Dataset):
    def __init__(
        self,
        json_file: str,
        root_dir: str,
        cache_path: Optional[str] = None,
        max_src_length: int = 64,
        glove_dim: int = 300,
        unk_vector: Optional[torch.Tensor] = None,
        max_output_length: int = 300,
        image_size: int = 256,
        frame_selection_fn: Optional[Callable[[List[int]], List[int]]] = None,
    ):
        """
        Args:
            json_file (str): Path to the JSON file containing the data.
            root_dir (str): Root directory containing image sequence folders.
            cache_path (str): Root directory containing the pre-cached image sequences.
                               If None, a cache will not be used/generated.
            max_src_length (int): Maximum number of tokens for the input text.
            glove_dim (int): Dimension of the GloVe vectors.
            unk_vector (torch.Tensor, optional): Vector to represent unknown tokens.
                                                  If None, uses a zero vector.
            max_output_length (int): Maximum number of images in the output sequence.
            image_size (int): Size for center cropping images.
        """
        self.samples = []
        with open(json_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = GloVe(name='840B', dim=glove_dim, unk_init=unk_vector)
        self.embed_dim = glove_dim
        self.max_src_length = max_src_length
        self.max_output_length = max_output_length
        self.root_dir = root_dir
        self.cache_path = cache_path
        self.image_size = image_size
        self.frame_selection_fn = frame_selection_fn

        # Define image transformations
        self.transform = v2.Compose([
            v2.CenterCrop(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],  # Recommended normalization
                         std=[0.229, 0.224, 0.225])
        ])

        # Process each entry in the JSON file
        for entry in raw_data:
            video_id = entry['videoID']
            captions = entry['enCap']
            image_dir = os.path.join(root_dir, video_id)

            if not os.path.isdir(image_dir):
                # print(f"Warning: Image directory {image_dir} does not exist. Skipping.")
                continue

            for caption in captions:
                self.samples.append({
                    'caption': caption,
                    'video_id': video_id,
                })

    def __len__(self):
        return len(self.samples)

    def _get_cached_images(self, video_id: str):
        if self.cache_path:
            cache = os.path.join(self.cache_path, video_id+".pt")

            if os.path.exists(cache):
                return torch.load(cache)

        return None

    def _save_cache(self, video_id, images):
        if self.cache_path:
            os.makedirs(self.cache_path, exist_ok=True)
            cache = os.path.join(self.cache_path, video_id+".pt")
            torch.save(images, cache, pickle_protocol=HIGHEST_PROTOCOL)

    def _get_images(self, video_id):
        # Return the cached image sequence tensor if it exists
        cache = self._get_cached_images(video_id)
        if cache:
            return cache

        image_dir = os.path.join(self.root_dir, video_id)

        # Sort the images in order
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        image_paths = [os.path.join(image_dir, f) for f in image_files]

        # Use frame selection function if present
        if self.frame_selection_fn:
            selected_indices = self.frame_selection_fn(list(range(len(image_paths))))
            image_paths = [image_paths[i] for i in selected_indices]

        # Limit the number of images per sequence
        image_paths = image_paths[:self.max_output_length]

        # Load all images
        images = []
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)  # Apply transformations
                images.append(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue

        # Ensure a fixed number of images
        # Padding with black images
        # TODO: Ignore padded images in loss function
        while len(images) < self.max_output_length:
            images.append(torch.zeros(3, self.image_size, self.image_size))

        images = torch.stack(images)  # (max_output_length, C, H, W)

        # Save image sequence to cache if enabled
        self._save_cache(video_id, images)

        return images

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item['caption']
        video_id = item['video_id']  # (max_output_length, C, H, W)
        images = self._get_images(video_id)

        # Tokenize the text
        tokens = self.tokenizer(text)
        if isinstance(tokens, list):
            tokens = tokens[:self.max_src_length]
        else:
            raise TypeError("Tokenizer must return a list of token strings")

        # Initialize src tensor - pad with zeros
        src = torch.zeros(self.max_src_length, self.embed_dim)
        token_vectors = self.vocab.get_vecs_by_tokens(tokens)
        src[:token_vectors.size(0), :] = token_vectors

        return {
            'src': src,               # (max_src_length, embed_dim)
            'tgt': images,            # (max_output_length, C, H, W)
            'prompt': text,           # The prompt as a human-readable string
            'videoID': video_id       # The video ID so the output can be checked
        }

def collate_fn(batch) -> dict[str, torch.Tensor | list[str]]:
    """
    Custom collate function to handle batches.
    Pads the source sequences to the maximum length in the batch.
    Ensures target image sequences are of the same length.
    """
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]
    prompt_batch = [item['prompt'] for item in batch]
    video_ids = [item['videoID'] for item in batch]

    batch_size = len(src_batch)
    embed_dim = src_batch[0].size(1)
    # C, H, W = tgt_batch[0].size(1), tgt_batch[0].size(2), tgt_batch[0].size(3)
    max_src_length = max([src.size(0) for src in src_batch])

    # Pad src sequences
    src_padded = torch.zeros(batch_size, max_src_length, embed_dim)
    for i, src in enumerate(src_batch):
        src_padded[i, :src.size(0), :] = src

    # Stack tgt sequences (already fixed length via dataset class)
    tgt_padded = torch.stack(tgt_batch)  # (batch_size, max_output_length, C, H, W)

    src_lengths = torch.tensor([s.size(0) for s in src_batch])

    return {
        'src': src_padded,               # (batch_size, max_src_length, embed_dim)
        'tgt': tgt_padded,               # (batch_size, max_output_length, C, H, W)
        'src_lengths': src_lengths,
        'prompt': prompt_batch,
        'videoID': video_ids,
    }
