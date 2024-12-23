
import pickle
import torch
from torch.utils.data import Dataset

from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer


class VideoCaptionDataset(Dataset):
    def __init__(
        self,
        pickle_file: str,
        max_src_length: int = 64,
        glove_dim: int = 300,
        unk_vector: torch.Tensor | None = None
        ):
        """
        Args:
            pickle_file (str): Path to the pickle file containing the data.
            max_src_length (int): Maximum number of tokens for the input text.
            glove_dim (int): Dimension of the GloVe vectors.
            unk_vector (torch.Tensor, optional): Vector to represent unknown tokens.
                                                  If None, uses a zero vector.
        """
        self.vectors = []
        self.captions = []
        with open(pickle_file, 'rb') as f:
            raw_data = pickle.load(f)

        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = GloVe(name='840B', dim=glove_dim, unk_init=unk_vector)
        self.embed_dim = glove_dim
        self.max_src_length = max_src_length

        # Process each object in the pickle file
        for obj in raw_data:
            enCaps = obj['enCap']
            self.vectors.append(obj['latent_vectors'])

            for enCap in enCaps:
                self.captions.append({
                    'enCap': enCap,
                    'vectors': len(self.vectors) - 1
                })

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        item = self.captions[idx]
        text = item['enCap']
        vector_idx = item['vectors']
        latent_vectors = self.vectors[vector_idx]

        # Tokenize the text
        tokens: list[str] = self.tokenizer(text)
        tokens = tokens[:self.max_src_length]

        # Initialize src tensor - pad with zeros
        src = torch.zeros(self.max_src_length, self.embed_dim)
        for i, token in enumerate(self.vocab.get_vecs_by_tokens(tokens)):
            src[i] = token

        return {
            'src': src,  # (max_src_length, embed_dim)
            'tgt': torch.tensor(latent_vectors, dtype=torch.float)  # (seq_length, vector_dim)
        }

def collate_fn(batch):
    """
    Custom collate function to handle batches of varying target lengths.
    Pads the target sequences to the maximum length in the batch.
    """
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]

    batch_size = len(src_batch)
    embed_dim = src_batch[0].size(1)
    vector_dim = tgt_batch[0].size(1)

    max_src_length = max([src.size(0) for src in src_batch])
    max_tgt_length = max([tgt.size(0) for tgt in tgt_batch])

    # Initialize padded tensors
    src_padded = torch.zeros(batch_size, max_src_length, embed_dim)
    for i, src in enumerate(src_batch):
        src_padded[i, :src.size(0), :] = src

    tgt_padded = torch.zeros(batch_size, max_tgt_length, vector_dim)
    for i, tgt in enumerate(tgt_batch):
        tgt_padded[i, :tgt.size(0), :] = tgt

    return {
        'src': src_padded,                   # (batch_size, max_src_length, embed_dim)
        'tgt': tgt_padded,                   # (batch_size, max_tgt_length, vector_dim)
        'src_lengths': torch.tensor([s.size(0) for s in src_batch]),
        'tgt_lengths': torch.tensor([t.size(0) for t in tgt_batch]),
    }
