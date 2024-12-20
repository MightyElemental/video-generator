
import pickle
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModel


class VideoCaptionDataset(Dataset):
    def __init__(self, pickle_file, tokenizer_name='bert-base-uncased', max_src_length=64):
        """
        Args:
            pickle_file: Path to the pickle file containing the data.
            tokenizer_name: Name of the pre-trained tokenizer.
            max_src_length: Maximum number of tokens for the input text.
        """
        self.vectors = []
        self.captions = []
        with open(pickle_file, 'rb') as f:
            raw_data = pickle.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.embedder = AutoModel.from_pretrained(tokenizer_name)
        self.embedder.eval()  # Freeze embedder

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

        # TODO: Pre-process data to speed up training?

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        item = self.captions[idx]
        text = item['enCap']
        vector_idx = item['latent_vectors']
        latent_vectors = self.vectors[vector_idx]  # List of lists or numpy arrays

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_src_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)         # (max_src_length)
        attention_mask = encoding['attention_mask'].squeeze(0)  # (max_src_length)

        # Get the embeddings from the embedder
        with torch.no_grad():
            embeddings = self.embedder(input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
            last_hidden_state = embeddings.last_hidden_state.squeeze(0)  # (max_src_length, embed_dim)

        # Convert latent_vectors to tensor
        latent_vectors = torch.tensor(latent_vectors, dtype=torch.float)  # (seq_length, vector_dim)

        return {
            'src': last_hidden_state,          # (max_src_length, embed_dim)
            'src_mask': None,                  # Placeholder (can implement if needed)
            'tgt': latent_vectors               # (seq_length, vector_dim)
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
