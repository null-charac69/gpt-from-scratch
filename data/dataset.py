"""
Dataset loading and preprocessing for GPT training.
Handles text data loading, tokenization, and batching.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class TextDataset(Dataset):
    """
    Dataset for language modeling.
    Loads text data and provides tokenized sequences.
    """
    
    def __init__(self, text_file, tokenizer, block_size=1024):
        """
        Args:
            text_file: Path to text file or raw text string
            tokenizer: Tokenizer instance
            block_size: Length of sequences (context window)
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Load text
        if os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = text_file  # Assume it's raw text
        
        # Tokenize entire text
        self.tokens = self.tokenizer.encode(text)
        print(f"Loaded {len(text)} characters, {len(self.tokens)} tokens")
        
        # Calculate number of samples
        self.num_samples = len(self.tokens) - block_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns a block of tokens and the next token as target.
        Input: tokens[idx:idx+block_size]
        Target: tokens[idx+1:idx+block_size+1]
        """
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class InfiniteDataLoader:
    """
    Wrapper around DataLoader that provides infinite batches.
    Useful for training without explicit epochs.
    """
    
    def __init__(self, dataset, batch_size, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loader = None
        self._iterator = None
    
    def _create_loader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.loader is None:
            self.loader = self._create_loader()
            self._iterator = iter(self.loader)
        
        try:
            batch = next(self._iterator)
        except StopIteration:
            # Recreate iterator when epoch ends
            self._iterator = iter(self.loader)
            batch = next(self._iterator)
        
        return batch


class DataLoaderWrapper:
    """
    Manages train and validation data loaders.
    """
    
    def __init__(self, train_data, val_data, tokenizer, config):
        """
        Args:
            train_data: Training text file path or string
            val_data: Validation text file path or string  
            tokenizer: Tokenizer instance
            config: GPTConfig instance
        """
        self.train_dataset = TextDataset(
            train_data, 
            tokenizer, 
            block_size=config.n_positions
        )
        
        if val_data:
            self.val_dataset = TextDataset(
                val_data,
                tokenizer,
                block_size=config.n_positions
            )
        else:
            self.val_dataset = None
    
    def get_train_loader(self, batch_size, num_workers=0, infinite=True):
        """Get training data loader."""
        if infinite:
            return InfiniteDataLoader(
                self.train_dataset,
                batch_size=batch_size,
                num_workers=num_workers
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
    
    def get_val_loader(self, batch_size, num_workers=0):
        """Get validation data loader."""
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )


def download_shakespeare():
    """
    Download Shakespeare dataset as example.
    Returns path to the downloaded file.
    """
    import requests
    
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    save_path = '/tmp/shakespeare.txt'
    
    if not os.path.exists(save_path):
        print("Downloading Shakespeare dataset...")
        response = requests.get(url)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Downloaded to {save_path}")
    
    return save_path


def prepare_sample_data():
    """
    Prepare a small sample dataset for testing.
    Returns train and validation text.
    """
    # Small sample text
    text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
""" * 20  # Repeat to have more data
    
    # Split into train and val
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    return train_text, val_text