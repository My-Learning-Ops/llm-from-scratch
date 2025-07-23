
""" 
dataset.py
This module defines a class used for training a character-level language model. It tokenizes
a given text into sequences of fixed length to be used as input-target pairs for autoregressive training.

Brendan Dileo, July 2025
"""

import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """ Character-level Dataset for autoregressive language modeling.
    Converts a raw text string into sequences of integer token ids where each sample
    is a pair (x, y).
    """
    
    # Creates an instance of the dataset
    def __init__(self, text, block_size):
        """ Creates an instance of the CharDataset """
        
        # Finds all unique characters in the text and sorts
        chars = sorted(set(text))
        
        # Creates a dictionary that maps each character to a unique index
        # Converts characters to integers as models require numerical input
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        
        # Saves size of vocabulary, block size, and full text as instance variables
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.data = text

    def __len__(self):
        """ Returns the number of training samples in the dataset. 
        Each sample is a sequence of length block_size + 1, so the total number
        of samples is the length of the text minus the block size.
        """
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        """Returns the idx-th sample from the dataset.
        It extracts a chunk of text starting from idx to idx + block_size + 1
        """
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.tensor([self.stoi[c] for c in chunk[:-1]], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in chunk[1:]], dtype=torch.long)
        return x, y
    
    
    