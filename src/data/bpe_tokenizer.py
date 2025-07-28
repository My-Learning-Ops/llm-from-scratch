
""" 
bpe_tokenizer.py

Brendan Dileo, July 2025
"""

import torch
from torch.utils.data import Dataset
import sentencepiece as spm

class BPEDataset(Dataset):
    def __init__(self, text, block_size, sp_model_path):
        """
        text: raw training text (string)
        block_size: length of input sequences in tokens
        sp_model_path: path to SentencePiece model file (e.g., "bpe_tokenizer.model")
        """
        self.block_size = block_size
        
        # Load SentencePiece model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        
        # Tokenize entire text into token ids (list of ints)
        self.token_ids = self.sp.encode(text, out_type=int)
        
        # Vocabulary size is size of SentencePiece vocab
        self.vocab_size = self.sp.get_piece_size()
        
        # Special token IDs
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        
        print(f"Loaded BPE tokenizer:")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  PAD ID: {self.pad_id}")
        print(f"  UNK ID: {self.unk_id}")
        print(f"  BOS ID: {self.bos_id}")
        print(f"  EOS ID: {self.eos_id}")
        
    def __len__(self):
        # number of samples is length of token_ids minus block_size
        return len(self.token_ids) - self.block_size
    
    def __getitem__(self, idx):
        # Extract sequence chunk of length block_size + 1 (input + target)
        chunk = self.token_ids[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
    
    def encode(self, text):
        """Encode raw text into list of token IDs"""
        return self.sp.encode(text, out_type=int)

    def decode(self, token_ids):
        """Decode list of token IDs back to text, filtering out special tokens"""
        filtered_ids = [tid for tid in token_ids if tid not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.decode(filtered_ids)
