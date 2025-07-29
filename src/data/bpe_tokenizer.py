
""" 
bpe_tokenizer.py
This module defines the BPEDataset class, which is a combined dataset and tokenizer built on top of SentencePiece's BPE model.
It handles tokenization of raw text, manages special tokens, and provides a PyTorch Dataset interface for feeding tokenized sequences into language models.

Brendan Dileo, July 2025
"""

import os
import torch
import logging
import sentencepiece as spm
from torch.utils.data import Dataset
from typing import List, Union, Optional, Dict

class BPEDataset(Dataset):
    
    def __init__(
        self, 
        text: str, 
        block_size: int, 
        sp_model_path: str,
        add_special_tokens: bool = True,
        stride: Optional[int] = None,
        return_attention_mask: bool = False,
        cache_tokens: bool = True
    ):
        """ 
        Initializes the BPE dataset.

        Args:
            text: raw training text (string)
            block_size: length of input sequences in tokens
            sp_model_path: path to SentencePiece model file
            add_special_tokens: whether to add BOS/EOS tokens to sequences
            stride: overlap between consecutive samples (for better context)
            return_attention_mask: whether to return attention masks
            cache_tokens: whether to cache tokenized text in memory
        """
        
        # Define instance variables
        self.block_size = block_size
        self.add_special_tokens = add_special_tokens
        self.stride = stride or block_size
        self.return_attention_mask = return_attention_mask
        
        # Validate model file path
        if not os.path.exists(sp_model_path):
            raise FileNotFoundError(f"SentencePiece model not found: {sp_model_path}")
        
        if self.stride > block_size:
            raise ValueError(f"Stride ({stride}) cannot be larger than block_size ({block_size})")
        
        # Load SentencePiece model with error handling
        try:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(sp_model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load SentencePiece model: {e}")
        
        
        # Get special token IDs
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.vocab_size = self.sp.get_piece_size()
        
        # Tokenize text with optional caching
        if cache_tokens:
            self.token_ids = self._tokenize_text(text)
        else:
            self.text = text
            self.token_ids = None
            
        # Calculate number of samples based on stride
        if self.token_ids:
            text_length = len(self.token_ids)
        else:
            # Estimate length for non-cached mode
            text_length = len(self.sp.encode(text[:10000])) * (len(text) / 10000)
            text_length = int(text_length)
        
        if self.add_special_tokens:
            # Account for BOS/EOS tokens
            effective_block_size = block_size - 2
        else:
            effective_block_size = block_size
            
        self.num_samples = max(0, (text_length - effective_block_size - 1) // self.stride + 1)
        
        self._log_info()
    
    def _tokenize_text(self, text: str) -> List[int]:
        """ Tokenize text with progress indication for larger bodies of text. """
        
        # For large texts, show progress
        if len(text) > 1_000_000:
            logging.info("Tokenizing large text corpus...")
        
        token_ids = self.sp.encode(text, out_type=int)
        
        if len(text) > 1_000_000:
            logging.info(f"Tokenization complete. Generated {len(token_ids)} tokens.")
        
        return token_ids
    
    def _log_info(self):
        """ Log tokenizer info """
        print(f"Enhanced BPE Tokenizer loaded:")
        print(f"  Vocab size: {self.vocab_size:,}")
        print(f"  Block size: {self.block_size}")
        print(f"  Stride: {self.stride}")
        print(f"  Add special tokens: {self.add_special_tokens}")
        print(f"  Number of samples: {self.num_samples:,}")
        print(f"  Special token IDs:")
        print(f"    PAD: {self.pad_id}")
        print(f"    UNK: {self.unk_id}")
        print(f"    BOS: {self.bos_id}")
        print(f"    EOS: {self.eos_id}")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Union[tuple, Dict[str, torch.Tensor]]:
        # Handle non-cached mode
        if self.token_ids is None:
            # This is less efficient but saves memory
            start_char = idx * self.stride * 4  # Rough estimate
            end_char = start_char + self.block_size * 6  # Rough estimate with buffer
            chunk_text = self.text[start_char:end_char]
            chunk_tokens = self.sp.encode(chunk_text, out_type=int)
        else:
            # Use cached tokens
            start_idx = idx * self.stride
            end_idx = start_idx + self.block_size + 1
            
            if end_idx > len(self.token_ids):
                # Handle edge case by padding
                chunk_tokens = self.token_ids[start_idx:]
                while len(chunk_tokens) < self.block_size + 1:
                    chunk_tokens.append(self.pad_id)
            else:
                chunk_tokens = self.token_ids[start_idx:end_idx]
        
        # Add special tokens if requested
        if self.add_special_tokens:
            if self.bos_id >= 0:  # Check if BOS token exists
                chunk_tokens = [self.bos_id] + chunk_tokens
            if self.eos_id >= 0 and len(chunk_tokens) < self.block_size + 1:
                chunk_tokens = chunk_tokens + [self.eos_id]
        
        # Ensure we have exactly block_size + 1 tokens
        if len(chunk_tokens) > self.block_size + 1:
            chunk_tokens = chunk_tokens[:self.block_size + 1]
        elif len(chunk_tokens) < self.block_size + 1:
            # Pad if necessary
            chunk_tokens.extend([self.pad_id] * (self.block_size + 1 - len(chunk_tokens)))
        
        # Create input and target sequences
        x = torch.tensor(chunk_tokens[:-1], dtype=torch.long)
        y = torch.tensor(chunk_tokens[1:], dtype=torch.long)
        
        if self.return_attention_mask:
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.tensor([1 if token != self.pad_id else 0 for token in x], 
                                        dtype=torch.long)
            return {
                'input_ids': x,
                'labels': y,
                'attention_mask': attention_mask
            }
        
        return x, y
    
    def encode(self, text: str, add_special_tokens: bool = None) -> List[int]:
        """
        Encode raw text into list of token IDs
        
        Args:
            text: input text to encode
            add_special_tokens: override default special token behavior
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")
        
        token_ids = self.sp.encode(text, out_type=int)
        
        if add_special_tokens is None:
            add_special_tokens = self.add_special_tokens
        
        if add_special_tokens:
            if self.bos_id >= 0:
                token_ids = [self.bos_id] + token_ids
            if self.eos_id >= 0:
                token_ids = token_ids + [self.eos_id]
        
        return token_ids

    def decode(self, token_ids: Union[List[int], torch.Tensor], 
               skip_special_tokens: bool = True) -> str:
        """
        Decode list of token IDs back to text
        
        Args:
            token_ids: token IDs to decode
            skip_special_tokens: whether to filter out special tokens
        """
        # Handle PyTorch tensors
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if not isinstance(token_ids, list):
            raise TypeError("token_ids must be a list or torch.Tensor")
        
        if skip_special_tokens:
            # Filter out special tokens
            special_tokens = {self.pad_id, self.bos_id, self.eos_id}
            filtered_ids = [tid for tid in token_ids if tid not in special_tokens]
        else:
            filtered_ids = token_ids
        
        try:
            return self.sp.decode(filtered_ids)
        except Exception as e:
            logging.warning(f"Decode error: {e}")
            return ""
    
    def batch_encode(self, texts: List[str], 
                    add_special_tokens: bool = None,
                    max_length: Optional[int] = None,
                    padding: bool = True,
                    truncation: bool = True) -> Dict[str, torch.Tensor]:
        """
        Batch encode multiple texts with padding and truncation
        
        Args:
            texts: list of input texts
            add_special_tokens: whether to add special tokens
            max_length: maximum sequence length
            padding: whether to pad sequences to same length
            truncation: whether to truncate long sequences
        """
        if add_special_tokens is None:
            add_special_tokens = self.add_special_tokens
        
        if max_length is None:
            max_length = self.block_size
        
        encoded_batch = []
        attention_masks = []
        
        for text in texts:
            token_ids = self.encode(text, add_special_tokens=add_special_tokens)
            
            # Truncate if necessary
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                if add_special_tokens and self.eos_id >= 0:
                    token_ids[-1] = self.eos_id  # Ensure EOS at end
            
            # Create attention mask
            attention_mask = [1] * len(token_ids)
            
            # Pad if necessary
            if padding and len(token_ids) < max_length:
                pad_length = max_length - len(token_ids)
                token_ids.extend([self.pad_id] * pad_length)
                attention_mask.extend([0] * pad_length)
            
            encoded_batch.append(token_ids)
            attention_masks.append(attention_mask)
        
        return {
            'input_ids': torch.tensor(encoded_batch, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
        }
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary as dictionary mapping tokens to IDs"""
        vocab = {}
        for i in range(self.vocab_size):
            piece = self.sp.id_to_piece(i)
            vocab[piece] = i
        return vocab
    
    def get_special_tokens_dict(self) -> Dict[str, int]:
        """Get special tokens dictionary"""
        return {
            'pad_token': self.pad_id,
            'unk_token': self.unk_id,
            'bos_token': self.bos_id,
            'eos_token': self.eos_id
        }