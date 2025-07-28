
""" 
attention.py
Defines the CasualSelfAttention module, implementing multi head self-attention with casual masking
to prevent postitions from attending to future tokens. This is essential for autoregressive models like GPT.

Brendan Dileo, July 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    """ Multi-head casual self attention mechanism.
    This module computes scaled dot-product attention seperately for multiple heads, masks future
    tokens to ensure casuality, and then projects concatenated outputs.

    Args:
        embed_dim (int): Dimensionality of input embeddings.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout probability applied to attention weights.
    """
    
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        
        # Check that embed_dim is divisible by number of heads for even splitting
        assert embed_dim % n_heads == 0
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads 
        # Dimension per head
        self.head_dim = embed_dim // n_heads
        
        # Linear layer that simultaneously projects input into queries, keys, and values
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        
        # Output linear projection after concatenating attention heads
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout applied to attention weights for regularization
        self.dropout = nn.Dropout(dropout)
                
        # Register a buffer (non-trainable tensor) for the causal mask
        # The mask is a lower-triangular matrix with ones on and below the diagonal
        # This prevents attention to future tokens
        self.register_buffer("mask", torch.tril(torch.ones(1000, 1000)))
    
    def forward(self, x):
        """ Forward pass for self attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embed_dim)

        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        
        # Batch size, sequence length, embedding dim
        B, T, C = x.size()
        
        # Compute queries, keys, and values by linear projection
        qkv = self.qkv(x)  # (B, T, 3 * embed_dim)
        
        # Split qkv into separate tensors for queries, keys, and values
        q, k, v = qkv.chunk(3, dim=-1)  # Each is (B, T, embed_dim)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
                
        # Compute scaled dot-product attention scores
        # Matrix multiplication between queries and keys transpose: (B, n_head
        att_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_heads, T, T)
        
        # Apply causal mask to prevent attending to future tokens
        # Mask shape is (T, T), broadcasted over batches and heads
        att_scores = att_scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        
        # Normalize attention scores with softmax to get attention weights
        att_weights = F.softmax(att_scores, dim=-1)
        
        # Apply dropout on attention weights for regularization
        att_weights = self.dropout(att_weights)
        
        # Multiply attention weights by values to get weighted sum (context)
        out = att_weights @ v  # (B, n_heads, T, head_dim)
        
        # Concatenate multiple heads by reversing the transpose and reshaping
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, embed_dim)
        # Final linear projection after concatenation of heads
        out = self.proj(out)
        
        return out