
""" 
transformer_block.py
Defines a single Transformer block used in the model.
The block contains a normalization layer, a multi head casual self-attention layer,
a feed forward MLP sublayer with dropout and residual connections around each sublayer.

Brendan Dileo, July 2025
"""

import torch.nn as nn
from src.models.attention import CausalSelfAttention
from src.models.mlp import MLP

class CausalTransformerBlock(nn.Module):
    """ A single transformer block with causal self-attention

    Args:
        embed_dim (int): Embedding dimensionality.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout rate applied in attention and MLP.
    """
    
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()

        # Layer normalization before self-attention
        self.ln1 = nn.LayerNorm(embed_dim)
        
        # Multi-head causal self-attention module
        self.attn = CausalSelfAttention(embed_dim, n_heads, dropout)
        
        # Layer normalization before MLP
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward MLP module
        self.mlp = MLP(embed_dim, dropout)
    
    def forward(self, x):
        """ Forwards pass through the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embed_dim)

        Returns:
            torch.Tensor: Output tensor of same shape
        """
        # Pre-norm architecture: normalize before sublayer
        # Apply self-attention with residual connection
        x = x + self.attn(self.ln1(x))
        
        # Apply feed-forward MLP with residual connection
        x = x + self.mlp(self.ln2(x))
        return x