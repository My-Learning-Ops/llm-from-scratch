
""" 
simple_gpt.py
A simplified Transformer-based language model similar to GPT.

This version only uses encoder layers of the Transformer architecture and dosent include casual masking.
In a true GPT-like model, the model would be prevented from attending to future tokens during training,
using a lower-triangular attention mask. This model skips that for clarity and simplicity.

It embeds input token indices, adds positional encoding, passes them through a stack of Transformer encoder blocks, 
normalizes the output, and projects it back into vocabulary space for logits, suitable for language modeling tasks if masking were added.

Brendan Dileo, July 2025

"""

import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    """ A simplified transformer based language model """
    
    
    def __init__(self, vocab_size, embed_dim=128, block_size=128, n_heads=4, n_layers=2):
        super().__init__()
        
        # Embeds token ids into continuous vectors of size embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # Learnable positional embeddings of shape (1, block_size, embed_dim)
        # These are added to the token embeddings to inject order information
        self.pos_embed = nn.Parameter(torch.zeros(1, block_size, embed_dim))
        
        # Stack of n_layers Transfer encoder blocks (not decoder, and no causal masking)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                # Ensures inputs are (B, T, C) instead of (T, B, C)
                batch_first=True
            )
            for _ in range(n_layers)
        ])
        
        # Normalizes the output across the embedding dimension
        # This helps stabilize training by ensuring the output has zero mean and unit variance
        self.ln = nn.LayerNorm(embed_dim)
        
        # Final linear layer that projects the output to vocabulary logits
        self.fc = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        """ Forward pass through the model """

        # Get batch size and sequence length from input tensor shape
        B, T = x.size()
        
        # Embed tokens and add positional embeddings, which are truncated to match the input length
        # This allows the model to learn not just what the tokens are, but where they are
        x = self.embed(x) + self.pos_embed[:, :T, :]
        
        # Pass through each transformer encoder layer sequentially
        for layer in self.layers:
            # Pass through each transformer encoder layer
            x = layer(x)
            
        # Normalize the output and project to vocabulary dimension
        x = self.ln(x)
        return self.fc(x)

            
    