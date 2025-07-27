
""" 
mlp.py
This module defines a simple Multi Layer Perceptron (mlp) block used in transformer models.
The MLP consists of two linear layers with a GELU activation in between, and dropout regularization
to help prevent overfitting.

Brendan Dileo, July 2025
"""

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """ Multi layer perceptron with GELU activation used as the feed forward network component 
    inside transformer blocks.

    Args:
        embed_dim (int): The input and output embedding dimensionality.
        dropout (float): Dropout rate for regularization (default 0.1).
    """
    
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        
        # First linear layer expands the embedding dimension by 4x
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        
        # Second linear layer projects back to the original embedding dimension
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        
        # Dropout layer for regularization after GELU activation
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """ Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embed_dim)

        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        # Apply first linear layer followed by GELU activation
        x = F.gelu(self.fc1(x))
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # Apply second linear layer to project back to embed_dim
        x = self.fc2(x)
        return x