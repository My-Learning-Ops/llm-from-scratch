
""" 
samplers.py
Defines sampling utilities for text generation. The two functions provided implement top-k and top-p (nucleus) sampling strategies.

The top-k sampling restricts the selection to the k most likely tokens, and the top-p nucleus sampling dynamically selects the
smallest set of tokens whose cumulative probability exceeds a threshold p. These strategies control the creativity and randomness 
of the generated text.

Brendan Dileo, July 2025
"""

import torch
import torch.nn.functional as F

def sample_top_k(logits, k):
    """ Sample from the top-k most likely tokens.

    Args:
        logits (Tensor): The raw logits from the model (shape: [batch, vocab_size]).
        k (int): Number of top tokens to keep. If None or larger than vocab size, 
                 fallback to full distribution.
    """
    
    # If no k is given or k exceeds the vocabulary size, sample normally
    if k is None or k >= logits.size(-1):
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        # Sample 1 token from the entire distribution
        return torch.multinomial(probs, num_samples=1)

    # Take only the top-k logits and their indices
    top_logits, top_indices = torch.topk(logits, k, dim=-1)

    # Normalize top-k logits to probabilities
    probs = F.softmax(top_logits, dim=-1)
    
    # Randomly sample 1 token from these top-k candidates
    sampled_index = torch.multinomial(probs, num_samples=1)
    
    # Map back to the original indices
    return top_indices.gather(-1, sampled_index)

def sample_top_p(logits, p):
    """ Sample the next token using nucleus (top-p) sampling.

    Args:
        logits (Tensor): The raw logits from the model (shape: [batch, vocab_size]).
        p (float): Cumulative probability threshold (0 < p ≤ 1).
                   Tokens are selected until their combined probability exceeds p.
                   
    Returns:
        Tensor: Index of the sampled token (shape: [batch, 1]).
    """
    
    # If p is None or 1.0, sample from the entire distribution (no filtering)
    if p is None or p >= 1.0:
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # Sort tokens by probability
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(probs, dim=-1)

    # Find cutoff where cumulative prob exceeds p
    cutoff = torch.searchsorted(cumulative_probs, p, right=True)
    cutoff = max(1, cutoff.item())  # always keep at least one token

    # Keep only top-p logits
    top_p_logits = sorted_logits[:, :cutoff]
    top_p_indices = sorted_indices[:, :cutoff]

    # Sample from that set
    probs = F.softmax(top_p_logits, dim=-1)
    sampled_index = torch.multinomial(probs, num_samples=1)

    # Map sampled token back to full vocabulary space
    return top_p_indices.gather(-1, sampled_index)
