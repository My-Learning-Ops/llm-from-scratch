
""" 
generate.py
The core text generation logic using multiple sampling strategies.

It supports greedy sampling, temperature based sampling, top-k sampling, and top-p sampling.
This module serves as the main generation engine and uses sampling utilities from samplers.py.

Brendan Dileo, July 2025
"""

import torch
import torch.nn.functional as F
from .samplers import sample_top_k, sample_top_p

def generate_text(model, prompt, max_length, stoi, itos, device='cpu',
                 method='temperature', temperature=1.0, top_k=None, top_p=None):
    """ 
    Generate text using a trained language model and various sampling strategies.

    Args:
        model (nn.Module): The trained transformer model.
        prompt (str): The initial text to start generation from.
        max_length (int): Number of tokens to generate after the prompt.
        stoi (dict): Mapping from characters to indices.
        itos (dict): Mapping from indices back to characters.
        device (str): Device to run inference on ('cpu' or 'cuda').
        method (str): Sampling strategy: 'greedy', 'temperature', 'top_k', or 'top_p'.
        temperature (float): Temperature for scaling logits (only used for temperature sampling).
        top_k (int): Number of top tokens to keep for top-k sampling.
        top_p (float): Probability mass cutoff for nucleus (top-p) sampling.

    Raises:
        ValueError: If an unknown sampling method is provided (not one of 
                    'greedy', 'temperature', 'top_k', or 'top_p').

    Returns:
        str: The full generated text including the prompt.
    """

    model.eval()
    model.to(device)

    # Convert prompt text into token IDs
    tokens = [stoi.get(c, stoi.get(' ', 0)) for c in prompt]  # fallback to space if char not found
    if not tokens:
        tokens = [0]

    generated = torch.tensor([tokens], dtype=torch.long, device=device)

    print(f"Starting prompt: '{prompt}'")
    print("Generated text:")
    print(prompt, end='', flush=True)

    with torch.no_grad():
        for _ in range(max_length):
            # Look at the most recent block of tokens (model.block_size context window)
            context = generated[:, -model.block_size:]
            
            # Forward pass through model to get next-token logits
            logits = model(context)
            
            # Take only the logits for the most recent token and apply temperature scaling
            logits = logits[:, -1, :] / temperature

            # Pick sampling method
            if method == 'greedy':
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            elif method == 'temperature':
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            elif method == 'top_k':
                next_token = sample_top_k(logits, top_k)
            elif method == 'top_p':
                next_token = sample_top_p(logits, top_p)
            else:
                raise ValueError(f"Unknown sampling method: {method}")

            # Append the token
            generated = torch.cat([generated, next_token], dim=1)

            # Print the character
            print(itos[next_token.item()], end='', flush=True)

    print("\n" + "="*50)
    return ''.join([itos[i.item()] for i in generated[0]])