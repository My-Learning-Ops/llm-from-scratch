
""" 
quick_start.py
A simple script to generate text from a trained language model.

It starts generation from a given initial character and produces a sequence of specified
length by repeatedly predicting the next character by choosing the most likely token each step.

Brendan Dileo, July 2025
"""

import torch
from src.models.simple_gpt import SimpleTransformer
from src.data.dataset import CharDataset

def generate(model, start_char, length, stoi, itos, device='cpu'):
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize the input tensor with the index of the starting character
    idx = torch.tensor([[stoi[start_char]]], dtype=torch.long).to(device)
    
    for _ in range(length):
        
        # Slice input tokens to the last block_size tokens
        input_ids = idx[:, -block_size:]
        
        # Forward pass, get the logits for the enitre current sequence
        logits = model(input_ids)
        
        # Pick the next token with the highest probability from last time step
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        
        # Append the predicted token to the input sequence
        # and increase the sequence length by 1
        idx = torch.cat([idx, next_token], dim=1)

    # Convert all indices in the generated sequence back to characters
    output = ''.join([itos[i.item()] for i in idx[0]])
    print("Generated text:")
    print(output)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Use same training text and block size
    text = """
    Once upon a time in a land far away, there lived a clever fox who loved to explore the forest.
    Every morning, the fox would greet the sun and begin a new adventure.
    """
    
    block_size = 64
    embed_dim = 128
    
    dataset = CharDataset(text, block_size)

    # Recreate model
    model = SimpleTransformer(dataset.vocab_size, embed_dim=embed_dim, block_size=block_size)
    model.load_state_dict(torch.load("checkpoints/simple_gpt.pth", map_location=device))
    model.to(device)

    # Generate
    generate(model, start_char='O', length=200, stoi=dataset.stoi, itos=dataset.itos, device=device, block_size=block_size)