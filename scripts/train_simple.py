
""" 
train_simple.py
A minimal script to train a SimpleTransformer language model on a character-level dataset.

Brendan Dileo, July 2025
"""

import torch
from src.models.simple_gpt import SimpleTransformer
from src.data.dataset import CharDataset
from src.training.trainer import train

text = """
Once upon a time in a land far away, there lived a clever fox who loved to explore the forest.
Every morning, the fox would greet the sun and begin a new adventure.
"""

if __name__ == "__main__":
    # Number of tokens per training example
    block_size = 64
    
    # Create the dataset from text, split into input-output pairs of length block_size
    dataset = CharDataset(text, block_size)
    
    # Instantiate the model with the vocabulary and block size from the dataset
    model = SimpleTransformer(dataset.vocab_size, block_size=block_size)
    
    # Train the model on the dataset
    train(model, dataset)
    
    # Save the trained model weights to a file
    torch.save(model.state_dict(), "checkpoints/simple_gpt.pth")
    print("Model saved to checkpoints/simple_gpt.pth")