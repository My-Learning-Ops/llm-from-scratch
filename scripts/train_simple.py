
""" 
train_simple.py
A minimal script to train a SimpleTransformer language model on a character-level dataset.

Brendan Dileo, July 2025
"""

import torch
from src.models.simple_gpt import SimpleTransformer
from src.data.dataset import CharDataset
from src.training.trainer import train
from src.data.load_text import load_training_text


if __name__ == "__main__":
    
    # Load and sanitize the training text data
    text = load_training_text("src/data/training.txt",
                                lowercase=True,
                                remove_non_ascii=True,
                                remove_punctuation=True,
                                log_stats=True)
    
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