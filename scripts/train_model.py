
""" 
train_model.py
A minimal script to train a SimpleTransformer language model on a character-level dataset.

Brendan Dileo, July 2025
"""

import os
import torch
from src.models.simple_gpt import SimpleTransformer
from src.data.bpe_tokenizer import BPEDataset
from src.training.trainer import train_improved
from src.utils.text_processing import load_training_text
from src.config.config import MODEL_CONFIG, TRAINING_CONFIG


if __name__ == "__main__":
    
    # Set device to gpu otherwise cpu based on availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load and sanitize the training text data
    text = load_training_text(
        "src/data/training.txt",
        lowercase=True,
        remove_non_ascii=True,
        remove_punctuation=True,
        log_stats=True
    )
    
    
    # Create the dataset from text, split into input-output pairs of length block_size
    dataset = BPEDataset(
        text, 
        MODEL_CONFIG['block_size'],
        stride=128,
        sp_model_path="src/data/bpe_tokenizer.model"
    )
    
    # Instantiate the model with the vocabulary and block size from the dataset
    model = SimpleTransformer(
        dataset.vocab_size, 
        embed_dim=MODEL_CONFIG['embed_dim'],
        block_size=MODEL_CONFIG['block_size'],
        n_heads=MODEL_CONFIG['n_heads'],
        n_layers=MODEL_CONFIG['n_layers'],
        dropout=MODEL_CONFIG['dropout']
    )
    
    checkpoint_path = "checkpoints/best_model.pth"
    
    # Load the model state from the checkpoint if it exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint weights from {checkpoint_path}")

    
    # Count params before training
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params/1e6:.2f}M parameters")
    
    # Print training device
    print(f"Training on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Train the model on the dataset
    train_improved(model, dataset, device=device, **TRAINING_CONFIG)
    
    # Save the trained model weights to a file
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")