
""" 
config.py
Defines the configuration for the model and training parameters.

Brendan Dileo, July 2025
"""

# Defines the model hyperparameters
MODEL_CONFIG = {
    'vocab_size': 50000,
    'embed_dim': 512,
    'block_size': 512,
    'n_heads': 8,
    'n_layers': 8,
    'dropout': 0.1
}

# Training related hyperparameters for the model
TRAINING_CONFIG = {
    'epochs': 10,
    'batch_size': 16,
    'lr': 3e-4,
    'warmup_steps': 4000,
    'max_lr': 3e-4,
    'min_lr': 1e-6,
    'grad_clip': 1.0,
    'eval_interval': 1000,
    'weight_decay': 0.01
}