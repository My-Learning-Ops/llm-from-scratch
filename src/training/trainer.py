
""" 
trainer.py
Defines the training loop for the simple transformer like language model.

Brendan Dileo, July 2025
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model, dataset, epochs=5, batch_size=32, lr=1e-3, device='cpu'):
    """ Trains the given model on the provided dataset.

    Args:
        model (torch.nn.Module): The neural network model to train.
        dataset (torch.utils.data.Dataset): Dataset providing (x, y) pairs.
        epochs (int): Number of full passes over the dataset.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the optimizer.
        device (str or torch.device): Device to run training on ('cpu' or 'cuda').
    """

    # Move model parameters to the specified device
    model.to(device)
    
    # DataLoader wraps the dataset and provides batches of data
    # 'shuffle=True' ensures batches are sampled randomly each epoch
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # AdamW optimizer is used for transformer training
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # CrossEntropyLoss computes softmax + negatuve log likelihood loss
    # for classification tasks like predicitng next token
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Loop over specified number of epochs
    for epoch in range(epochs):
        # Puts model in training mode
        model.train()
        
        # Accumulate loss over batches
        total_loss = 0.0
        
        # tqdm wraps the loader to show progress bar during training iteration
        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}"):
            # Move input and target batches to the specified device
            x, y = x.to(device), y.to(device)
            
            # Forward pass to compute model predictions (logits)
            # logits shape: (B, T, vocab_size)
            logits = model(x)
            
            # Unpack batch size, sequence length, and vocab size
            B, T, C = logits.shape
            
            # Reshape logits and targets to 2D tensors for loss calculation
            # CrossEntropyLoss expects input of shape (N, C) and target of shape (N,)
            loss = loss_fn(logits.view(B * T, C), y.view(B * T))
            
            # Zero gradients before backward pass
            optimizer.zero_grad()
            
            # Compute gradients via backpropagation
            loss.backward()
            
            # Update model parameters using optimizer
            optimizer.step()
            
            # Add batch loss to total
            total_loss += loss.item()
    
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
