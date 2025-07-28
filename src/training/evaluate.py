
"""
evaluate.py
Defines the validation loop for the model.

Brendan Dileo, July 2025
"""

import torch
import torch.nn.functional as F

def evaluate(model, val_loader, device):
    """ Evaluates the model on validation data and computes average loss. 

    Args:
        model (torch.nn.Module): Model to evaluate.
        val_loader (DataLoader): DataLoader for validation dataset.
        device (str): Device to run evaluation on ('cpu' or 'cuda').

    Returns:
        float: Average cross-entropy loss on validation set.
    """
    
    # Evaluation mode for model
    model.eval()
    
    # No gradient tracking needed during evaluation
    total_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))
            total_loss += loss.item()
    return total_loss / len(val_loader)