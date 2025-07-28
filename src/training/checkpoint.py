
"""
checkpoint.py
Defines checkpoint related utilities for saving and loading model states during training.

Brendan Dileo, July 2025
"""

import torch
import os

def save_checkpoint(model, optimizer, scheduler, step, loss, checkpoint_dir, best=False, epoch=None):
    """ Saves the model, optimizer, and scheduler state to a checkpoint file on disk.

    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        scheduler (object): Scheduler to save.
        step (int): Current training step.
        loss (float): Current loss value.
        checkpoint_dir (str): Directory to save checkpoints.
        best (bool): Flag if this is the best model checkpoint.
        epoch (int, optional): Current epoch number for regular checkpoints.    
    """
    
    # Determine checkpoint filename
    checkpoint_name = "best_model.pth" if best else f"checkpoint_epoch_{epoch}.pth"
    
    # Save the training state dictionary
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.__dict__,
        'loss': loss,
    }, os.path.join(checkpoint_dir, checkpoint_name))