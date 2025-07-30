
""" 
trainer.py
Defines the training loop for the simple transformer like language model.

Brendan Dileo, July 2025
"""


import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm

from .evaluate import evaluate
from .checkpoint import save_checkpoint
from .scheduler import CosineAnnealingLRWithWarmup

def train(model, dataset, epochs=10, batch_size=32, lr=1e-3, device='cpu'):
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
        

def train_improved(model, dataset, epochs=10, batch_size=64, lr=3e-4,
                   device='cpu', warmup_steps=1000, max_lr=3e-4, min_lr=1e-5,
                   grad_clip=1.0, eval_interval=500, checkpoint_dir="checkpoints",
                   weight_decay=0.01):
    """ An enhanced training loop with warmup, cosine annealing learning rate scheduling, gradient clipping, validation, and 
    model checkpointing.

    Args:
        model (torch.nn.Module): Model to train.
        dataset (torch.utils.data.Dataset): Dataset of (input, target) pairs.
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size for training.
        lr (float): Base learning rate (used initially by optimizer).
        device (str): Device for training ('cpu' or 'cuda').
        warmup_steps (int): Number of steps for linear learning rate warmup.
        max_lr (float): Peak learning rate after warmup.
        min_lr (float): Minimum learning rate at end of cosine annealing.
        grad_clip (float): Maximum gradient norm for clipping.
        eval_interval (int): Number of steps between validation & checkpoint.
        checkpoint_dir (str): Directory path to save checkpoints.
    """
    
    # Ensure checkpoint directory exists and move model to target device
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)

    # Split dataset into 90% training and 10% validation subsets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoader for training with shuffling and multiple workers for optimized data loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Create DataLoader for validation without shuffling
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize AdamW optimizer with weight decay and specified hyperparameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    
    # Calculate total training steps across all epochs (used for scheduler)
    total_steps = epochs * len(train_loader)
    
    # Create custom cosine annealing LR scheduler with warmup
    scheduler = CosineAnnealingLRWithWarmup(optimizer, warmup_steps, total_steps, max_lr, min_lr)

    # Global step counter and best validation loss tracker for checkpointing
    step = 0
    best_val_loss = float('inf')

    print(f"Training for {epochs} epochs ({total_steps} steps)")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    for epoch in range(epochs):
        # Set model to training mode and accumulate epoch loss
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for x, y in pbar:
            # Move batch data to the device
            x, y = x.to(device), y.to(device)

            # Forward pass: compute predicted logits from model
            logits = model(x)
            B, T, C = logits.shape
            
            # Compute cross-entropy loss between logits and targets
            loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))

            # Zero previous gradients before backward pass
            optimizer.zero_grad()
            
            # Backpropagate gradients from loss
            loss.backward()

            # Clip gradients to prevent exploding gradients
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Take optimization step to update model parameters
            optimizer.step()
            
            # Update learning rate scheduler
            scheduler.step()

            # Increment global step and accumulate batch loss
            step += 1
            epoch_loss += loss.item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}', 'step': step})
            
            # Run validation and checkpointing every eval_interval steps
            if step % eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"\nStep {step}: Train Loss {loss.item():.4f} | Val Loss {val_loss:.4f}")

                # Save checkpoint if this validation loss is the best so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, scheduler, step, val_loss, checkpoint_dir, best=True)
                    print(f"New best model saved! (Val Loss: {val_loss:.4f})")

                # Return model to training mode after validation
                model.train()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} done. Avg Loss: {avg_epoch_loss:.4f}")
        
        # Save regular checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, scheduler, step, avg_epoch_loss, checkpoint_dir, epoch=epoch+1)

    # Save final model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'final_model.pth'))
    print("Training completed! Final model saved.")