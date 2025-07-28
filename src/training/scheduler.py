
"""
scheduler.py
Implements a customl earning rate scheduler that combines linear warmup with cosine annealing decay.

Cosine Annealing is a technique for gradually decreasing the learning rate during training by following
a half-cosine curve from a maximum learning rate (max_lr) down to a minimum learning rate (min_lr). This
schedule helps improve convergence and final model performance by allowing large updates early on and
smaller, fine-tuned updates later.

This scheduler first increases the learning rate linearly from zero to max_lr during an initial warmup
phase (warmup_steps), which helps stabilize training and prevents sudden large updates at the start.
After warmup, the learning rate smoothly decreases following a cosine curve until reaching min_lr at the
end of training (total_steps).

Brendan Dileo, July 2025
"""

import math

class CosineAnnealingLRWithWarmup:
    """Custom LR scheduler with warmup and cosine decay."""

    def __init__(self, optimizer, warmup_steps, total_steps, max_lr, min_lr):
        """

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to adjust LR for.
            warmup_steps (int): Number of steps for linear LR warmup.
            total_steps (int): Total number of training steps.
            max_lr (float): Peak learning rate after warmup.
            min_lr (float): Minimum learning rate at end of training.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_count = 0

    def step(self):
        """ Updates the learning rate based on the current step. """
        
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            
            # Linear increase during warmup
            lr = self.max_lr * self.step_count / self.warmup_steps
        else:
            # Cosine decay after warmup
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        # Set learning rate for all parameter groups in optimizer
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def get_last_lr(self):
        """ Get the last computed learning rate. """
        return [group['lr'] for group in self.optimizer.param_groups]