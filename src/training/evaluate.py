
"""
evaluate.py

Brendan Dileo, July 2025"""

import torch
import torch.nn.functional as F

def evaluate(model, val_loader, device):