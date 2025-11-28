"""
utils.py
---------
Helper functions for visualization, reproducibility, and general utilities
used across training, evaluation, and inference scripts.
"""

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_batch(dl, nrow=8, figsize=(12, 8), save_path=None):
    """
    Display a grid of images from a single batch.

    Args:
        dl (DataLoader): PyTorch DataLoader.
        nrow (int): Number of images per row in the grid.
        figsize (tuple): Figure size.
        save_path (str): Optional path to save the figure.
    """
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=nrow).permute(1, 2, 0))
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close(fig)
        break


def set_seed(seed=42):
    """
    Set random seed for reproducibility across torch, numpy, and random.

    Args:
        seed (int): Seed value.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model (nn.Module): PyTorch model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, path):
    """
    Save model checkpoint with optimizer state.

    Args:
        model (nn.Module): PyTorch model.
        optimizer (Optimizer): PyTorch optimizer.
        epoch (int): Current epoch.
        path (str): File path to save checkpoint.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


def load_checkpoint(model, optimizer, path):
    """
    Load model checkpoint into model and optimizer.

    Args:
        model (nn.Module): PyTorch model.
        optimizer (Optimizer): PyTorch optimizer.
        path (str): File path to checkpoint.

    Returns:
        int: Epoch number from checkpoint.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
