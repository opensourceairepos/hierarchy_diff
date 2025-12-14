"""
Utility Functions

Common utilities for training, visualization, and evaluation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def plot_training_curves(history, save_path=None):
    """Plot training and validation curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Progress', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_layout(layout, saliency=None, save_path=None):
    """Visualize a layout with optional saliency map."""
    fig, axes = plt.subplots(1, 2 if saliency is not None else 1, figsize=(12, 6))
    
    if saliency is not None:
        ax_layout, ax_saliency = axes
    else:
        ax_layout = axes
    
    # Plot layout
    width = layout.get('width', 256)
    height = layout.get('height', 384)
    
    ax_layout.set_xlim(0, width)
    ax_layout.set_ylim(height, 0)
    ax_layout.set_aspect('equal')
    ax_layout.set_title('Layout', fontsize=14)
    
    colors = {'underlay': 'lightblue', 'text': 'lightgreen', 'logo': 'lightcoral'}
    
    for elem in layout.get('elements', []):
        x, y = elem['x'], elem['y']
        w, h = elem['width'], elem['height']
        elem_type = elem.get('type', 'underlay')
        
        rect = plt.Rectangle((x, y), w, h, 
                            facecolor=colors.get(elem_type, 'gray'),
                            edgecolor='black', linewidth=2)
        ax_layout.add_patch(rect)
        
        # Add label
        ax_layout.text(x + w/2, y + h/2, elem_type,
                      ha='center', va='center', fontsize=8)
    
    # Plot saliency
    if saliency is not None:
        im = ax_saliency.imshow(saliency, cmap='hot', aspect='auto')
        ax_saliency.set_title('Saliency Map', fontsize=14)
        plt.colorbar(im, ax=ax_saliency)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def compute_metrics(pred, target):
    """Compute evaluation metrics."""
    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    
    return {'mse': mse, 'mae': mae}
