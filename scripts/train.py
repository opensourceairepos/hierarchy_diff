"""
Training Script for HierarchyDiff

Train hierarchical diffusion model for layout generation.

Usage:
    python scripts/train.py --config configs/hierarchydiff_base.yaml
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hierarchydiff.models.hierarchical_diffusion import HierarchicalDiffusion, DiffusionProcess
from hierarchydiff.data.layout_dataset import LayoutDataset, collate_fn


def train_epoch(model, dataloader, optimizer, diffusion_process, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        features = batch['features'].to(device)
        
        # Add noise
        noisy_features, noise, sigma = diffusion_process.add_noise(features)
        
        # Predict noise
        noise_pred = model(noisy_features)
        
        # Compute loss
        loss = nn.MSELoss()(noise_pred, noise)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, diffusion_process, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            
            # Add noise
            noisy_features, noise, sigma = diffusion_process.add_noise(features)
            
            # Predict noise
            noise_pred = model(noisy_features)
            
            # Compute loss
            loss = nn.MSELoss()(noise_pred, noise)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = HierarchicalDiffusion(
        input_dim=config['model']['input_dim'],
        hidden_dims=config['model']['hidden_dims'],
        output_dim=config['model']['output_dim']
    ).to(device)
    
    # Print parameter count
    params = model.get_param_count()
    print("\nParameter Count:")
    for key, value in params.items():
        print(f"  {key}: {value:,}")
    
    # Create datasets
    train_dataset = LayoutDataset(
        data_dir=config['dataset']['data_dir'],
        split='train'
    )
    
    val_dataset = LayoutDataset(
        data_dir=config['dataset']['data_dir'],
        split='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create optimizer
    if config['training']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate']
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate']
        )
    
    # Create diffusion process
    diffusion_process = DiffusionProcess(
        sigma_min=config['diffusion']['sigma_min'],
        sigma_max=config['diffusion']['sigma_max']
    )
    
    # Training loop
    best_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0
    }
    
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    
    for epoch in range(config['training']['epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, diffusion_process, device)
        
        # Validate
        val_loss = validate(model, val_loader, diffusion_process, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']}: "
              f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            history['best_epoch'] = epoch + 1
            
            checkpoint_dir = os.path.join(config['paths']['checkpoint_dir'], 'best')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config
            }, os.path.join(checkpoint_dir, 'model.pth'))
    
    # Save final results
    results = {
        'initial_loss': history['train_loss'][0],
        'final_loss': history['train_loss'][-1],
        'best_loss': min(history['train_loss']),
        'best_epoch': history['train_loss'].index(min(history['train_loss'])) + 1,
        'loss_reduction': (history['train_loss'][0] - history['train_loss'][-1]) / history['train_loss'][0] * 100,
        'history': history,
        'parameters': params
    }
    
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Initial Loss: {results['initial_loss']:.4f}")
    print(f"Final Loss: {results['final_loss']:.4f}")
    print(f"Best Loss: {results['best_loss']:.4f} (Epoch {results['best_epoch']})")
    print(f"Loss Reduction: {results['loss_reduction']:.1f}%")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HierarchyDiff')
    parser.add_argument('--config', type=str, default='configs/hierarchydiff_base.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    main(args)
