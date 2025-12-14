"""
Evaluation Script for HierarchyDiff

Evaluate trained models on test data.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best/model.pth
"""

import argparse
import os
import yaml
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hierarchydiff.models.hierarchical_diffusion import HierarchicalDiffusion, DiffusionProcess
from hierarchydiff.data.layout_dataset import LayoutDataset, collate_fn
from hierarchydiff.utils.common import compute_metrics, visualize_layout


def evaluate(model, dataloader, diffusion_process, device, output_dir='outputs/eval'):
    """Evaluate model on dataset."""
    model.eval()
    
    all_losses = []
    all_metrics = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            features = batch['features'].to(device)
            
            # Add noise
            noisy_features, noise, sigma = diffusion_process.add_noise(features)
            
            # Predict noise
            noise_pred = model(noisy_features)
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            all_losses.append(loss.item())
            
            # Compute metrics
            metrics = compute_metrics(noise_pred, noise)
            all_metrics.append(metrics)
            
            # Visualize some samples
            if i < 5:  # Save first 5 samples
                layout = batch['layouts'][0]
                saliency = batch['saliency'][0].cpu().numpy()
                
                visualize_layout(
                    layout,
                    saliency,
                    save_path=os.path.join(output_dir, f'sample_{i:03d}.png')
                )
    
    # Aggregate results
    results = {
        'mean_loss': np.mean(all_losses),
        'std_loss': np.std(all_losses),
        'mean_mse': np.mean([m['mse'] for m in all_metrics]),
        'mean_mae': np.mean([m['mae'] for m in all_metrics]),
        'num_samples': len(all_losses)
    }
    
    return results


def main(args):
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    config = checkpoint.get('config', {})
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = HierarchicalDiffusion(
        input_dim=config.get('model', {}).get('input_dim', 71),
        hidden_dims=config.get('model', {}).get('hidden_dims', [128, 64, 32]),
        output_dim=config.get('model', {}).get('output_dim', 70)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create dataset
    test_dataset = LayoutDataset(
        data_dir=args.data_dir,
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create diffusion process
    diffusion_process = DiffusionProcess(
        sigma_min=config.get('diffusion', {}).get('sigma_min', 0.1),
        sigma_max=config.get('diffusion', {}).get('sigma_max', 0.5)
    )
    
    # Evaluate
    print("\nEvaluating on test set...")
    results = evaluate(model, test_loader, diffusion_process, device, args.output_dir)
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print(f"Mean Loss: {results['mean_loss']:.4f} ± {results['std_loss']:.4f}")
    print(f"Mean MSE: {results['mean_mse']:.4f}")
    print(f"Mean MAE: {results['mean_mae']:.4f}")
    print(f"Num Samples: {results['num_samples']}")
    print("="*50)
    
    # Save results
    with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}/eval_results.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate HierarchyDiff')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='datasets/layout_dataset',
                        help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='outputs/eval',
                        help='Output directory')
    args = parser.parse_args()
    
    main(args)
