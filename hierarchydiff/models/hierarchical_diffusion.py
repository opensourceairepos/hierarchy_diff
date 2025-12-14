"""
Hierarchical Diffusion Model

Three-level hierarchical architecture for layout generation:
- Level 1 (Coarse): 128 dimensions, 9,216 parameters
- Level 2 (Medium): 64 dimensions, 8,256 parameters  
- Level 3 (Fine): 32 dimensions, 2,080 parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalDiffusion(nn.Module):
    """
    Hierarchical diffusion model for layout generation.
    
    Args:
        input_dim (int): Input dimension (default: 71)
        hidden_dims (list): Hidden dimensions for each level (default: [128, 64, 32])
        output_dim (int): Output dimension (default: 70)
    """
    
    def __init__(self, input_dim=71, hidden_dims=[128, 64, 32], output_dim=70):
        super(HierarchicalDiffusion, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Level 1: Coarse (128 dims)
        self.level1 = nn.Linear(input_dim, hidden_dims[0])
        
        # Level 2: Medium (64 dims)
        self.level2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        
        # Level 3: Fine (32 dims)
        self.level3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        
        # Output layer
        self.output = nn.Linear(hidden_dims[2], output_dim)
        
        # Activation
        self.activation = nn.ReLU()
        
    def forward(self, x, t=None):
        """
        Forward pass through hierarchical levels.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            t (torch.Tensor, optional): Timestep tensor
            
        Returns:
            torch.Tensor: Predicted noise [batch_size, output_dim]
        """
        # Level 1: Coarse processing
        h1 = self.activation(self.level1(x))
        
        # Level 2: Medium processing
        h2 = self.activation(self.level2(h1))
        
        # Level 3: Fine processing
        h3 = self.activation(self.level3(h2))
        
        # Output: Noise prediction
        out = self.output(h3)
        
        return out
    
    def get_param_count(self):
        """Get parameter count for each level."""
        params = {
            'level1': sum(p.numel() for p in self.level1.parameters()),
            'level2': sum(p.numel() for p in self.level2.parameters()),
            'level3': sum(p.numel() for p in self.level3.parameters()),
            'output': sum(p.numel() for p in self.output.parameters()),
            'total': sum(p.numel() for p in self.parameters())
        }
        return params


class DiffusionProcess:
    """
    Diffusion process for training and sampling.
    
    Implements forward noising and reverse denoising.
    """
    
    def __init__(self, sigma_min=0.1, sigma_max=0.5):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
    def add_noise(self, x, sigma=None):
        """
        Add Gaussian noise to clean data.
        
        Args:
            x (torch.Tensor): Clean data
            sigma (float, optional): Noise level
            
        Returns:
            tuple: (noisy_data, noise, sigma)
        """
        if sigma is None:
            sigma = torch.rand(1).item() * (self.sigma_max - self.sigma_min) + self.sigma_min
        
        noise = torch.randn_like(x)
        noisy = x + sigma * noise
        
        return noisy, noise, sigma
    
    def denoise(self, model, x_noisy, sigma):
        """
        Denoise using the model.
        
        Args:
            model (nn.Module): Trained diffusion model
            x_noisy (torch.Tensor): Noisy data
            sigma (float): Noise level
            
        Returns:
            torch.Tensor: Denoised data
        """
        with torch.no_grad():
            noise_pred = model(x_noisy)
            x_clean = x_noisy - sigma * noise_pred
        
        return x_clean
