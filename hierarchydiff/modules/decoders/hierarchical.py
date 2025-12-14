"""
Decoder Modules for HierarchyDiff

Implements various decoder architectures for layout generation.
"""

import torch
import torch.nn as nn


class HierarchicalDecoder(nn.Module):
    """
    Hierarchical decoder for multi-scale layout decoding.
    
    Args:
        hidden_dims (list): List of hidden dimensions (reversed for decoder)
        output_dim (int): Output dimension
        activation (str): Activation function
    """
    
    def __init__(self, hidden_dims=[32, 64, 128], output_dim=70, activation='relu'):
        super(HierarchicalDecoder, self).__init__()
        
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build decoder layers (reverse order)
        self.layers = nn.ModuleList()
        prev_dim = hidden_dims[0]
        
        for dim in hidden_dims[1:]:
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [B, hidden_dims[0]]
            
        Returns:
            torch.Tensor: Decoded output [B, output_dim]
        """
        h = x
        for layer in self.layers:
            h = self.activation(layer(h))
        return self.output_layer(h)


class ConvDecoder(nn.Module):
    """
    Convolutional decoder for spatial layout generation.
    
    Args:
        latent_dim (int): Latent dimension
        hidden_channels (list): List of hidden channel sizes
        output_size (tuple): Output spatial size (H, W)
    """
    
    def __init__(self, latent_dim=256, hidden_channels=[256, 128, 64], output_size=(384, 256)):
        super(ConvDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_size = output_size
        
        # Initial projection
        init_size = output_size[0] // (2 ** len(hidden_channels))
        self.fc = nn.Linear(latent_dim, hidden_channels[0] * init_size * init_size)
        self.init_size = init_size
        
        # Decoder blocks
        self.layers = nn.ModuleList()
        prev_ch = hidden_channels[0]
        
        for ch in hidden_channels[1:]:
            self.layers.append(nn.Sequential(
                nn.ConvTranspose2d(prev_ch, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU()
            ))
            prev_ch = ch
        
        # Output layer
        self.output = nn.Sequential(
            nn.ConvTranspose2d(prev_ch, prev_ch, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(prev_ch, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        """
        Forward pass.
        
        Args:
            z (torch.Tensor): Latent code [B, latent_dim]
            
        Returns:
            torch.Tensor: Generated layout [B, 3, H, W]
        """
        h = self.fc(z)
        h = h.view(-1, self.hidden_channels[0], self.init_size, self.init_size)
        
        for layer in self.layers:
            h = layer(h)
        
        return self.output(h)


class MLPDecoder(nn.Module):
    """
    MLP decoder for layout parameter generation.
    
    Args:
        input_dim (int): Input dimension
        hidden_dim (int): Hidden dimension
        output_dim (int): Output dimension
        num_layers (int): Number of layers
    """
    
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=70, num_layers=3):
        super(MLPDecoder, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.mlp(x)
