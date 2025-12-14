"""
Encoder Modules for HierarchyDiff

Implements various encoder architectures for layout encoding.
"""

import torch
import torch.nn as nn


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical encoder for multi-scale layout encoding.
    
    Args:
        input_dim (int): Input dimension
        hidden_dims (list): List of hidden dimensions for each level
        activation (str): Activation function ('relu', 'gelu', 'silu')
    """
    
    def __init__(self, input_dim=71, hidden_dims=[128, 64, 32], activation='relu'):
        super(HierarchicalEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build encoder layers
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        
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
            x (torch.Tensor): Input tensor [B, input_dim]
            
        Returns:
            torch.Tensor: Encoded features [B, hidden_dims[-1]]
        """
        h = x
        for layer in self.layers:
            h = self.activation(layer(h))
        return h


class ConvEncoder(nn.Module):
    """
    Convolutional encoder for spatial layout encoding.
    
    Args:
        in_channels (int): Number of input channels
        hidden_channels (list): List of hidden channel sizes
    """
    
    def __init__(self, in_channels=3, hidden_channels=[64, 128, 256]):
        super(ConvEncoder, self).__init__()
        
        self.layers = nn.ModuleList()
        prev_ch = in_channels
        
        for ch in hidden_channels:
            self.layers.append(nn.Sequential(
                nn.Conv2d(prev_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ))
            prev_ch = ch
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Encoded features
        """
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for sequence-based layout encoding.
    
    Args:
        d_model (int): Model dimension
        nhead (int): Number of attention heads
        num_layers (int): Number of transformer layers
    """
    
    def __init__(self, d_model=128, nhead=8, num_layers=6):
        super(TransformerEncoder, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
    
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input sequence [B, L, D]
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Encoded sequence [B, L, D]
        """
        return self.transformer(x, src_key_padding_mask=mask)
