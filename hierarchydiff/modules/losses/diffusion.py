"""
Loss Functions for HierarchyDiff

Implements various loss functions for diffusion model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """
    Standard diffusion loss for noise prediction.
    
    Args:
        loss_type (str): Type of loss ('mse', 'l1', 'huber')
        reduction (str): Reduction method ('mean', 'sum')
    """
    
    def __init__(self, loss_type='mse', reduction='mean'):
        super(DiffusionLoss, self).__init__()
        self.loss_type = loss_type
        self.reduction = reduction
    
    def forward(self, pred_noise, target_noise):
        """
        Compute diffusion loss.
        
        Args:
            pred_noise (torch.Tensor): Predicted noise
            target_noise (torch.Tensor): Target noise
            
        Returns:
            torch.Tensor: Loss value
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(pred_noise, target_noise, reduction=self.reduction)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(pred_noise, target_noise, reduction=self.reduction)
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(pred_noise, target_noise, reduction=self.reduction)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class HierarchicalLoss(nn.Module):
    """
    Hierarchical loss for multi-level supervision.
    
    Args:
        level_weights (list): Weights for each hierarchy level
    """
    
    def __init__(self, level_weights=[1.0, 0.5, 0.25]):
        super(HierarchicalLoss, self).__init__()
        self.level_weights = level_weights
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        """
        Compute hierarchical loss.
        
        Args:
            predictions (list): List of predictions for each level
            targets (list): List of targets for each level
            
        Returns:
            torch.Tensor: Weighted loss
        """
        total_loss = 0.0
        
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            weight = self.level_weights[i] if i < len(self.level_weights) else 1.0
            total_loss += weight * self.mse(pred, target)
        
        return total_loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using feature similarity.
    
    Args:
        feature_dim (int): Dimension of features
    """
    
    def __init__(self, feature_dim=128):
        super(PerceptualLoss, self).__init__()
        self.feature_dim = feature_dim
    
    def forward(self, pred_features, target_features):
        """
        Compute perceptual loss.
        
        Args:
            pred_features (torch.Tensor): Predicted features
            target_features (torch.Tensor): Target features
            
        Returns:
            torch.Tensor: Perceptual loss
        """
        return F.mse_loss(pred_features, target_features)


class CompositeLoss(nn.Module):
    """
    Composite loss combining multiple loss terms.
    
    Args:
        weights (dict): Dictionary of loss weights
    """
    
    def __init__(self, weights={'diffusion': 1.0, 'hierarchical': 0.1}):
        super(CompositeLoss, self).__init__()
        self.weights = weights
        
        self.diffusion_loss = DiffusionLoss()
        self.hierarchical_loss = HierarchicalLoss()
    
    def forward(self, outputs, targets):
        """
        Compute composite loss.
        
        Args:
            outputs (dict): Dictionary of model outputs
            targets (dict): Dictionary of targets
            
        Returns:
            dict: Dictionary of losses
        """
        losses = {}
        total_loss = 0.0
        
        # Diffusion loss
        if 'noise_pred' in outputs and 'noise_target' in targets:
            diff_loss = self.diffusion_loss(outputs['noise_pred'], targets['noise_target'])
            losses['diffusion'] = diff_loss
            total_loss += self.weights.get('diffusion', 1.0) * diff_loss
        
        # Hierarchical loss
        if 'level_preds' in outputs and 'level_targets' in targets:
            hier_loss = self.hierarchical_loss(outputs['level_preds'], targets['level_targets'])
            losses['hierarchical'] = hier_loss
            total_loss += self.weights.get('hierarchical', 0.1) * hier_loss
        
        losses['total'] = total_loss
        return losses


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss with per-element weights.
    
    Args:
        reduction (str): Reduction method
    """
    
    def __init__(self, reduction='mean'):
        super(WeightedMSELoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, target, weights=None):
        """
        Compute weighted MSE loss.
        
        Args:
            pred (torch.Tensor): Predictions
            target (torch.Tensor): Targets
            weights (torch.Tensor, optional): Per-element weights
            
        Returns:
            torch.Tensor: Weighted loss
        """
        loss = (pred - target) ** 2
        
        if weights is not None:
            loss = loss * weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
