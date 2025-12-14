"""
Diffusion Model Components

Noise schedulers and diffusion processes.
"""

import torch
import torch.nn as nn
import numpy as np


class NoiseScheduler:
    """
    Noise scheduler for diffusion process.
    
    Args:
        schedule_type (str): Type of schedule ('linear', 'cosine', 'quadratic')
        num_steps (int): Number of diffusion steps
        beta_start (float): Starting beta value
        beta_end (float): Ending beta value
    """
    
    def __init__(self, schedule_type='linear', num_steps=1000, beta_start=0.0001, beta_end=0.02):
        self.schedule_type = schedule_type
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Compute schedule
        if schedule_type == 'linear':
            self.betas = np.linspace(beta_start, beta_end, num_steps)
        elif schedule_type == 'cosine':
            self.betas = self._cosine_schedule()
        elif schedule_type == 'quadratic':
            self.betas = np.linspace(beta_start**0.5, beta_end**0.5, num_steps) ** 2
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Compute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])
        
        # Compute other parameters
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = np.sqrt(1.0 / self.alphas)
    
    def _cosine_schedule(self, s=0.008):
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = self.num_steps + 1
        x = np.linspace(0, self.num_steps, steps)
        alphas_cumprod = np.cos(((x / self.num_steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, x0, t, noise=None):
        """
        Add noise to data at timestep t.
        
        Args:
            x0 (torch.Tensor): Original data
            t (torch.Tensor): Timesteps
            noise (torch.Tensor, optional): Noise to add
            
        Returns:
            torch.Tensor: Noisy data
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_prod = torch.from_numpy(self.sqrt_alphas_cumprod[t]).float().to(x0.device)
        sqrt_one_minus_alpha_prod = torch.from_numpy(self.sqrt_one_minus_alphas_cumprod[t]).float().to(x0.device)
        
        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, *([1] * (len(x0.shape) - 1)))
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, *([1] * (len(x0.shape) - 1)))
        
        return sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise


class DDPMSampler:
    """
    DDPM sampling algorithm.
    
    Args:
        noise_scheduler (NoiseScheduler): Noise scheduler
    """
    
    def __init__(self, noise_scheduler):
        self.scheduler = noise_scheduler
    
    @torch.no_grad()
    def sample(self, model, shape, device='cuda', num_steps=None):
        """
        Sample from the model.
        
        Args:
            model (nn.Module): Trained diffusion model
            shape (tuple): Shape of samples to generate
            device (str): Device to use
            num_steps (int, optional): Number of sampling steps
            
        Returns:
            torch.Tensor: Generated samples
        """
        if num_steps is None:
            num_steps = self.scheduler.num_steps
        
        # Start from random noise
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion
        for t in reversed(range(num_steps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            pred_noise = model(x, t_batch)
            
            # Compute mean
            alpha = self.scheduler.alphas[t]
            alpha_cumprod = self.scheduler.alphas_cumprod[t]
            alpha_cumprod_prev = self.scheduler.alphas_cumprod_prev[t]
            
            beta = self.scheduler.betas[t]
            sqrt_one_minus_alpha_cumprod = self.scheduler.sqrt_one_minus_alphas_cumprod[t]
            sqrt_recip_alpha = self.scheduler.sqrt_recip_alphas[t]
            
            # Compute x_{t-1}
            pred_x0 = sqrt_recip_alpha * (x - beta / sqrt_one_minus_alpha_cumprod * pred_noise)
            
            if t > 0:
                noise = torch.randn_like(x)
                variance = (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * beta
                x = pred_x0 * np.sqrt(alpha_cumprod_prev) + np.sqrt(variance) * noise + \
                    np.sqrt(1 - alpha_cumprod_prev - variance) * pred_noise
            else:
                x = pred_x0
        
        return x


class DDIMSampler:
    """
    DDIM sampling algorithm (deterministic).
    
    Args:
        noise_scheduler (NoiseScheduler): Noise scheduler
        eta (float): Stochasticity parameter (0 for deterministic)
    """
    
    def __init__(self, noise_scheduler, eta=0.0):
        self.scheduler = noise_scheduler
        self.eta = eta
    
    @torch.no_grad()
    def sample(self, model, shape, device='cuda', num_steps=50):
        """
        Sample using DDIM.
        
        Args:
            model (nn.Module): Trained diffusion model
            shape (tuple): Shape of samples
            device (str): Device
            num_steps (int): Number of sampling steps
            
        Returns:
            torch.Tensor: Generated samples
        """
        # Create sampling schedule
        step_size = self.scheduler.num_steps // num_steps
        timesteps = list(range(0, self.scheduler.num_steps, step_size))
        timesteps.reverse()
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            pred_noise = model(x, t_batch)
            
            # Get alpha values
            alpha_cumprod = self.scheduler.alphas_cumprod[t]
            
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_cumprod_prev = self.scheduler.alphas_cumprod[t_prev]
            else:
                alpha_cumprod_prev = 1.0
            
            # Compute x0 prediction
            pred_x0 = (x - np.sqrt(1 - alpha_cumprod) * pred_noise) / np.sqrt(alpha_cumprod)
            
            # Compute direction pointing to x_t
            dir_xt = np.sqrt(1 - alpha_cumprod_prev - self.eta ** 2 * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * (1 - alpha_cumprod / alpha_cumprod_prev)) * pred_noise
            
            # Compute x_{t-1}
            x = np.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt
            
            if self.eta > 0:
                noise = torch.randn_like(x)
                variance = self.eta * np.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod)) * np.sqrt(1 - alpha_cumprod / alpha_cumprod_prev)
                x = x + variance * noise
        
        return x
