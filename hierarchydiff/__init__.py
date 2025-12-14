"""
HierarchyDiff: Deep Learning for Multi-Scale Hierarchical Layout Generation

A hierarchical neural architecture for layout generation using diffusion models.
"""

__version__ = "0.1.0"
__author__ = "Anonymous"

from hierarchydiff.models import HierarchicalDiffusion
from hierarchydiff.data import LayoutDataset

__all__ = ['HierarchicalDiffusion', 'LayoutDataset']
