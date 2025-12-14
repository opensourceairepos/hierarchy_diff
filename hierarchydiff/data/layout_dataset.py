"""
Layout Dataset

Dataset class for hierarchical layout generation.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class LayoutDataset(Dataset):
    """
    Dataset for hierarchical layouts.
    
    Each layout contains:
    - Elements: position (x, y), size (w, h), type
    - Saliency map: highlighting important regions
    - Hierarchy levels: underlay, text, logos
    
    Args:
        data_dir (str): Directory containing layout data
        split (str): 'train', 'val', or 'test'
        transform (callable, optional): Transform to apply
    """
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load metadata
        metadata_path = os.path.join(data_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Get file list
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            self.layout_files = sorted([f for f in os.listdir(split_dir) if f.startswith('layout_') and f.endswith('.json')])
        else:
            self.layout_files = []
    
    def __len__(self):
        return len(self.layout_files)
    
    def __getitem__(self, idx):
        # Load layout
        layout_file = self.layout_files[idx]
        layout_path = os.path.join(self.data_dir, self.split, layout_file)
        
        with open(layout_path, 'r') as f:
            layout_data = json.load(f)
        
        # Load saliency
        saliency_file = layout_file.replace('layout_', 'saliency_').replace('.json', '.npy')
        saliency_path = os.path.join(self.data_dir, self.split, saliency_file)
        
        if os.path.exists(saliency_path):
            saliency = np.load(saliency_path)
        else:
            saliency = np.zeros((256, 384))
        
        # Convert to feature vector
        features = self._layout_to_features(layout_data, saliency)
        
        # Apply transform
        if self.transform:
            features = self.transform(features)
        
        return {
            'features': torch.FloatTensor(features),
            'layout': layout_data,
            'saliency': torch.FloatTensor(saliency)
        }
    
    def _layout_to_features(self, layout, saliency):
        """
        Convert layout to feature vector.
        
        Args:
            layout (dict): Layout data
            saliency (np.ndarray): Saliency map
            
        Returns:
            np.ndarray: Feature vector [71 dims]
        """
        elements = layout.get('elements', [])
        
        # Pad or truncate to 5 elements
        num_elements = 5
        features = []
        
        for i in range(num_elements):
            if i < len(elements):
                elem = elements[i]
                # [x, y, w, h, type_0, type_1, type_2] = 7 dims per element
                x = elem.get('x', 0.0) / layout.get('width', 256.0)
                y = elem.get('y', 0.0) / layout.get('height', 384.0)
                w = elem.get('width', 0.0) / layout.get('width', 256.0)
                h = elem.get('height', 0.0) / layout.get('height', 384.0)
                
                # One-hot encode type
                elem_type = elem.get('type', 'underlay')
                type_vec = [0.0, 0.0, 0.0]
                if elem_type == 'underlay':
                    type_vec[0] = 1.0
                elif elem_type == 'text':
                    type_vec[1] = 1.0
                elif elem_type == 'logo':
                    type_vec[2] = 1.0
                
                features.extend([x, y, w, h] + type_vec)
            else:
                # Padding
                features.extend([0.0] * 7)
        
        # Add saliency mean (1 dim)
        saliency_mean = np.mean(saliency)
        features.append(saliency_mean)
        
        return np.array(features, dtype=np.float32)


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    features = torch.stack([item['features'] for item in batch])
    layouts = [item['layout'] for item in batch]
    saliency = torch.stack([item['saliency'] for item in batch])
    
    return {
        'features': features,
        'layouts': layouts,
        'saliency': saliency
    }
