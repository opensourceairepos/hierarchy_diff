"""
Hierarchical Layout Data Generator
===================================

This script generates synthetic layout data with hierarchical structure.
The data mimics professional design layouts with three distinct levels.

ALGORITHM 1: Hierarchical Layout Generation
-------------------------------------------
Input: 
    - N: number of samples to generate
    - canvas_size: (height, width) in pixels
    
Output:
    - Dataset with layouts, saliency maps, and metadata
    
Procedure:
    For each sample i = 1 to N:
        1. Generate Level 0 (Coarse): 1-2 underlay rectangles
           - Size: 30-60% of canvas area
           - Position: Random with 5% margin from edges
           - Purpose: Define global composition and color blocks
           
        2. Generate Level 1 (Medium): 2-4 text rectangles  
           - Size: 15-30% width, 5-15% height
           - Position: Prefer central regions (10-70% from edges)
           - Purpose: Create content structure and reading hierarchy
           
        3. Generate Level 2 (Fine): 1-2 logo rectangles
           - Size: 8-15% (approximately square)
           - Position: Strategic placement avoiding overlaps
           - Purpose: Add brand identity and visual accents
           
        4. Generate saliency map with 1-3 Gaussian peaks
           - Random centers across canvas
           - Radius: 30-80 pixels
           - Smooth with Gaussian filter (sigma=10)
           
        5. Compute relationships between all element pairs
           - Spatial: above, below, left_of, right_of, near
           - Size: larger, smaller, similar
           
        6. Save layout JSON and saliency NPY files
    
    Return complete dataset with train/val/test splits (70/15/15)


ALGORITHM 2: Spatial Relationship Computation
---------------------------------------------
Input:
    - layout: list of N elements with bounding boxes
    
Output:
    - relationships: list of (elem1, elem2, spatial_rel, size_rel)
    
Procedure:
    For each pair (i, j) where i < j:
        1. Calculate element centers:
           center_i = ((x1_i + x2_i)/2, (y1_i + y2_i)/2)
           center_j = ((x1_j + x2_j)/2, (y1_j + y2_j)/2)
           
        2. Determine spatial relationship (threshold = 0.05):
           If center_i.y < center_j.y - 0.05: spatial = "above"
           Else if center_i.y > center_j.y + 0.05: spatial = "below"
           Else if center_i.x < center_j.x - 0.05: spatial = "left_of"
           Else if center_i.x > center_j.x + 0.05: spatial = "right_of"
           Else: spatial = "near"
           
        3. Calculate areas:
           area_i = (x2_i - x1_i) * (y2_i - y1_i)
           area_j = (x2_j - x1_j) * (y2_j - y1_j)
           
        4. Determine size relationship:
           If area_i > 1.2 * area_j: size = "larger"
           Else if area_i < 0.8 * area_j: size = "smaller"
           Else: size = "similar"
           
        5. Store: relationships.append((i, j, spatial, size))
    
    Return relationships
"""

import numpy as np
import json
from pathlib import Path
from scipy.ndimage import gaussian_filter

# Set seed for reproducibility
np.random.seed(42)

class LayoutGenerator:
    """
    Generates hierarchical layout data.
    
    Attributes:
        canvas_height: Canvas height in pixels
        canvas_width: Canvas width in pixels
        num_samples: Total number of layouts to generate
    """
    
    def __init__(self, canvas_size=(256, 384), num_samples=100):
        self.canvas_height = canvas_size[0]
        self.canvas_width = canvas_size[1]
        self.num_samples = num_samples
        
        print("="*60)
        print("HIERARCHICAL LAYOUT DATA GENERATOR")
        print("="*60)
        print(f"Canvas size: {self.canvas_height} x {self.canvas_width}")
        print(f"Total samples: {self.num_samples}")
        print()
        
    def generate_underlay(self):
        """Generate Level 0 - Underlay elements (global structure)"""
        num_elements = np.random.randint(1, 3)
        elements = []
        
        for _ in range(num_elements):
            # Large size: 30-60% width, 20-40% height
            w = np.random.uniform(0.3, 0.6)
            h = np.random.uniform(0.2, 0.4)
            
            # Random position with 5% margin
            x1 = np.random.uniform(0.05, 0.5)
            y1 = np.random.uniform(0.05, 0.5)
            x2 = min(x1 + w, 0.95)
            y2 = min(y1 + h, 0.95)
            
            area = (x2 - x1) * (y2 - y1)
            elements.append({
                'category': 'underlay',
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'level': 0,
                'area': float(area)
            })
        
        return elements
    
    def generate_text(self):
        """Generate Level 1 - Text elements (medium structure)"""
        num_elements = np.random.randint(2, 5)
        elements = []
        
        for _ in range(num_elements):
            # Medium size: 15-30% width, 5-15% height  
            w = np.random.uniform(0.15, 0.3)
            h = np.random.uniform(0.05, 0.15)
            
            # Prefer central regions
            x1 = np.random.uniform(0.1, 0.7)
            y1 = np.random.uniform(0.1, 0.7)
            x2 = min(x1 + w, 0.9)
            y2 = min(y1 + h, 0.9)
            
            area = (x2 - x1) * (y2 - y1)
            elements.append({
                'category': 'text',
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'level': 1,
                'area': float(area)
            })
        
        return elements
    
    def generate_logo(self):
        """Generate Level 2 - Logo elements (fine details)"""
        num_elements = np.random.randint(1, 3)
        elements = []
        
        for _ in range(num_elements):
            # Small, square-ish size: 8-15%
            size = np.random.uniform(0.08, 0.15)
            
            # Strategic placement
            x1 = np.random.uniform(0.1, 0.8)
            y1 = np.random.uniform(0.1, 0.8)
            x2 = min(x1 + size, 0.95)
            y2 = min(y1 + size, 0.95)
            
            area = (x2 - x1) * (y2 - y1)
            elements.append({
                'category': 'logo',
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'level': 2,
                'area': float(area)
            })
        
        return elements
    
    def generate_layout(self):
        """
        Generate complete hierarchical layout.
        Implements ALGORITHM 1 (steps 1-3).
        """
        layout = []
        
        # Level 0: Underlays
        layout.extend(self.generate_underlay())
        
        # Level 1: Text
        layout.extend(self.generate_text())
        
        # Level 2: Logos
        layout.extend(self.generate_logo())
        
        return layout
    
    def compute_relationships(self, layout):
        """
        Compute spatial and size relationships.
        Implements ALGORITHM 2.
        """
        relationships = []
        threshold = 0.05
        
        for i in range(len(layout)):
            for j in range(i + 1, len(layout)):
                bbox_i = layout[i]['bbox']
                bbox_j = layout[j]['bbox']
                
                # Calculate centers
                center_i_x = (bbox_i[0] + bbox_i[2]) / 2
                center_i_y = (bbox_i[1] + bbox_i[3]) / 2
                center_j_x = (bbox_j[0] + bbox_j[2]) / 2
                center_j_y = (bbox_j[1] + bbox_j[3]) / 2
                
                # Spatial relationship
                if center_i_y < center_j_y - threshold:
                    spatial = 'above'
                elif center_i_y > center_j_y + threshold:
                    spatial = 'below'
                elif center_i_x < center_j_x - threshold:
                    spatial = 'left_of'
                elif center_i_x > center_j_x + threshold:
                    spatial = 'right_of'
                else:
                    spatial = 'near'
                
                # Size relationship
                area_i = layout[i]['area']
                area_j = layout[j]['area']
                
                if area_i > 1.2 * area_j:
                    size = 'larger'
                elif area_i < 0.8 * area_j:
                    size = 'smaller'
                else:
                    size = 'similar'
                
                relationships.append({
                    'elem1_idx': i,
                    'elem2_idx': j,
                    'spatial': spatial,
                    'size': size
                })
        
        return relationships
    
    def generate_saliency(self):
        """
        Generate saliency map with Gaussian peaks.
        Implements ALGORITHM 1 (step 4).
        """
        saliency = np.zeros((self.canvas_height, self.canvas_width))
        
        # Add 1-3 salient regions
        num_regions = np.random.randint(1, 4)
        
        for _ in range(num_regions):
            # Random center
            cx = np.random.randint(0, self.canvas_width)
            cy = np.random.randint(0, self.canvas_height)
            
            # Random radius
            radius = np.random.randint(30, 80)
            
            # Create circular region
            y_coords, x_coords = np.ogrid[:self.canvas_height, :self.canvas_width]
            mask = ((x_coords - cx)**2 + (y_coords - cy)**2) <= radius**2
            
            # Add with random intensity
            saliency[mask] = np.random.uniform(0.5, 1.0)
        
        # Smooth with Gaussian filter
        saliency = gaussian_filter(saliency, sigma=10)
        
        # Normalize to [0, 1]
        if saliency.max() > 0:
            saliency = saliency / saliency.max()
        
        return saliency
    
    def generate_dataset(self, output_dir):
        """
        Generate complete dataset with train/val/test splits.
        Implements complete ALGORITHM 1.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            (output_dir / split).mkdir(exist_ok=True)
        
        # Calculate split indices (70% train, 15% val, 15% test)
        train_end = int(0.7 * self.num_samples)
        val_end = int(0.85 * self.num_samples)
        
        print("Generating dataset...")
        print(f"  Train: 0 to {train_end}")
        print(f"  Val: {train_end} to {val_end}")
        print(f"  Test: {val_end} to {self.num_samples}")
        print()
        
        all_data = []
        
        for idx in range(self.num_samples):
            # Generate layout
            layout = self.generate_layout()
            
            # Compute relationships
            relationships = self.compute_relationships(layout)
            
            # Generate saliency
            saliency = self.generate_saliency()
            
            # Determine split
            if idx < train_end:
                split = 'train'
            elif idx < val_end:
                split = 'val'
            else:
                split = 'test'
            
            # Count elements by type
            counts = {
                'underlay': sum(1 for e in layout if e['category'] == 'underlay'),
                'text': sum(1 for e in layout if e['category'] == 'text'),
                'logo': sum(1 for e in layout if e['category'] == 'logo')
            }
            
            # Create data item
            data_item = {
                'id': idx,
                'split': split,
                'layout': layout,
                'relationships': relationships,
                'num_elements': len(layout),
                'counts': counts
            }
            
            # Save files
            layout_file = output_dir / split / f'layout_{idx:05d}.json'
            saliency_file = output_dir / split / f'saliency_{idx:05d}.npy'
            
            with open(layout_file, 'w') as f:
                json.dump(data_item, f, indent=2)
            
            np.save(saliency_file, saliency)
            
            all_data.append(data_item)
            
            if (idx + 1) % 20 == 0:
                print(f"  Generated {idx + 1}/{self.num_samples} samples")
        
        # Calculate statistics
        stats = {
            'total_samples': self.num_samples,
            'canvas_size': [self.canvas_height, self.canvas_width],
            'splits': {
                'train': train_end,
                'val': val_end - train_end,
                'test': self.num_samples - val_end
            },
            'avg_elements_per_layout': np.mean([d['num_elements'] for d in all_data]),
            'avg_underlays': np.mean([d['counts']['underlay'] for d in all_data]),
            'avg_text': np.mean([d['counts']['text'] for d in all_data]),
            'avg_logos': np.mean([d['counts']['logo'] for d in all_data]),
            'total_relationships': sum(len(d['relationships']) for d in all_data)
        }
        
        # Save metadata
        metadata = {
            'dataset_info': 'Hierarchical layout dataset for content-aware generation',
            'generation_algorithm': 'Three-level hierarchical generation (ALGORITHM 1)',
            'statistics': stats
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print()
        print("="*60)
        print("GENERATION COMPLETE")
        print("="*60)
        print(f"Total samples: {self.num_samples}")
        print(f"  Train: {stats['splits']['train']}")
        print(f"  Val: {stats['splits']['val']}")
        print(f"  Test: {stats['splits']['test']}")
        print(f"Average elements per layout: {stats['avg_elements_per_layout']:.2f}")
        print(f"  Underlays: {stats['avg_underlays']:.2f}")
        print(f"  Text: {stats['avg_text']:.2f}")
        print(f"  Logos: {stats['avg_logos']:.2f}")
        print(f"Total relationships computed: {stats['total_relationships']}")
        print(f"\nData saved to: {output_dir}")
        print("="*60)
        print()
        
        return all_data, metadata


if __name__ == '__main__':
    # Initialize generator
    generator = LayoutGenerator(
        canvas_size=(256, 384),
        num_samples=100
    )
    
    # Generate dataset
    data, metadata = generator.generate_dataset('../data/layout_dataset')
    
    print("Dataset generation successful!")
    print(f"Files: ../data/layout_dataset/")
