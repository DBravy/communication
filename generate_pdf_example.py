"""
Quick script to generate a single PDF example for demonstration.
"""

import torch
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

import config
from model import ARCEncoder, ARCAutoencoder
from dataset import ARCDataset
from visualize_cnn_features import load_model, visualize_cnn_features_comprehensive
import matplotlib.pyplot as plt


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = 'checkpoints/slot_attention.pth'
    model = load_model(checkpoint_path, device)
    encoder = model.encoder
    print("Model loaded!")
    
    # Load a single interesting grid
    dataset_version = config.DATASET_VERSION
    dataset_split = config.DATASET_SPLIT
    data_path = os.path.join(dataset_version, 'data', dataset_split)
    
    dataset = ARCDataset(
        data_path,
        min_size=config.MIN_GRID_SIZE,
        filter_size=config.FILTER_GRID_SIZE,
        max_grids=30,
        num_distractors=0,
        track_puzzle_ids=False,
        use_input_output_pairs=False
    )
    
    # Pick an interesting grid (grid 22 - the 12×12 one)
    grid_idx = 22
    grid_tensor, actual_size = dataset[grid_idx]
    
    print(f"\nGenerating PDF example for grid {grid_idx} (size: {actual_size[0]}×{actual_size[1]})")
    
    # Generate visualization as PDF
    save_path = 'cnn_features_example.pdf'
    fig, stats = visualize_cnn_features_comprehensive(
        encoder, grid_tensor, actual_size, device, save_path
    )
    plt.close(fig)
    
    print(f"\n✓ PDF generated: {save_path}")
    print("\nYou can now open this PDF and zoom in - text will remain crisp!")
    print("PDF advantages:")
    print("  - Vector graphics (infinite zoom)")
    print("  - Better text rendering")
    print("  - Smaller file size for complex figures")


if __name__ == '__main__':
    main()


