"""
Visualize Slot Attention Segmentation on ARC Puzzles

This script loads a trained slot attention model and visualizes how it
segments ARC puzzle grids into object-centric representations.

For each grid, it shows:
- The original input grid
- The reconstructed output
- Each slot's attention mask (showing what regions it focuses on)
- Each slot's individual reconstruction
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os
from pathlib import Path

import config
from model import ARCEncoder, ARCAutoencoder
from dataset import ARCDataset


# ARC color palette (10 colors)
ARC_COLORS = [
    '#000000',  # 0: Black
    '#0074D9',  # 1: Blue
    '#FF4136',  # 2: Red
    '#2ECC40',  # 3: Green
    '#FFDC00',  # 4: Yellow
    '#AAAAAA',  # 5: Grey
    '#F012BE',  # 6: Fuchsia
    '#FF851B',  # 7: Orange
    '#7FDBFF',  # 8: Teal
    '#870C25',  # 9: Brown
]


def load_model(checkpoint_path, device, num_slots=None, slot_dim=None, slot_iterations=None):
    """Load the trained slot attention model.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: torch device
        num_slots: Override number of slots (None = use config or checkpoint value)
        slot_dim: Override slot dimension (None = use config or checkpoint value)
        slot_iterations: Override slot iterations (None = use config or checkpoint value)
    """
    
    # Load checkpoint first to check for stored config
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration from checkpoint if available
    ckpt_config = {}
    if isinstance(checkpoint, dict):
        # Look for config keys in checkpoint
        for key in ['num_slots', 'slot_dim', 'slot_iterations', 'slot_hidden_dim', 'slot_eps',
                    'hidden_dim', 'latent_dim', 'num_conv_layers']:
            if key in checkpoint:
                ckpt_config[key] = checkpoint[key]
        
        print(f"\nCheckpoint info:")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Bottleneck type: {checkpoint.get('bottleneck_type', 'unknown')}")
        if ckpt_config:
            print(f"  Config from checkpoint: {ckpt_config}")
    
    # Determine final configuration (priority: function args > checkpoint > config.py)
    final_num_slots = num_slots if num_slots is not None else ckpt_config.get('num_slots', config.NUM_SLOTS)
    final_slot_dim = slot_dim if slot_dim is not None else ckpt_config.get('slot_dim', config.SLOT_DIM)
    final_slot_iterations = slot_iterations if slot_iterations is not None else ckpt_config.get('slot_iterations', config.SLOT_ITERATIONS)
    final_slot_hidden_dim = ckpt_config.get('slot_hidden_dim', config.SLOT_HIDDEN_DIM)
    final_slot_eps = ckpt_config.get('slot_eps', config.SLOT_EPS)
    final_hidden_dim = ckpt_config.get('hidden_dim', config.HIDDEN_DIM)
    final_latent_dim = ckpt_config.get('latent_dim', config.LATENT_DIM)
    final_num_conv_layers = ckpt_config.get('num_conv_layers', config.NUM_CONV_LAYERS)
    
    print(f"\nUsing configuration:")
    print(f"  num_slots: {final_num_slots}")
    print(f"  slot_dim: {final_slot_dim}")
    print(f"  slot_iterations: {final_slot_iterations}")
    print(f"  slot_hidden_dim: {final_slot_hidden_dim}")
    print(f"  hidden_dim: {final_hidden_dim}")
    print(f"  latent_dim: {final_latent_dim}")
    print(f"  num_conv_layers: {final_num_conv_layers}")
    
    # Create encoder with extracted config
    encoder = ARCEncoder(
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=final_hidden_dim,
        latent_dim=final_latent_dim,
        num_conv_layers=final_num_conv_layers,
        use_beta_vae=config.USE_BETA_VAE
    )
    
    # Create model with slot attention
    model = ARCAutoencoder(
        encoder=encoder,
        vocab_size=config.VOCAB_SIZE,
        max_length=config.MAX_MESSAGE_LENGTH,
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=final_hidden_dim,
        max_grid_size=config.MAX_GRID_SIZE,
        bottleneck_type='slot_attention',  # Force slot attention
        task_type='reconstruction',
        num_conv_layers=final_num_conv_layers,
        use_beta_vae=config.USE_BETA_VAE,
        beta=config.BETA_VAE_BETA,
        num_slots=final_num_slots,
        slot_dim=final_slot_dim,
        slot_iterations=final_slot_iterations,
        slot_hidden_dim=final_slot_hidden_dim,
        slot_eps=final_slot_eps
    ).to(device)
    
    # Load state dict
    try:
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"\n✓ Successfully loaded model state dict")
            else:
                model.load_state_dict(checkpoint)
                print(f"\n✓ Successfully loaded checkpoint as state dict")
        else:
            model.load_state_dict(checkpoint)
            print(f"\n✓ Successfully loaded checkpoint")
    except Exception as e:
        print(f"\n⚠️  Error loading checkpoint: {e}")
        print("This might be due to architecture mismatch. Try adjusting num_slots, slot_dim, or other parameters.")
        raise
    
    model.eval()
    
    # Update config module with actual values for use in visualization
    config.NUM_SLOTS = final_num_slots
    config.SLOT_DIM = final_slot_dim
    config.SLOT_ITERATIONS = final_slot_iterations
    config.SLOT_HIDDEN_DIM = final_slot_hidden_dim
    config.HIDDEN_DIM = final_hidden_dim
    config.LATENT_DIM = final_latent_dim
    config.NUM_CONV_LAYERS = final_num_conv_layers
    
    return model


def visualize_grid(ax, grid, actual_size=None, title='', show_grid=True):
    """Visualize an ARC grid with proper colors."""
    if actual_size is not None:
        h, w = actual_size
        grid = grid[:h, :w]
    
    # Create color map
    cmap = ListedColormap(ARC_COLORS)
    
    # Display grid
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')
    
    if show_grid:
        # Add grid lines
        h, w = grid.shape
        for i in range(h + 1):
            ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
        for j in range(w + 1):
            ax.axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.3)


def visualize_attention_mask(ax, mask, actual_size=None, title='', cmap='hot'):
    """Visualize an attention mask as a heatmap."""
    if actual_size is not None:
        h, w = actual_size
        mask = mask[:h, :w]
    
    im = ax.imshow(mask, cmap=cmap, interpolation='bilinear', vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    return im


def visualize_slot_reconstruction(ax, slot_recon, actual_size=None, title=''):
    """Visualize a slot's reconstruction (before combining)."""
    if actual_size is not None:
        h, w = actual_size
        slot_recon = slot_recon[:, :h, :w]
    
    # Convert logits to predicted colors
    pred_grid = slot_recon.argmax(dim=0).cpu().numpy()
    
    # Create color map
    cmap = ListedColormap(ARC_COLORS)
    
    ax.imshow(pred_grid, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def visualize_single_grid(model, grid_tensor, actual_size, device, save_path=None):
    """
    Visualize slot attention decomposition for a single grid.
    
    Args:
        model: Trained slot attention model
        grid_tensor: Input grid [30, 30]
        actual_size: (H, W) actual size of grid
        device: torch device
        save_path: Optional path to save figure
    """
    with torch.no_grad():
        # Prepare input
        grid_batch = grid_tensor.unsqueeze(0).to(device)  # [1, 30, 30]
        sizes = [actual_size]
        
        # Get reconstruction
        logits_list, _, _ = model(grid_batch, sizes)
        recon_logits = logits_list[0].squeeze(0)  # [num_colors, H, W]
        recon_grid = recon_logits.argmax(dim=0).cpu().numpy()
        
        # Get slot visualization
        slot_vis = model.visualize_slots(grid_batch, sizes=sizes)
        slots = slot_vis['slots']  # [1, num_slots, slot_dim]
        slot_recons = slot_vis['slot_reconstructions']  # [1, num_slots, num_colors, H, W]
        masks = slot_vis['masks']  # [1, num_slots, 1, H, W]
        
        num_slots = slots.shape[1]
        
        # Extract data
        slot_recons = slot_recons[0]  # [num_slots, num_colors, H, W]
        masks = masks[0, :, 0, :, :]  # [num_slots, H, W]
        
        # Move to CPU
        masks_np = masks.cpu().numpy()
        
        # Create figure with subplots
        # Layout: 
        # Row 1: Original | Reconstruction | [empty slots]
        # Row 2+: Slot masks and reconstructions
        
        cols = max(4, num_slots)  # At least 4 columns
        rows = 3  # Row 0: original + recon, Rows 1-2: slots
        
        fig = plt.figure(figsize=(cols * 2.5, rows * 2.5))
        gs = fig.add_gridspec(rows, cols, hspace=0.3, wspace=0.3)
        
        # Original grid
        ax_orig = fig.add_subplot(gs[0, 0])
        visualize_grid(ax_orig, grid_tensor.cpu().numpy(), actual_size, 
                      title='Original Input', show_grid=True)
        
        # Reconstruction
        ax_recon = fig.add_subplot(gs[0, 1])
        visualize_grid(ax_recon, recon_grid, actual_size,
                      title='Reconstruction', show_grid=True)
        
        # Compute reconstruction accuracy
        orig_grid = grid_tensor.cpu().numpy()
        h, w = actual_size
        orig_cropped = orig_grid[:h, :w]
        recon_cropped = recon_grid[:h, :w]
        acc = 100 * (orig_cropped == recon_cropped).sum() / (h * w)
        
        # Add accuracy text
        ax_acc = fig.add_subplot(gs[0, 2])
        ax_acc.text(0.5, 0.5, f'Pixel Accuracy:\n{acc:.1f}%', 
                   ha='center', va='center', fontsize=14, fontweight='bold')
        ax_acc.axis('off')
        
        # Info text
        ax_info = fig.add_subplot(gs[0, 3])
        info_text = f'Grid Size: {h}×{w}\n'
        info_text += f'Num Slots: {num_slots}\n'
        info_text += f'Slot Dim: {config.SLOT_DIM}\n'
        info_text += f'Iterations: {config.SLOT_ITERATIONS}'
        ax_info.text(0.1, 0.5, info_text, ha='left', va='center', 
                    fontsize=10, family='monospace')
        ax_info.axis('off')
        
        # Visualize each slot
        # Row 1: Attention masks
        # Row 2: Slot reconstructions
        
        for slot_idx in range(num_slots):
            col_idx = slot_idx % cols
            
            # Attention mask (row 1)
            ax_mask = fig.add_subplot(gs[1, col_idx])
            mask_data = masks_np[slot_idx]
            mask_mean = mask_data.mean()
            im = visualize_attention_mask(ax_mask, mask_data, actual_size,
                                         title=f'Slot {slot_idx+1} Attention\n(avg: {mask_mean:.3f})',
                                         cmap='hot')
            
            # Slot reconstruction (row 2)
            ax_slot_recon = fig.add_subplot(gs[2, col_idx])
            visualize_slot_reconstruction(ax_slot_recon, slot_recons[slot_idx],
                                         actual_size,
                                         title=f'Slot {slot_idx+1} Output')
        
        # Add colorbar for attention masks
        cbar_ax = fig.add_axes([0.92, 0.4, 0.01, 0.15])
        plt.colorbar(im, cax=cbar_ax, label='Attention Weight')
        
        # Main title
        fig.suptitle(f'Slot Attention Visualization\n{num_slots} Slots × {config.SLOT_DIM}D with {config.SLOT_ITERATIONS} Iterations',
                    fontsize=14, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        plt.tight_layout()
        return fig


def main():
    """Main function to visualize slot attention on ARC grids."""
    
    # Configuration
    checkpoint_path = 'checkpoints/slot_attention_32.pth'
    output_dir = 'slot_attention_visualizations'
    num_grids = 10  # Number of grids to visualize
    
    # Optional: Override config parameters here if needed
    # Set to None to auto-detect from checkpoint or use config.py values
    override_num_slots = None  # e.g., 32 if checkpoint has 32 slots
    override_slot_dim = None   # e.g., 64 if checkpoint has 64D slots
    override_slot_iterations = None  # e.g., 3 for 3 iterations
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nCheckpoint: {checkpoint_path}")
    
    # Load model
    if not os.path.exists(checkpoint_path):
        print(f"\nERROR: Checkpoint not found at {checkpoint_path}")
        print("Please ensure the checkpoint file exists.")
        return
    
    try:
        model = load_model(checkpoint_path, device, 
                          num_slots=override_num_slots,
                          slot_dim=override_slot_dim,
                          slot_iterations=override_slot_iterations)
        print(f"\n✓ Model loaded successfully!")
        print(f"✓ Configuration: {config.NUM_SLOTS} slots, {config.SLOT_DIM}D, {config.SLOT_ITERATIONS} iterations")
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        print("\nTroubleshooting:")
        print("1. Check if the checkpoint was saved with different config parameters")
        print("2. Try setting override parameters manually in main() function")
        print("3. Common mismatches: num_slots, slot_dim, hidden_dim, latent_dim")
        return
    
    # Load dataset
    # Use the same dataset configuration as training
    dataset_version = config.DATASET_VERSION
    dataset_split = config.DATASET_SPLIT
    data_path = os.path.join(dataset_version, 'data', dataset_split)
    
    print(f"\nLoading dataset from: {data_path}")
    
    dataset = ARCDataset(
        data_path,
        min_size=config.MIN_GRID_SIZE,
        filter_size=config.FILTER_GRID_SIZE,
        max_grids=num_grids * 2,  # Load more than needed in case some are similar
        num_distractors=0,
        track_puzzle_ids=False,
        use_input_output_pairs=False
    )
    
    print(f"\nGenerating visualizations for {num_grids} grids...")
    print(f"Saving to: {output_dir}/")
    print("=" * 80)
    
    # Visualize a subset of grids
    indices = np.linspace(0, len(dataset) - 1, num_grids, dtype=int)
    
    for i, idx in enumerate(indices, 1):
        grid_tensor, actual_size = dataset[idx]
        
        print(f"\n[{i}/{num_grids}] Visualizing grid {idx} (size: {actual_size[0]}×{actual_size[1]})")
        
        save_path = os.path.join(output_dir, f'grid_{idx:04d}.png')
        fig = visualize_single_grid(model, grid_tensor, actual_size, device, save_path)
        plt.close(fig)
    
    print("\n" + "=" * 80)
    print(f"✓ Visualization complete!")
    print(f"✓ Generated {num_grids} visualizations in: {output_dir}/")
    print("=" * 80)
    
    # Create a summary figure with multiple examples in one image
    print("\nCreating summary visualization with multiple examples...")
    create_summary_visualization(model, dataset, device, output_dir, num_examples=6)


def create_summary_visualization(model, dataset, device, output_dir, num_examples=6):
    """Create a summary figure showing multiple examples side-by-side."""
    
    # Select diverse examples (spread across dataset)
    indices = np.linspace(0, len(dataset) - 1, num_examples, dtype=int)
    
    fig, axes = plt.subplots(num_examples, 2 + config.NUM_SLOTS, 
                            figsize=(2.5 * (2 + config.NUM_SLOTS), 2.5 * num_examples))
    
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, dataset_idx in enumerate(indices):
        grid_tensor, actual_size = dataset[dataset_idx]
        
        with torch.no_grad():
            # Prepare input
            grid_batch = grid_tensor.unsqueeze(0).to(device)
            sizes = [actual_size]
            
            # Get reconstruction and slots
            logits_list, _, _ = model(grid_batch, sizes)
            recon_logits = logits_list[0].squeeze(0)
            recon_grid = recon_logits.argmax(dim=0).cpu().numpy()
            
            slot_vis = model.visualize_slots(grid_batch, sizes=sizes)
            masks = slot_vis['masks'][0, :, 0, :, :]  # [num_slots, H, W]
            masks_np = masks.cpu().numpy()
            
            # Original
            ax = axes[row_idx, 0]
            visualize_grid(ax, grid_tensor.cpu().numpy(), actual_size, 
                          title=f'Input {dataset_idx}', show_grid=False)
            
            # Reconstruction
            ax = axes[row_idx, 1]
            visualize_grid(ax, recon_grid, actual_size,
                          title='Recon', show_grid=False)
            
            # Slot masks
            for slot_idx in range(config.NUM_SLOTS):
                ax = axes[row_idx, 2 + slot_idx]
                visualize_attention_mask(ax, masks_np[slot_idx], actual_size,
                                        title=f'S{slot_idx+1}' if row_idx == 0 else '',
                                        cmap='hot')
    
    fig.suptitle(f'Slot Attention Overview: {num_examples} Examples\n'
                f'{config.NUM_SLOTS} Slots × {config.SLOT_DIM}D with {config.SLOT_ITERATIONS} Iterations',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, 'summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved summary visualization to: {summary_path}")


if __name__ == '__main__':
    main()

