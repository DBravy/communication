"""
Visualize CNN Feature Extraction for Slot Attention

This script visualizes the intermediate CNN features to diagnose whether
the encoder is extracting meaningful spatial features for slot attention.

For each grid, it shows:
- Original input
- Feature maps after each convolutional layer
- Feature statistics (mean, std, entropy)
- Spatial feature diversity analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from scipy.stats import entropy

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
    """Load the trained model.
    
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
    
    # Create model
    model = ARCAutoencoder(
        encoder=encoder,
        vocab_size=config.VOCAB_SIZE,
        max_length=config.MAX_MESSAGE_LENGTH,
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=final_hidden_dim,
        max_grid_size=config.MAX_GRID_SIZE,
        bottleneck_type='slot_attention',
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


def extract_cnn_features(encoder, grid_tensor, actual_size, device):
    """
    Extract CNN features at each layer.
    
    Returns:
        Dictionary with:
        - 'input': one-hot encoded input [C, H, W]
        - 'conv1', 'conv2', etc.: feature maps [C, H, W]
        - 'spatial_features': final spatial features [C, H, W]
        - 'pooled': pooled features [C, 4, 4]
    """
    encoder.eval()
    with torch.no_grad():
        grid_batch = grid_tensor.unsqueeze(0).to(device)
        
        # Get intermediate features
        features = encoder.extract_feature_maps(grid_batch, sizes=[actual_size])
        
        # Get spatial features (before pooling)
        spatial_features = encoder.get_spatial_features(grid_batch, sizes=[actual_size])
        features['spatial_features'] = spatial_features[0]  # Remove batch dim
        
        # Move to CPU
        for key in features:
            if isinstance(features[key], torch.Tensor):
                features[key] = features[key].cpu()
        
        return features


def compute_feature_statistics(feature_map, actual_size=None):
    """
    Compute statistics about a feature map.
    
    Args:
        feature_map: [C, H, W] or [B, C, H, W] feature tensor
        actual_size: Optional (h, w) to focus on actual content
        
    Returns:
        Dictionary of statistics
    """
    # Handle batch dimension if present
    if feature_map.dim() == 4:
        feature_map = feature_map[0]  # Remove batch dimension
    
    if actual_size is not None:
        h, w = actual_size
        feature_map = feature_map[:, :h, :w]
    
    C, H, W = feature_map.shape
    
    # Flatten spatial dimensions
    features_flat = feature_map.reshape(C, -1)  # [C, H*W]
    
    # Statistics per channel
    channel_means = features_flat.mean(dim=1).numpy()
    channel_stds = features_flat.std(dim=1).numpy()
    channel_maxs = features_flat.max(dim=1)[0].numpy()
    channel_mins = features_flat.min(dim=1)[0].numpy()
    
    # Statistics per spatial location
    spatial_flat = feature_map.reshape(C, -1).T  # [H*W, C]
    spatial_means = spatial_flat.mean(dim=1).numpy()
    spatial_stds = spatial_flat.std(dim=1).numpy()
    
    # Spatial diversity: how different are features across locations?
    # Compute pairwise cosine similarities between spatial locations
    spatial_norm = F.normalize(spatial_flat, dim=1)
    similarity_matrix = torch.mm(spatial_norm, spatial_norm.T)  # [H*W, H*W]
    avg_similarity = similarity_matrix.mean().item()
    
    # Feature activation sparsity
    activation_sparsity = (features_flat.abs() < 0.01).float().mean().item()
    
    # Information content: entropy of normalized activations
    # Normalize to probabilities per channel
    channel_entropies = []
    for c in range(C):
        activations = features_flat[c].numpy()
        # Normalize to positive values and make it a distribution
        activations = activations - activations.min()
        if activations.sum() > 0:
            probs = activations / activations.sum()
            # Add small epsilon to avoid log(0)
            probs = probs + 1e-10
            probs = probs / probs.sum()
            ent = entropy(probs)
            channel_entropies.append(ent)
        else:
            channel_entropies.append(0)
    
    return {
        'num_channels': C,
        'spatial_size': (H, W),
        'channel_mean_avg': channel_means.mean(),
        'channel_mean_std': channel_means.std(),
        'channel_std_avg': channel_stds.mean(),
        'activation_range': (channel_mins.min(), channel_maxs.max()),
        'spatial_diversity': 1 - avg_similarity,  # Higher = more diverse
        'activation_sparsity': activation_sparsity,
        'avg_channel_entropy': np.mean(channel_entropies),
        'channel_means': channel_means,
        'channel_stds': channel_stds,
    }


def visualize_grid(ax, grid, actual_size=None, title='', show_grid=True):
    """Visualize an ARC grid."""
    if actual_size is not None:
        h, w = actual_size
        grid = grid[:h, :w]
    
    cmap = ListedColormap(ARC_COLORS)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')
    
    if show_grid:
        h, w = grid.shape
        for i in range(h + 1):
            ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
        for j in range(w + 1):
            ax.axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.3)


def visualize_feature_map(ax, feature_map, actual_size=None, title='', channel_idx=None):
    """
    Visualize a feature map.
    
    Args:
        ax: matplotlib axis
        feature_map: [C, H, W], [B, C, H, W], or [H, W] tensor
        actual_size: Optional (h, w) to crop
        title: Title for plot
        channel_idx: If None, show average across channels. Otherwise show specific channel.
    """
    # Handle batch dimension if present
    if feature_map.dim() == 4:
        feature_map = feature_map[0]
    
    if feature_map.dim() == 3:
        if channel_idx is not None:
            # Show specific channel
            vis_map = feature_map[channel_idx].numpy()
        else:
            # Average across channels
            vis_map = feature_map.mean(dim=0).numpy()
    else:
        vis_map = feature_map.numpy()
    
    if actual_size is not None:
        h, w = actual_size
        vis_map = vis_map[:h, :w]
    
    im = ax.imshow(vis_map, cmap='viridis', interpolation='bilinear')
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    return im


def visualize_feature_channels(ax, feature_map, actual_size=None, num_channels=16):
    """
    Visualize multiple channels of a feature map in a grid.
    
    Args:
        ax: matplotlib axis
        feature_map: [C, H, W] or [B, C, H, W] tensor
        actual_size: Optional (h, w) to crop
        num_channels: Number of channels to display
    """
    # Handle batch dimension if present
    if feature_map.dim() == 4:
        feature_map = feature_map[0]
    
    C, H, W = feature_map.shape
    
    if actual_size is not None:
        h, w = actual_size
        feature_map = feature_map[:, :h, :w]
        H, W = h, w
    
    # Select channels to display (evenly spaced)
    num_display = min(num_channels, C)
    channel_indices = np.linspace(0, C - 1, num_display, dtype=int)
    
    # Create a grid of channel visualizations
    grid_size = int(np.ceil(np.sqrt(num_display)))
    
    # Create composite image
    composite = np.zeros((grid_size * H, grid_size * W))
    
    for idx, c_idx in enumerate(channel_indices):
        row = idx // grid_size
        col = idx % grid_size
        
        channel_data = feature_map[c_idx].numpy()
        # Normalize each channel independently
        c_min, c_max = channel_data.min(), channel_data.max()
        if c_max > c_min:
            channel_data = (channel_data - c_min) / (c_max - c_min)
        
        composite[row*H:(row+1)*H, col*W:(col+1)*W] = channel_data
    
    im = ax.imshow(composite, cmap='viridis', interpolation='nearest')
    ax.set_title(f'Feature Channels (showing {num_display}/{C})', fontsize=9)
    ax.axis('off')
    
    # Add grid lines between channels
    for i in range(1, grid_size):
        ax.axhline(i * H - 0.5, color='white', linewidth=1)
        ax.axvline(i * W - 0.5, color='white', linewidth=1)
    
    return im


def visualize_cnn_features_comprehensive(encoder, grid_tensor, actual_size, device, save_path=None):
    """
    Create a comprehensive visualization of CNN features.
    """
    # Extract features
    features = extract_cnn_features(encoder, grid_tensor, actual_size, device)
    
    # Compute statistics for each layer
    stats = {}
    for key in ['embed', 'conv1', 'conv2', 'conv3', 'spatial_features']:
        if key in features:
            stats[key] = compute_feature_statistics(features[key], actual_size)
    
    # Create figure with better spacing
    num_conv_layers = sum(1 for k in features if k.startswith('conv'))
    
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(5, 5, hspace=0.5, wspace=0.4, 
                         left=0.05, right=0.95, top=0.93, bottom=0.05)
    
    # Row 0: Original input and embedding
    ax_orig = fig.add_subplot(gs[0, 0])
    visualize_grid(ax_orig, grid_tensor.cpu().numpy(), actual_size, 
                  title='Original Input', show_grid=True)
    
    ax_embed = fig.add_subplot(gs[0, 1])
    if 'embed' in features:
        visualize_feature_map(ax_embed, features['embed'], actual_size,
                            title=f'One-Hot Embedding\n{stats["embed"]["num_channels"]} channels')
    
    # Statistics box - now in row 0, taking full width
    ax_stats = fig.add_subplot(gs[0, 2:5])
    ax_stats.axis('off')
    
    h, w = actual_size
    stats_text = f'Grid Size: {h}×{w}  |  '
    stats_text += f'CNN: {config.NUM_CONV_LAYERS} conv layers  |  '
    stats_text += f'Hidden Dim: {config.HIDDEN_DIM}\n\n'
    stats_text += 'Layer Statistics:\n'
    stats_text += '-' * 70 + '\n'
    
    for key in ['embed', 'conv1', 'conv2', 'conv3', 'spatial_features']:
        if key in stats:
            s = stats[key]
            stats_text += f'\n{key.upper()}: '
            stats_text += f'Ch={s["num_channels"]}, Size={s["spatial_size"]}, '
            stats_text += f'Mean={s["channel_mean_avg"]:.3f}±{s["channel_mean_std"]:.3f}, '
            stats_text += f'Range=[{s["activation_range"][0]:.2f},{s["activation_range"][1]:.2f}]\n'
            stats_text += f'  Diversity={s["spatial_diversity"]:.4f}, '
            stats_text += f'Sparsity={s["activation_sparsity"]:.3f}, '
            stats_text += f'Entropy={s["avg_channel_entropy"]:.2f}\n'
    
    ax_stats.text(0.02, 0.98, stats_text, fontsize=8, family='monospace',
                 verticalalignment='top', transform=ax_stats.transAxes)
    
    # Row 1-2: Conv layer visualizations
    layer_names = ['conv1', 'conv2', 'conv3']
    for i, layer_name in enumerate(layer_names):
        if layer_name not in features:
            continue
        
        # Average activation
        ax_avg = fig.add_subplot(gs[1, i])
        visualize_feature_map(ax_avg, features[layer_name], actual_size,
                            title=f'{layer_name.upper()} Average\n{features[layer_name].shape[0]} channels')
        
        # Individual channels grid
        ax_channels = fig.add_subplot(gs[2, i])
        visualize_feature_channels(ax_channels, features[layer_name], actual_size, num_channels=16)
    
    # Row 1-2, col 3-4: Spatial features (what goes into slot attention)
    if 'spatial_features' in features:
        ax_spatial_avg = fig.add_subplot(gs[1, 3:5])
        visualize_feature_map(ax_spatial_avg, features['spatial_features'], actual_size,
                            title=f'Spatial Features (avg)\n→ Input to Slot Attention')
        
        ax_spatial_channels = fig.add_subplot(gs[2, 3:5])
        visualize_feature_channels(ax_spatial_channels, features['spatial_features'], 
                                  actual_size, num_channels=16)
    
    # Row 4: Channel activation distributions
    ax_dist = fig.add_subplot(gs[4, :3])
    
    if 'spatial_features' in stats:
        s = stats['spatial_features']
        x = np.arange(len(s['channel_means']))
        
        ax_dist.bar(x, s['channel_means'], alpha=0.7, label='Mean activation', color='steelblue')
        ax_dist.errorbar(x, s['channel_means'], yerr=s['channel_stds'], 
                        fmt='none', ecolor='red', alpha=0.5, capsize=2)
        ax_dist.set_xlabel('Channel Index', fontsize=9)
        ax_dist.set_ylabel('Activation', fontsize=9)
        ax_dist.set_title('Spatial Feature Channel Activations (mean ± std)', fontsize=10)
        ax_dist.legend(fontsize=8)
        ax_dist.grid(True, alpha=0.3)
        ax_dist.tick_params(labelsize=8)
    
    # Row 4, col 3-5: Spatial diversity heatmap
    ax_diversity = fig.add_subplot(gs[4, 3:])
    
    if 'spatial_features' in features:
        # Compute spatial correlation matrix
        feat = features['spatial_features']
        if actual_size is not None:
            h, w = actual_size
            feat = feat[:, :h, :w]
        
        C, H, W = feat.shape
        spatial_flat = feat.reshape(C, -1).T  # [H*W, C]
        
        # Sample some spatial locations for visualization
        num_samples = min(50, H * W)
        sample_indices = np.linspace(0, H * W - 1, num_samples, dtype=int)
        sampled = spatial_flat[sample_indices]
        
        # Compute correlation
        sampled_norm = F.normalize(sampled, dim=1)
        corr = torch.mm(sampled_norm, sampled_norm.T).numpy()
        
        im = ax_diversity.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax_diversity.set_title(f'Spatial Location Similarity\n(sample of {num_samples} locations)', fontsize=10)
        ax_diversity.set_xlabel('Spatial Location', fontsize=9)
        ax_diversity.set_ylabel('Spatial Location', fontsize=9)
        plt.colorbar(im, ax=ax_diversity, label='Cosine Similarity')
    
    # Add diagnosis summary box in row 3, col 3-5
    ax_diagnosis = fig.add_subplot(gs[3, 3:])
    ax_diagnosis.axis('off')
    
    if 'spatial_features' in stats:
        s = stats['spatial_features']
        diag_text = 'DIAGNOSIS SUMMARY:\n'
        diag_text += '=' * 50 + '\n'
        
        issues = []
        if s['spatial_diversity'] < 0.1:
            issues.append('⚠️  LOW DIVERSITY: Features too similar')
        elif s['spatial_diversity'] > 0.5:
            issues.append('✅ GOOD DIVERSITY')
        
        if s['activation_sparsity'] > 0.5:
            issues.append('⚠️  HIGH SPARSITY: Many dead neurons')
        else:
            issues.append('✅ GOOD SPARSITY')
        
        if s['avg_channel_entropy'] < 2.0:
            issues.append('⚠️  LOW ENTROPY: Low information')
        else:
            issues.append('✅ GOOD ENTROPY')
        
        for issue in issues:
            diag_text += issue + '\n'
        
        ax_diagnosis.text(0.05, 0.95, diag_text, fontsize=9, family='monospace',
                         verticalalignment='top', transform=ax_diagnosis.transAxes,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Main title
    fig.suptitle(f'CNN Feature Analysis: Grid {h}×{w}',
                fontsize=16, fontweight='bold')
    
    if save_path:
        # Support both PNG and PDF output
        if save_path.endswith('.pdf'):
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
        else:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    return fig, stats


def analyze_feature_quality(encoder, dataset, device, num_samples=20):
    """
    Analyze feature quality across multiple grids to identify systematic issues.
    """
    print("\n" + "="*80)
    print("FEATURE QUALITY ANALYSIS")
    print("="*80)
    
    all_stats = []
    
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    
    for idx in indices:
        grid_tensor, actual_size = dataset[idx]
        features = extract_cnn_features(encoder, grid_tensor, actual_size, device)
        
        if 'spatial_features' in features:
            stats = compute_feature_statistics(features['spatial_features'], actual_size)
            all_stats.append(stats)
    
    # Aggregate statistics
    print(f"\nAnalyzing {len(all_stats)} grids...")
    print("\nAGGREGATE STATISTICS:")
    print("-" * 80)
    
    spatial_diversities = [s['spatial_diversity'] for s in all_stats]
    activation_sparsities = [s['activation_sparsity'] for s in all_stats]
    mean_activations = [s['channel_mean_avg'] for s in all_stats]
    entropies = [s['avg_channel_entropy'] for s in all_stats]
    
    print(f"Spatial Diversity: {np.mean(spatial_diversities):.4f} ± {np.std(spatial_diversities):.4f}")
    print(f"  → Higher is better (0 = all locations identical, 1 = maximally diverse)")
    print(f"  → Range: [{np.min(spatial_diversities):.4f}, {np.max(spatial_diversities):.4f}]")
    print()
    
    print(f"Activation Sparsity: {np.mean(activation_sparsities):.4f} ± {np.std(activation_sparsities):.4f}")
    print(f"  → Lower is better (0 = all active, 1 = all near zero)")
    print(f"  → Range: [{np.min(activation_sparsities):.4f}, {np.max(activation_sparsities):.4f}]")
    print()
    
    print(f"Mean Activation: {np.mean(mean_activations):.4f} ± {np.std(mean_activations):.4f}")
    print(f"  → Should be non-zero and varied")
    print(f"  → Range: [{np.min(mean_activations):.4f}, {np.max(mean_activations):.4f}]")
    print()
    
    print(f"Average Channel Entropy: {np.mean(entropies):.2f} ± {np.std(entropies):.2f}")
    print(f"  → Higher is better (more information content)")
    print(f"  → Range: [{np.min(entropies):.2f}, {np.max(entropies):.2f}]")
    print()
    
    # Diagnosis
    print("DIAGNOSIS:")
    print("-" * 80)
    
    issues = []
    
    if np.mean(spatial_diversities) < 0.1:
        issues.append("⚠️  LOW SPATIAL DIVERSITY: Features are too similar across locations")
        issues.append("   → Slot attention needs diverse features to learn object separation")
        issues.append("   → Consider: increasing CNN depth, using dilated convolutions, or skip connections")
    
    if np.mean(activation_sparsities) > 0.5:
        issues.append("⚠️  HIGH ACTIVATION SPARSITY: Many neurons are inactive")
        issues.append("   → May indicate dying ReLU problem or poor initialization")
        issues.append("   → Consider: using LeakyReLU, adjusting learning rate, or batch norm")
    
    if np.mean(mean_activations) < 0.01:
        issues.append("⚠️  LOW MEAN ACTIVATION: Features have very small magnitudes")
        issues.append("   → Network may not be learning effectively")
        issues.append("   → Consider: checking learning rate, gradient flow, or initialization")
    
    if np.mean(entropies) < 2.0:
        issues.append("⚠️  LOW ENTROPY: Features have low information content")
        issues.append("   → Features may be collapsing to similar patterns")
        issues.append("   → Consider: regularization, data augmentation, or architecture changes")
    
    if not issues:
        print("✓ No major issues detected!")
        print("  Features appear to have reasonable diversity and activation levels.")
    else:
        for issue in issues:
            print(issue)
    
    print("\n" + "="*80)
    
    return {
        'spatial_diversity': (np.mean(spatial_diversities), np.std(spatial_diversities)),
        'activation_sparsity': (np.mean(activation_sparsities), np.std(activation_sparsities)),
        'mean_activation': (np.mean(mean_activations), np.std(mean_activations)),
        'entropy': (np.mean(entropies), np.std(entropies)),
    }


def main():
    """Main function to visualize CNN features."""
    
    # Configuration
    checkpoint_path = 'checkpoints/slot_attention_32.pth'
    output_dir = 'cnn_feature_visualizations'
    num_grids = 10
    output_format = 'pdf'  # Options: 'png' or 'pdf'
    
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
        encoder = model.encoder
        print(f"\n✓ Model loaded successfully!")
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        print("\nTroubleshooting:")
        print("1. Check if the checkpoint was saved with different config parameters")
        print("2. Try setting override parameters manually in main() function")
        print("3. Common mismatches: num_slots, slot_dim, hidden_dim, latent_dim")
        return
    
    # Load dataset
    dataset_version = config.DATASET_VERSION
    dataset_split = config.DATASET_SPLIT
    data_path = os.path.join(dataset_version, 'data', dataset_split)
    
    print(f"\nLoading dataset from: {data_path}")
    
    dataset = ARCDataset(
        data_path,
        min_size=config.MIN_GRID_SIZE,
        filter_size=config.FILTER_GRID_SIZE,
        max_grids=num_grids * 3,
        num_distractors=0,
        track_puzzle_ids=False,
        use_input_output_pairs=False
    )
    
    # First, run aggregate analysis
    analyze_feature_quality(encoder, dataset, device, num_samples=20)
    
    # Generate visualizations
    print(f"\nGenerating detailed visualizations for {num_grids} grids...")
    print(f"Saving to: {output_dir}/")
    print("=" * 80)
    
    indices = np.linspace(0, len(dataset) - 1, num_grids, dtype=int)
    
    for i, idx in enumerate(indices, 1):
        grid_tensor, actual_size = dataset[idx]
        
        print(f"\n[{i}/{num_grids}] Visualizing grid {idx} (size: {actual_size[0]}×{actual_size[1]})")
        
        save_path = os.path.join(output_dir, f'cnn_features_grid_{idx:04d}.{output_format}')
        fig, stats = visualize_cnn_features_comprehensive(encoder, grid_tensor, actual_size, 
                                                         device, save_path)
        plt.close(fig)
    
    print("\n" + "=" * 80)
    print(f"✓ Visualization complete!")
    print(f"✓ Generated {num_grids} visualizations in: {output_dir}/")
    print("=" * 80)


if __name__ == '__main__':
    main()

