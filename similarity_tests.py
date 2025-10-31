"""Similarity tests for ARC encoder - to be integrated into train.py

Add these functions to train.py and call run_similarity_test() at the same rate
as generalization tests.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def apply_transformation(grid, transform_type):
    """Apply a geometric or color transformation to a grid.
    
    Args:
        grid: [H, W] grid tensor
        transform_type: one of:
            Geometric: ['rot90', 'rot180', 'rot270', 'fliph', 'flipv']
            Color: ['color_swap', 'color_shift', 'color_invert', 'color_remap']
    
    Returns:
        Transformed grid (and new_size if size changes, otherwise None)
    """
    # Geometric transformations
    if transform_type == 'rot90':
        return torch.rot90(grid, k=1, dims=(0, 1))
    elif transform_type == 'rot180':
        return torch.rot90(grid, k=2, dims=(0, 1))
    elif transform_type == 'rot270':
        return torch.rot90(grid, k=3, dims=(0, 1))
    elif transform_type == 'fliph':
        return torch.flip(grid, dims=[1])  # horizontal flip
    elif transform_type == 'flipv':
        return torch.flip(grid, dims=[0])  # vertical flip
    
    # Color transformations
    elif transform_type == 'color_swap':
        # Swap two random colors
        unique_colors = torch.unique(grid)
        if len(unique_colors) >= 2:
            # Pick two different colors
            colors = unique_colors[torch.randperm(len(unique_colors))[:2]]
            c1, c2 = colors[0].item(), colors[1].item()
            
            # Create a copy and swap
            transformed = grid.clone()
            mask1 = grid == c1
            mask2 = grid == c2
            transformed[mask1] = c2
            transformed[mask2] = c1
            return transformed
        return grid  # Not enough colors to swap
    
    elif transform_type == 'color_shift':
        # Shift all colors by a random amount (modulo 10)
        shift = np.random.randint(1, 10)
        return (grid + shift) % 10
    
    elif transform_type == 'color_invert':
        # Invert colors: color -> (9 - color)
        return 9 - grid
    
    elif transform_type == 'color_remap':
        # Randomly remap all colors consistently
        # Create a random permutation of colors 0-9
        perm = torch.randperm(10)
        
        # Apply permutation
        transformed = grid.clone()
        for old_color in range(10):
            mask = grid == old_color
            transformed[mask] = perm[old_color]
        return transformed
    
    else:
        raise ValueError(f"Unknown transform: {transform_type}")


def create_similar_pairs(dataset, num_pairs=50, transform_types=['rot90', 'fliph', 'color_swap'], device='cpu'):
    """Create pairs of (original, transformed) grids for similarity testing.
    
    Args:
        dataset: Dataset to sample from
        num_pairs: Number of pairs to create
        transform_types: List of transformations to apply (geometric and/or color)
        device: Device to move tensors to
    
    Returns:
        List of (grid1, size1, grid2, size2, transform_name, transform_category) tuples
    """
    pairs = []
    
    # Categorize transforms
    geometric_transforms = {'rot90', 'rot180', 'rot270', 'fliph', 'flipv'}
    color_transforms = {'color_swap', 'color_shift', 'color_invert', 'color_remap'}
    
    # Sample random grids
    indices = np.random.choice(len(dataset), size=min(num_pairs, len(dataset)), replace=False)
    
    for idx in indices:
        # Get original grid
        if hasattr(dataset, 'dataset'):  # Handle Subset wrapper
            item = dataset.dataset[dataset.indices[idx]]
        else:
            item = dataset[idx]
        
        # Unpack based on dataset type
        if len(item) == 2:  # (grid, size)
            original_grid, original_size = item
        elif len(item) >= 4:  # input-output pairs
            original_grid, original_size = item[0], item[1]
        else:
            continue
        
        # Apply random transformation
        transform = np.random.choice(transform_types)
        transformed_grid = apply_transformation(original_grid, transform)
        
        # Determine category
        if transform in geometric_transforms:
            category = 'geometric'
        elif transform in color_transforms:
            category = 'color'
        else:
            category = 'unknown'
        
        # Size may change for rotations only
        if transform in ['rot90', 'rot270']:
            transformed_size = (original_size[1], original_size[0])  # swap h, w
        else:
            transformed_size = original_size
        
        pairs.append((
            original_grid.to(device),
            original_size,
            transformed_grid.to(device),
            transformed_size,
            transform,
            category
        ))
    
    return pairs


def calculate_encoding_similarity(encoder, grid1, size1, grid2, size2, metric='cosine'):
    """Calculate similarity between encodings of two grids.
    
    Args:
        encoder: The encoder model
        grid1, grid2: Grid tensors
        size1, size2: Actual sizes
        metric: 'cosine' or 'euclidean'
    
    Returns:
        Similarity score (higher = more similar for cosine, lower = more similar for euclidean)
    """
    with torch.no_grad():
        # Encode both grids
        enc1 = encoder(grid1.unsqueeze(0), sizes=[size1])
        enc2 = encoder(grid2.unsqueeze(0), sizes=[size2])
        
        if metric == 'cosine':
            # Cosine similarity: ranges from -1 to 1, higher is more similar
            similarity = F.cosine_similarity(enc1, enc2, dim=1).item()
        elif metric == 'euclidean':
            # Euclidean distance: lower is more similar
            similarity = torch.norm(enc1 - enc2, p=2, dim=1).item()
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    return similarity


def create_random_dissimilar_pairs(dataset, num_pairs=50, device='cpu'):
    """Create pairs of random (dissimilar) grids for baseline comparison.
    
    Args:
        dataset: Dataset to sample from
        num_pairs: Number of pairs to create
        device: Device to move tensors to
    
    Returns:
        List of (grid1, size1, grid2, size2) tuples
    """
    pairs = []
    
    # Sample random pairs
    indices = np.random.choice(len(dataset), size=min(num_pairs * 2, len(dataset)), replace=False)
    
    for i in range(0, len(indices) - 1, 2):
        # Get two different grids
        if hasattr(dataset, 'dataset'):  # Handle Subset wrapper
            item1 = dataset.dataset[dataset.indices[indices[i]]]
            item2 = dataset.dataset[dataset.indices[indices[i+1]]]
        else:
            item1 = dataset[indices[i]]
            item2 = dataset[indices[i+1]]
        
        # Unpack grids
        if len(item1) == 2:
            grid1, size1 = item1
        elif len(item1) >= 4:
            grid1, size1 = item1[0], item1[1]
        else:
            continue
            
        if len(item2) == 2:
            grid2, size2 = item2
        elif len(item2) >= 4:
            grid2, size2 = item2[0], item2[1]
        else:
            continue
        
        pairs.append((
            grid1.to(device),
            size1,
            grid2.to(device),
            size2
        ))
    
    return pairs


def run_similarity_test(model, device, dataset=None, num_similar_pairs=50, num_dissimilar_pairs=50):
    """Run similarity test to check if similar grids have similar encodings.
    
    This test:
    1. Creates pairs of similar grids (via geometric and color transformations)
    2. Creates pairs of dissimilar grids (random pairs)
    3. Compares encoding similarities
    4. Reports whether similar grids are encoded more similarly than random pairs
    
    Args:
        model: The model containing the encoder
        device: Device to run on
        dataset: Dataset to sample from (if None, skip test)
        num_similar_pairs: Number of similar pairs to test
        num_dissimilar_pairs: Number of dissimilar pairs to test
    
    Returns:
        Dictionary with test results
    """
    # Check if similarity testing is enabled
    if not getattr(config, 'SIMILARITY_TEST_ENABLED', False):
        return None
    
    if dataset is None:
        print("Warning: No dataset provided for similarity test")
        return None
    
    print(f'\n{"="*80}')
    print(f'SIMILARITY TEST: Testing encoding consistency')
    print(f'{"="*80}')
    
    model.eval()
    encoder = model.encoder
    
    # Define transformations to test (both geometric and color)
    transform_types = ['rot90', 'rot180', 'rot270', 'fliph', 'flipv',
                      'color_swap', 'color_shift', 'color_invert', 'color_remap']
    
    try:
        # Create similar pairs (transformed versions)
        print(f'Creating {num_similar_pairs} similar pairs (geometric + color transformations)...')
        similar_pairs = create_similar_pairs(
            dataset, 
            num_pairs=num_similar_pairs,
            transform_types=transform_types,
            device=device
        )
        
        # Create dissimilar pairs (random)
        print(f'Creating {num_dissimilar_pairs} dissimilar pairs (random)...')
        dissimilar_pairs = create_random_dissimilar_pairs(
            dataset,
            num_pairs=num_dissimilar_pairs,
            device=device
        )
        
        # Calculate similarities for similar pairs
        print('Calculating encoding similarities for similar pairs...')
        similar_cosine_sims = []
        similar_euclidean_dists = []
        transform_stats = {t: [] for t in transform_types}
        geometric_stats = []
        color_stats = []
        
        for grid1, size1, grid2, size2, transform, category in tqdm(similar_pairs):
            cosine_sim = calculate_encoding_similarity(
                encoder, grid1, size1, grid2, size2, metric='cosine'
            )
            euclidean_dist = calculate_encoding_similarity(
                encoder, grid1, size1, grid2, size2, metric='euclidean'
            )
            
            similar_cosine_sims.append(cosine_sim)
            similar_euclidean_dists.append(euclidean_dist)
            transform_stats[transform].append(cosine_sim)
            
            # Track by category
            if category == 'geometric':
                geometric_stats.append(cosine_sim)
            elif category == 'color':
                color_stats.append(cosine_sim)
        
        # Calculate similarities for dissimilar pairs
        print('Calculating encoding similarities for dissimilar pairs...')
        dissimilar_cosine_sims = []
        dissimilar_euclidean_dists = []
        
        for grid1, size1, grid2, size2 in tqdm(dissimilar_pairs):
            cosine_sim = calculate_encoding_similarity(
                encoder, grid1, size1, grid2, size2, metric='cosine'
            )
            euclidean_dist = calculate_encoding_similarity(
                encoder, grid1, size1, grid2, size2, metric='euclidean'
            )
            
            dissimilar_cosine_sims.append(cosine_sim)
            dissimilar_euclidean_dists.append(euclidean_dist)
        
        # Calculate statistics
        similar_cosine_mean = np.mean(similar_cosine_sims)
        similar_cosine_std = np.std(similar_cosine_sims)
        dissimilar_cosine_mean = np.mean(dissimilar_cosine_sims)
        dissimilar_cosine_std = np.std(dissimilar_cosine_sims)
        
        similar_euclidean_mean = np.mean(similar_euclidean_dists)
        similar_euclidean_std = np.std(similar_euclidean_dists)
        dissimilar_euclidean_mean = np.mean(dissimilar_euclidean_dists)
        dissimilar_euclidean_std = np.std(dissimilar_euclidean_dists)
        
        # Calculate per-transform statistics
        transform_means = {t: np.mean(sims) if sims else 0.0 
                          for t, sims in transform_stats.items()}
        
        # Calculate category statistics
        geometric_mean = np.mean(geometric_stats) if geometric_stats else 0.0
        geometric_std = np.std(geometric_stats) if geometric_stats else 0.0
        color_mean = np.mean(color_stats) if color_stats else 0.0
        color_std = np.std(color_stats) if color_stats else 0.0
        
        # Report results
        print(f'\nSimilarity Test Results:')
        print(f'  Cosine Similarity (higher = more similar):')
        print(f'    Similar pairs (all): {similar_cosine_mean:.4f} ± {similar_cosine_std:.4f}')
        print(f'    Dissimilar pairs:    {dissimilar_cosine_mean:.4f} ± {dissimilar_cosine_std:.4f}')
        print(f'    Difference:          {similar_cosine_mean - dissimilar_cosine_mean:.4f}')
        
        print(f'\n  By Category:')
        print(f'    Geometric transforms: {geometric_mean:.4f} ± {geometric_std:.4f}')
        print(f'    Color transforms:     {color_mean:.4f} ± {color_std:.4f}')
        
        print(f'\n  Euclidean Distance (lower = more similar):')
        print(f'    Similar pairs (all): {similar_euclidean_mean:.4f} ± {similar_euclidean_std:.4f}')
        print(f'    Dissimilar pairs:    {dissimilar_euclidean_mean:.4f} ± {dissimilar_euclidean_std:.4f}')
        print(f'    Difference:          {dissimilar_euclidean_mean - similar_euclidean_mean:.4f}')
        
        print(f'\n  Per-Transform Cosine Similarity:')
        print(f'    Geometric:')
        for transform in ['rot90', 'rot180', 'rot270', 'fliph', 'flipv']:
            if transform in transform_means:
                print(f'      {transform:12s}: {transform_means[transform]:.4f}')
        print(f'    Color:')
        for transform in ['color_swap', 'color_shift', 'color_invert', 'color_remap']:
            if transform in transform_means:
                print(f'      {transform:12s}: {transform_means[transform]:.4f}')
        
        # Determine if test passed
        # Good encoders should have higher cosine similarity for similar pairs
        # and lower euclidean distance for similar pairs
        cosine_passed = similar_cosine_mean > dissimilar_cosine_mean
        euclidean_passed = similar_euclidean_mean < dissimilar_euclidean_mean
        
        test_passed = cosine_passed and euclidean_passed
        
        if test_passed:
            print(f'\n  ✓ PASSED: Similar grids have more similar encodings!')
        else:
            print(f'\n  ✗ FAILED: Similar grids do not have consistently more similar encodings')
            if not cosine_passed:
                print(f'    - Cosine similarity not higher for similar pairs')
            if not euclidean_passed:
                print(f'    - Euclidean distance not lower for similar pairs')
        
        print(f'{"="*80}\n')
        
        # Prepare results dictionary
        results = {
            'num_similar_pairs': len(similar_pairs),
            'num_dissimilar_pairs': len(dissimilar_pairs),
            'cosine_similarity': {
                'similar_mean': float(similar_cosine_mean),
                'similar_std': float(similar_cosine_std),
                'dissimilar_mean': float(dissimilar_cosine_mean),
                'dissimilar_std': float(dissimilar_cosine_std),
                'difference': float(similar_cosine_mean - dissimilar_cosine_mean),
                'geometric_mean': float(geometric_mean),
                'geometric_std': float(geometric_std),
                'color_mean': float(color_mean),
                'color_std': float(color_std)
            },
            'euclidean_distance': {
                'similar_mean': float(similar_euclidean_mean),
                'similar_std': float(similar_euclidean_std),
                'dissimilar_mean': float(dissimilar_euclidean_mean),
                'dissimilar_std': float(dissimilar_euclidean_std),
                'difference': float(dissimilar_euclidean_mean - similar_euclidean_mean)
            },
            'per_transform_cosine': {t: float(m) for t, m in transform_means.items()},
            'test_passed': test_passed,
            'timestamp': time.time()
        }
        
        return results
        
    except Exception as e:
        print(f'\nError during similarity test: {e}')
        import traceback
        traceback.print_exc()
        return None


# Integration code - add to main training loop in train.py:
"""
# Add this at the top of main() with other tracking variables:
similarity_history = []
similarity_results_path = os.path.join(config.SAVE_DIR, 'similarity_test_results.json')

# Add this in the training loop, right after the generalization test (around line 1600):
# Run similarity test at same rate as generalization test
sim_interval = getattr(config, 'SIMILARITY_TEST_INTERVAL', gen_interval)  # Same as gen by default
if getattr(config, 'SIMILARITY_TEST_ENABLED', False) and (epoch + 1) % sim_interval == 0:
    # Use validation dataset for similarity testing
    sim_results = run_similarity_test(
        model, 
        device, 
        dataset=val_dataset,
        num_similar_pairs=getattr(config, 'SIMILARITY_TEST_NUM_PAIRS', 50),
        num_dissimilar_pairs=getattr(config, 'SIMILARITY_TEST_NUM_PAIRS', 50)
    )
    if sim_results is not None:
        # Add epoch info
        sim_results['epoch'] = epoch + 1
        sim_results['train_loss'] = train_loss
        sim_results['train_acc'] = train_acc
        sim_results['val_loss'] = val_loss
        sim_results['val_acc'] = val_acc
        
        # Add to history
        similarity_history.append(sim_results)
        
        # Save results to JSON
        with open(similarity_results_path, 'w') as f:
            json.dump({
                'training_dataset': config.DATASET_VERSION,
                'training_split': config.DATASET_SPLIT,
                'task_type': task_type,
                'bottleneck_type': config.BOTTLENECK_TYPE,
                'history': similarity_history
            }, f, indent=2)
        print(f'✓ Saved similarity test results to {similarity_results_path}')
"""

# Configuration options to add to your config file:
"""
# Similarity Testing
SIMILARITY_TEST_ENABLED = True  # Enable similarity testing
SIMILARITY_TEST_INTERVAL = 20  # Test every N epochs (same as GENERALIZATION_TEST_INTERVAL by default)
SIMILARITY_TEST_NUM_PAIRS = 50  # Number of pairs to test
"""