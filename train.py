"""Training script for ARC communication system with optional pretrained encoder."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import json
from tqdm import tqdm

import config
from dataset import ARCDataset, collate_fn, collate_fn_puzzle_classification
from model import ARCEncoder, ARCAutoencoder, ARCPuzzleSolver
from live_plotter import LivePlotter
from puzzle_training_dataset import ARCPuzzleTrainingDataset, collate_fn_puzzle_training


def load_all_datasets_with_holdout(min_size, filter_size, max_grids, num_distractors, track_puzzle_ids, use_input_output_pairs, holdout_per_category=25, holdout_seed=42):
    """Load all datasets (V1+V2, training+evaluation) with holdout grids for generalization.
    
    Args:
        min_size: Minimum grid size
        filter_size: Filter for specific grid size
        max_grids: Maximum number of grids per split (None = all)
        num_distractors: Number of distractor grids
        track_puzzle_ids: Whether to track puzzle IDs
        use_input_output_pairs: Whether to use input-output pairs
        holdout_per_category: Number of grids to hold out from each category
        holdout_seed: Random seed for consistent holdout selection
    
    Returns:
        Tuple of (training_dataset, holdout_dataset)
    """
    import random
    import numpy as np
    from torch.utils.data import Subset
    
    print('='*80)
    print('LOADING ALL DATASETS WITH HOLDOUT')
    print('='*80)
    print(f'Holdout: {holdout_per_category} grids per category (4 categories = {holdout_per_category * 4} total)')
    print(f'Random seed: {holdout_seed}')
    print()
    
    # Set random seed for reproducibility
    random.seed(holdout_seed)
    np.random.seed(holdout_seed)
    
    # Define all 4 categories
    categories = [
        ('V1', 'training'),
        ('V1', 'evaluation'),
        ('V2', 'training'),
        ('V2', 'evaluation')
    ]
    
    all_datasets = []
    holdout_indices_all = []  # Global indices for holdout
    current_offset = 0
    
    # Load each category and select holdout grids
    for version, split in categories:
        data_path = os.path.join(version, 'data', split)
        
        if not os.path.exists(data_path):
            print(f'Warning: {data_path} not found, skipping...')
            continue
        
        print(f'Loading {version}/{split}...')
        
        # Load the dataset
        dataset = ARCDataset(
            data_path,
            min_size=min_size,
            filter_size=filter_size,
            max_grids=max_grids,
            num_distractors=num_distractors,
            track_puzzle_ids=track_puzzle_ids,
            use_input_output_pairs=use_input_output_pairs
        )
        
        # Randomly select holdout indices for this category
        n_grids = len(dataset)
        n_holdout = min(holdout_per_category, n_grids)
        
        if n_grids == 0:
            print(f'  Warning: No grids found in {version}/{split}')
            continue
        
        # Select random indices for holdout
        local_indices = list(range(n_grids))
        random.shuffle(local_indices)
        holdout_local = local_indices[:n_holdout]
        
        # Convert to global indices
        holdout_global = [idx + current_offset for idx in holdout_local]
        holdout_indices_all.extend(holdout_global)
        
        print(f'  Loaded {n_grids} grids, holding out {n_holdout} for generalization')
        
        all_datasets.append(dataset)
        current_offset += n_grids
    
    # Combine all datasets
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset(all_datasets)
    
    print()
    print(f'Total grids: {len(combined_dataset)}')
    print(f'Holdout grids: {len(holdout_indices_all)}')
    print(f'Training grids: {len(combined_dataset) - len(holdout_indices_all)}')
    
    # Create training dataset (all indices except holdout)
    all_indices = set(range(len(combined_dataset)))
    holdout_indices_set = set(holdout_indices_all)
    training_indices = sorted(list(all_indices - holdout_indices_set))
    
    training_dataset = Subset(combined_dataset, training_indices)
    holdout_dataset = Subset(combined_dataset, holdout_indices_all)
    
    # Handle puzzle_id_map if tracking puzzle IDs
    if track_puzzle_ids:
        # Merge puzzle_id_maps from all datasets
        combined_puzzle_id_map = {}
        next_id = 0
        
        for dataset in all_datasets:
            if hasattr(dataset, 'puzzle_id_map'):
                for puzzle_id, _ in dataset.puzzle_id_map.items():
                    if puzzle_id not in combined_puzzle_id_map:
                        combined_puzzle_id_map[puzzle_id] = next_id
                        next_id += 1
        
        # Attach to both training and holdout datasets
        training_dataset.puzzle_id_map = combined_puzzle_id_map
        holdout_dataset.puzzle_id_map = combined_puzzle_id_map
        
        print(f'Total unique puzzles: {len(combined_puzzle_id_map)}')
    
    print('='*80)
    print()
    
    return training_dataset, holdout_dataset


def load_dataset_with_splits(dataset_version, dataset_split, use_combined_splits, min_size, filter_size, max_grids, num_distractors, track_puzzle_ids, use_input_output_pairs, use_all_datasets=False, holdout_per_category=25, holdout_seed=42):
    """Load dataset(s) based on split configuration.
    
    Args:
        dataset_version: 'V1' or 'V2' (ignored if use_all_datasets=True)
        dataset_split: 'training' or 'evaluation' (ignored if use_combined_splits=True or use_all_datasets=True)
        use_combined_splits: If True, load and combine both training and evaluation splits
        min_size: Minimum grid size
        filter_size: Filter for specific grid size
        max_grids: Maximum number of grids per split (None = all)
        num_distractors: Number of distractor grids
        track_puzzle_ids: Whether to track puzzle IDs
        use_input_output_pairs: Whether to use input-output pairs
        use_all_datasets: If True, load ALL datasets (V1+V2, train+eval) with holdout for generalization
        holdout_per_category: Number of grids to hold out from each category for generalization
        holdout_seed: Random seed for holdout selection
    
    Returns:
        Combined dataset (or tuple of (training_dataset, holdout_dataset) if use_all_datasets=True)
    """
    # NEW MODE: Load all datasets with holdout
    if use_all_datasets:
        return load_all_datasets_with_holdout(
            min_size=min_size,
            filter_size=filter_size,
            max_grids=max_grids,
            num_distractors=num_distractors,
            track_puzzle_ids=track_puzzle_ids,
            use_input_output_pairs=use_input_output_pairs,
            holdout_per_category=holdout_per_category,
            holdout_seed=holdout_seed
        )
    if not use_combined_splits:
        # Load single split as before
        if dataset_version in ['V1', 'V2']:
            data_path = os.path.join(dataset_version, 'data', dataset_split)
        else:
            data_path = config.DATA_PATH
        
        return ARCDataset(
            data_path,
            min_size=min_size,
            filter_size=filter_size,
            max_grids=max_grids,
            num_distractors=num_distractors,
            track_puzzle_ids=track_puzzle_ids,
            use_input_output_pairs=use_input_output_pairs
        )
    else:
        # Load both training and evaluation splits and combine them
        print(f'Loading combined splits: training + evaluation from {dataset_version}')
        
        if dataset_version in ['V1', 'V2']:
            train_path = os.path.join(dataset_version, 'data', 'training')
            eval_path = os.path.join(dataset_version, 'data', 'evaluation')
        else:
            # Fallback to single path if not V1/V2
            print("Warning: USE_COMBINED_SPLITS not supported for legacy DATA_PATH format")
            return ARCDataset(
                config.DATA_PATH,
                min_size=min_size,
                filter_size=filter_size,
                max_grids=max_grids,
                num_distractors=num_distractors,
                track_puzzle_ids=track_puzzle_ids,
                use_input_output_pairs=use_input_output_pairs
            )
        
        # Load training split
        train_dataset = ARCDataset(
            train_path,
            min_size=min_size,
            filter_size=filter_size,
            max_grids=max_grids,
            num_distractors=num_distractors,
            track_puzzle_ids=track_puzzle_ids,
            use_input_output_pairs=use_input_output_pairs
        )
        
        # Load evaluation split
        eval_dataset = ARCDataset(
            eval_path,
            min_size=min_size,
            filter_size=filter_size,
            max_grids=max_grids,
            num_distractors=num_distractors,
            track_puzzle_ids=track_puzzle_ids,
            use_input_output_pairs=use_input_output_pairs
        )
        
        # Combine datasets using ConcatDataset
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset([train_dataset, eval_dataset])
        
        # If tracking puzzle IDs, we need to merge the puzzle_id_map
        if track_puzzle_ids and hasattr(train_dataset, 'puzzle_id_map') and hasattr(eval_dataset, 'puzzle_id_map'):
            # Create a combined puzzle_id_map
            combined_puzzle_id_map = {}
            next_id = 0
            
            # Add training puzzles
            for puzzle_id, _ in train_dataset.puzzle_id_map.items():
                combined_puzzle_id_map[puzzle_id] = next_id
                next_id += 1
            
            # Add evaluation puzzles (that aren't already in training)
            for puzzle_id, _ in eval_dataset.puzzle_id_map.items():
                if puzzle_id not in combined_puzzle_id_map:
                    combined_puzzle_id_map[puzzle_id] = next_id
                    next_id += 1
            
            # Attach the combined map to the dataset
            combined_dataset.puzzle_id_map = combined_puzzle_id_map
            print(f'Combined {len(train_dataset)} training grids + {len(eval_dataset)} evaluation grids')
            print(f'Total unique puzzles: {len(combined_puzzle_id_map)}')
        else:
            print(f'Combined {len(train_dataset)} training grids + {len(eval_dataset)} evaluation grids')
        
        return combined_dataset


def visualize_grid(grid, title="Grid"):
    """Print a grid with colored blocks."""
    color_codes = {
        0: '\033[40m  \033[0m',  # Black
        1: '\033[44m  \033[0m',  # Blue
        2: '\033[41m  \033[0m',  # Red
        3: '\033[42m  \033[0m',  # Green
        4: '\033[43m  \033[0m',  # Yellow
        5: '\033[47m  \033[0m',  # Gray/White
        6: '\033[45m  \033[0m',  # Magenta
        7: '\033[46m  \033[0m',  # Cyan
        8: '\033[100m  \033[0m', # Dark gray
        9: '\033[101m  \033[0m', # Light red
    }
    
    char_map = {
        0: '  ', 1: '██', 2: '▓▓', 3: '▒▒', 4: '░░',
        5: '▀▀', 6: '▄▄', 7: '▪▪', 8: '●●', 9: '◆◆',
    }
    
    print(f"\n{title} ({grid.shape[0]}x{grid.shape[1]}):")
    print("┌" + "─" * (grid.shape[1] * 2) + "┐")
    
    for row in grid:
        print("│", end="")
        for cell in row:
            cell_val = int(cell)
            try:
                print(color_codes.get(cell_val, '  '), end="")
            except:
                print(char_map.get(cell_val, '??'), end="")
        print("│")
    
    print("└" + "─" * (grid.shape[1] * 2) + "┘")


def visualize_reconstruction(model, dataloader, device, num_samples=3, task_type='reconstruction', use_input_output_pairs=False, plotter=None):
    """Show input grids, messages (if communication mode), and reconstructions/selections side by side."""
    # Force plot update to ensure display is synchronized with current model state
    if plotter is not None:
        plotter.force_update()
    
    model.eval()
    print("\n" + "="*80)
    if task_type == 'selection':
        print("SELECTION VISUALIZATION")
    else:
        print("RECONSTRUCTION VISUALIZATION")
    if use_input_output_pairs:
        print("MODE: Input → Output pairs (training on actual ARC transformations)")
    else:
        print("MODE: Self-supervised (reconstructing/selecting same grid)")
    print("="*80)
    
    with torch.no_grad():
        for batch_data in dataloader:
            # Unpack batch based on task type and use_input_output_pairs
            if use_input_output_pairs:
                if task_type == 'selection':
                    input_grids, input_sizes, output_grids, output_sizes, candidates_list, candidates_sizes_list, target_indices = batch_data
                    input_grids = input_grids.to(device)
                    output_grids = output_grids.to(device)
                    candidates_list = [c.to(device) for c in candidates_list]
                    target_indices = target_indices.to(device)
                else:  # reconstruction
                    input_grids, input_sizes, output_grids, output_sizes = batch_data
                    input_grids = input_grids.to(device)
                    output_grids = output_grids.to(device)
                    candidates_list = None
                    target_indices = None
            else:
                # Original behavior (self-supervised)
                if task_type == 'selection':
                    grids, sizes, candidates_list, candidates_sizes_list, target_indices = batch_data
                    grids = grids.to(device)
                    candidates_list = [c.to(device) for c in candidates_list]
                    target_indices = target_indices.to(device)
                    input_grids = grids
                    input_sizes = sizes
                else:
                    grids, sizes = batch_data
                    grids = grids.to(device)
                    input_grids = grids
                    input_sizes = sizes
            
            for i in range(min(num_samples, len(input_grids))):
                grid = input_grids[i]
                actual_h, actual_w = input_sizes[i]
                
                # Get the actual INPUT grid (without padding)
                input_grid = grid[:actual_h, :actual_w].cpu().numpy()
                
                print(f"\n{'─'*80}")
                print(f"SAMPLE {i+1}")
                print(f"{'─'*80}")
                
                # Show INPUT grid
                visualize_grid(input_grid, f"INPUT GRID (Size: {actual_h}x{actual_w})")
                
                # If using input-output pairs, also show expected output
                if use_input_output_pairs:
                    output_h, output_w = output_sizes[i]
                    expected_output = output_grids[i][:output_h, :output_w].cpu().numpy()
                    visualize_grid(expected_output, f"EXPECTED OUTPUT (Size: {output_h}x{output_w})")
                
                if task_type == 'selection':
                    # Selection task
                    single_grid = grid.unsqueeze(0)
                    candidates = candidates_list[i]
                    target_idx = target_indices[i]
                    
                    # Forward pass for single sample
                    single_candidates_sizes = candidates_sizes_list[i]
                    model_output = model(
                        single_grid, [(actual_h, actual_w)], temperature=1.0,
                        candidates_list=[candidates], candidates_sizes_list=[single_candidates_sizes],
                        target_indices=target_idx.unsqueeze(0)
                    )
                    
                    # Unpack output based on bottleneck type
                    if len(model_output) == 5:  # communication mode with message_lengths
                        selection_logits_list, reconstruction_logits_list, _, messages, message_lengths = model_output
                    else:  # autoencoder mode
                        selection_logits_list, reconstruction_logits_list, _, messages = model_output
                        message_lengths = None
                    
                    # Show message if in communication mode
                    if messages is not None:
                        msg = messages[0].cpu().tolist()
                        if message_lengths is not None:
                            actual_length = message_lengths[0].item()
                            # Show only the actual message (up to stop token)
                            actual_msg = msg[:actual_length]
                            print(f"\n📨 MESSAGE (length {actual_length}/{len(msg)}): {actual_msg}")
                            if actual_length < len(msg):
                                print(f"   (Stopped early - max length is {len(msg)})")
                        else:
                            print(f"\n📨 MESSAGE: {msg}")
                    else:
                        print(f"\n🔄 BOTTLENECK: Continuous latent vector (dim={model.encoder.latent_dim})")
                    
                    # Show candidates and selection
                    sel_logits = selection_logits_list[0]
                    probs = torch.softmax(sel_logits, dim=0).cpu().numpy()
                    pred_idx = sel_logits.argmax().item()
                    
                    # Determine sizes for candidates
                    if use_input_output_pairs:
                        # Candidates are output grids, so we need to show their actual sizes
                        print(f"\n🎯 CANDIDATES ({len(candidates)} options) - selecting OUTPUT:")
                    else:
                        print(f"\n🎯 CANDIDATES ({len(candidates)} options):")
                    
                    for c_idx in range(len(candidates)):
                        c_h, c_w = single_candidates_sizes[c_idx]
                        cand_grid = candidates[c_idx][:c_h, :c_w].cpu().numpy()
                        is_target = "✓ TARGET" if c_idx == target_idx.item() else ""
                        is_predicted = "← SELECTED" if c_idx == pred_idx else ""
                        label = f"Candidate {c_idx} (prob: {probs[c_idx]:.2%}) {is_target} {is_predicted}"
                        visualize_grid(cand_grid, label)
                    
                    # Calculate accuracy for this sample
                    correct = (pred_idx == target_idx.item())
                    print(f"\n📊 METRICS:")
                    print(f"   Selection: {'✓ CORRECT' if correct else '✗ INCORRECT'}")
                    print(f"   Confidence: {probs[pred_idx]:.2%}")
                    
                    # Show reconstruction if selection was correct
                    if correct:
                        recon_logits = reconstruction_logits_list[0]
                        recon = recon_logits.argmax(dim=1).squeeze(0).cpu().numpy()
                        visualize_grid(recon, f"RECONSTRUCTION (from message) - Size: {recon.shape[0]}x{recon.shape[1]}")
                        
                        # Calculate reconstruction accuracy
                        if use_input_output_pairs:
                            # Compare against selected candidate (output)
                            target_grid = candidates[target_idx.item()][:recon.shape[0], :recon.shape[1]].cpu().numpy()
                        else:
                            # Compare against input
                            target_grid = input_grid[:recon.shape[0], :recon.shape[1]]
                        
                        min_h = min(target_grid.shape[0], recon.shape[0])
                        min_w = min(target_grid.shape[1], recon.shape[1])
                        correct_pixels = (target_grid[:min_h, :min_w] == recon[:min_h, :min_w]).sum()
                        total_pixels = min_h * min_w
                        recon_accuracy = 100.0 * correct_pixels / total_pixels
                        print(f"   Reconstruction accuracy: {recon_accuracy:.2f}% ({correct_pixels}/{total_pixels} correct)")
                else:
                    # Reconstruction task
                    single_grid = grid.unsqueeze(0)
                    model_output = model(single_grid, [(actual_h, actual_w)], temperature=1.0)
                    
                    # Unpack output based on bottleneck type
                    if len(model_output) == 4:  # communication mode with message_lengths
                        logits_list, _, messages, message_lengths = model_output
                    else:  # autoencoder mode
                        logits_list, _, messages = model_output
                        message_lengths = None
                    
                    # Get reconstruction
                    recon = logits_list[0].argmax(dim=1).squeeze(0).cpu().numpy()
                    
                    # Show message if in communication mode
                    if messages is not None:
                        msg = messages[0].cpu().tolist()
                        if message_lengths is not None:
                            actual_length = message_lengths[0].item()
                            # Show only the actual message (up to stop token)
                            actual_msg = msg[:actual_length]
                            print(f"\n📨 MESSAGE (length {actual_length}/{len(msg)}): {actual_msg}")
                            if actual_length < len(msg):
                                print(f"   (Stopped early - max length is {len(msg)})")
                        else:
                            print(f"\n📨 MESSAGE: {msg}")
                    else:
                        print(f"\n🔄 BOTTLENECK: Continuous latent vector (dim={model.encoder.latent_dim})")
                    
                    # Show reconstruction
                    visualize_grid(recon, f"RECONSTRUCTION (Size: {recon.shape[0]}x{recon.shape[1]})")
                    
                    # Calculate accuracy for this sample
                    if use_input_output_pairs:
                        # Compare against expected output
                        output_h, output_w = output_sizes[i]
                        expected_output = output_grids[i][:output_h, :output_w].cpu().numpy()
                        min_h = min(output_h, recon.shape[0])
                        min_w = min(output_w, recon.shape[1])
                        correct_pixels = (expected_output[:min_h, :min_w] == recon[:min_h, :min_w]).sum()
                        total_pixels = min_h * min_w
                        accuracy = 100.0 * correct_pixels / total_pixels
                        print(f"\n📊 METRICS:")
                        print(f"   Pixel accuracy (vs expected output): {accuracy:.2f}% ({correct_pixels}/{total_pixels} correct)")
                    else:
                        # Compare against input (self-supervised)
                        min_h = min(actual_h, recon.shape[0])
                        min_w = min(actual_w, recon.shape[1])
                        correct_pixels = (input_grid[:min_h, :min_w] == recon[:min_h, :min_w]).sum()
                        total_pixels = min_h * min_w
                        accuracy = 100.0 * correct_pixels / total_pixels
                        print(f"\n📊 METRICS:")
                        print(f"   Pixel accuracy: {accuracy:.2f}% ({correct_pixels}/{total_pixels} correct)")
                
            break  # Only show first batch
    
    print("\n" + "="*80 + "\n")
    model.train()


def train_epoch(model, dataloader, optimizer, criterion, device, temperature, plotter=None, task_type='reconstruction', use_input_output_pairs=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, batch_data in enumerate(pbar):
        # Unpack batch based on task type and use_input_output_pairs
        if use_input_output_pairs:
            if task_type == 'selection':
                input_grids, input_sizes, output_grids, output_sizes, candidates_list, candidates_sizes_list, target_indices = batch_data
                input_grids = input_grids.to(device)
                output_grids = output_grids.to(device)
                candidates_list = [c.to(device) for c in candidates_list]
                target_indices = target_indices.to(device)
            else:  # reconstruction
                input_grids, input_sizes, output_grids, output_sizes = batch_data
                input_grids = input_grids.to(device)
                output_grids = output_grids.to(device)
                candidates_list = None
                target_indices = None
        else:
            # Original behavior (self-supervised)
            if task_type == 'selection':
                grids, sizes, candidates_list, candidates_sizes_list, target_indices = batch_data
                grids = grids.to(device)
                candidates_list = [c.to(device) for c in candidates_list]
                target_indices = target_indices.to(device)
                input_grids = grids
                input_sizes = sizes
            elif task_type == 'puzzle_classification':
                grids, sizes, labels = batch_data
                grids = grids.to(device)
                labels = labels.to(device)
                input_grids = grids
                input_sizes = sizes
            else:
                grids, sizes = batch_data
                grids = grids.to(device)
                candidates_list = None
                target_indices = None
                input_grids = grids
                input_sizes = sizes
        
        # Forward pass
        optimizer.zero_grad()
        
        if task_type == 'puzzle_classification':
            # Puzzle classification task (no I/O pairs)
            model_output = model(input_grids, input_sizes, temperature=temperature, labels=labels)
            if len(model_output) == 4:  # communication mode with message_lengths
                classification_logits, _, messages, message_lengths = model_output
            else:
                classification_logits, _, messages = model_output
            
            # Compute classification loss
            loss = criterion(classification_logits, labels)
            
            # Calculate accuracy
            pred = classification_logits.argmax(dim=1)
            batch_correct = (pred == labels).sum().item()
            batch_total = labels.size(0)
        elif task_type == 'selection':
            # Selection task (now also computes reconstruction in background)
            if use_input_output_pairs:
                # Use input to generate message, select from candidates based on output
                model_output = model(
                    input_grids, input_sizes, temperature=temperature, 
                    candidates_list=candidates_list, candidates_sizes_list=candidates_sizes_list,
                    target_indices=target_indices
                )
            else:
                # Original: select the same grid
                model_output = model(
                    input_grids, input_sizes, temperature=temperature, 
                    candidates_list=candidates_list, candidates_sizes_list=candidates_sizes_list,
                    target_indices=target_indices
                )
            
            # Unpack output based on bottleneck type
            if len(model_output) == 5:  # communication mode with message_lengths
                selection_logits_list, reconstruction_logits_list, actual_sizes, messages, message_lengths = model_output
            else:  # autoencoder mode
                selection_logits_list, reconstruction_logits_list, actual_sizes, messages = model_output
            
            # Compute selection loss for each sample
            batch_loss = 0
            batch_correct = 0
            batch_total = 0
            reconstruction_loss_total = 0
            reconstruction_correct = 0
            reconstruction_total = 0
            num_reconstructions = 0
            
            for sample_idx, sel_logits in enumerate(selection_logits_list):
                target_idx = target_indices[sample_idx]
                sample_loss = criterion(sel_logits.unsqueeze(0), target_idx.unsqueeze(0))
                batch_loss += sample_loss
                
                pred_idx = sel_logits.argmax()
                is_correct = (pred_idx == target_idx).item()
                batch_correct += is_correct
                batch_total += 1
                
                # Compute reconstruction loss when selection is correct
                if is_correct:
                    recon_logits = reconstruction_logits_list[sample_idx]
                    actual_h, actual_w = actual_sizes[sample_idx]
                    H, W = recon_logits.shape[2], recon_logits.shape[3]
                    
                    # Get the target grid for reconstruction and its actual size
                    if use_input_output_pairs:
                        # Reconstruct the selected output grid
                        target_grid_idx = target_indices[sample_idx]
                        target_grid = candidates_list[sample_idx][target_grid_idx:target_grid_idx+1, :H, :W]
                        # Get actual size of the output grid (target candidate)
                        recon_actual_h, recon_actual_w = candidates_sizes_list[sample_idx][target_grid_idx.item()]
                    else:
                        # Reconstruct the input grid (self-supervised)
                        target_grid = input_grids[sample_idx:sample_idx+1, :H, :W]
                        recon_actual_h, recon_actual_w = input_sizes[sample_idx]
                    
                    # Compute reconstruction loss
                    logits_flat = recon_logits.permute(0, 2, 3, 1).reshape(-1, recon_logits.shape[1])
                    targets_flat = target_grid.reshape(-1)
                    
                    recon_loss = criterion(logits_flat, targets_flat)
                    reconstruction_loss_total += recon_loss
                    num_reconstructions += 1
                    
                    # Calculate reconstruction accuracy (only on actual non-padded pixels)
                    pred = recon_logits.argmax(dim=1).squeeze(0)
                    target = target_grid.squeeze(0)
                    reconstruction_correct += (pred[:recon_actual_h, :recon_actual_w] == target[:recon_actual_h, :recon_actual_w]).sum().item()
                    reconstruction_total += recon_actual_h * recon_actual_w
            
            # Combine losses: selection loss for all samples + reconstruction loss for correct selections
            loss = batch_loss / len(selection_logits_list)
            if num_reconstructions > 0:
                reconstruction_loss_avg = reconstruction_loss_total / num_reconstructions
                # Add reconstruction loss (already detached from sender/encoder via .detach())
                loss = loss + reconstruction_loss_avg
        else:
            # Reconstruction task
            if use_input_output_pairs:
                # Use input to generate message, reconstruct output
                model_output = model(input_grids, input_sizes, temperature=temperature)
                
                # Unpack output based on bottleneck type
                if len(model_output) == 4:  # communication mode with message_lengths
                    logits_list, actual_sizes, messages, message_lengths = model_output
                else:  # autoencoder mode
                    logits_list, actual_sizes, messages = model_output
                
                # Compute reconstruction loss for each sample (target is OUTPUT)
                batch_loss = 0
                batch_correct = 0
                batch_total = 0
                
                for sample_idx, (logits, (actual_h, actual_w)) in enumerate(zip(logits_list, actual_sizes)):
                    actual_h, actual_w = output_sizes[sample_idx]  # Use OUTPUT size
                    H, W = logits.shape[2], logits.shape[3]
                    
                    target_grid = output_grids[sample_idx:sample_idx+1, :H, :W]  # Use OUTPUT grid
                    
                    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
                    targets_flat = target_grid.reshape(-1)
                    
                    sample_loss = criterion(logits_flat, targets_flat)
                    batch_loss += sample_loss
                    
                    # Only calculate accuracy on actual (non-padded) pixels
                    pred = logits.argmax(dim=1).squeeze(0)
                    target = target_grid.squeeze(0)
                    batch_correct += (pred[:actual_h, :actual_w] == target[:actual_h, :actual_w]).sum().item()
                    batch_total += actual_h * actual_w
                
                loss = batch_loss / len(logits_list)
            else:
                # Original: reconstruct the same grid
                model_output = model(input_grids, input_sizes, temperature=temperature)
                
                # Unpack output based on bottleneck type
                if len(model_output) == 4:  # communication mode with message_lengths
                    logits_list, actual_sizes, messages, message_lengths = model_output
                else:  # autoencoder mode
                    logits_list, actual_sizes, messages = model_output
                
                # Compute reconstruction loss for each sample
                batch_loss = 0
                batch_correct = 0
                batch_total = 0
                
                for sample_idx, (logits, (actual_h, actual_w)) in enumerate(zip(logits_list, actual_sizes)):
                    actual_h, actual_w = input_sizes[sample_idx]
                    H, W = logits.shape[2], logits.shape[3]
                    
                    target_grid = input_grids[sample_idx:sample_idx+1, :H, :W]
                    
                    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
                    targets_flat = target_grid.reshape(-1)
                    
                    sample_loss = criterion(logits_flat, targets_flat)
                    batch_loss += sample_loss
                    
                    # Only calculate accuracy on actual (non-padded) pixels
                    pred = logits.argmax(dim=1).squeeze(0)
                    target = target_grid.squeeze(0)
                    batch_correct += (pred[:actual_h, :actual_w] == target[:actual_h, :actual_w]).sum().item()
                    batch_total += actual_h * actual_w
                
                loss = batch_loss / len(logits_list)
        
        # For β-VAE: add KL divergence term to the loss
        kl_div = torch.tensor(0.0)
        if hasattr(model, 'use_beta_vae') and model.use_beta_vae and task_type == 'reconstruction':
            loss, kl_div = model.compute_beta_vae_loss(loss)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        correct += batch_correct
        total += batch_total
        
        # Calculate current metrics
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = 100. * correct / total if total > 0 else 0.0
        
        # Update live plot
        if plotter is not None:
            plotter.update(avg_loss, accuracy)
        
        # Update progress bar
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            postfix = {
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%'
            }
            if config.BOTTLENECK_TYPE == 'communication':
                postfix['temp'] = f'{temperature:.2f}'
            # Add KL divergence to progress bar for β-VAE
            if hasattr(model, 'use_beta_vae') and model.use_beta_vae and task_type == 'reconstruction':
                postfix['kl'] = f'{kl_div.item():.4f}'
            pbar.set_postfix(postfix)
    
    return total_loss / len(dataloader), 100. * correct / total if total > 0 else 0.0


def run_generalization_test(model, device, task_type='reconstruction', use_input_output_pairs=False, holdout_dataset=None):
    """
    Test model on unseen dataset for generalization evaluation.
    Returns dict with metrics and saves results to JSON file.
    
    Args:
        model: The model to test
        device: Device to run on
        task_type: Type of task
        use_input_output_pairs: Whether using input-output pairs
        holdout_dataset: Pre-loaded holdout dataset (for USE_ALL_DATASETS mode)
    """
    # Check if generalization testing is enabled
    if not getattr(config, 'GENERALIZATION_TEST_ENABLED', False):
        return None
    
    print(f'\n{"="*80}')
    
    # Determine num_distractors and track_puzzle_ids based on task type
    # (needed for both holdout dataset and regular generalization test)
    num_distractors = getattr(config, 'NUM_DISTRACTORS', 0) if task_type == 'selection' else 0
    track_puzzle_ids = task_type == 'puzzle_classification'
    
    # If holdout_dataset is provided (USE_ALL_DATASETS mode), use it directly
    if holdout_dataset is not None:
        print(f'GENERALIZATION TEST: Testing on HOLDOUT grids')
        print(f'{"="*80}')
        gen_dataset = holdout_dataset
        gen_dataset_version = 'ALL (V1+V2)'
        gen_dataset_split = 'holdout'
    else:
        # Original behavior: load a separate generalization test dataset
        # Get generalization test dataset configuration
        gen_dataset_version = getattr(config, 'GENERALIZATION_TEST_DATASET_VERSION', 'V2')
        gen_dataset_split = getattr(config, 'GENERALIZATION_TEST_DATASET_SPLIT', 'training')
        gen_max_grids = getattr(config, 'GENERALIZATION_TEST_MAX_GRIDS', 100)
        
        # Construct path to generalization test dataset
        if gen_dataset_version in ['V1', 'V2']:
            gen_data_path = os.path.join(gen_dataset_version, 'data', gen_dataset_split)
        else:
            print(f"Warning: Unknown generalization test dataset version: {gen_dataset_version}")
            return None
        
        # Check if path exists
        if not os.path.exists(gen_data_path):
            print(f"Warning: Generalization test dataset not found at {gen_data_path}")
            return None
        
        print(f'GENERALIZATION TEST: Testing on {gen_dataset_version}/{gen_dataset_split}')
        print(f'{"="*80}')
        
        # Load generalization test dataset
        try:
            gen_dataset = ARCDataset(
                gen_data_path,
                min_size=config.MIN_GRID_SIZE,
                filter_size=getattr(config, 'FILTER_GRID_SIZE', None),
                max_grids=gen_max_grids,
                num_distractors=num_distractors,
                track_puzzle_ids=track_puzzle_ids,
                use_input_output_pairs=use_input_output_pairs
            )
        except Exception as e:
            print(f"Error loading generalization test dataset: {e}")
            return None
    
    # Create dataloader
    from functools import partial
    if task_type == 'puzzle_classification':
        collate_fn_for_task = collate_fn_puzzle_classification
    else:
        collate_fn_for_task = partial(collate_fn, num_distractors=num_distractors, use_input_output_pairs=use_input_output_pairs)
    
    gen_loader = DataLoader(
        gen_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_for_task,
        num_workers=0
    )
    
    # Run validation on generalization dataset
    criterion = nn.CrossEntropyLoss()
    gen_loss, gen_acc = validate(model, gen_loader, criterion, device, task_type=task_type, use_input_output_pairs=use_input_output_pairs)
    
    # Prepare results
    results = {
        'dataset_version': gen_dataset_version,
        'dataset_split': gen_dataset_split,
        'num_grids': len(gen_dataset),
        'loss': float(gen_loss),
        'accuracy': float(gen_acc),
        'timestamp': time.time()
    }
    
    print(f'Generalization Test Loss: {gen_loss:.4f}, Accuracy: {gen_acc:.2f}%')
    print(f'{"="*80}\n')
    
    return results


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
        import numpy as np
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
    import numpy as np
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
    import torch.nn.functional as F
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
    import numpy as np
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
    import numpy as np
    import torch.nn.functional as F
    
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


def validate(model, dataloader, criterion, device, task_type='reconstruction', use_input_output_pairs=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc='Validation'):
            # Unpack batch based on task type and use_input_output_pairs
            if use_input_output_pairs:
                if task_type == 'selection':
                    input_grids, input_sizes, output_grids, output_sizes, candidates_list, candidates_sizes_list, target_indices = batch_data
                    input_grids = input_grids.to(device)
                    output_grids = output_grids.to(device)
                    candidates_list = [c.to(device) for c in candidates_list]
                    target_indices = target_indices.to(device)
                else:  # reconstruction
                    input_grids, input_sizes, output_grids, output_sizes = batch_data
                    input_grids = input_grids.to(device)
                    output_grids = output_grids.to(device)
                    candidates_list = None
                    target_indices = None
            else:
                # Original behavior (self-supervised)
                if task_type == 'selection':
                    grids, sizes, candidates_list, candidates_sizes_list, target_indices = batch_data
                    grids = grids.to(device)
                    candidates_list = [c.to(device) for c in candidates_list]
                    target_indices = target_indices.to(device)
                    input_grids = grids
                    input_sizes = sizes
                elif task_type == 'puzzle_classification':
                    grids, sizes, labels = batch_data
                    grids = grids.to(device)
                    labels = labels.to(device)
                    input_grids = grids
                    input_sizes = sizes
                else:
                    grids, sizes = batch_data
                    grids = grids.to(device)
                    candidates_list = None
                    target_indices = None
                    input_grids = grids
                    input_sizes = sizes
            
            # Forward pass
            if task_type == 'puzzle_classification':
                # Puzzle classification task
                model_output = model(input_grids, input_sizes, temperature=1.0, labels=labels)
                if len(model_output) == 4:  # communication mode with message_lengths
                    classification_logits, _, messages, message_lengths = model_output
                else:
                    classification_logits, _, messages = model_output
                
                # Compute classification loss
                loss = criterion(classification_logits, labels)
                
                # Calculate accuracy
                pred = classification_logits.argmax(dim=1)
                batch_correct = (pred == labels).sum().item()
                batch_total = labels.size(0)
                
                total_loss += loss.item()
                correct += batch_correct
                total += batch_total
            elif task_type == 'selection':
                if use_input_output_pairs:
                    model_output = model(
                        input_grids, input_sizes, temperature=1.0,
                        candidates_list=candidates_list, candidates_sizes_list=candidates_sizes_list,
                        target_indices=target_indices
                    )
                else:
                    model_output = model(
                        input_grids, input_sizes, temperature=1.0,
                        candidates_list=candidates_list, candidates_sizes_list=candidates_sizes_list,
                        target_indices=target_indices
                    )
                
                # Unpack output based on bottleneck type
                if len(model_output) == 5:  # communication mode with message_lengths
                    selection_logits_list, reconstruction_logits_list, actual_sizes, messages, message_lengths = model_output
                else:  # autoencoder mode
                    selection_logits_list, reconstruction_logits_list, actual_sizes, messages = model_output
                
                # Compute selection loss for each sample
                batch_loss = 0
                batch_correct = 0
                batch_total = 0
                reconstruction_loss_total = 0
                num_reconstructions = 0
                
                for sample_idx, sel_logits in enumerate(selection_logits_list):
                    target_idx = target_indices[sample_idx]
                    sample_loss = criterion(sel_logits.unsqueeze(0), target_idx.unsqueeze(0))
                    batch_loss += sample_loss
                    
                    pred_idx = sel_logits.argmax()
                    is_correct = (pred_idx == target_idx).item()
                    batch_correct += is_correct
                    batch_total += 1
                    
                    # Compute reconstruction loss when selection is correct (for validation tracking)
                    if is_correct:
                        recon_logits = reconstruction_logits_list[sample_idx]
                        actual_h, actual_w = actual_sizes[sample_idx]
                        H, W = recon_logits.shape[2], recon_logits.shape[3]
                        
                        # Get the target grid for reconstruction and its actual size
                        if use_input_output_pairs:
                            # Reconstruct the selected output grid
                            target_grid_idx = target_indices[sample_idx]
                            target_grid = candidates_list[sample_idx][target_grid_idx:target_grid_idx+1, :H, :W]
                            # Get actual size of the output grid (target candidate)
                            recon_actual_h, recon_actual_w = candidates_sizes_list[sample_idx][target_grid_idx.item()]
                        else:
                            # Reconstruct the input grid (self-supervised)
                            target_grid = input_grids[sample_idx:sample_idx+1, :H, :W]
                            recon_actual_h, recon_actual_w = input_sizes[sample_idx]
                        
                        # Compute reconstruction loss
                        logits_flat = recon_logits.permute(0, 2, 3, 1).reshape(-1, recon_logits.shape[1])
                        targets_flat = target_grid.reshape(-1)
                        
                        recon_loss = criterion(logits_flat, targets_flat)
                        reconstruction_loss_total += recon_loss
                        num_reconstructions += 1
                
                loss = batch_loss / len(selection_logits_list)
                if num_reconstructions > 0:
                    reconstruction_loss_avg = reconstruction_loss_total / num_reconstructions
                    loss = loss + reconstruction_loss_avg
            else:
                if use_input_output_pairs:
                    # Use input to generate message, reconstruct output
                    model_output = model(input_grids, input_sizes, temperature=1.0)
                    
                    # Unpack output based on bottleneck type
                    if len(model_output) == 4:  # communication mode with message_lengths
                        logits_list, actual_sizes, messages, message_lengths = model_output
                    else:  # autoencoder mode
                        logits_list, actual_sizes, messages = model_output
                    
                    # Compute reconstruction loss for each sample (target is OUTPUT)
                    batch_loss = 0
                    batch_correct = 0
                    batch_total = 0
                    
                    for sample_idx, (logits, (actual_h, actual_w)) in enumerate(zip(logits_list, actual_sizes)):
                        actual_h, actual_w = output_sizes[sample_idx]  # Use OUTPUT size
                        H, W = logits.shape[2], logits.shape[3]
                        
                        target_grid = output_grids[sample_idx:sample_idx+1, :H, :W]  # Use OUTPUT grid
                        
                        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
                        targets_flat = target_grid.reshape(-1)
                        
                        sample_loss = criterion(logits_flat, targets_flat)
                        batch_loss += sample_loss
                        
                        # Only calculate accuracy on actual (non-padded) pixels
                        pred = logits.argmax(dim=1).squeeze(0)
                        target = target_grid.squeeze(0)
                        batch_correct += (pred[:actual_h, :actual_w] == target[:actual_h, :actual_w]).sum().item()
                        batch_total += actual_h * actual_w
                    
                    loss = batch_loss / len(logits_list)
                else:
                    # Original: reconstruct the same grid
                    model_output = model(input_grids, input_sizes, temperature=1.0)
                    
                    # Unpack output based on bottleneck type
                    if len(model_output) == 4:  # communication mode with message_lengths
                        logits_list, actual_sizes, messages, message_lengths = model_output
                    else:  # autoencoder mode
                        logits_list, actual_sizes, messages = model_output
                    
                    # Compute reconstruction loss for each sample
                    batch_loss = 0
                    batch_correct = 0
                    batch_total = 0
                    
                    for sample_idx, (logits, (actual_h, actual_w)) in enumerate(zip(logits_list, actual_sizes)):
                        actual_h, actual_w = input_sizes[sample_idx]
                        H, W = logits.shape[2], logits.shape[3]
                        
                        target_grid = input_grids[sample_idx:sample_idx+1, :H, :W]
                        
                        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
                        targets_flat = target_grid.reshape(-1)
                        
                        sample_loss = criterion(logits_flat, targets_flat)
                        batch_loss += sample_loss
                        
                        # Only calculate accuracy on actual (non-padded) pixels
                        pred = logits.argmax(dim=1).squeeze(0)
                        target = target_grid.squeeze(0)
                        batch_correct += (pred[:actual_h, :actual_w] == target[:actual_h, :actual_w]).sum().item()
                        batch_total += actual_h * actual_w
                    
                    loss = batch_loss / len(logits_list)
            
            total_loss += loss.item()
            correct += batch_correct
            total += batch_total
    
    return total_loss / len(dataloader), 100. * correct / total if total > 0 else 0.0


def analyze_messages(model, dataloader, device, num_samples=5):
    """Print some example messages to see what the agents are communicating."""
    if config.BOTTLENECK_TYPE != 'communication':
        print("\n--- Message analysis only available in communication mode ---")
        return
        
    model.eval()
    print("\n--- Sample Messages ---")
    use_stop_token = getattr(config, 'USE_STOP_TOKEN', False)
    
    with torch.no_grad():
        for grids, sizes in dataloader:
            grids = grids.to(device)
            num_samples = min(num_samples, len(grids))
            
            # Get messages from sender
            sender_output = model.sender(grids[:num_samples], sizes=sizes[:num_samples], temperature=1.0)
            if len(sender_output) == 3:  # with message_lengths
                message, _, message_lengths = sender_output
            else:  # old format
                message, _ = sender_output
                message_lengths = None
            
            for i in range(num_samples):
                msg = message[i].cpu().tolist()
                actual_size = sizes[i]
                print(f"Grid {i} - Size: {actual_size}")
                if message_lengths is not None and use_stop_token:
                    actual_length = message_lengths[i].item()
                    actual_msg = msg[:actual_length]
                    print(f"  Message (length {actual_length}/{len(msg)}): {actual_msg}")
                    if actual_length < len(msg):
                        print(f"  (Stopped early - max length is {len(msg)})")
                else:
                    print(f"  Message: {msg}")
            
            break  # Only show first batch
    print("---")


def load_pretrained_encoder(encoder, pretrained_path):
    """Load pretrained encoder weights."""
    if os.path.exists(pretrained_path):
        print(f'\nLoading pretrained encoder from {pretrained_path}...')
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Try to load, but handle architecture mismatches gracefully
        try:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print(f"✓ Loaded pretrained encoder (Val Acc: {checkpoint.get('val_acc', 'N/A')}%)")
            return True
        except RuntimeError as e:
            print(f"✗ Failed to load pretrained encoder: architecture mismatch")
            print(f"  Error: {e}")
            print("  Training from scratch instead...")
            return False
    else:
        print(f'\nPretrained encoder not found at {pretrained_path}')
        print('Training from scratch...')
        return False


def infer_num_conv_layers_from_checkpoint(checkpoint_path):
    """Infer the number of convolutional layers from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['encoder_state_dict']
        
        # Count conv layers by looking for conv_layers.X.weight keys
        max_layer = -1
        for key in state_dict.keys():
            if key.startswith('conv_layers.') and '.weight' in key:
                layer_num = int(key.split('.')[1])
                max_layer = max(max_layer, layer_num)
        
        if max_layer >= 0:
            num_layers = max_layer + 1
            return num_layers
        return None
    except Exception as e:
        print(f"Warning: Could not infer architecture from checkpoint: {e}")
        return None


def cleanup_old_checkpoints(save_dir, keep_last=3):
    """Remove old checkpoint files, keeping only the most recent ones.
    
    Args:
        save_dir: Directory containing checkpoint files
        keep_last: Number of most recent checkpoints to keep
    """
    import glob
    import re
    
    # Find all checkpoint files (but not best_model.pth or pretrained_*.pth)
    checkpoint_pattern = os.path.join(save_dir, 'checkpoint_epoch_*.pth')
    checkpoints = glob.glob(checkpoint_pattern)
    
    if len(checkpoints) <= keep_last:
        return  # Nothing to clean up
    
    # Extract epoch numbers and sort
    checkpoint_info = []
    for cp in checkpoints:
        match = re.search(r'checkpoint_epoch_(\d+)\.pth', cp)
        if match:
            epoch_num = int(match.group(1))
            checkpoint_info.append((epoch_num, cp))
    
    # Sort by epoch number (oldest first)
    checkpoint_info.sort(key=lambda x: x[0])
    
    # Remove oldest checkpoints
    num_to_remove = len(checkpoint_info) - keep_last
    for i in range(num_to_remove):
        _, checkpoint_path = checkpoint_info[i]
        try:
            os.remove(checkpoint_path)
            print(f'  Removed old checkpoint: {os.path.basename(checkpoint_path)}')
        except Exception as e:
            print(f'  Warning: Could not remove {checkpoint_path}: {e}')

def train_epoch_puzzle_solving(model, dataloader, optimizer, criterion, device, temperature, plotter=None):
    """Training epoch for puzzle solving task."""
    model.train()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    
    pbar = tqdm(dataloader, desc='Training Puzzles')
    for batch_idx, puzzle_data in enumerate(pbar):
        (train_inputs, train_input_sizes, train_outputs, train_output_sizes,
         test_inputs, test_input_sizes, test_outputs, test_output_sizes,
         puzzle_id) = puzzle_data
        
        # Move to device
        train_inputs = [inp.to(device) for inp in train_inputs]
        train_outputs = [out.to(device) for out in train_outputs]
        test_inputs = [inp.to(device) for inp in test_inputs]
        test_outputs = [out.to(device) for out in test_outputs]
        
        optimizer.zero_grad()
        
        # Process each test example
        batch_loss = 0
        for test_idx, (test_inp, test_inp_size, test_out, test_out_size) in enumerate(
            zip(test_inputs, test_input_sizes, test_outputs, test_output_sizes)):
            
            # Forward pass
            logits, message, soft_message, message_lengths, rule = model(
                train_inputs, train_input_sizes,
                train_outputs, train_output_sizes,
                test_inp, test_inp_size, test_out_size,
                temperature=temperature
            )
            
            # Compute loss
            H, W = logits.shape[2], logits.shape[3]
            target_grid = test_out.unsqueeze(0)[:, :H, :W]
            
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
            targets_flat = target_grid.reshape(-1)
            
            loss = criterion(logits_flat, targets_flat)
            batch_loss += loss
            
            # Calculate accuracy on actual (non-padded) pixels
            out_h, out_w = test_out_size
            pred = logits.argmax(dim=1).squeeze(0)
            target = test_out
            correct_pixels += (pred[:out_h, :out_w] == target[:out_h, :out_w]).sum().item()
            total_pixels += out_h * out_w
        
        # Average loss over test examples
        loss = batch_loss / len(test_inputs)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate metrics
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = 100. * correct_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # Update live plot
        if plotter is not None:
            plotter.update(avg_loss, accuracy)
        
        # Update progress bar
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%',
                'puzzle': puzzle_id
            })
    
    return total_loss / len(dataloader), 100. * correct_pixels / total_pixels if total_pixels > 0 else 0.0


def validate_puzzle_solving(model, dataloader, criterion, device):
    """Validation for puzzle solving task."""
    model.eval()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for puzzle_data in tqdm(dataloader, desc='Validation'):
            (train_inputs, train_input_sizes, train_outputs, train_output_sizes,
             test_inputs, test_input_sizes, test_outputs, test_output_sizes,
             puzzle_id) = puzzle_data
            
            # Move to device
            train_inputs = [inp.to(device) for inp in train_inputs]
            train_outputs = [out.to(device) for out in train_outputs]
            test_inputs = [inp.to(device) for inp in test_inputs]
            test_outputs = [out.to(device) for out in test_outputs]
            
            # Process each test example
            batch_loss = 0
            for test_idx, (test_inp, test_inp_size, test_out, test_out_size) in enumerate(
                zip(test_inputs, test_input_sizes, test_outputs, test_output_sizes)):
                
                logits, _, _, _, _ = model(
                    train_inputs, train_input_sizes,
                    train_outputs, train_output_sizes,
                    test_inp, test_inp_size, test_out_size,
                    temperature=1.0
                )
                
                # Compute loss
                H, W = logits.shape[2], logits.shape[3]
                target_grid = test_out.unsqueeze(0)[:, :H, :W]
                
                logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
                targets_flat = target_grid.reshape(-1)
                
                loss = criterion(logits_flat, targets_flat)
                batch_loss += loss
                
                # Calculate accuracy
                out_h, out_w = test_out_size
                pred = logits.argmax(dim=1).squeeze(0)
                target = test_out
                correct_pixels += (pred[:out_h, :out_w] == target[:out_h, :out_w]).sum().item()
                total_pixels += out_h * out_w
            
            total_loss += (batch_loss / len(test_inputs)).item()
    
    return total_loss / len(dataloader), 100. * correct_pixels / total_pixels if total_pixels > 0 else 0.0


def main():
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create save directory
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # Determine task type and num distractors FIRST (before using them)
    task_type = getattr(config, 'TASK_TYPE', 'reconstruction')
        # ADD THIS CHECK:
    if task_type == 'puzzle_solving':
        # Puzzle solving uses different dataset and training loop
        train_puzzle_solving_mode(device)
        return  # Exit early, puzzle solving handled separately

    num_distractors = getattr(config, 'NUM_DISTRACTORS', 0) if task_type == 'selection' else 0
    track_puzzle_ids = task_type == 'puzzle_classification'
    use_input_output_pairs = getattr(config, 'USE_INPUT_OUTPUT_PAIRS', False)
    
    # Display bottleneck type and task type
    print(f'\n{"="*80}')
    print(f'BOTTLENECK TYPE: {config.BOTTLENECK_TYPE.upper()}')
    if config.BOTTLENECK_TYPE == 'communication':
        use_stop_token = getattr(config, 'USE_STOP_TOKEN', False)
        print(f'  - Vocabulary size: {config.VOCAB_SIZE}')
        if use_stop_token:
            stop_token_id = getattr(config, 'STOP_TOKEN_ID', config.VOCAB_SIZE)
            print(f'  - Stop token enabled: True (ID: {stop_token_id}, effective vocab size: {config.VOCAB_SIZE + 1})')
        else:
            print(f'  - Stop token enabled: False')
        print(f'  - Max message length: {config.MAX_MESSAGE_LENGTH}')
        receiver_gets_input = getattr(config, 'RECEIVER_GETS_INPUT_PUZZLE', False)
        print(f'  - Receiver gets input puzzle: {receiver_gets_input}')
    else:
        print(f'  - Latent dimension: {config.LATENT_DIM}')
    
    print(f'\nTASK TYPE: {task_type.upper()}')
    if task_type == 'selection':
        print(f'  - Number of distractors: {num_distractors}')
        if use_input_output_pairs:
            print(f'  - Task: Sender sees INPUT → Message → Receiver selects OUTPUT from {num_distractors + 1} candidates')
        else:
            print(f'  - Task: Sender → Message → Receiver selects from {num_distractors + 1} candidates (self-supervised)')
    elif task_type == 'puzzle_classification':
        print(f'  - Task: Sender → Message → Receiver classifies puzzle category')
    else:
        if use_input_output_pairs:
            print(f'  - Task: Sender sees INPUT → Message → Receiver reconstructs OUTPUT')
        else:
            print(f'  - Task: Sender → Message → Receiver reconstructs grid (self-supervised)')
    print(f'{"="*80}\n')
    
    # Load dataset
    print('Loading dataset...')
    use_combined_splits = getattr(config, 'USE_COMBINED_SPLITS', False)
    use_all_datasets = getattr(config, 'USE_ALL_DATASETS', False)
    dataset_version = getattr(config, 'DATASET_VERSION', 'V2')
    dataset_split = getattr(config, 'DATASET_SPLIT', 'training')
    
    dataset_result = load_dataset_with_splits(
        dataset_version=dataset_version,
        dataset_split=dataset_split,
        use_combined_splits=use_combined_splits,
        min_size=config.MIN_GRID_SIZE,
        filter_size=getattr(config, 'FILTER_GRID_SIZE', None),
        max_grids=getattr(config, 'MAX_GRIDS', None),
        num_distractors=num_distractors,
        track_puzzle_ids=track_puzzle_ids,
        use_input_output_pairs=use_input_output_pairs,
        use_all_datasets=use_all_datasets,
        holdout_per_category=getattr(config, 'HOLDOUT_GRIDS_PER_CATEGORY', 25),
        holdout_seed=getattr(config, 'HOLDOUT_SEED', 42)
    )
    
    # Handle return value - either a single dataset or (training, holdout) tuple
    if use_all_datasets:
        dataset, holdout_dataset = dataset_result
        print(f'\n✓ Using ALL datasets mode with {len(holdout_dataset)} grids held out for generalization')
    else:
        dataset = dataset_result
        holdout_dataset = None
    
    # For puzzle classification, get number of classes
    num_classes = None
    if task_type == 'puzzle_classification':
        num_puzzles = len(dataset.puzzle_id_map)
        num_classes = num_puzzles * 2
        print(f'Puzzle classification setup:')
        print(f'  - Number of puzzles: {num_puzzles}')
        print(f'  - Number of classes (inputs + outputs): {num_classes}')
        print()
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create collate function based on task type
    from functools import partial
    if task_type == 'puzzle_classification':
        collate_fn_for_task = collate_fn_puzzle_classification
    else:
        collate_fn_for_task = partial(collate_fn, num_distractors=num_distractors, use_input_output_pairs=use_input_output_pairs)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_for_task,
        num_workers=0  # Set to 0 to avoid issues with random sampling in multiprocessing
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_for_task,
        num_workers=0  # Set to 0 to avoid issues with random sampling in multiprocessing
    )
    
    # Create encoder
    print('Creating model...')
    
    # If using pretrained encoder, infer the architecture from checkpoint
    pretrained_path = os.path.join(config.SAVE_DIR, 'pretrained_encoder.pth')
    num_conv_layers = config.NUM_CONV_LAYERS if hasattr(config, 'NUM_CONV_LAYERS') else 3
    
    if hasattr(config, 'USE_PRETRAINED') and config.USE_PRETRAINED:
        inferred_layers = infer_num_conv_layers_from_checkpoint(pretrained_path)
        if inferred_layers is not None:
            print(f'Pretrained encoder has {inferred_layers} convolutional layers')
            print(f'Config specifies {num_conv_layers} convolutional layers')
            if inferred_layers != num_conv_layers:
                print(f'⚠ Architecture mismatch detected!')
                print(f'  Using {inferred_layers} layers to match pretrained encoder')
                num_conv_layers = inferred_layers
    
    encoder = ARCEncoder(
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_conv_layers=num_conv_layers,
        conv_channels=getattr(config, 'ENCODER_CONV_CHANNELS', None),
        use_beta_vae=getattr(config, 'USE_BETA_VAE', False)
    )
    
    # Load pretrained encoder if available and enabled
    if hasattr(config, 'USE_PRETRAINED') and config.USE_PRETRAINED:
        load_pretrained_encoder(encoder, pretrained_path)
    
    # Create full model with specified bottleneck type and task type
    receiver_gets_input_puzzle = getattr(config, 'RECEIVER_GETS_INPUT_PUZZLE', False)
    use_stop_token = getattr(config, 'USE_STOP_TOKEN', False)
    stop_token_id = getattr(config, 'STOP_TOKEN_ID', None)
    model = ARCAutoencoder(
        encoder=encoder,
        vocab_size=config.VOCAB_SIZE if config.BOTTLENECK_TYPE == 'communication' else None,
        max_length=config.MAX_MESSAGE_LENGTH if config.BOTTLENECK_TYPE == 'communication' else None,
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        max_grid_size=config.MAX_GRID_SIZE,
        bottleneck_type=config.BOTTLENECK_TYPE,
        task_type=task_type,
        num_conv_layers=num_conv_layers,  # Use the same num_conv_layers as encoder
        num_classes=num_classes,  # For puzzle_classification task
        receiver_gets_input_puzzle=receiver_gets_input_puzzle,
        use_stop_token=use_stop_token,
        stop_token_id=stop_token_id,
        lstm_hidden_dim=getattr(config, 'LSTM_HIDDEN_DIM', None),
        use_beta_vae=getattr(config, 'USE_BETA_VAE', False),
        beta=getattr(config, 'BETA_VAE_BETA', 4.0),
        # Slot attention parameters
        num_slots=getattr(config, 'NUM_SLOTS', 7),
        slot_dim=getattr(config, 'SLOT_DIM', 64),
        slot_iterations=getattr(config, 'SLOT_ITERATIONS', 3),
        slot_hidden_dim=getattr(config, 'SLOT_HIDDEN_DIM', 128),
        slot_eps=getattr(config, 'SLOT_EPS', 1e-8)
    ).to(device)
    
    # Freeze encoder if configured
    if hasattr(config, 'FREEZE_ENCODER') and config.FREEZE_ENCODER:
        print('\n🔒 Freezing encoder weights (encoder will not be updated during training)')
        for param in model.encoder.parameters():
            param.requires_grad = False
        frozen_params = sum(p.numel() for p in model.encoder.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  - Frozen parameters: {frozen_params:,}')
        print(f'  - Trainable parameters: {trainable_params:,}')
    else:
        print('\n🔓 Encoder weights will be updated during training')
    
    print(f'Total model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Create live plotter with memory-efficient settings
    print('Initializing live plotter...')
    max_plot_points = getattr(config, 'MAX_PLOT_POINTS', 10000)
    max_epoch_markers = getattr(config, 'MAX_EPOCH_MARKERS', 50)
    plotter = LivePlotter(
        update_interval=1,
        max_points=max_plot_points,
        max_epoch_markers=max_epoch_markers
    )
    
    # Training loop
    best_val_loss = float('inf')
    temperature = config.TEMPERATURE if config.BOTTLENECK_TYPE == 'communication' else 1.0
    
    # Initialize generalization test history
    generalization_history = []
    generalization_results_path = os.path.join(config.SAVE_DIR, 'generalization_test_results.json')
    
    # Initialize similarity test history
    similarity_history = []
    similarity_results_path = os.path.join(config.SAVE_DIR, 'similarity_test_results.json')
    
    print('Starting training...')
    if config.BOTTLENECK_TYPE == 'communication':
        print('Communication bottleneck: grids → discrete symbols → reconstruction')
    elif config.BOTTLENECK_TYPE == 'slot_attention':
        print(f'Slot Attention bottleneck: grids → {config.NUM_SLOTS} object slots → reconstruction')
        print(f'  Slot dimension: {config.SLOT_DIM}, Iterations: {config.SLOT_ITERATIONS}')
    else:
        print('Autoencoder bottleneck: grids → continuous latent → reconstruction')
    print('Decoder/Receiver knows the target size!')
    
    # Display generalization test configuration
    if getattr(config, 'GENERALIZATION_TEST_ENABLED', False):
        gen_dataset_version = getattr(config, 'GENERALIZATION_TEST_DATASET_VERSION', 'V2')
        gen_interval = getattr(config, 'GENERALIZATION_TEST_INTERVAL', 20)
        print(f'\n📊 Generalization Testing Enabled:')
        print(f'   Testing on {gen_dataset_version} dataset every {gen_interval} epochs')
    
    # Display similarity test configuration
    if getattr(config, 'SIMILARITY_TEST_ENABLED', False):
        sim_interval = getattr(config, 'SIMILARITY_TEST_INTERVAL', gen_interval if getattr(config, 'GENERALIZATION_TEST_ENABLED', False) else 20)
        sim_num_pairs = getattr(config, 'SIMILARITY_TEST_NUM_PAIRS', 50)
        print(f'\n🔍 Similarity Testing Enabled:')
        print(f'   Testing encoding consistency every {sim_interval} epochs')
        print(f'   Using {sim_num_pairs} similar pairs and {sim_num_pairs} dissimilar pairs')
    
    # Show initial reconstructions/selections (before training)
    if task_type == 'selection':
        print("\n🔍 INITIAL SELECTIONS (before training):")
    else:
        print("\n🔍 INITIAL RECONSTRUCTIONS (before training):")
    visualize_reconstruction(model, val_loader, device, num_samples=2, task_type=task_type, use_input_output_pairs=use_input_output_pairs, plotter=plotter)
    
    try:
        for epoch in range(config.NUM_EPOCHS):
            print(f'\nEpoch {epoch+1}/{config.NUM_EPOCHS}')
            
            # Train with live plotting
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, temperature,
                plotter=plotter, task_type=task_type, use_input_output_pairs=use_input_output_pairs
            )
            
            # Mark epoch boundary on plot
            plotter.add_epoch_marker(epoch + 1)
            
            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device, task_type=task_type, use_input_output_pairs=use_input_output_pairs)
            
            # Run generalization test every N epochs
            gen_interval = getattr(config, 'GENERALIZATION_TEST_INTERVAL', 20)
            if getattr(config, 'GENERALIZATION_TEST_ENABLED', False) and (epoch + 1) % gen_interval == 0:
                gen_results = run_generalization_test(model, device, task_type=task_type, use_input_output_pairs=use_input_output_pairs, holdout_dataset=holdout_dataset)
                if gen_results is not None:
                    # Add epoch info
                    gen_results['epoch'] = epoch + 1
                    gen_results['train_loss'] = train_loss
                    gen_results['train_acc'] = train_acc
                    gen_results['val_loss'] = val_loss
                    gen_results['val_acc'] = val_acc
                    
                    # Add to history
                    generalization_history.append(gen_results)
                    
                    # Save results to JSON
                    with open(generalization_results_path, 'w') as f:
                        json.dump({
                            'training_dataset': config.DATASET_VERSION,
                            'training_split': config.DATASET_SPLIT,
                            'task_type': task_type,
                            'bottleneck_type': config.BOTTLENECK_TYPE,
                            'history': generalization_history
                        }, f, indent=2)
                    print(f'✓ Saved generalization test results to {generalization_results_path}')
            
            # Run similarity test every N epochs
            sim_interval = getattr(config, 'SIMILARITY_TEST_INTERVAL', gen_interval)
            if getattr(config, 'SIMILARITY_TEST_ENABLED', False) and (epoch + 1) % sim_interval == 0:
                sim_num_pairs = getattr(config, 'SIMILARITY_TEST_NUM_PAIRS', 50)
                sim_results = run_similarity_test(
                    model, 
                    device, 
                    dataset=val_dataset,
                    num_similar_pairs=sim_num_pairs,
                    num_dissimilar_pairs=sim_num_pairs
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
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Print statistics
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Show reconstructions/selections every 5 epochs (or more frequently early on)
            if (epoch + 1) % 5 == 0 or epoch < 3:
                visualize_reconstruction(model, val_loader, device, num_samples=3, task_type=task_type, use_input_output_pairs=use_input_output_pairs, plotter=plotter)
            
            # Show messages every 10 epochs (only in communication mode)
            if (epoch + 1) % 10 == 0:
                analyze_messages(model, val_loader, device)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'encoder_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'bottleneck_type': config.BOTTLENECK_TYPE,
                    # Save model architecture config for easy loading
                    'hidden_dim': config.HIDDEN_DIM,
                    'latent_dim': config.LATENT_DIM,
                    'num_conv_layers': config.NUM_CONV_LAYERS,
                }
                # Add slot attention config if using slot attention
                if config.BOTTLENECK_TYPE == 'slot_attention':
                    checkpoint_data.update({
                        'num_slots': config.NUM_SLOTS,
                        'slot_dim': config.SLOT_DIM,
                        'slot_iterations': config.SLOT_ITERATIONS,
                        'slot_hidden_dim': config.SLOT_HIDDEN_DIM,
                        'slot_eps': config.SLOT_EPS,
                    })
                torch.save(checkpoint_data, os.path.join(config.SAVE_DIR, 'best_model.pth'))
                print('✓ Saved best model')
            
            # Save checkpoint every 10 epochs (keep only last 3 to save disk space)
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(config.SAVE_DIR, f'checkpoint_epoch_{epoch+1}.pth')
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'encoder_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'bottleneck_type': config.BOTTLENECK_TYPE,
                    # Save model architecture config for easy loading
                    'hidden_dim': config.HIDDEN_DIM,
                    'latent_dim': config.LATENT_DIM,
                    'num_conv_layers': config.NUM_CONV_LAYERS,
                }
                # Add slot attention config if using slot attention
                if config.BOTTLENECK_TYPE == 'slot_attention':
                    checkpoint_data.update({
                        'num_slots': config.NUM_SLOTS,
                        'slot_dim': config.SLOT_DIM,
                        'slot_iterations': config.SLOT_ITERATIONS,
                        'slot_hidden_dim': config.SLOT_HIDDEN_DIM,
                        'slot_eps': config.SLOT_EPS,
                    })
                torch.save(checkpoint_data, checkpoint_path)
                
                # Clean up old checkpoints
                keep_last = getattr(config, 'KEEP_LAST_CHECKPOINTS', 3)
                cleanup_old_checkpoints(config.SAVE_DIR, keep_last=keep_last)
        
        # Final visualization
        if task_type == 'selection':
            print("\n🎯 FINAL SELECTIONS:")
        else:
            print("\n🎯 FINAL RECONSTRUCTIONS:")
        visualize_reconstruction(model, val_loader, device, num_samples=5, task_type=task_type, use_input_output_pairs=use_input_output_pairs, plotter=plotter)
        
        # Save final plot
        plotter.save(os.path.join(config.SAVE_DIR, 'training_progress.png'))
        print(f'Training plot saved to {config.SAVE_DIR}/training_progress.png')
        
    except KeyboardInterrupt:
        print('\n\nTraining interrupted by user!')
        print('Saving progress...')
        plotter.save(os.path.join(config.SAVE_DIR, 'training_progress_interrupted.png'))
        
    except Exception as e:
        print(f'\n\nError during training: {e}')
        import traceback
        traceback.print_exc()
        plotter.save(os.path.join(config.SAVE_DIR, 'training_progress_error.png'))
        
    finally:
        # Always close the plotter
        plotter.close()
    
    print('\nTraining complete!')
    if task_type == 'selection':
        if config.BOTTLENECK_TYPE == 'communication':
            print(f'Agents learned to select from candidates using {config.VOCAB_SIZE} discrete symbols')
        else:
            print(f'Autoencoder learned to select from candidates using {config.LATENT_DIM}-dimensional latent space')
        print(f'Selection accuracy on {num_distractors + 1} candidates!')
    else:
        if config.BOTTLENECK_TYPE == 'communication':
            print(f'Agents learned to communicate using {config.VOCAB_SIZE} discrete symbols')
        else:
            print(f'Autoencoder learned to compress grids to {config.LATENT_DIM}-dimensional latent space')
        print('The decoder/receiver was given target sizes directly!')

def train_puzzle_solving_mode(device):
    """Training mode for puzzle solving."""
    print('='*80)
    print('PUZZLE SOLVING MODE')
    print('='*80)
    
    # Create save directory
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # Get dataset configuration
    dataset_version = getattr(config, 'DATASET_VERSION', 'V2')
    dataset_split = getattr(config, 'DATASET_SPLIT', 'training')
    
    # Construct data path (same logic as app.py)
    if dataset_version in ['V1', 'V2']:
        data_path = os.path.join(dataset_version, 'data', dataset_split)
    else:
        data_path = config.DATA_PATH
    
    # Display configuration
    print(f'\nBOTTLENECK TYPE: {config.BOTTLENECK_TYPE.upper()}')
    if config.BOTTLENECK_TYPE == 'communication':
        use_stop_token = getattr(config, 'USE_STOP_TOKEN', False)
        print(f'  - Vocabulary size: {config.VOCAB_SIZE}')
        if use_stop_token:
            stop_token_id = getattr(config, 'STOP_TOKEN_ID', config.VOCAB_SIZE)
            print(f'  - Stop token enabled: True (ID: {stop_token_id})')
        else:
            print(f'  - Stop token enabled: False')
        print(f'  - Max message length: {config.MAX_MESSAGE_LENGTH}')
    
    rule_dim = getattr(config, 'RULE_DIM', 256)
    pair_combination = getattr(config, 'PAIR_COMBINATION', 'concat')
    max_puzzles = getattr(config, 'MAX_PUZZLES', None)
    max_train_examples = getattr(config, 'MAX_TRAIN_EXAMPLES_PER_PUZZLE', None)
    
    print(f'\nPUZZLE SOLVING CONFIG:')
    print(f'  - Dataset: {dataset_version}/{dataset_split}')
    print(f'  - Data path: {data_path}')
    print(f'  - Rule dimension: {rule_dim}')
    print(f'  - Pair combination: {pair_combination}')
    print(f'  - Max puzzles: {max_puzzles or "all"}')
    print(f'  - Max train examples per puzzle: {max_train_examples or "all"}')
    print('='*80 + '\n')
    
    # Load dataset
    print('Loading puzzle dataset...')
    dataset = ARCPuzzleTrainingDataset(
        data_path,
        max_puzzles=max_puzzles,
        max_train_examples=max_train_examples,
        max_test_examples=1
    )
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders (batch_size MUST be 1 for puzzle training)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn_puzzle_training,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_puzzle_training,
        num_workers=0
    )
    
    # Create encoder
    print('Creating model...')
    encoder = ARCEncoder(
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_conv_layers=getattr(config, 'NUM_CONV_LAYERS', 3),
        conv_channels=getattr(config, 'ENCODER_CONV_CHANNELS', None),
        use_beta_vae=getattr(config, 'USE_BETA_VAE', False)
    )
    
    # Load pretrained encoder if available
    if hasattr(config, 'USE_PRETRAINED') and config.USE_PRETRAINED:
        pretrained_path = os.path.join(config.SAVE_DIR, 'pretrained_encoder.pth')
        load_pretrained_encoder(encoder, pretrained_path)
    
    # Create puzzle solver
    use_stop_token = getattr(config, 'USE_STOP_TOKEN', False)
    stop_token_id = getattr(config, 'STOP_TOKEN_ID', None)
    
    model = ARCPuzzleSolver(
        encoder=encoder,
        vocab_size=config.VOCAB_SIZE if config.BOTTLENECK_TYPE == 'communication' else None,
        max_length=config.MAX_MESSAGE_LENGTH if config.BOTTLENECK_TYPE == 'communication' else None,
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        max_grid_size=config.MAX_GRID_SIZE,
        bottleneck_type=config.BOTTLENECK_TYPE,
        rule_dim=rule_dim,
        pair_combination=pair_combination,
        num_conv_layers=getattr(config, 'NUM_CONV_LAYERS', 2),
        use_stop_token=use_stop_token,
        stop_token_id=stop_token_id,
        lstm_hidden_dim=getattr(config, 'LSTM_HIDDEN_DIM', None)
    ).to(device)
    
    # Freeze encoder if configured
    if hasattr(config, 'FREEZE_ENCODER') and config.FREEZE_ENCODER:
        print('\n🔒 Freezing encoder weights')
        for param in model.encoder.parameters():
            param.requires_grad = False
        frozen_params = sum(p.numel() for p in model.encoder.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  - Frozen parameters: {frozen_params:,}')
        print(f'  - Trainable parameters: {trainable_params:,}')
    
    print(f'Total model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Create live plotter
    print('Initializing live plotter...')
    plotter = LivePlotter(
        update_interval=1,
        max_points=getattr(config, 'MAX_PLOT_POINTS', 10000),
        max_epoch_markers=getattr(config, 'MAX_EPOCH_MARKERS', 50)
    )
    
    # Training loop
    best_val_loss = float('inf')
    temperature = config.TEMPERATURE if config.BOTTLENECK_TYPE == 'communication' else 1.0
    
    print('Starting training...')
    
    try:
        for epoch in range(config.NUM_EPOCHS):
            print(f'\nEpoch {epoch+1}/{config.NUM_EPOCHS}')
            
            # Train
            train_loss, train_acc = train_epoch_puzzle_solving(
                model, train_loader, optimizer, criterion, device, temperature,
                plotter=plotter
            )
            
            # Mark epoch boundary
            plotter.add_epoch_marker(epoch + 1)
            
            # Validate
            val_loss, val_acc = validate_puzzle_solving(
                model, val_loader, criterion, device
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Print statistics
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'encoder_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'bottleneck_type': config.BOTTLENECK_TYPE,
                    'task_type': 'puzzle_solving',
                    'rule_dim': rule_dim,
                    'pair_combination': pair_combination,
                }, os.path.join(config.SAVE_DIR, 'best_model_puzzle_solving.pth'))
                print('✓ Saved best model')
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(config.SAVE_DIR, f'checkpoint_puzzle_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'encoder_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'bottleneck_type': config.BOTTLENECK_TYPE,
                    'task_type': 'puzzle_solving',
                }, checkpoint_path)
                
                # Clean up old checkpoints
                keep_last = getattr(config, 'KEEP_LAST_CHECKPOINTS', 3)
                cleanup_old_checkpoints(config.SAVE_DIR, keep_last=keep_last)
        
        # Save final plot
        plotter.save(os.path.join(config.SAVE_DIR, 'training_progress_puzzle.png'))
        print(f'Training plot saved to {config.SAVE_DIR}/training_progress_puzzle.png')
        
    except KeyboardInterrupt:
        print('\n\nTraining interrupted by user!')
        print('Saving progress...')
        plotter.save(os.path.join(config.SAVE_DIR, 'training_progress_puzzle_interrupted.png'))
        
    except Exception as e:
        print(f'\n\nError during training: {e}')
        import traceback
        traceback.print_exc()
        plotter.save(os.path.join(config.SAVE_DIR, 'training_progress_puzzle_error.png'))
        
    finally:
        plotter.close()
    
    print('\nPuzzle solving training complete!')

if __name__ == '__main__':
    main()