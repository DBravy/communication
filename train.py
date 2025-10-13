"""Training script for ARC communication system with optional pretrained encoder."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

import config
from dataset import ARCDataset, collate_fn, collate_fn_puzzle_classification
from model import ARCEncoder, ARCAutoencoder
from live_plotter import LivePlotter


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


def visualize_reconstruction(model, dataloader, device, num_samples=3, task_type='reconstruction', use_input_output_pairs=False):
    """Show input grids, messages (if communication mode), and reconstructions/selections side by side."""
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
                    
                    # Get the target grid for reconstruction
                    if use_input_output_pairs:
                        # Reconstruct the selected output grid
                        target_grid_idx = target_indices[sample_idx]
                        target_grid = candidates_list[sample_idx][target_grid_idx:target_grid_idx+1, :H, :W]
                    else:
                        # Reconstruct the input grid (self-supervised)
                        target_grid = input_grids[sample_idx:sample_idx+1, :H, :W]
                    
                    # Compute reconstruction loss
                    logits_flat = recon_logits.permute(0, 2, 3, 1).reshape(-1, recon_logits.shape[1])
                    targets_flat = target_grid.reshape(-1)
                    
                    recon_loss = criterion(logits_flat, targets_flat)
                    reconstruction_loss_total += recon_loss
                    num_reconstructions += 1
                    
                    # Calculate reconstruction accuracy
                    pred = recon_logits.argmax(dim=1).squeeze(0)
                    target = target_grid.squeeze(0)
                    reconstruction_correct += (pred == target).sum().item()
                    reconstruction_total += target.numel()
            
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
                    
                    pred = logits.argmax(dim=1).squeeze(0)
                    target = target_grid.squeeze(0)
                    batch_correct += (pred == target).sum().item()
                    batch_total += target.numel()
                
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
                    
                    pred = logits.argmax(dim=1).squeeze(0)
                    target = target_grid.squeeze(0)
                    batch_correct += (pred == target).sum().item()
                    batch_total += target.numel()
                
                loss = batch_loss / len(logits_list)
        
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
            pbar.set_postfix(postfix)
    
    return total_loss / len(dataloader), 100. * correct / total if total > 0 else 0.0


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
                        
                        # Get the target grid for reconstruction
                        if use_input_output_pairs:
                            # Reconstruct the selected output grid
                            target_grid_idx = target_indices[sample_idx]
                            target_grid = candidates_list[sample_idx][target_grid_idx:target_grid_idx+1, :H, :W]
                        else:
                            # Reconstruct the input grid (self-supervised)
                            target_grid = input_grids[sample_idx:sample_idx+1, :H, :W]
                        
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
                        
                        pred = logits.argmax(dim=1).squeeze(0)
                        target = target_grid.squeeze(0)
                        batch_correct += (pred == target).sum().item()
                        batch_total += target.numel()
                    
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
                        
                        pred = logits.argmax(dim=1).squeeze(0)
                        target = target_grid.squeeze(0)
                        batch_correct += (pred == target).sum().item()
                        batch_total += target.numel()
                    
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


def main():
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create save directory
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # Determine task type and num distractors FIRST (before using them)
    task_type = getattr(config, 'TASK_TYPE', 'reconstruction')
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
    dataset = ARCDataset(
        config.DATA_PATH, 
        min_size=config.MIN_GRID_SIZE,
        filter_size=getattr(config, 'FILTER_GRID_SIZE', None),
        max_grids=getattr(config, 'MAX_GRIDS', None),
        num_distractors=num_distractors,
        track_puzzle_ids=track_puzzle_ids,
        use_input_output_pairs=use_input_output_pairs
    )
    
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
        num_conv_layers=num_conv_layers
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
        stop_token_id=stop_token_id
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
    
    print('Starting training...')
    if config.BOTTLENECK_TYPE == 'communication':
        print('Communication bottleneck: grids → discrete symbols → reconstruction')
    else:
        print('Autoencoder bottleneck: grids → continuous latent → reconstruction')
    print('Decoder/Receiver knows the target size!')
    
    # Show initial reconstructions/selections (before training)
    if task_type == 'selection':
        print("\n🔍 INITIAL SELECTIONS (before training):")
    else:
        print("\n🔍 INITIAL RECONSTRUCTIONS (before training):")
    visualize_reconstruction(model, val_loader, device, num_samples=2, task_type=task_type, use_input_output_pairs=use_input_output_pairs)
    
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
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Print statistics
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Show reconstructions/selections every 5 epochs (or more frequently early on)
            if (epoch + 1) % 5 == 0 or epoch < 3:
                visualize_reconstruction(model, val_loader, device, num_samples=3, task_type=task_type, use_input_output_pairs=use_input_output_pairs)
            
            # Show messages every 10 epochs (only in communication mode)
            if (epoch + 1) % 10 == 0:
                analyze_messages(model, val_loader, device)
            
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
                }, os.path.join(config.SAVE_DIR, 'best_model.pth'))
                print('✓ Saved best model')
            
            # Save checkpoint every 10 epochs (keep only last 3 to save disk space)
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(config.SAVE_DIR, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'encoder_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'bottleneck_type': config.BOTTLENECK_TYPE,
                }, checkpoint_path)
                
                # Clean up old checkpoints
                keep_last = getattr(config, 'KEEP_LAST_CHECKPOINTS', 3)
                cleanup_old_checkpoints(config.SAVE_DIR, keep_last=keep_last)
        
        # Final visualization
        if task_type == 'selection':
            print("\n🎯 FINAL SELECTIONS:")
        else:
            print("\n🎯 FINAL RECONSTRUCTIONS:")
        visualize_reconstruction(model, val_loader, device, num_samples=5, task_type=task_type, use_input_output_pairs=use_input_output_pairs)
        
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


if __name__ == '__main__':
    main()