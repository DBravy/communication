"""Finetune model on a single ARC puzzle."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm

import config
from puzzle_dataset import ARCSinglePuzzleDataset, collate_fn_puzzle
from model import ARCEncoder, ARCAutoencoder


def compute_accuracy(logits, target_grid, target_size):
    """Compute pixel accuracy for a single grid."""
    pred = logits.argmax(dim=1).squeeze(0)
    target = target_grid.squeeze(0)
    
    H, W = target_size
    
    # Only count pixels within actual size
    correct = (pred[:H, :W] == target[:H, :W]).sum().item()
    total = H * W
    
    return correct, total


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for input_grids, input_sizes, output_grids, output_sizes in tqdm(dataloader, desc='Training'):
        input_grids = input_grids.to(device)
        output_grids = output_grids.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits_list, _, _, _ = model(input_grids, input_sizes, temperature=1.0, 
                                    output_sizes=output_sizes)        
        # Compute loss for each sample
        batch_loss = 0
        batch_correct = 0
        batch_total = 0
        
        for i, logits in enumerate(logits_list):
            output_h, output_w = output_sizes[i]
            H, W = logits.shape[2], logits.shape[3]
            
            target_grid = output_grids[i:i+1, :H, :W]
            
            # Flatten for loss computation
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
            targets_flat = target_grid.reshape(-1)
            
            sample_loss = criterion(logits_flat, targets_flat)
            batch_loss += sample_loss
            
            # Compute accuracy
            sample_correct, sample_total = compute_accuracy(logits, target_grid, (output_h, output_w))
            batch_correct += sample_correct
            batch_total += sample_total
        
        loss = batch_loss / len(logits_list)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += batch_correct
        total += batch_total
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for input_grids, input_sizes, output_grids, output_sizes in dataloader:
            input_grids = input_grids.to(device)
            output_grids = output_grids.to(device)
            
            # Forward pass
            logits_list, _, _, _ = model(input_grids, input_sizes, temperature=1.0,
                                        output_sizes=output_sizes)
            
            # Compute loss for each sample
            batch_loss = 0
            batch_correct = 0
            batch_total = 0
            
            for i, logits in enumerate(logits_list):
                output_h, output_w = output_sizes[i]
                H, W = logits.shape[2], logits.shape[3]
                
                target_grid = output_grids[i:i+1, :H, :W]
                
                # Flatten for loss computation
                logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
                targets_flat = target_grid.reshape(-1)
                
                sample_loss = criterion(logits_flat, targets_flat)
                batch_loss += sample_loss
                
                # Compute accuracy
                sample_correct, sample_total = compute_accuracy(logits, target_grid, (output_h, output_w))
                batch_correct += sample_correct
                batch_total += sample_total
            
            loss = batch_loss / len(logits_list)
            total_loss += loss.item()
            correct += batch_correct
            total += batch_total
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Finetune on a single ARC puzzle')
    parser.add_argument('--puzzle_id', type=str, required=True, help='Puzzle ID to finetune on')
    parser.add_argument('--data_path', type=str, default='arc-agi_training_challenges.json', help='Path to ARC data')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to pretrained checkpoint')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--receiver_lr', type=float, default=None, help='Learning rate for receiver (if None, uses same as --lr)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--save_dir', type=str, default='puzzle_checkpoints', help='Directory to save finetuned models')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset
    print(f'\nLoading puzzle {args.puzzle_id}...')
    train_dataset = ARCSinglePuzzleDataset(args.data_path, args.puzzle_id, split='train')
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_puzzle,
        num_workers=0
    )
    
    # Create model
    print('\nCreating model...')
    receiver_gets_input_puzzle = getattr(config, 'RECEIVER_GETS_INPUT_PUZZLE', False)
    if config.BOTTLENECK_TYPE == 'communication' and receiver_gets_input_puzzle:
        print(f'Receiver will get input puzzle in addition to message symbols')
    
    encoder = ARCEncoder(
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_conv_layers=getattr(config, 'NUM_CONV_LAYERS', 3),
        conv_channels=getattr(config, 'ENCODER_CONV_CHANNELS', None)
    )
    model = ARCAutoencoder(
        encoder=encoder,
        vocab_size=config.VOCAB_SIZE if config.BOTTLENECK_TYPE == 'communication' else None,
        max_length=config.MAX_MESSAGE_LENGTH if config.BOTTLENECK_TYPE == 'communication' else None,
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        max_grid_size=config.MAX_GRID_SIZE,
        bottleneck_type=config.BOTTLENECK_TYPE,
        task_type='reconstruction',  # Always use reconstruction for puzzle solving
        num_conv_layers=getattr(config, 'NUM_CONV_LAYERS', 3),
        receiver_gets_input_puzzle=receiver_gets_input_puzzle,
        use_stop_token=getattr(config, 'USE_STOP_TOKEN', False),
        stop_token_id=getattr(config, 'STOP_TOKEN_ID', None),
        lstm_hidden_dim=getattr(config, 'LSTM_HIDDEN_DIM', None)
    ).to(device)
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f'\nLoading checkpoint from {args.checkpoint}...')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Check if this is a selection task checkpoint (has receiver_reconstructor)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        if 'receiver_reconstructor.symbol_embed.weight' in state_dict or \
           'decoder_reconstructor.fc_decode.weight' in state_dict:
            print('Detected selection task checkpoint - mapping background reconstruction weights...')
            
            # Map receiver_reconstructor -> receiver OR decoder_reconstructor -> decoder
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('receiver_reconstructor.'):
                    # Map background reconstruction receiver to main receiver
                    new_key = k.replace('receiver_reconstructor.', 'receiver.')
                    new_state_dict[new_key] = v
                    print(f'  Mapping: {k} -> {new_key}')
                elif k.startswith('decoder_reconstructor.'):
                    # Map background reconstruction decoder to main decoder
                    new_key = k.replace('decoder_reconstructor.', 'decoder.')
                    new_state_dict[new_key] = v
                    print(f'  Mapping: {k} -> {new_key}')
                elif k.startswith('encoder.'):
                    # Keep encoder weights
                    new_state_dict[k] = v
                elif k.startswith('sender.'):
                    # Keep sender weights (for communication mode)
                    new_state_dict[k] = v
                # Skip receiver.* and decoder.* keys (those are for selection, not reconstruction)
            
            # Load the mapped weights
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                print(f'⚠️  Missing keys (will be randomly initialized): {missing_keys[:5]}...')
            if unexpected_keys:
                print(f'⚠️  Unexpected keys (ignored): {unexpected_keys[:5]}...')
            
            print('✓ Loaded and mapped weights from selection checkpoint')
        else:
            # Standard reconstruction checkpoint
            try:
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print('✓ Loaded model weights')
                elif 'encoder_state_dict' in checkpoint:
                    # Load just the encoder
                    encoder.load_state_dict(checkpoint['encoder_state_dict'])
                    print('✓ Loaded encoder weights (rest initialized randomly)')
            except Exception as e:
                print(f'⚠️  Warning: Could not load checkpoint completely: {e}')
                print('  Continuing with partial weights...')
    else:
        print('\nTraining from scratch (no checkpoint provided)')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Use separate learning rate for receiver if specified
    if args.receiver_lr is not None and hasattr(model, 'receiver'):
        # Separate receiver parameters from other parameters
        receiver_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'receiver' in name:
                receiver_params.append(param)
            else:
                other_params.append(param)
        
        # Create optimizer with parameter groups
        optimizer = optim.Adam([
            {'params': other_params, 'lr': args.lr},
            {'params': receiver_params, 'lr': args.receiver_lr}
        ])
        print(f'Using separate learning rates: base={args.lr}, receiver={args.receiver_lr}')
    else:
        # Standard optimizer with single learning rate
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        if args.receiver_lr is not None:
            print(f'Warning: receiver_lr specified but model has no receiver, using single lr={args.lr}')
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True
    )
    
    # Training loop
    print(f'\nFinetuning on puzzle {args.puzzle_id}...')
    print(f'Training examples: {len(train_dataset)}')
    print(f'Epochs: {args.epochs}')
    if args.receiver_lr is not None and hasattr(model, 'receiver'):
        print(f'Learning rate: {args.lr} (base), {args.receiver_lr} (receiver)')
    else:
        print(f'Learning rate: {args.lr}')
    print(f'Early stopping: Will stop if accuracy reaches 99%')
    
    best_loss = float('inf')
    best_acc = 0.0
    early_stop_threshold = 99.0  # Stop if accuracy reaches 99%
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Update learning rate
        scheduler.step(train_loss)
        
        # Print statistics
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            best_acc = train_acc
            save_path = os.path.join(args.save_dir, f'{args.puzzle_id}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'puzzle_id': args.puzzle_id,
                'bottleneck_type': config.BOTTLENECK_TYPE,
            }, save_path)
            
            if (epoch + 1) % 50 == 0:
                print(f'  ✓ Saved best model (loss: {best_loss:.4f}, acc: {best_acc:.2f}%)')
        
        # Early stopping check
        if train_acc >= early_stop_threshold:
            print(f'\n✓ Early stopping triggered! Reached {train_acc:.2f}% accuracy (threshold: {early_stop_threshold}%)')
            print(f'  Stopped at epoch {epoch+1}/{args.epochs}')
            break
    
    # Save final model
    final_path = os.path.join(args.save_dir, f'{args.puzzle_id}_final.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'puzzle_id': args.puzzle_id,
        'bottleneck_type': config.BOTTLENECK_TYPE,
    }, final_path)
    
    print(f'\n✓ Finetuning complete!')
    print(f'  Best loss: {best_loss:.4f}')
    print(f'  Final model saved to: {final_path}')
    print(f'  Best model saved to: {save_path}')


if __name__ == '__main__':
    main()