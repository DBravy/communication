"""Utility script for loading and resuming training from checkpoints.

This script demonstrates how to:
1. Load a checkpoint with full config
2. Resume training from a checkpoint
3. Fine-tune a model from a checkpoint with different settings
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse

import config
from dataset import ARCDataset, collate_fn, collate_fn_puzzle_classification
from model import ARCEncoder, ARCAutoencoder


def load_checkpoint_info(checkpoint_path):
    """Load and display checkpoint information."""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return None
    
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n" + "="*80)
    print("CHECKPOINT INFORMATION")
    print("="*80)
    
    # Training state
    print("\nTraining State:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Batch: {checkpoint.get('batch', 'N/A')}")
    if checkpoint.get('val_loss') is not None:
        print(f"  Validation Loss: {checkpoint['val_loss']:.4f}")
    if checkpoint.get('val_acc') is not None:
        print(f"  Validation Accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Task configuration
    print("\nTask Configuration:")
    print(f"  Task Type: {checkpoint.get('task_type', 'N/A')}")
    print(f"  Bottleneck Type: {checkpoint.get('bottleneck_type', 'N/A')}")
    print(f"  Num Distractors: {checkpoint.get('num_distractors', 'N/A')}")
    print(f"  Use Input-Output Pairs: {checkpoint.get('use_input_output_pairs', 'N/A')}")
    
    # Model architecture
    print("\nModel Architecture:")
    print(f"  Hidden Dim: {checkpoint.get('hidden_dim', 'N/A')}")
    print(f"  Latent Dim: {checkpoint.get('latent_dim', 'N/A')}")
    print(f"  Num Conv Layers: {checkpoint.get('num_conv_layers', 'N/A')}")
    if checkpoint.get('vocab_size') is not None:
        print(f"  Vocab Size: {checkpoint['vocab_size']}")
        print(f"  Max Message Length: {checkpoint.get('max_message_length', 'N/A')}")
    
    # Training hyperparameters
    print("\nTraining Hyperparameters:")
    print(f"  Batch Size: {checkpoint.get('batch_size', 'N/A')}")
    print(f"  Learning Rate: {checkpoint.get('learning_rate', 'N/A')}")
    
    # Data configuration
    print("\nData Configuration:")
    print(f"  Max Grids: {checkpoint.get('max_grids', 'N/A')}")
    print(f"  Filter Grid Size: {checkpoint.get('filter_grid_size', 'N/A')}")
    
    print("="*80 + "\n")
    
    return checkpoint


def create_model_from_checkpoint(checkpoint, device='cpu'):
    """Create and load a model from checkpoint."""
    
    # Create encoder with checkpoint architecture
    encoder = ARCEncoder(
        num_colors=checkpoint.get('num_colors', config.NUM_COLORS),
        embedding_dim=checkpoint.get('embedding_dim', config.EMBEDDING_DIM),
        hidden_dim=checkpoint['hidden_dim'],
        latent_dim=checkpoint['latent_dim'],
        num_conv_layers=checkpoint['num_conv_layers']
    )
    
    # Determine num_classes for puzzle classification
    num_classes = None
    task_type = checkpoint.get('task_type', 'reconstruction')
    
    # Create full model
    model = ARCAutoencoder(
        encoder=encoder,
        vocab_size=checkpoint.get('vocab_size'),
        max_length=checkpoint.get('max_message_length'),
        num_colors=checkpoint.get('num_colors', config.NUM_COLORS),
        embedding_dim=checkpoint.get('embedding_dim', config.EMBEDDING_DIM),
        hidden_dim=checkpoint['hidden_dim'],
        max_grid_size=checkpoint.get('max_grid_size', config.MAX_GRID_SIZE),
        bottleneck_type=checkpoint['bottleneck_type'],
        task_type=task_type,
        num_conv_layers=checkpoint['num_conv_layers'],
        num_classes=num_classes
    ).to(device)
    
    # Load model weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Successfully loaded model weights")
    except Exception as e:
        print(f"✗ Error loading model weights: {e}")
        return None
    
    return model


def resume_training_from_checkpoint(checkpoint_path, num_additional_epochs=10):
    """Resume training from a checkpoint."""
    
    checkpoint = load_checkpoint_info(checkpoint_path)
    if checkpoint is None:
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create model from checkpoint
    model = create_model_from_checkpoint(checkpoint, device)
    if model is None:
        return
    
    # Create optimizer and load state
    optimizer = optim.Adam(model.parameters(), lr=checkpoint['learning_rate'])
    if checkpoint.get('optimizer_state_dict') is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Successfully loaded optimizer state")
        except Exception as e:
            print(f"⚠ Warning: Could not load optimizer state: {e}")
            print("  Using fresh optimizer state")
    
    # Load dataset with checkpoint configuration
    task_type = checkpoint['task_type']
    num_distractors = checkpoint.get('num_distractors', 0) if task_type == 'selection' else 0
    track_puzzle_ids = task_type == 'puzzle_classification'
    use_input_output_pairs = checkpoint.get('use_input_output_pairs', False)
    
    print(f"\nLoading dataset with checkpoint configuration...")
    dataset = ARCDataset(
        config.DATA_PATH,
        min_size=config.MIN_GRID_SIZE,
        filter_size=checkpoint.get('filter_grid_size'),
        max_grids=checkpoint.get('max_grids'),
        num_distractors=num_distractors,
        track_puzzle_ids=track_puzzle_ids,
        use_input_output_pairs=use_input_output_pairs
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    from functools import partial
    if task_type == 'puzzle_classification':
        collate_fn_for_task = collate_fn_puzzle_classification
    else:
        collate_fn_for_task = partial(collate_fn, num_distractors=num_distractors, 
                                      use_input_output_pairs=use_input_output_pairs)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=checkpoint['batch_size'],
        shuffle=True,
        collate_fn=collate_fn_for_task,
        num_workers=0
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} total samples")
    print(f"  Train: {train_size}, Val: {val_size}")
    
    # Continue training
    starting_epoch = checkpoint['epoch']
    print(f"\n{'='*80}")
    print(f"RESUMING TRAINING FROM EPOCH {starting_epoch}")
    print(f"Will train for {num_additional_epochs} more epochs")
    print(f"{'='*80}\n")
    
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    # Training loop (simplified example)
    for epoch in range(num_additional_epochs):
        total_loss = 0
        for batch_idx, batch_data in enumerate(train_loader):
            # This is a simplified training loop
            # Implement full training logic based on task_type
            # (Similar to train_worker in app.py)
            pass
        
        print(f"Epoch {starting_epoch + epoch + 1}: Loss = {total_loss / len(train_loader):.4f}")
    
    print("\nTraining resumed successfully!")
    print("(Note: This is a simplified example. Implement full training logic as needed.)")


def main():
    parser = argparse.ArgumentParser(description='Load and inspect checkpoints')
    parser.add_argument('checkpoint_path', type=str, help='Path to checkpoint file')
    parser.add_argument('--info-only', action='store_true', 
                       help='Only display checkpoint info without loading model')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of additional epochs to train (if --resume)')
    
    args = parser.parse_args()
    
    if args.info_only:
        load_checkpoint_info(args.checkpoint_path)
    elif args.resume:
        resume_training_from_checkpoint(args.checkpoint_path, args.epochs)
    else:
        # Load and display info, then create model
        checkpoint = load_checkpoint_info(args.checkpoint_path)
        if checkpoint:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = create_model_from_checkpoint(checkpoint, device)
            if model:
                print("✓ Model created and loaded successfully!")
                print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == '__main__':
    main()

