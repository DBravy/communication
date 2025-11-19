"""
Utility script to inspect checkpoint contents and configuration.

This helps debug loading issues by showing:
- Checkpoint structure and keys
- Model architecture configuration
- Saved epoch and training metrics
- State dict layer names and shapes
"""

import torch
import sys
import os


def inspect_checkpoint(checkpoint_path):
    """Inspect and display checkpoint contents."""
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    print("=" * 80)
    print(f"CHECKPOINT INSPECTION: {checkpoint_path}")
    print("=" * 80)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("\n✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"\n✗ Failed to load checkpoint: {e}")
        return
    
    # Check if it's a dict or raw state_dict
    print(f"\nCheckpoint type: {type(checkpoint)}")
    
    if not isinstance(checkpoint, dict):
        print("\n⚠️  Checkpoint is not a dictionary (might be raw state_dict)")
        print(f"   Available keys: {list(checkpoint.keys())[:10]}...")
        return
    
    # Display main keys
    print(f"\nMain checkpoint keys:")
    for key in checkpoint.keys():
        if key.endswith('_state_dict'):
            print(f"  - {key}: <state dict>")
        else:
            value = checkpoint[key]
            if isinstance(value, (int, float, str, bool)):
                print(f"  - {key}: {value}")
            elif isinstance(value, (list, tuple)):
                print(f"  - {key}: {value}")
            else:
                print(f"  - {key}: {type(value)}")
    
    # Extract training info
    print("\n" + "-" * 80)
    print("TRAINING INFORMATION")
    print("-" * 80)
    
    info_keys = ['epoch', 'val_loss', 'val_acc', 'train_loss', 'train_acc', 
                 'best_val_acc', 'best_val_loss']
    for key in info_keys:
        if key in checkpoint:
            print(f"  {key}: {checkpoint[key]}")
    
    # Extract model configuration
    print("\n" + "-" * 80)
    print("MODEL CONFIGURATION")
    print("-" * 80)
    
    config_keys = ['bottleneck_type', 'task_type', 'num_slots', 'slot_dim', 
                   'slot_iterations', 'slot_hidden_dim', 'slot_eps',
                   'hidden_dim', 'latent_dim', 'num_conv_layers', 
                   'num_colors', 'embedding_dim', 'max_grid_size',
                   'use_beta_vae', 'beta', 'vocab_size', 'max_message_length']
    
    found_config = False
    for key in config_keys:
        if key in checkpoint:
            print(f"  {key}: {checkpoint[key]}")
            found_config = True
    
    if not found_config:
        print("  ⚠️  No configuration parameters found in checkpoint")
        print("     You may need to manually specify parameters when loading")
    
    # Inspect state dicts
    print("\n" + "-" * 80)
    print("STATE DICTIONARIES")
    print("-" * 80)
    
    for key in ['model_state_dict', 'encoder_state_dict', 'optimizer_state_dict']:
        if key in checkpoint:
            state_dict = checkpoint[key]
            print(f"\n{key}:")
            print(f"  Total parameters: {len(state_dict)}")
            
            # Group by module
            modules = {}
            for param_name in state_dict.keys():
                module_name = param_name.split('.')[0]
                if module_name not in modules:
                    modules[module_name] = 0
                modules[module_name] += 1
            
            print(f"  Parameters by module:")
            for module_name, count in sorted(modules.items()):
                print(f"    - {module_name}: {count} parameters")
            
            # Show first few parameter names and shapes
            print(f"\n  First 10 parameters:")
            for i, (param_name, param_tensor) in enumerate(list(state_dict.items())[:10]):
                if hasattr(param_tensor, 'shape'):
                    print(f"    {i+1}. {param_name}: {param_tensor.shape}")
                else:
                    print(f"    {i+1}. {param_name}: {type(param_tensor)}")
    
    # Try to infer configuration from state dict structure
    if 'model_state_dict' in checkpoint and not found_config:
        print("\n" + "-" * 80)
        print("INFERRED CONFIGURATION (from state dict)")
        print("-" * 80)
        
        state_dict = checkpoint['model_state_dict']
        
        # Try to infer num_slots
        slot_keys = [k for k in state_dict.keys() if 'slot' in k.lower()]
        if slot_keys:
            print(f"\n  Found {len(slot_keys)} slot-related parameters")
            # Look for slots_mu or slots_logsigma to infer num_slots
            for key in slot_keys:
                if 'slots_mu' in key or 'slots_logsigma' in key:
                    shape = state_dict[key].shape
                    print(f"    {key}: {shape}")
                    if len(shape) >= 2:
                        print(f"    → Inferred num_slots: {shape[0]}, slot_dim: {shape[1]}")
                        break
        
        # Try to infer hidden_dim from conv layers
        conv_keys = [k for k in state_dict.keys() if 'conv' in k and 'weight' in k]
        if conv_keys:
            print(f"\n  Found {len(conv_keys)} conv layer weights")
            for key in conv_keys[:3]:
                shape = state_dict[key].shape
                print(f"    {key}: {shape}")
    
    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)
    
    # Provide recommendations
    print("\nRECOMMENDATIONS:")
    
    if found_config:
        print("  ✓ Checkpoint contains configuration - visualization scripts should auto-detect it")
    else:
        print("  ⚠️  No configuration in checkpoint - you may need to manually set parameters:")
        print("     - Set override_num_slots, override_slot_dim, etc. in main() function")
        print("     - Match parameters to the architecture used during training")
    
    if 'bottleneck_type' in checkpoint:
        bottleneck = checkpoint['bottleneck_type']
        print(f"  ✓ Bottleneck type: {bottleneck}")
        if bottleneck != 'slot_attention':
            print(f"     ⚠️  This checkpoint uses '{bottleneck}', not 'slot_attention'")
    
    print("\n")


def main():
    """Main function."""
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = 'checkpoints/slot_attention_32.pth'
        print(f"No checkpoint path provided, using default: {checkpoint_path}\n")
        print("Usage: python inspect_checkpoint.py <checkpoint_path>\n")
    
    inspect_checkpoint(checkpoint_path)


if __name__ == '__main__':
    main()

