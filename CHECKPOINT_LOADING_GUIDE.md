# Checkpoint Loading Guide

This guide explains how to work with checkpoints, especially when loading models trained with different configurations.

## Problem

When loading checkpoints, the visualization scripts need to know what model architecture was used during training (e.g., number of slots, slot dimensions, hidden dimensions). If these parameters don't match, you'll get errors like:

```
RuntimeError: Error(s) in loading state_dict...
size mismatch for slots_mu: copying a param with shape torch.Size([32, 64]) from checkpoint, 
the shape in current model is torch.Size([7, 64]).
```

## Solution

The visualization scripts have been updated to:

1. **Auto-detect configuration from checkpoint** (if saved)
2. **Use config.py as fallback**
3. **Allow manual override** of parameters

### Updated Files

- `visualize_slot_attention.py` - Updated with flexible checkpoint loading
- `visualize_cnn_features.py` - Updated with flexible checkpoint loading  
- `train.py` - Now saves configuration to checkpoints (for future training runs)
- `inspect_checkpoint.py` - New utility to inspect checkpoint contents

## Usage

### 1. Inspect Your Checkpoint

First, use the inspection utility to see what's in your checkpoint:

```bash
python inspect_checkpoint.py checkpoints/slot_attention_32.pth
```

This will show:
- Configuration parameters (if saved)
- Model architecture details
- Training epoch and metrics
- State dict structure

### 2. Run Visualization Scripts

The scripts will now try to auto-detect the configuration:

```bash
python visualize_slot_attention.py
python visualize_cnn_features.py
```

### 3. Manual Override (if needed)

If auto-detection fails or you know the correct parameters, you can manually override them in the `main()` function of each script:

```python
def main():
    # Configuration
    checkpoint_path = 'checkpoints/slot_attention_32.pth'
    output_dir = 'slot_attention_visualizations'
    num_grids = 10
    
    # Manual overrides (set to None for auto-detection)
    override_num_slots = 32      # If checkpoint has 32 slots
    override_slot_dim = 64       # If checkpoint has 64D slots
    override_slot_iterations = 3 # If checkpoint uses 3 iterations
    
    # ... rest of the code
```

## Common Checkpoint Configurations

Based on the filename `slot_attention_32.pth`, likely configurations:

### Option 1: 32 Slots (most likely)
```python
override_num_slots = 32
override_slot_dim = 64  # or check with inspect_checkpoint.py
override_slot_iterations = 3
```

### Option 2: 32-dimensional Slots
```python
override_num_slots = 7  # default
override_slot_dim = 32
override_slot_iterations = 3
```

### Option 3: Batch Size 32 (just a naming convention)
```python
override_num_slots = 7
override_slot_dim = 64
override_slot_iterations = 3
```

## Configuration Priority

The loading system uses this priority order:

1. **Manual override** (passed to `load_model()`)
2. **Checkpoint configuration** (if saved in checkpoint)
3. **config.py defaults** (fallback)

## Troubleshooting

### Error: "size mismatch for slots_mu"

This means the number of slots or slot dimension doesn't match.

**Solution:**
1. Run `python inspect_checkpoint.py <checkpoint_path>`
2. Look for the shape of `slots_mu` or `slots_logsigma`
3. Set `override_num_slots` and `override_slot_dim` accordingly

Example:
```
slots_mu: torch.Size([32, 64])
→ num_slots=32, slot_dim=64
```

### Error: "size mismatch for encoder.conv_layers"

This means the encoder architecture doesn't match (hidden_dim, num_conv_layers, etc.).

**Solution:**
1. Check your config.py for HIDDEN_DIM, NUM_CONV_LAYERS
2. These should match what was used during training
3. If you don't know, try common values:
   - HIDDEN_DIM: 128, 256, 512
   - NUM_CONV_LAYERS: 3 (most common)

### No Configuration in Checkpoint

If you have an old checkpoint without saved configuration:

**Solution:**
1. Check your git history or training logs for the config used
2. Use `inspect_checkpoint.py` to infer parameters from state dict shapes
3. Try different parameter combinations
4. For future training, use the updated `train.py` which saves config

## Future Checkpoints

Going forward, all checkpoints saved by `train.py` will include configuration parameters:

```python
checkpoint = {
    'epoch': ...,
    'model_state_dict': ...,
    'bottleneck_type': 'slot_attention',
    'num_slots': 7,
    'slot_dim': 64,
    'slot_iterations': 3,
    # ... etc
}
```

This makes loading much easier and more reliable!

## Example Workflow

```bash
# 1. Inspect your checkpoint
python inspect_checkpoint.py checkpoints/slot_attention_32.pth

# 2. Check what configuration was found
# If configuration is present: ✓ Scripts should work automatically
# If not: Note the inferred parameters from state dict shapes

# 3. Run visualizations (with manual override if needed)
python visualize_slot_attention.py

# 4. If you get errors, edit the main() function:
#    - Set override_num_slots, override_slot_dim, etc.
#    - Re-run the script

# 5. Once working, run CNN features visualization
python visualize_cnn_features.py
```

## Questions?

If you continue to have issues:

1. Share the output of `inspect_checkpoint.py`
2. Share the error message from the visualization script
3. Check your training logs for the config used when creating the checkpoint

