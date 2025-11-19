# Conv Channels Auto-Detection Fix

## Problem

The checkpoint `slot_attention_322.pth` was trained with 3 convolutional layers, each with **32 channels** instead of the default 128 channels. However, the checkpoint metadata only saved `hidden_dim=128`, not the actual per-layer channel configuration.

This caused a size mismatch error when loading:
```
RuntimeError: Error(s) in loading state_dict for ARCAutoencoder:
size mismatch for encoder.conv_layers.0.weight: copying a param with shape torch.Size([32, 10, 3, 3]) from checkpoint,
the shape in current model is torch.Size([128, 10, 3, 3]).
```

## Root Cause

The `ARCEncoder` supports two ways to specify convolutional layer dimensions:

1. **Simple**: Use `hidden_dim` (all layers have same dimension)
   ```python
   encoder = ARCEncoder(hidden_dim=128, num_conv_layers=3)
   # Creates: [128, 128, 128]
   ```

2. **Advanced**: Use `conv_channels` list (per-layer dimensions)
   ```python
   encoder = ARCEncoder(conv_channels=[32, 32, 32], num_conv_layers=3)
   # Creates: [32, 32, 32]
   ```

Your checkpoint used `conv_channels=[32, 32, 32]`, but this wasn't saved in the checkpoint metadata, so the loading code tried to use `hidden_dim=128` instead.

## Solution

Updated both visualization scripts to **automatically infer** the conv channels from the checkpoint's state dict:

### How It Works

1. **Load checkpoint** and check for saved `conv_channels` config
2. **If not found**, inspect the actual weight tensor shapes:
   ```python
   for i in range(num_conv_layers):
       weight = state_dict[f'encoder.conv_layers.{i}.weight']
       # Weight shape is [out_channels, in_channels, kernel_h, kernel_w]
       channels.append(weight.shape[0])  # Extract out_channels
   ```
3. **Use inferred channels** to create the encoder with correct architecture
4. **Load state dict** successfully!

### Files Modified

- ✅ `visualize_slot_attention.py` - Added conv_channels inference
- ✅ `visualize_cnn_features.py` - Added conv_channels inference  
- ✅ `train.py` - Now saves conv_channels to future checkpoints

### Example Output

When you run the visualization now, you'll see:

```
Loading checkpoint from: checkpoints/slot_attention_322.pth

Checkpoint info:
  Epoch: 322
  Bottleneck type: slot_attention
  Config from checkpoint: {'hidden_dim': 128, 'num_slots': 7, ...}
  Inferred conv_channels from state dict: [32, 32, 32]

Using configuration:
  num_slots: 7
  slot_dim: 64
  slot_iterations: 3
  hidden_dim: 128
  num_conv_layers: 3
  conv_channels: [32, 32, 32]  ← Automatically detected!

✓ Successfully loaded model state dict
✓ Model loaded successfully!
```

## Testing

Try running the visualization scripts now:

```bash
python visualize_slot_attention.py
python visualize_cnn_features.py
```

They should now automatically detect the correct architecture from your `slot_attention_322.pth` checkpoint without any manual configuration!

## For Future Checkpoints

Future training runs will save the `conv_channels` configuration, so this auto-detection won't be needed. But it's a good fallback for existing checkpoints.

## Manual Override (If Needed)

If auto-detection fails for some reason, you can still manually specify:

```python
# In main() function of visualization script:
override_conv_channels = [32, 32, 32]  # Specify explicitly
```

## Technical Details

### Conv Layer Weight Shape

PyTorch Conv2d weights have shape: `[out_channels, in_channels, kernel_h, kernel_w]`

For your checkpoint:
- Layer 0: `[32, 10, 3, 3]` → 32 output channels (input is 10 one-hot colors)
- Layer 1: `[32, 32, 3, 3]` → 32 output channels
- Layer 2: `[32, 32, 3, 3]` → 32 output channels

The inference code reads `shape[0]` to get the output channels for each layer: `[32, 32, 32]`

### Why This Matters

Using 32 channels instead of 128 makes the model:
- **Smaller**: ~4x fewer parameters in conv layers
- **Faster**: Less computation
- **Different capacity**: May affect performance

Understanding the actual architecture is important for interpreting results!

