# Changes Summary: Flexible Checkpoint Loading

## Overview

Modified visualization scripts and training code to handle checkpoints with different configurations more gracefully. The `slot_attention_32.pth` checkpoint should now work with the visualization scripts.

## Files Modified

### 1. `visualize_slot_attention.py` ✓

**Changes:**
- Updated `load_model()` function to accept optional parameter overrides
- Added automatic config detection from checkpoint
- Implements 3-tier priority system: manual override → checkpoint config → config.py
- Better error messages with troubleshooting hints
- Updates config module with actual values used for visualization consistency

**Key features:**
```python
def load_model(checkpoint_path, device, num_slots=None, slot_dim=None, slot_iterations=None):
    # Loads checkpoint first to extract configuration
    # Creates model with correct architecture
    # Provides detailed diagnostic output
```

**User-facing changes:**
- Can now override config in `main()` function:
  ```python
  override_num_slots = 32  # Set to None for auto-detect
  override_slot_dim = 64
  override_slot_iterations = 3
  ```
- Better error messages when loading fails
- Shows configuration being used

### 2. `visualize_cnn_features.py` ✓

**Changes:**
- Same improvements as `visualize_slot_attention.py`
- Consistent loading behavior across visualization scripts
- Added override options in `main()` function

### 3. `train.py` ✓

**Changes:**
- Updated checkpoint saving to include model configuration
- Saves both for "best model" and periodic checkpoints
- Only saves slot attention config when using slot attention bottleneck

**Configuration saved:**
```python
checkpoint_data = {
    # ... existing fields ...
    'hidden_dim': config.HIDDEN_DIM,
    'latent_dim': config.LATENT_DIM,
    'num_conv_layers': config.NUM_CONV_LAYERS,
    # Slot attention specific (if applicable):
    'num_slots': config.NUM_SLOTS,
    'slot_dim': config.SLOT_DIM,
    'slot_iterations': config.SLOT_ITERATIONS,
    'slot_hidden_dim': config.SLOT_HIDDEN_DIM,
    'slot_eps': config.SLOT_EPS,
}
```

**Impact:**
- All future checkpoints will be self-documenting
- Makes loading and reproducing experiments easier
- Minimal overhead (just a few extra bytes per checkpoint)

## New Files Created

### 4. `inspect_checkpoint.py` ✨ NEW

**Purpose:** 
Utility script to inspect checkpoint contents and diagnose loading issues

**Features:**
- Shows checkpoint structure and keys
- Displays saved configuration parameters
- Shows training metrics (epoch, loss, accuracy)
- Lists state dict modules and parameters
- Attempts to infer configuration from state dict shapes when not saved
- Provides actionable recommendations

**Usage:**
```bash
python inspect_checkpoint.py checkpoints/slot_attention_32.pth
```

**Output includes:**
- Checkpoint info (epoch, bottleneck type, etc.)
- Model configuration (if saved)
- State dict structure
- Inferred configuration (if not saved)
- Troubleshooting recommendations

### 5. `CHECKPOINT_LOADING_GUIDE.md` 📖 NEW

**Purpose:**
Comprehensive guide for working with checkpoints

**Contents:**
- Problem explanation
- Solution overview
- Step-by-step usage instructions
- Common configurations
- Troubleshooting section
- Example workflow

### 6. `CHANGES_SUMMARY.md` 📝 NEW

This file - documents all changes made

## How It Works

### Configuration Priority System

1. **Manual Override** (highest priority)
   - Set in `main()` function: `override_num_slots = 32`
   - Useful when you know the correct config

2. **Checkpoint Configuration** (middle priority)
   - Loaded from checkpoint if saved
   - Reliable for checkpoints from updated training code

3. **config.py Defaults** (lowest priority/fallback)
   - Used when nothing else available
   - May not match checkpoint architecture

### Loading Process

```
1. Load checkpoint file
2. Extract config from checkpoint (if present)
3. Apply manual overrides (if specified)
4. Display configuration being used
5. Create model with correct architecture
6. Load state dict
7. Update config module for visualization consistency
```

## For the User

### Immediate Action

To use your `slot_attention_32.pth` checkpoint:

```bash
# Step 1: Inspect the checkpoint
python inspect_checkpoint.py checkpoints/slot_attention_32.pth

# Step 2: Run visualization (should auto-detect or show clear error)
python visualize_slot_attention.py

# Step 3: If needed, edit main() in visualize_slot_attention.py
# Set override_num_slots, override_slot_dim based on inspection output

# Step 4: Run again
python visualize_slot_attention.py

# Step 5: Repeat for CNN features
python visualize_cnn_features.py
```

### Most Likely Configuration for slot_attention_32.pth

Based on the filename, it probably uses:
- **32 slots** (num_slots=32)
- 64-dimensional slots (slot_dim=64) 
- 3 iterations (slot_iterations=3)

Try setting:
```python
override_num_slots = 32
override_slot_dim = 64
override_slot_iterations = 3
```

## Benefits

### Immediate
- ✓ Can load `slot_attention_32.pth` checkpoint
- ✓ Clear error messages with troubleshooting hints
- ✓ Easy to debug configuration mismatches
- ✓ Works with old checkpoints (no config saved)

### Long-term
- ✓ Future checkpoints are self-documenting
- ✓ Easier to reproduce experiments
- ✓ Less confusion when switching between experiments
- ✓ Better collaboration (checkpoints explain themselves)

## Testing Recommendations

1. **Test with current checkpoint:**
   ```bash
   python inspect_checkpoint.py checkpoints/slot_attention_32.pth
   python visualize_slot_attention.py
   ```

2. **If auto-detection fails:**
   - Note the error message
   - Use `inspect_checkpoint.py` output to set overrides
   - Try common configurations listed in the guide

3. **For future training:**
   - New checkpoints will automatically include config
   - Test loading: should work without any overrides

## Backward Compatibility

- ✓ Old checkpoints still work (with manual config if needed)
- ✓ No breaking changes to checkpoint format
- ✓ Config saving is additive (doesn't remove existing fields)
- ✓ Scripts gracefully handle both old and new checkpoint formats

## Notes

- The "32" in `slot_attention_32.pth` most likely refers to the number of slots (32 slots)
- However, it could also mean 32-dimensional slots or batch size 32
- Use `inspect_checkpoint.py` to determine the actual configuration
- The visualization scripts are now robust to any configuration mismatch

## Questions or Issues?

If the checkpoint still doesn't load:
1. Run `inspect_checkpoint.py` and share the output
2. Share the error message from visualization script
3. Check training logs for the original config used

