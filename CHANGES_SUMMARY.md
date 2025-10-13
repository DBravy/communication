# Background Reconstruction Training - Changes Summary

## What Was Changed

This implementation allows the model to train reconstruction in the background while in selection mode. The reconstruction loss only applies to the reconstructor module, not the entire system, ensuring that selection training isn't disrupted.

## Files Modified

### 1. `app.py`

#### Changes to `get_selections()` (line 453)
- **Updated model call**: Now unpacks 4 values instead of 3 for selection mode
- Added `reconstruction_logits_list` to the unpacking

#### Changes to `train_worker()` - Selection Mode (line 1118)
- **Updated model call**: Now unpacks 4 values instead of 3 for selection mode
- Added `reconstruction_logits_list` to the unpacking

### 2. `model.py`

#### Changes to `ARCAutoencoder.__init__` (lines 525-564)
- **Added**: When `task_type='selection'`, now creates both receivers/decoders:
  - Communication mode: Creates `receiver_reconstructor` (ReceiverAgent) alongside `receiver` (ReceiverSelector)
  - Autoencoder mode: Creates `decoder_reconstructor` (ARCDecoder) alongside `decoder` (DecoderSelector)

#### Changes to `ARCAutoencoder.forward` - Selection Mode (lines 611-680)
- **Changed return signature**: Now returns 4 values instead of 3:
  - `selection_logits_list` (as before)
  - `reconstruction_logits_list` (NEW)
  - `actual_sizes` (as before)
  - `messages` (as before)

- **Added reconstruction pass**: For each sample, also computes reconstruction:
  ```python
  # During training, detach soft messages to prevent gradient flow to sender
  if self.training:
      soft_message=soft_single.detach()
  ```

### 3. `train.py`

#### Changes to `train_epoch` - Selection Mode (lines 268-338)
- **Updated model call**: Now unpacks 4 values instead of 3
- **Added reconstruction loss computation**: 
  - Only computed when selection is **correct**
  - Applied to the selected grid (output in I/O mode, input in self-supervised)
  - Tracked separately: `reconstruction_loss_total`, `reconstruction_correct`, `reconstruction_total`
- **Combined loss**: `loss = selection_loss + reconstruction_loss_avg`

#### Changes to `validate` - Selection Mode (lines 487-544)
- **Updated model call**: Now unpacks 4 values instead of 3
- **Added reconstruction loss tracking**: Same logic as training, for validation metrics

#### Changes to `visualize_reconstruction` (lines 117-183)
- **Updated model call**: Now unpacks 4 values instead of 3
- **Added reconstruction visualization**: When selection is correct, shows:
  - The reconstructed grid
  - Reconstruction accuracy metrics

## How It Works

### Gradient Flow

```
Training in Selection Mode:

1. Selection Path (full gradient flow):
   Input → Encoder → Sender → Message → ReceiverSelector → Selection Loss
   └─────────────── All modules updated ──────────────────┘

2. Reconstruction Path (isolated gradient flow):
   Input → Encoder → Sender → Message → .detach() → ReceiverReconstructor → Recon Loss
                                         └─── Only this module updated ────────┘
```

### Key Mechanism: `.detach()`

In `model.py`, line 639:
```python
if self.training:
    soft_message=soft_single.detach()  # Prevents gradients to sender
```

This ensures that reconstruction loss only updates the `receiver_reconstructor`, not the sender or encoder.

### Loss Combination

In selection mode:
- **Selection loss**: Always computed for all samples
- **Reconstruction loss**: Only computed for correctly selected samples
- **Total loss**: `selection_loss + reconstruction_loss` (1:1 ratio)

This means:
- Selection learning is the primary objective
- Reconstruction learns as a secondary task, only on successful selections
- Reconstruction doesn't interfere with selection learning (isolated gradients)

## Benefits

1. **Dual Learning**: Model learns both selection and reconstruction simultaneously
2. **Gradient Isolation**: Reconstruction training doesn't interfere with selection learning
3. **Transfer Ready**: When switching to puzzle solving mode, the reconstructor is already trained
4. **Quality Signals**: Reconstruction only trains on correct selections, ensuring good training signals

## Usage

No changes needed to existing workflow! Simply use the model as before:

```python
# config.py
TASK_TYPE = 'selection'
USE_INPUT_OUTPUT_PAIRS = True  # Train on input→output pairs
```

The background reconstruction happens automatically.

## Testing

Run the test script to verify the implementation:
```bash
python test_background_reconstruction.py
```

This will verify:
1. Both receivers/decoders are created
2. Forward pass returns correct outputs
3. Gradient flow is properly isolated

## Training Output

During training, you'll now see:
- Selection accuracy (as before)
- When visualizing, reconstruction is shown for correct selections
- Combined loss (selection + reconstruction)

Example visualization output for correct selections:
```
📊 METRICS:
   Selection: ✓ CORRECT
   Confidence: 95.32%
   Reconstruction accuracy: 87.45% (1748/2000 correct)
```

## Future Workflow

1. **Train in selection mode** (current):
   - Model learns to select correct outputs from candidates
   - Reconstructor learns to decode messages (in background)

2. **Fine-tune for puzzle solving** (future):
   - Switch to reconstruction mode
   - Load the trained model (sender + encoder + reconstructor)
   - Fine-tune on specific puzzle transformations
   - The reconstructor already knows how to decode the message format!

## Notes

- **Memory overhead**: Adds one additional receiver/decoder (~same size as selector)
- **Computation overhead**: Additional forward pass through reconstructor for all samples
- **Loss weighting**: Currently 1:1 ratio (can be adjusted by modifying line 337-338 in train.py)

