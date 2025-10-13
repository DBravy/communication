# Background Reconstruction Training - Implementation Summary

## Overview
This implementation modifies the training pipeline to train reconstruction in the background during selection mode. This allows the model to learn both selection and reconstruction capabilities simultaneously, preparing it for later fine-tuning on specific puzzles.

## Key Changes

### 1. Model Architecture (`model.py`)

#### ARCAutoencoder Initialization
- **Before**: In selection mode, only created `ReceiverSelector` (communication) or `DecoderSelector` (autoencoder)
- **After**: Now creates BOTH the selector and a reconstructor:
  - Communication mode: `receiver` (selector) + `receiver_reconstructor` (reconstruction)
  - Autoencoder mode: `decoder` (selector) + `decoder_reconstructor` (reconstruction)

#### Forward Pass in Selection Mode
- **Returns**: Now returns 4 values instead of 3:
  - `selection_logits_list` - logits for selecting from candidates
  - `reconstruction_logits_list` - logits for reconstructing grids (NEW)
  - `actual_sizes` - grid sizes
  - `messages` - communication messages (or None for autoencoder)

- **Gradient Control**: During training, the soft messages/latent representations are **detached** before passing to the reconstructor:
  ```python
  if self.training:
      soft_message=soft_single.detach()  # Prevents gradients to sender
  ```
  This ensures reconstruction loss only updates the reconstructor, not the sender/encoder.

### 2. Training Loop (`train.py`)

#### train_epoch Function
- **Selection Loss**: Computed for all samples as before
- **Reconstruction Loss**: NEW - computed only for correctly selected samples:
  ```python
  if is_correct:  # Only when selection is correct
      # Compute reconstruction loss
      # Gradients already detached in forward pass
  ```
- **Combined Loss**: `total_loss = selection_loss + reconstruction_loss`
- **Gradient Flow**:
  - Selection loss → Updates sender + encoder + receiver (selector)
  - Reconstruction loss → Updates ONLY receiver_reconstructor (via detached gradients)

#### validate Function
- Updated to handle new 4-value return signature
- Tracks reconstruction loss for correctly selected samples during validation
- Combines selection and reconstruction losses for total validation loss

#### visualize_reconstruction Function
- Updated to handle new return signature
- Now displays reconstruction alongside selection when correct:
  - Shows selected candidates
  - Shows reconstruction from the message
  - Displays reconstruction accuracy metrics

### 3. Gradient Flow Diagram

```
Input Grid
    ↓
Encoder (trainable from selection loss)
    ↓
Sender (trainable from selection loss)
    ↓
Message (soft representation)
    ├─────────────────────────┐
    │                         │
    │ (no detach)             │ (.detach() during training)
    │                         │
    ↓                         ↓
ReceiverSelector          ReceiverReconstructor
(selection task)          (reconstruction task)
    ↓                         ↓
Selection Logits          Reconstruction Logits
    ↓                         ↓
Selection Loss            Reconstruction Loss
(if correct: used)        (only if selection correct)
    ↓                         ↓
Backprop through          Backprop ONLY to
entire chain              ReceiverReconstructor
```

## Benefits

1. **Simultaneous Training**: The model learns both tasks at once
2. **Isolated Gradient Flow**: Reconstruction training doesn't interfere with selection learning
3. **Transfer Learning Ready**: When switching to puzzle-solving (which requires reconstruction), the reconstructor is already trained
4. **Conditional Training**: Reconstruction only trains on correct selections, ensuring quality signals

## Usage

Simply set the config as usual:
```python
TASK_TYPE = 'selection'
USE_INPUT_OUTPUT_PAIRS = True  # or False for self-supervised
```

The background reconstruction happens automatically. During training, you'll see:
- Selection accuracy metrics (as before)
- Reconstruction visualizations when selection is correct
- Combined loss (selection + reconstruction)

## Future Use Case

After training in selection mode with background reconstruction:
1. The sender and encoder learn to create good representations
2. The receiver_reconstructor learns to decode those representations
3. When fine-tuning on a specific puzzle:
   - Load the trained model
   - Switch to reconstruction mode
   - The decoder already knows how to reconstruct from the learned message format
   - Fine-tune the entire system for the specific puzzle transformation

## Technical Notes

- **Memory**: Adds one additional receiver/decoder (approximately same size as the selector)
- **Computation**: Adds reconstruction forward pass for all samples (but loss only for correct ones)
- **Loss Weighting**: Currently 1:1 ratio between selection and reconstruction loss (can be adjusted if needed)

