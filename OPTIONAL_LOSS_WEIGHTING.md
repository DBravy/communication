# Optional: Configuring Reconstruction Loss Weight

By default, the reconstruction loss and selection loss have equal weight (1:1 ratio). If you want to adjust this, you can modify the weighting.

## Current Implementation

In `train.py`, lines 334-338:
```python
# Combine losses: selection loss for all samples + reconstruction loss for correct selections
loss = batch_loss / len(selection_logits_list)
if num_reconstructions > 0:
    reconstruction_loss_avg = reconstruction_loss_total / num_reconstructions
    # Add reconstruction loss (already detached from sender/encoder via .detach())
    loss = loss + reconstruction_loss_avg
```

## Option 1: Add Configuration Parameter

Add to `config.py`:
```python
# Loss weighting for background reconstruction
RECONSTRUCTION_LOSS_WEIGHT = 0.5  # Weight for reconstruction loss in selection mode
```

Then modify `train.py` (lines 337-338):
```python
# Add weighted reconstruction loss
loss = loss + config.RECONSTRUCTION_LOSS_WEIGHT * reconstruction_loss_avg
```

Also update in `validate()` function (similar location).

## Option 2: Adaptive Weighting

You could make the weight depend on selection accuracy:
```python
# In train.py, after computing batch_correct
selection_accuracy = batch_correct / batch_total if batch_total > 0 else 0

# Weight reconstruction more as selection improves
reconstruction_weight = selection_accuracy  # 0 to 1

if num_reconstructions > 0:
    reconstruction_loss_avg = reconstruction_loss_total / num_reconstructions
    loss = loss + reconstruction_weight * reconstruction_loss_avg
```

This would:
- Start with low reconstruction weight when selection is poor
- Gradually increase reconstruction weight as selection improves
- Ensure selection learning dominates early training

## Option 3: Curriculum Learning

Start with selection only, then gradually add reconstruction:
```python
# In config.py
RECONSTRUCTION_START_EPOCH = 10  # Start reconstruction training after this epoch

# In train.py train_epoch function
def train_epoch(model, dataloader, optimizer, criterion, device, temperature, 
                plotter=None, task_type='reconstruction', use_input_output_pairs=False,
                current_epoch=0):
    # ... existing code ...
    
    # Only add reconstruction loss after warmup period
    if num_reconstructions > 0 and current_epoch >= config.RECONSTRUCTION_START_EPOCH:
        reconstruction_loss_avg = reconstruction_loss_total / num_reconstructions
        loss = loss + reconstruction_loss_avg
```

## Recommended Settings

Based on the goals:

1. **Equal Priority** (default): `weight = 1.0`
   - Use when both tasks are equally important
   - Current implementation

2. **Selection First**: `weight = 0.1` to `0.5`
   - Use when selection learning is more important
   - Reconstruction is just auxiliary

3. **Adaptive**: Use Option 2
   - Use when you want reconstruction to gradually become more important
   - Good for curriculum learning

4. **Delayed Start**: Use Option 3 with `RECONSTRUCTION_START_EPOCH = 5-10`
   - Let selection stabilize first
   - Then add reconstruction training

## Implementation Example

To implement Option 1 (simplest):

1. Add to `config.py`:
```python
RECONSTRUCTION_LOSS_WEIGHT = 1.0  # Adjust as needed (0.1 to 1.0)
```

2. Modify `train.py` line 337-338:
```python
if num_reconstructions > 0:
    reconstruction_loss_avg = reconstruction_loss_total / num_reconstructions
    loss = loss + config.RECONSTRUCTION_LOSS_WEIGHT * reconstruction_loss_avg
```

3. Do the same in `validate()` function around line 542-544

## Monitoring

To decide on the right weight, monitor during training:
- Selection accuracy (primary metric)
- Reconstruction accuracy (when selection is correct)
- Combined loss

If selection accuracy is suffering, reduce reconstruction weight.
If selection is good but reconstruction is poor, increase reconstruction weight.

