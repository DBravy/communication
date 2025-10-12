# Checkpoint Saving Guide

This guide explains how to save and load checkpoints during training with the ARC communication system.

## Features

- **Save at any time**: Save checkpoints during training without stopping the training process
- **Named checkpoints**: Give your checkpoints custom names for easy identification
- **Full configuration**: All model architecture and training settings are saved with each checkpoint
- **Easy resumption**: Load checkpoints to resume training or fine-tune with different settings

## Saving Checkpoints from the Web App

### 1. During Training

When training is running (not pretraining), you can save a checkpoint at any time:

**Using curl:**
```bash
curl -X POST http://localhost:5002/save_checkpoint \
  -H "Content-Type: application/json" \
  -d '{"name": "my_checkpoint_name"}'
```

**Using Python requests:**
```python
import requests
response = requests.post('http://localhost:5002/save_checkpoint',
                        json={'name': 'my_checkpoint_name'})
print(response.json())
```

The checkpoint will be saved as `checkpoints/my_checkpoint_name.pth`.

### 2. Default Name

If you don't provide a name, a default name will be generated:
```bash
curl -X POST http://localhost:5002/save_checkpoint
```

This will save as `checkpoints/manual_checkpoint_epoch_X.pth`.

### 3. List All Checkpoints

View all saved checkpoints:
```bash
curl http://localhost:5002/list_checkpoints
```

This returns detailed information about each checkpoint including:
- Filename and file size
- Epoch and batch when saved
- Task type and bottleneck type
- Model architecture details
- Validation metrics (if available)

## What's Saved in Each Checkpoint

Each checkpoint contains:

### Training State
- Current epoch and batch number
- Validation loss and accuracy (when available)

### Model Weights
- Full model state dict
- Encoder state dict (separately for easy access)
- Optimizer state dict (for exact training resumption)

### Task Configuration
- Task type (reconstruction, selection, puzzle_classification)
- Bottleneck type (communication, autoencoder)
- Number of distractors
- Whether using input-output pairs

### Model Architecture
- Hidden dimensions
- Latent dimensions
- Number of convolutional layers
- Vocabulary size (for communication mode)
- Max message length (for communication mode)

### Training Hyperparameters
- Batch size
- Learning rate
- Temperature (for Gumbel-softmax)

### Data Configuration
- Max grids used
- Grid size filter settings

## Loading and Using Checkpoints

### Inspect Checkpoint Info

```bash
python load_checkpoint.py checkpoints/my_checkpoint_name.pth --info-only
```

This displays all the configuration and metadata stored in the checkpoint.

### Load Checkpoint in Python

```python
import torch
from model import ARCEncoder, ARCAutoencoder

# Load checkpoint
checkpoint = torch.load('checkpoints/my_checkpoint_name.pth')

# Create encoder with saved architecture
encoder = ARCEncoder(
    num_colors=checkpoint['num_colors'],
    embedding_dim=checkpoint['embedding_dim'],
    hidden_dim=checkpoint['hidden_dim'],
    latent_dim=checkpoint['latent_dim'],
    num_conv_layers=checkpoint['num_conv_layers']
)

# Create full model
model = ARCAutoencoder(
    encoder=encoder,
    vocab_size=checkpoint['vocab_size'],
    max_length=checkpoint['max_message_length'],
    num_colors=checkpoint['num_colors'],
    embedding_dim=checkpoint['embedding_dim'],
    hidden_dim=checkpoint['hidden_dim'],
    max_grid_size=checkpoint['max_grid_size'],
    bottleneck_type=checkpoint['bottleneck_type'],
    task_type=checkpoint['task_type'],
    num_conv_layers=checkpoint['num_conv_layers'],
    num_classes=None  # Set if using puzzle_classification
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode

print(f"Loaded model from epoch {checkpoint['epoch']}")
```

### Resume Training from Checkpoint

```python
import torch.optim as optim

# ... (load model as shown above) ...

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=checkpoint['learning_rate'])

# Load optimizer state for exact resumption
if checkpoint.get('optimizer_state_dict'):
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Continue training from the saved epoch
starting_epoch = checkpoint['epoch']
# ... continue training loop ...
```

### Fine-tune with Different Settings

You can load a checkpoint but use different hyperparameters for fine-tuning:

```python
# Load model (as shown above)
model.load_state_dict(checkpoint['model_state_dict'])

# Create optimizer with NEW learning rate
new_lr = 1e-6  # Lower learning rate for fine-tuning
optimizer = optim.Adam(model.parameters(), lr=new_lr)

# Don't load old optimizer state - start fresh
# ... train with new settings ...
```

## Best Practices

1. **Naming Convention**: Use descriptive names that include:
   - Task type
   - Key hyperparameters
   - Performance indicators
   - Example: `selection_vocab25_len2_acc85`

2. **Save Before Major Changes**: Before changing config or stopping training, save a checkpoint.

3. **Disk Space**: Checkpoints are ~100-500 MB each. Clean up old checkpoints periodically:
   ```bash
   # List checkpoints by size
   ls -lh checkpoints/*.pth
   
   # Remove old checkpoints
   rm checkpoints/old_checkpoint_name.pth
   ```

4. **Testing**: After saving, verify the checkpoint loads correctly:
   ```bash
   python load_checkpoint.py checkpoints/my_checkpoint_name.pth --info-only
   ```

## Comparing with Pretrained Encoder Checkpoints

The system has two types of checkpoints:

1. **Pretrained Encoder Checkpoints** (from pretraining):
   - Saved automatically during pretraining
   - Only contain encoder weights
   - Named: `pretrained_encoder_binary.pth`, `pretrained_encoder_selection.pth`, etc.
   - Used as initialization for main training

2. **Full System Checkpoints** (this feature):
   - Saved on-demand during main training
   - Contain entire model (sender + receiver) and optimizer
   - Include complete configuration
   - Used for resuming training or fine-tuning

## API Reference

### POST `/save_checkpoint`

Request a checkpoint save during training.

**Request Body:**
```json
{
  "name": "checkpoint_name"  // optional
}
```

**Response:**
```json
{
  "status": "requested",
  "checkpoint_name": "checkpoint_name",
  "message": "Checkpoint save requested: checkpoint_name"
}
```

**Error Cases:**
- Training not running: 400 error
- Pretraining mode: 400 error (only works during main training)

### GET `/list_checkpoints`

List all saved checkpoints with metadata.

**Response:**
```json
{
  "checkpoints": [
    {
      "filename": "my_checkpoint.pth",
      "path": "checkpoints/my_checkpoint.pth",
      "size_mb": 234.5,
      "modified": "2025-10-12 14:30:00",
      "epoch": 50,
      "batch": 120,
      "task_type": "selection",
      "bottleneck_type": "communication",
      "val_loss": 0.234,
      "val_acc": 87.5,
      "hidden_dim": 128,
      "latent_dim": 128,
      "vocab_size": 25
    }
  ]
}
```

## Troubleshooting

### Checkpoint not saving
- Check that training is running: `curl http://localhost:5002/status`
- Make sure you're in main training mode, not pretraining
- Check disk space: `df -h`

### Can't load checkpoint
- Verify file exists: `ls checkpoints/`
- Check file isn't corrupted: `python load_checkpoint.py checkpoints/name.pth --info-only`
- Ensure you have the correct model.py version

### Architecture mismatch
If you get an error loading weights, the checkpoint was saved with a different model architecture. Check:
- `num_conv_layers` matches
- `hidden_dim` and `latent_dim` match
- Task type is compatible

## Example Workflow

```bash
# 1. Start training via web app
curl -X POST http://localhost:5002/start_train

# 2. Monitor training in browser at http://localhost:5002

# 3. When you see good performance, save a checkpoint
curl -X POST http://localhost:5002/save_checkpoint \
  -H "Content-Type: application/json" \
  -d '{"name": "selection_good_performance_epoch_45"}'

# 4. Continue training...

# 5. Later, inspect the checkpoint
python load_checkpoint.py checkpoints/selection_good_performance_epoch_45.pth --info-only

# 6. Load in your own script for inference or fine-tuning
```

## Notes

- Checkpoint saving is **non-blocking** - training continues while saving
- Saves typically take 1-2 seconds
- You can save multiple checkpoints during a single training run
- Each checkpoint is independent - no incremental/delta saving

