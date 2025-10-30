# Generalization Testing Feature

## Overview
The system now supports automatic generalization testing during reconstruction training. This allows you to test how well your model trained on one dataset (e.g., V1) generalizes to an unseen dataset (e.g., V2).

## Configuration

### In `config.py`:

```python
# Generalization Testing (test on unseen dataset during training)
GENERALIZATION_TEST_ENABLED = True  # Whether to run generalization tests on unseen dataset
GENERALIZATION_TEST_DATASET_VERSION = 'V2'  # Dataset version to test on (e.g., 'V2' when training on 'V1')
GENERALIZATION_TEST_DATASET_SPLIT = 'training'  # Split to use for generalization testing
GENERALIZATION_TEST_INTERVAL = 20  # Run generalization test every N epochs
GENERALIZATION_TEST_MAX_GRIDS = 100  # Maximum number of grids to test (None = all grids)
```

### Key Settings:

- **GENERALIZATION_TEST_ENABLED**: Set to `True` to enable generalization testing
- **GENERALIZATION_TEST_DATASET_VERSION**: The dataset to test on (different from your training dataset)
- **GENERALIZATION_TEST_DATASET_SPLIT**: Which split to use ('training' or 'evaluation')
- **GENERALIZATION_TEST_INTERVAL**: How often to run tests (every N epochs)
- **GENERALIZATION_TEST_MAX_GRIDS**: Limit the number of test grids (for faster testing)

## How It Works

1. **During Training**: Every N epochs (default: 20), the system automatically:
   - Loads the unseen test dataset (e.g., V2 if training on V1)
   - Runs validation on this dataset without training on it
   - Records the loss and accuracy metrics
   - Saves results to `checkpoints/generalization_test_results.json`

2. **Results Storage**: All generalization test results are saved in JSON format with:
   - Epoch number when test was run
   - Generalization test loss and accuracy
   - Corresponding training and validation metrics
   - Number of test grids used
   - Timestamp

3. **Web Interface**: View results in real-time at `http://localhost:5000`:
   - Automatically displays when results are available
   - Shows comparison table with training, validation, and generalization metrics
   - Updates every 10 seconds during training
   - Displays most recent 10 tests

## Example Use Cases

### Training on V1, Testing on V2:
```python
# config.py
DATASET_VERSION = 'V1'
DATASET_SPLIT = 'training'

GENERALIZATION_TEST_ENABLED = True
GENERALIZATION_TEST_DATASET_VERSION = 'V2'
GENERALIZATION_TEST_DATASET_SPLIT = 'training'
GENERALIZATION_TEST_INTERVAL = 20
```

This will:
- Train your model on V1/training
- Every 20 epochs, test on V2/training (without training on it)
- Show how well the model generalizes to completely unseen puzzles

### Training on V2, Testing on V1:
```python
# config.py
DATASET_VERSION = 'V2'
DATASET_SPLIT = 'training'

GENERALIZATION_TEST_ENABLED = True
GENERALIZATION_TEST_DATASET_VERSION = 'V1'
GENERALIZATION_TEST_DATASET_SPLIT = 'evaluation'
GENERALIZATION_TEST_INTERVAL = 10
```

This will:
- Train your model on V2/training
- Every 10 epochs, test on V1/evaluation
- Useful for testing on the original ARC dataset

## Viewing Results

### In the Web App:
1. Start training with `python app.py`
2. Navigate to `http://localhost:5000`
3. Start reconstruction training
4. After the first generalization test (epoch 20 by default), a new section will appear showing:
   - Training and testing dataset information
   - Table comparing generalization test metrics with training/validation metrics
   - History of all tests run

### In the JSON File:
Results are also saved to `checkpoints/generalization_test_results.json`:
```json
{
  "training_dataset": "V1",
  "training_split": "training",
  "task_type": "reconstruction",
  "bottleneck_type": "communication",
  "history": [
    {
      "epoch": 20,
      "loss": 0.523,
      "accuracy": 82.5,
      "train_loss": 0.412,
      "train_acc": 88.3,
      "val_loss": 0.445,
      "val_acc": 86.1,
      "num_grids": 100,
      "dataset_version": "V2",
      "dataset_split": "training",
      "timestamp": 1234567890.123
    },
    ...
  ]
}
```

## Interpreting Results

### Good Generalization:
- Generalization test accuracy is close to validation accuracy
- Both metrics improve together over epochs
- Loss on test dataset decreases over time

### Poor Generalization (Overfitting):
- Validation accuracy is high but generalization test accuracy is low
- Training accuracy >> generalization test accuracy
- Model learned patterns specific to training dataset

### Example Output:
```
================================================================================
GENERALIZATION TEST: Testing on V2/training
================================================================================
Generalization Test Loss: 0.5234, Accuracy: 82.50%
================================================================================

Epoch 20:
  Train Loss: 0.4120, Train Acc: 88.30%
  Val Loss: 0.4450, Val Acc: 86.10%
  Gen Test Loss: 0.5234, Gen Acc: 82.50%
```

## Tips

1. **Start with fewer test grids**: Set `GENERALIZATION_TEST_MAX_GRIDS = 50` for faster testing during development

2. **Adjust test interval**: Use smaller intervals (e.g., 10) early in training to catch overfitting sooner

3. **Compare datasets**: Try both directions:
   - Train on V1, test on V2
   - Train on V2, test on V1
   
4. **Watch the gap**: Large gaps between validation and generalization accuracy suggest the model is learning dataset-specific patterns

5. **Disable for final training**: Set `GENERALIZATION_TEST_ENABLED = False` if you want faster training without periodic testing

## Technical Notes

- Generalization testing only works with reconstruction and selection tasks
- The test dataset is loaded fresh each time (not cached)
- Testing is done in evaluation mode (`model.eval()`)
- No gradients are computed during testing
- Results are saved after each test, so they persist even if training is interrupted

