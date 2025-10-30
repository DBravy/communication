# Quick Start: Generalization Testing

## Setup (1 minute)

1. Open `config.py` and configure:

```python
# Your main training dataset
DATASET_VERSION = 'V1'
DATASET_SPLIT = 'training'

# Generalization testing
GENERALIZATION_TEST_ENABLED = True
GENERALIZATION_TEST_DATASET_VERSION = 'V2'  # Test on V2 while training on V1
GENERALIZATION_TEST_DATASET_SPLIT = 'training'
GENERALIZATION_TEST_INTERVAL = 20  # Test every 20 epochs
GENERALIZATION_TEST_MAX_GRIDS = 100  # Use 100 grids for testing (faster)

# Make sure you're doing reconstruction
TASK_TYPE = 'reconstruction'
BOTTLENECK_TYPE = 'communication'  # or 'autoencoder'
```

2. Start the web app:
```bash
python app.py
```

3. Navigate to `http://localhost:5000`

4. Click "Start Main Training"

## What Happens

- Training proceeds normally on V1 dataset
- Every 20 epochs, the system automatically:
  - Pauses training briefly
  - Loads 100 grids from V2 dataset
  - Tests the model (no training, just evaluation)
  - Saves results
  - Resumes training

## Viewing Results

### In the Web Interface:
After epoch 20, a new green section appears showing:
- Which datasets are being used (training vs testing)
- A table with all test results
- Comparison of generalization metrics vs training/validation

### In the Terminal:
You'll see output like:
```
================================================================================
GENERALIZATION TEST: Testing on V2/training
================================================================================
Generalization Test Loss: 0.5234, Accuracy: 82.50%
================================================================================
```

### In the File System:
Results are saved to: `checkpoints/generalization_test_results.json`

## Understanding the Results

**Good Sign:**
```
Epoch 40:
  Train Acc: 88.30%
  Val Acc: 86.10%
  Gen Test Acc: 83.50%  ← Close to validation accuracy
```

**Warning Sign (Overfitting):**
```
Epoch 40:
  Train Acc: 95.30%
  Val Acc: 92.10%
  Gen Test Acc: 65.20%  ← Much lower than validation
```

## Common Configurations

### Fast testing (for debugging):
```python
GENERALIZATION_TEST_INTERVAL = 5  # Every 5 epochs
GENERALIZATION_TEST_MAX_GRIDS = 20  # Only 20 grids
```

### Thorough testing:
```python
GENERALIZATION_TEST_INTERVAL = 10  # Every 10 epochs
GENERALIZATION_TEST_MAX_GRIDS = None  # All available grids
```

### Disable testing:
```python
GENERALIZATION_TEST_ENABLED = False
```

## Tips

1. **Start with V1 → V2**: V1 has 400 training puzzles, V2 has 1000, so testing generalization from smaller to larger dataset is interesting

2. **Watch the gap**: If generalization accuracy is much lower than validation accuracy, your model might be overfitting to the training dataset

3. **Check early**: Use a smaller interval (e.g., 10 epochs) to catch problems sooner

4. **Limit grids initially**: Use 50-100 grids for testing while experimenting, increase for final runs

## Troubleshooting

**No results showing?**
- Check that `GENERALIZATION_TEST_ENABLED = True`
- Verify the test dataset path exists (V1/data/training or V2/data/training)
- Wait until epoch 20 (or your configured interval)

**Testing is slow?**
- Reduce `GENERALIZATION_TEST_MAX_GRIDS` to 50 or fewer
- Reduce `BATCH_SIZE` if memory is an issue

**Results don't make sense?**
- Ensure training and testing datasets are different
- Check that `TASK_TYPE = 'reconstruction'` (doesn't work with puzzle_solving yet)
- Verify both datasets exist and have grids in them

