# Generalization Testing Implementation - Summary of Changes

## Overview
Added the ability to test model generalization on an unseen dataset during training. For example, when training on V1, you can now automatically test on V2 every N epochs to see how well the model generalizes.

## Files Modified

### 1. `config.py` - Configuration Options
**Added:**
- `GENERALIZATION_TEST_ENABLED`: Enable/disable generalization testing
- `GENERALIZATION_TEST_DATASET_VERSION`: Dataset to test on (e.g., 'V2' when training on 'V1')
- `GENERALIZATION_TEST_DATASET_SPLIT`: Split to use for testing ('training' or 'evaluation')
- `GENERALIZATION_TEST_INTERVAL`: How often to run tests (default: every 20 epochs)
- `GENERALIZATION_TEST_MAX_GRIDS`: Limit number of test grids (default: 100, None = all)

### 2. `train.py` - Core Testing Logic
**Added:**
- Imported `time` and `json` modules
- `run_generalization_test()` function: Loads unseen dataset, runs validation, returns metrics
- Integration in main training loop: Calls test every N epochs, saves results to JSON
- Display of generalization test configuration at training start
- JSON file output: `checkpoints/generalization_test_results.json`

**Key Features:**
- Automatically loads and tests on unseen dataset
- Saves comprehensive results including:
  - Generalization test metrics (loss, accuracy)
  - Corresponding training metrics
  - Corresponding validation metrics
  - Dataset information
  - Timestamp
- No impact on training if disabled
- Graceful handling of missing datasets

### 3. `app.py` - Web API Endpoint
**Added:**
- `/generalization_test_results` endpoint (GET)
- Returns JSON with test results or availability status
- Handles missing files gracefully

### 4. `templates/index.html` - Web Interface
**Added:**
- New HTML section for displaying generalization test results
- JavaScript function `updateGeneralizationTestResults()`:
  - Fetches results from API every 10 seconds
  - Displays results in formatted table
  - Shows metadata (training/testing datasets, task type, etc.)
  - Displays most recent 10 tests
  - Auto-shows/hides based on availability
- Styled section with green theme to distinguish from training metrics

## Data Flow

```
Training Loop (train.py)
    ↓
Every N epochs (config.GENERALIZATION_TEST_INTERVAL)
    ↓
run_generalization_test()
    ↓
Load unseen dataset (e.g., V2)
    ↓
Run validation (no training)
    ↓
Collect metrics
    ↓
Save to JSON file (checkpoints/generalization_test_results.json)
    ↓
Web App (app.py)
    ↓
/generalization_test_results endpoint
    ↓
Read JSON file
    ↓
Return to web interface
    ↓
Display in browser (index.html)
```

## JSON Output Format

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
    }
  ]
}
```

## Web Interface Display

The web interface shows:
1. **Section Header**: "📊 Generalization Test Results (Testing on Unseen Dataset)"
2. **Metadata**:
   - Training Dataset: V1/training
   - Testing Dataset: V2/training
   - Task Type: reconstruction
   - Bottleneck Type: communication
3. **Results Table**:
   - Epoch
   - Gen Test Loss
   - Gen Test Acc (highlighted)
   - Train Loss
   - Train Acc
   - Val Loss
   - Val Acc
   - Number of Test Grids
4. **Footer**: Shows count if more than 10 tests

## Usage Examples

### Example 1: Train on V1, Test on V2
```python
# config.py
DATASET_VERSION = 'V1'
GENERALIZATION_TEST_ENABLED = True
GENERALIZATION_TEST_DATASET_VERSION = 'V2'
GENERALIZATION_TEST_INTERVAL = 20
```

### Example 2: Train on V2, Test on V1 Evaluation Set
```python
# config.py
DATASET_VERSION = 'V2'
GENERALIZATION_TEST_ENABLED = True
GENERALIZATION_TEST_DATASET_VERSION = 'V1'
GENERALIZATION_TEST_DATASET_SPLIT = 'evaluation'
GENERALIZATION_TEST_INTERVAL = 10
```

### Example 3: Fast Testing for Debugging
```python
# config.py
GENERALIZATION_TEST_ENABLED = True
GENERALIZATION_TEST_INTERVAL = 5  # Every 5 epochs
GENERALIZATION_TEST_MAX_GRIDS = 20  # Only 20 grids
```

## Benefits

1. **Early Overfitting Detection**: See when model stops generalizing
2. **Cross-Dataset Comparison**: Test how well patterns learned on one dataset transfer to another
3. **No Manual Intervention**: Fully automatic during training
4. **Real-Time Monitoring**: View results in web interface while training
5. **Historical Tracking**: All results saved for later analysis
6. **Minimal Performance Impact**: Testing is fast and only runs every N epochs
7. **Configurable**: Easy to enable/disable, adjust frequency, limit test size

## Limitations

- Only works with reconstruction and selection tasks (not puzzle_solving yet)
- Requires both datasets to exist on disk
- Testing pauses training briefly (typically 5-30 seconds depending on test size)
- Uses validation mode (no gradient computation)

## Future Enhancements (Possible)

- Add generalization testing for puzzle_solving task
- Create visualization plots comparing training vs generalization over time
- Add statistical significance tests
- Export results to CSV for external analysis
- Add alerts when generalization gap exceeds threshold
- Support testing on multiple datasets simultaneously

