# ALL DATASETS WITH HOLDOUT FEATURE

## Overview

This feature allows you to train on **all available data** (V1 + V2, training + evaluation splits) while automatically holding out a subset of grids for generalization testing.

## What This Feature Does

When `USE_ALL_DATASETS = True` is set in `config.py`, the training system will:

1. **Load all 4 dataset categories:**
   - V1/training
   - V1/evaluation
   - V2/training
   - V2/evaluation

2. **Hold out grids for generalization testing:**
   - Randomly selects 25 grids from each category (configurable)
   - Total: 100 grids held out by default
   - Uses a fixed random seed for reproducibility

3. **Train on remaining grids:**
   - All grids except the held-out ones are used for training
   - Significantly larger training dataset than single splits

4. **Test generalization on held-out grids:**
   - During training, generalization tests run on the held-out grids
   - These grids are never seen during training
   - Provides true out-of-sample generalization metrics

## Configuration

### In `config.py`:

```python
# Enable the feature
USE_ALL_DATASETS = False  # Set to True to enable

# Number of grids to hold out from each category
HOLDOUT_GRIDS_PER_CATEGORY = 25  # 25 × 4 categories = 100 total

# Random seed for reproducible holdout selection
HOLDOUT_SEED = 42

# Generalization testing (automatically uses holdout grids when USE_ALL_DATASETS=True)
GENERALIZATION_TEST_ENABLED = True
GENERALIZATION_TEST_INTERVAL = 20  # Test every 20 epochs
```

### Via Web Interface:

The new configuration options are available in the web UI:

- `use_all_datasets` (checkbox)
- `holdout_grids_per_category` (number)
- `holdout_seed` (number)

## How It Works

### Dataset Loading

When `USE_ALL_DATASETS = True`:

1. **Four datasets are loaded separately:**
   ```
   V1/training  → e.g., 400 grids
   V1/evaluation → e.g., 400 grids
   V2/training  → e.g., 1000 grids
   V2/evaluation → e.g., 120 grids
   Total: ~1920 grids
   ```

2. **Random holdout selection (with fixed seed):**
   ```
   V1/training:   25 grids held out
   V1/evaluation: 25 grids held out
   V2/training:   25 grids held out
   V2/evaluation: 25 grids held out
   Total held out: 100 grids
   ```

3. **Training dataset:**
   ```
   ~1820 grids available for training
   ```

### Generalization Testing

When `GENERALIZATION_TEST_ENABLED = True` and `USE_ALL_DATASETS = True`:

- **Traditional mode:** Loads a separate dataset (e.g., V2/training) for testing
- **All datasets mode:** Uses the 100 held-out grids for testing
  - No additional data loading required
  - True out-of-sample test (grids never seen during training)
  - Balanced across all dataset categories

## Example Usage

### Configuration for Maximum Data

```python
# config.py

# Use all available data
USE_ALL_DATASETS = True
HOLDOUT_GRIDS_PER_CATEGORY = 25  # 100 total held out
HOLDOUT_SEED = 42  # Reproducible

# Enable generalization testing on holdout
GENERALIZATION_TEST_ENABLED = True
GENERALIZATION_TEST_INTERVAL = 20

# Other settings
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1000
```

### Expected Output

When training starts, you'll see:

```
================================================================================
LOADING ALL DATASETS WITH HOLDOUT
================================================================================
Holdout: 25 grids per category (4 categories = 100 total)
Random seed: 42

Loading V1/training...
  Loaded 400 grids, holding out 25 for generalization
Loading V1/evaluation...
  Loaded 400 grids, holding out 25 for generalization
Loading V2/training...
  Loaded 1000 grids, holding out 25 for generalization
Loading V2/evaluation...
  Loaded 120 grids, holding out 25 for generalization

Total grids: 1920
Holdout grids: 100
Training grids: 1820
Total unique puzzles: 520
================================================================================

✓ Using ALL datasets mode with 100 grids held out for generalization
```

During training, generalization tests will show:

```
================================================================================
GENERALIZATION TEST: Testing on HOLDOUT grids
================================================================================
Generalization Test Loss: 0.2345, Accuracy: 87.50%
================================================================================
```

## Benefits

### 1. **Maximum Training Data**
- Uses all available ARC data
- ~1820 training grids (compared to ~400 or ~1000 for single splits)
- Better model generalization potential

### 2. **True Out-of-Sample Testing**
- Held-out grids never seen during training
- Balanced across all dataset categories
- Reproducible with fixed seed

### 3. **Comprehensive Coverage**
- Tests on diverse range of puzzles from V1 and V2
- Both training and evaluation splits represented
- Better assessment of true generalization

### 4. **Easy to Use**
- Single flag to enable: `USE_ALL_DATASETS = True`
- Automatic holdout management
- Compatible with all existing features

## Comparison with Other Modes

| Mode | Training Data | Test Data | Use Case |
|------|--------------|-----------|----------|
| **Single Split** | V2/training (1000) | None | Basic training |
| **Combined Splits** | V2/train+eval (1120) | V1 or V2 | More data from one version |
| **All Datasets (NEW)** | All - 100 (1820) | 100 holdout | Maximum data + true holdout |

## Implementation Details

### Random Selection
- Uses Python's `random.seed()` for reproducibility
- Shuffles indices and selects first N grids
- Same holdout grids across training runs with same seed

### Dataset Structure
- Returns tuple: `(training_dataset, holdout_dataset)`
- Both use PyTorch `Subset` for efficient indexing
- Maintains puzzle_id_map for classification tasks

### Memory Efficiency
- Only loads each dataset once
- Uses indexing to split training/holdout
- No data duplication

## Troubleshooting

### If a dataset is missing:
```
Warning: V1/evaluation not found, skipping...
```
The feature will continue with available datasets.

### If you get fewer than expected holdout grids:
The number of holdout grids is `min(HOLDOUT_GRIDS_PER_CATEGORY, available_grids)` per category. If a category has fewer grids, all will be used for training.

### To change holdout selection:
Modify `HOLDOUT_SEED` in config.py to get a different random selection.

## Notes

- **Pretraining:** Uses only the training portion (ignores holdout)
- **Validation Split:** The 1820 training grids are further split 80/20 for train/val
- **Checkpoint Compatibility:** Checkpoints save the holdout configuration
- **Web Interface:** All settings controllable via web UI

## Summary

The ALL DATASETS feature provides the maximum amount of training data while maintaining a proper holdout set for generalization testing. It's ideal for training models that need exposure to diverse ARC puzzles while still being able to measure true generalization performance.

