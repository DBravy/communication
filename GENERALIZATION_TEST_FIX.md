# Generalization Test Fix

## Issue
The generalization test (testing on unseen puzzles) was not running when training through the web app, even though:
1. The test function existed in `train.py`
2. An API endpoint existed to fetch results (`/generalization_test_results`)
3. The test was working when running `train.py` directly

## Root Cause
The generalization test was only integrated into `train.py`'s `main()` function, but web-based training uses `app.py`'s `train_worker()` function, which had **no generalization test logic**.

## Solution
Integrated the generalization test into the web app training flow:

### 1. Added Generalization Test to `train_worker()` (app.py lines 1469-1525)
- Runs every N epochs (configurable via `generalization_test_interval`)
- Only runs when `generalization_test_enabled` is True
- Saves results to JSON file in checkpoints directory
- Handles errors gracefully without stopping training

### 2. Exposed Configuration Settings via API
Added new configuration options to control generalization testing:
- `generalization_test_enabled` (bool) - Enable/disable testing
- `generalization_test_dataset_version` (str) - Which dataset to test on (V1/V2)
- `generalization_test_dataset_split` (str) - Which split to use (training/evaluation)
- `generalization_test_max_grids` (int) - Max number of grids to test
- `generalization_test_interval` (int) - Test every N epochs

These can now be:
- Set in `config.py` (loaded on startup)
- Modified via POST `/task_config` endpoint
- Retrieved via GET `/task_config` endpoint

### 3. Updated Global Training State (app.py lines 142-147)
Initialized the new configuration options with default values from `config.py`

## How to Use

### Option 1: Enable via config.py
```python
# Add to config.py
GENERALIZATION_TEST_ENABLED = True
GENERALIZATION_TEST_DATASET_VERSION = 'V2'
GENERALIZATION_TEST_DATASET_SPLIT = 'evaluation'  # Test on unseen data
GENERALIZATION_TEST_MAX_GRIDS = 100
GENERALIZATION_TEST_INTERVAL = 1  # Test every epoch
```

### Option 2: Enable via Web API
```javascript
// Update configuration via POST request
fetch('/task_config', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    generalization_test_enabled: true,
    generalization_test_dataset_version: 'V2',
    generalization_test_dataset_split: 'evaluation',
    generalization_test_max_grids: 100,
    generalization_test_interval: 1
  })
});
```

### Viewing Results
Results are saved to `checkpoints/generalization_test_results.json` and can be fetched via:
```javascript
fetch('/generalization_test_results')
  .then(res => res.json())
  .then(data => {
    if (data.available) {
      console.log('Generalization test history:', data.data.history);
    }
  });
```

## What Gets Tested
The generalization test:
- Works for **both communication and autoencoder bottlenecks**
- Tests on a separate dataset (different from training data)
- Reports loss and accuracy on unseen puzzles
- Saves results with training metrics for comparison
- Updates incrementally during training

## Results Format
```json
{
  "training_dataset": "V2",
  "training_split": "training",
  "task_type": "reconstruction",
  "bottleneck_type": "communication",
  "history": [
    {
      "dataset_version": "V2",
      "dataset_split": "evaluation",
      "num_grids": 100,
      "loss": 2.5,
      "accuracy": 45.2,
      "epoch": 1,
      "train_loss": 3.2,
      "train_acc": 38.5,
      "timestamp": 1234567890.0
    },
    ...
  ]
}
```

## Benefits
- **Monitors generalization** during training
- **Works with web app** training (not just command-line)
- **Configurable** via API or config file
- **Non-blocking** - errors don't stop training
- **Supports all task types** - reconstruction, selection, puzzle_classification

