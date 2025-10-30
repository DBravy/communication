# Generalization Test Display Fix

## Issue
The generalization test was running (visible in console logs), but the results weren't appearing on the web page.

## Root Cause
The JavaScript code in `index.html` was trying to access `val_loss` and `val_acc` fields that don't exist in the generalization test results. When it called `.toFixed()` on `undefined`, it threw an error that prevented the entire results table from rendering.

## Solution
Modified the JavaScript to gracefully handle missing fields by checking if they exist before calling `.toFixed()`, and displaying 'N/A' for undefined values.

**File Changed:** `templates/index.html` (lines 1796-1803)

## How to Verify It's Working

### 1. Check Browser Console
Open your browser's Developer Tools (F12) and look at the Console tab. You should see:
```
Generalization test data: {available: true, data: {...}}
```

Every 10 seconds (the polling interval).

### 2. Wait for Test to Run
Your `config.py` has `GENERALIZATION_TEST_INTERVAL = 20`, which means the test only runs **every 20 epochs**. 

- If you've trained for < 20 epochs: No results yet ✗
- If you've trained for ≥ 20 epochs: Results should appear ✓

### 3. Check Backend Logs
In your terminal running the Flask server, you should see:
```
✓ Generalization test: Loss=X.XXXX, Acc=XX.XX%
```

Every 20 epochs (or whatever your interval is set to).

### 4. Check the JSON File
Results are saved to `checkpoints/generalization_test_results.json`. You can check if this file exists and has data:
```bash
cat checkpoints/generalization_test_results.json | python -m json.tool
```

## Adjusting the Test Interval

If you want to see results sooner, you can reduce the interval in your config:

```python
# In config.py
GENERALIZATION_TEST_INTERVAL = 1  # Test every epoch (faster, but more computation)
```

Or via the web interface (once you add UI controls for it).

## What the Display Shows

When generalization test results are available, a green box will appear on the web page showing:

**Metadata:**
- Training Dataset & Split
- Testing Dataset & Split  
- Task Type & Bottleneck Type

**Results Table (most recent 10 tests):**
| Column | Description |
|--------|-------------|
| Epoch | Which training epoch the test was run |
| Gen Test Loss | Loss on unseen dataset |
| Gen Test Acc | Accuracy on unseen dataset |
| Train Loss | Training loss at that epoch |
| Train Acc | Training accuracy at that epoch |
| Val Loss | Validation loss (N/A - not computed) |
| Val Acc | Validation accuracy (N/A - not computed) |
| # Test Grids | Number of grids tested |

## Current Configuration

From your `config.py`:
- ✅ Enabled: `GENERALIZATION_TEST_ENABLED = True`
- 📊 Test Dataset: V2/training (testing while training on V1/training)
- 🔄 Interval: Every 20 epochs
- 📈 Max Grids: 100

## Testing It Right Now

If you want to see results immediately without waiting 20 epochs:

### Option 1: Change interval to 1 epoch
```python
# In config.py
GENERALIZATION_TEST_INTERVAL = 1
```

Then restart training.

### Option 2: Run a quick test
Start training and let it run for at least 20 epochs. After epoch 20, you should see:
1. Console message: `✓ Generalization test: Loss=X.XXXX, Acc=XX.XX%`
2. Browser console: `Generalization test data: {available: true, ...}`
3. Green box appears on the web page with results table

## Debug Checklist

If results still don't appear:

1. ✅ Check `config.py` has `GENERALIZATION_TEST_ENABLED = True`
2. ✅ Check you've trained for at least `GENERALIZATION_TEST_INTERVAL` epochs
3. ✅ Check browser console for errors (F12 → Console tab)
4. ✅ Check file exists: `ls -la checkpoints/generalization_test_results.json`
5. ✅ Check server logs for "✓ Generalization test" messages
6. ✅ Refresh the web page after training for 20+ epochs

## API Endpoint

You can also fetch results directly via API:
```bash
curl http://localhost:5002/generalization_test_results | python -m json.tool
```

This should return:
```json
{
  "available": true,
  "data": {
    "training_dataset": "V1",
    "training_split": "training",
    "task_type": "reconstruction",
    "bottleneck_type": "communication",
    "history": [
      {
        "dataset_version": "V2",
        "dataset_split": "training",
        "num_grids": 100,
        "loss": 2.5432,
        "accuracy": 45.67,
        "epoch": 20,
        "train_loss": 2.8901,
        "train_acc": 42.34,
        "timestamp": 1234567890.0
      }
    ]
  }
}
```

