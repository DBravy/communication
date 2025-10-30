# Clear Generalization Test Results on Training Start

## Changes Made

The generalization test results now automatically clear when you start a new training session (either pretraining or main training).

### Backend Changes (`app.py`)

**Modified Functions:**
1. `start_pretrain()` (lines 2273-2280)
2. `start_train()` (lines 2323-2330)

**What Happens:**
When you click "Start Pretraining" or "Start Main Training", the system now:
1. Deletes the `checkpoints/generalization_test_results.json` file
2. Prints a confirmation message: "Cleared previous generalization test results"
3. Starts fresh with no old results

**Error Handling:**
If the file can't be deleted (permissions issue, file locked, etc.), it prints a warning but continues training without failing.

### Frontend Changes (`templates/index.html`)

**Modified Functions:**
1. `startPretrainInternal()` (line 1570)
2. `startTrainInternal()` (line 1611)

**What Happens:**
When training starts, the generalization test section is immediately hidden:
```javascript
document.getElementById('generalizationTestSection').style.display = 'none';
```

This provides instant visual feedback - the green results box disappears as soon as you start training.

## User Experience

### Before Training Starts:
- Old generalization test results visible (if they exist)

### When You Click "Start Training":
1. ✨ Generalization test section **immediately disappears** from the page
2. 🗑️ Old results file is **deleted** from disk
3. 📊 Fresh metrics charts appear
4. 🚀 Training begins from epoch 1

### As Training Progresses:
- After your configured interval (e.g., every 20 epochs), new generalization test results will appear
- Only shows results from the **current** training run
- No confusion from mixing old and new results

## Why This Matters

**Previous Behavior:**
- Old generalization results would remain visible when starting new training
- Could be confusing - were you looking at current or previous results?
- Results file kept accumulating data across multiple training runs

**New Behavior:**
- Clean slate for each training session
- Results are always from the current run only
- Clear visual feedback that you're starting fresh

## Example Timeline

```
Epoch 0:  Start training → generalization results cleared
Epoch 20: First generalization test runs → results appear
Epoch 40: Second generalization test runs → results update
Epoch 60: Third generalization test runs → results update
...
Stop training → results remain visible

Start new training → results cleared again → fresh start
```

## Files Modified

1. **`app.py`** - Backend logic to delete results file
2. **`templates/index.html`** - Frontend logic to hide results section

## Testing

To verify it's working:

1. **Start training** and let it run for 20+ epochs (or whatever your interval is)
2. **See generalization results** appear in green box
3. **Stop training** - results remain visible
4. **Start training again** - results should **immediately disappear**
5. Check console: `"Cleared previous generalization test results"`
6. Check file system: `ls checkpoints/generalization_test_results.json` → "No such file"

## Configuration

This works with any generalization test configuration:
```python
# config.py
GENERALIZATION_TEST_ENABLED = True
GENERALIZATION_TEST_INTERVAL = 20  # Or any value
GENERALIZATION_TEST_DATASET_VERSION = 'V2'
GENERALIZATION_TEST_DATASET_SPLIT = 'training'
```

Results will always clear on start, regardless of these settings.

