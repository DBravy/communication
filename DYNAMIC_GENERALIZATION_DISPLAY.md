# Dynamic Generalization Test Display

## Overview
The generalization test results section now displays **dynamically during training**, updating as new results become available.

## How It Works

### 1. When Training Starts
- ✅ Backend deletes old `generalization_test_results.json` file
- ✅ Frontend immediately checks for results (finds none)
- ✅ Section is automatically **hidden** (no stale data shown)
- ✅ Training begins from epoch 1

### 2. During Training
- 🔄 Frontend polls for results **every 10 seconds** (automatic)
- 📊 When generalization test runs (e.g., at epoch 20), results are saved to JSON
- ⚡ Within 10 seconds, frontend detects new results
- ✨ Green results section **automatically appears**
- 📈 Table shows the latest test results

### 3. As Training Continues
- 📊 Every time generalization test runs (e.g., epochs 20, 40, 60...)
- 🔄 Results are appended to the JSON file
- ⚡ Within 10 seconds, table updates with new row
- 📈 You see the history building in real-time

### 4. When Training Stops
- ✅ Results remain visible
- 📊 You can review the full history
- 🔄 Polling continues (section stays visible)

## Timeline Example

```
00:00 - Click "Start Training" 
        → Results section hidden (no data)
        → Training starts

00:10 - Epoch 20 completes
        → Generalization test runs
        → Results saved to JSON
        
00:15 - Next poll cycle
        → Frontend detects results
        → GREEN BOX APPEARS with 1 row

00:20 - Epoch 40 completes
        → Generalization test runs
        → Results appended to JSON
        
00:25 - Next poll cycle
        → Table updates
        → Now shows 2 rows (epochs 20, 40)

...continues updating...

01:00 - Click "Stop Training"
        → Results remain visible
        → Can review full history
```

## Visual States

### State 1: No Results Yet
```
[Training running...]
[Charts showing metrics...]
← No generalization section visible
```

### State 2: First Results Available
```
[Training running...]
[Charts showing metrics...]

┌─────────────────────────────────────┐
│ 📊 Generalization Test Results     │
│                                     │
│ Training: V1/training              │
│ Testing: V2/training               │
│                                     │
│ Epoch | Gen Loss | Gen Acc | ...   │
│  20   |  2.5432  | 45.67%  | ...   │
└─────────────────────────────────────┘
```

### State 3: Multiple Results Building
```
[Training running...]
[Charts showing metrics...]

┌─────────────────────────────────────┐
│ 📊 Generalization Test Results     │
│                                     │
│ Training: V1/training              │
│ Testing: V2/training               │
│                                     │
│ Epoch | Gen Loss | Gen Acc | ...   │
│  60   |  2.1234  | 52.34%  | ...   │
│  40   |  2.3456  | 48.90%  | ...   │
│  20   |  2.5432  | 45.67%  | ...   │
└─────────────────────────────────────┘
```

## Key Features

### ✅ Real-Time Updates
- No need to stop training to see results
- Updates automatically every 10 seconds
- See progress as training continues

### ✅ Clean Start
- Old results cleared when starting new training
- No confusion from stale data
- Fresh slate for each training run

### ✅ History Display
- Shows most recent 10 tests
- Reverse chronological order (newest first)
- Includes epoch, loss, accuracy, and more

### ✅ Automatic Show/Hide
- Hidden when no results available
- Appears automatically when first results arrive
- No manual refresh needed

## Code Flow

```javascript
// When training starts:
startTrain() → 
  Backend: Delete results file → 
  Frontend: updateGeneralizationTestResults() → 
  No results found → 
  Hide section

// During training (every 10 seconds):
setInterval(updateGeneralizationTestResults, 10000) →
  Poll /generalization_test_results →
  If results available:
    - Show section
    - Populate table with history
  Else:
    - Hide section

// When test runs (e.g., epoch 20):
Backend: run_generalization_test() →
  Append results to JSON →
  Next frontend poll (within 10s) →
  Detect new data →
  Update table
```

## Files Modified

1. **`templates/index.html`** (lines 1570-1573, 1613-1616)
   - Removed forced hiding on training start
   - Added immediate update check
   - Let polling handle show/hide dynamically

2. **`app.py`** (unchanged - still clears results on start)

## Configuration

Works with any settings:
```python
# config.py
GENERALIZATION_TEST_ENABLED = True
GENERALIZATION_TEST_INTERVAL = 20  # How often to run test
GENERALIZATION_TEST_DATASET_VERSION = 'V2'
GENERALIZATION_TEST_DATASET_SPLIT = 'training'
GENERALIZATION_TEST_MAX_GRIDS = 100
```

## Polling Frequency

The frontend checks for new results **every 10 seconds**:
```javascript
// Line 1830-1831 in index.html
updateGeneralizationTestResults();
setInterval(updateGeneralizationTestResults, 10000);
```

This means:
- Test runs at epoch 20 → Results appear within 10 seconds
- Test runs at epoch 40 → Table updates within 10 seconds
- Minimal delay, fast feedback

## Browser Console Debugging

Open browser console (F12) to see:
```
Generalization test data: {available: false}  // No results yet
...
Generalization test data: {available: true, data: {...}}  // Results available!
```

## Behavior Summary

| Event | Section Visibility | Content |
|-------|-------------------|---------|
| Page load (no training) | Hidden | N/A |
| Start training | Hidden | No results yet |
| Epoch 20 (test runs) | Appears within 10s | 1 row of data |
| Epoch 40 (test runs) | Visible (updates) | 2 rows of data |
| Epoch 60 (test runs) | Visible (updates) | 3 rows of data |
| Stop training | Visible | Full history |
| Start new training | Hidden | Old results cleared |

## Perfect For

- 📊 Monitoring generalization during training
- 🔍 Catching overfitting early (train acc up, gen acc down)
- 📈 Seeing if model generalizes progressively
- ⏱️ Real-time feedback without interrupting training
- 🎯 Comparing training vs generalization accuracy live

