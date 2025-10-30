# Generalization Testing for ARC Training

## ✅ Implementation Complete

The system now supports automatic generalization testing during reconstruction training. You can train on one dataset (e.g., V1) and automatically test on an unseen dataset (e.g., V2) every N epochs.

## 🚀 Quick Start

### 1. Configure (`config.py`)

```python
# Your training dataset
DATASET_VERSION = 'V1'
DATASET_SPLIT = 'training'

# Enable generalization testing
GENERALIZATION_TEST_ENABLED = True
GENERALIZATION_TEST_DATASET_VERSION = 'V2'  # Test on V2 while training on V1
GENERALIZATION_TEST_DATASET_SPLIT = 'training'
GENERALIZATION_TEST_INTERVAL = 20  # Test every 20 epochs
GENERALIZATION_TEST_MAX_GRIDS = 100  # Use 100 grids (None = all grids)
```

### 2. Run Training

**Option A: Command Line**
```bash
python train.py
```

**Option B: Web App**
```bash
python app.py
# Navigate to http://localhost:5000
# Click "Start Main Training"
```

### 3. View Results

**In the Web App:**
- After epoch 20 (or your configured interval), a new green section appears
- Shows a table comparing generalization metrics with training/validation metrics
- Updates automatically every 10 seconds

**In the Terminal:**
```
================================================================================
GENERALIZATION TEST: Testing on V2/training
================================================================================
Generalization Test Loss: 0.5234, Accuracy: 82.50%
================================================================================
```

**In the File System:**
```bash
cat checkpoints/generalization_test_results.json
```

## 📊 What You'll See

### Web Interface
A new section displays:
- **Training Dataset**: V1/training
- **Testing Dataset**: V2/training
- **Task Type**: reconstruction
- **Bottleneck Type**: communication

Plus a table with columns:
- Epoch
- Gen Test Loss & Accuracy (highlighted)
- Train Loss & Accuracy
- Val Loss & Accuracy
- Number of Test Grids

### Example Output
```
Epoch 20:
  Train Loss: 0.4120, Train Acc: 88.30%
  Val Loss: 0.4450, Val Acc: 86.10%
  Gen Test Loss: 0.5234, Gen Acc: 82.50%  ← This is new!
```

## 🎯 Use Cases

### 1. Detecting Overfitting
Train on V1, test on V2. If validation accuracy is high but generalization accuracy is low, your model is overfitting to V1-specific patterns.

### 2. Cross-Dataset Generalization
See how well patterns learned from one dataset transfer to another:
- V1 → V2: Small dataset to large dataset
- V2 → V1: Large dataset to small dataset
- Training split → Evaluation split: Within-dataset generalization

### 3. Model Selection
Compare different architectures or hyperparameters based on generalization performance, not just validation accuracy.

## ⚙️ Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `GENERALIZATION_TEST_ENABLED` | `True` | Enable/disable testing |
| `GENERALIZATION_TEST_DATASET_VERSION` | `'V2'` | Dataset to test on |
| `GENERALIZATION_TEST_DATASET_SPLIT` | `'training'` | Split to use |
| `GENERALIZATION_TEST_INTERVAL` | `20` | Test every N epochs |
| `GENERALIZATION_TEST_MAX_GRIDS` | `100` | Max grids to test (None = all) |

## 📈 Interpreting Results

### ✅ Good Generalization
```
Train Acc: 88.3%
Val Acc: 86.1%
Gen Test Acc: 83.5%  ← Close to validation
```
Model learned general patterns that transfer across datasets.

### ⚠️ Overfitting
```
Train Acc: 95.3%
Val Acc: 92.1%
Gen Test Acc: 65.2%  ← Much lower
```
Model memorized training dataset, doesn't generalize well.

### 📊 Monitoring Over Time
Watch the trend:
- **Improving**: Gen test accuracy increases over epochs → good learning
- **Diverging**: Val accuracy increases but gen test decreases → overfitting
- **Plateauing**: Both metrics stop improving → reached model capacity

## 🛠️ Common Configurations

### Fast Testing (Debugging)
```python
GENERALIZATION_TEST_INTERVAL = 5
GENERALIZATION_TEST_MAX_GRIDS = 20
```

### Thorough Testing (Final Runs)
```python
GENERALIZATION_TEST_INTERVAL = 10
GENERALIZATION_TEST_MAX_GRIDS = None  # All grids
```

### Disable Testing
```python
GENERALIZATION_TEST_ENABLED = False
```

## 📁 Output Files

### JSON Results File
**Location**: `checkpoints/generalization_test_results.json`

**Structure**:
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

## 🔍 Technical Details

### Files Modified
1. **`config.py`**: Added 5 new configuration options
2. **`train.py`**: 
   - Added `run_generalization_test()` function
   - Integrated into training loop
   - Saves results to JSON
3. **`app.py`**: Added `/generalization_test_results` API endpoint
4. **`templates/index.html`**: 
   - Added results display section
   - Added JavaScript to fetch and display results

### How It Works
1. Every N epochs, training pauses briefly
2. System loads the unseen test dataset
3. Runs validation (no training, just evaluation)
4. Collects metrics (loss, accuracy)
5. Saves results to JSON file
6. Resumes training
7. Web interface polls for updates every 10 seconds

### Performance Impact
- Testing typically takes 5-30 seconds (depending on test size)
- No impact on training gradients or model updates
- Memory efficient (test dataset loaded fresh each time)
- Minimal disk space (JSON file is small)

## 🐛 Troubleshooting

### Results Not Showing?
- ✓ Check `GENERALIZATION_TEST_ENABLED = True`
- ✓ Verify test dataset path exists (e.g., `V2/data/training/`)
- ✓ Wait until epoch 20 (or your configured interval)
- ✓ Refresh browser or check `checkpoints/generalization_test_results.json`

### Testing is Slow?
- Reduce `GENERALIZATION_TEST_MAX_GRIDS` to 50 or fewer
- Increase `GENERALIZATION_TEST_INTERVAL` (e.g., 30 epochs)

### Memory Issues?
- Set `GENERALIZATION_TEST_MAX_GRIDS = 50` or lower
- Reduce `BATCH_SIZE` in config

### Wrong Results?
- Ensure training and testing datasets are DIFFERENT
- Check `TASK_TYPE = 'reconstruction'` (doesn't support puzzle_solving yet)
- Verify both datasets exist and contain grids

## 📚 Documentation Files

- **`GENERALIZATION_TESTING.md`**: Detailed documentation
- **`QUICK_START_GENERALIZATION_TESTING.md`**: 1-minute setup guide
- **`GENERALIZATION_TESTING_CHANGES.md`**: Implementation details
- **`GENERALIZATION_TESTING_README.md`**: This file

## 🎓 Best Practices

1. **Start with limited grids**: Use 50-100 grids initially, increase for final runs
2. **Test early and often**: Use interval of 10-20 epochs to catch issues quickly
3. **Compare both directions**: Try V1→V2 and V2→V1 to understand bidirectional transfer
4. **Monitor the gap**: Large val-gen gap indicates dataset-specific learning
5. **Save checkpoints**: Use checkpoints when generalization performance peaks

## 🚧 Limitations

- Only works with reconstruction and selection tasks (not puzzle_solving)
- Requires test dataset to exist on disk
- Testing pauses training briefly
- Web interface shows last 10 tests only (full history in JSON)

## 💡 Tips

- Use V1 evaluation split to test on original ARC puzzles
- Compare generalization across different model architectures
- Watch for the "sweet spot" where generalization peaks before overfitting
- Generalization gap can help decide when to stop training
- Use results to justify architecture choices in papers/reports

## ✨ Example Session

```bash
# 1. Configure
vim config.py  # Set GENERALIZATION_TEST_ENABLED = True

# 2. Start training
python app.py

# 3. Navigate to http://localhost:5000
# 4. Click "Start Main Training"
# 5. Watch training progress
# 6. After epoch 20, see generalization results appear
# 7. Compare metrics to detect overfitting
# 8. Stop training when generalization plateaus or decreases
```

## 🎉 Success!

You can now train on one dataset while continuously monitoring how well your model generalizes to completely unseen data!

