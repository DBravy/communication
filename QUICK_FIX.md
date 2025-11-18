# Quick Fix for CNN Activation Sparsity

## Problem
60.5% of your CNN neurons are dead (producing near-zero activations), which prevents slot attention from learning meaningful object segmentations.

## Solution
Replace ReLU with LeakyReLU and lower the learning rate.

---

## Changes to Make

### 1. Fix `model.py` - Replace ReLU with LeakyReLU

**Location**: Line ~73 in `ARCEncoder.__init__`

**Before:**
```python
self.relu = nn.ReLU()
self.dropout = nn.Dropout(0.1)
```

**After:**
```python
self.relu = nn.LeakyReLU(negative_slope=0.01)  # ← CHANGED
self.dropout = nn.Dropout(0.1)
```

**Why**: LeakyReLU allows small negative gradients, preventing neurons from dying permanently.

---

### 2. Fix `config.py` - Lower Learning Rate

**Location**: Line ~74 in config.py

**Before:**
```python
LEARNING_RATE = 1e-4
```

**After:**
```python
LEARNING_RATE = 5e-5  # ← CHANGED (50% reduction)
```

**Why**: High learning rates can cause gradient explosions that kill neurons.

---

## Optional but Recommended

### 3. Add Dropout Between Conv Layers

**Location**: In `ARCEncoder.forward()` method (line ~104)

**Before:**
```python
# Convolutional layers
for conv, bn in zip(self.conv_layers, self.bn_layers):
    x = self.relu(bn(conv(x)))
```

**After:**
```python
# Convolutional layers
for conv, bn in zip(self.conv_layers, self.bn_layers):
    x = self.relu(bn(conv(x)))
    x = self.dropout(x)  # ← ADD THIS LINE
```

**Why**: Prevents over-reliance on specific neurons, forcing redundancy.

---

### 4. Reduce Training Epochs

**Location**: Line ~75 in config.py

**Before:**
```python
NUM_EPOCHS = 100000
```

**After:**
```python
NUM_EPOCHS = 1000  # ← CHANGED (for initial testing)
```

**Why**: Training for 2688 epochs on 200 grids is likely overfitting and causing neuron death.

---

## After Making Changes

1. **Delete old checkpoint** (to ensure fresh start):
   ```bash
   mv checkpoints/slot_attention.pth checkpoints/slot_attention_old.pth
   ```

2. **Retrain the model**:
   ```bash
   conda activate comm
   python train.py  # Or your training script
   ```

3. **Verify improvements** after training completes:
   ```bash
   python visualize_cnn_features.py
   python visualize_slot_attention.py
   ```

4. **Check metrics**:
   - Activation Sparsity should drop from 60.5% → **< 40%**
   - More slots should have non-zero attention masks
   - Reconstruction accuracy should improve

---

## Expected Results

### Before (Current):
- ❌ Activation Sparsity: 60.5%
- ❌ Active Slots: 1-2 per image
- ❌ Reconstruction: ~79%

### After (Expected):
- ✅ Activation Sparsity: 20-40%
- ✅ Active Slots: 3-5 per image
- ✅ Reconstruction: 85-90%

---

## If This Doesn't Work

Try also replacing LeakyReLU with GELU (a smoother activation):

```python
self.relu = nn.GELU()
```

GELU has smooth gradients everywhere and is less prone to neuron death.

---

## Summary

**Minimal fix** (2 lines):
1. Change `nn.ReLU()` → `nn.LeakyReLU(0.01)`
2. Change `LEARNING_RATE = 1e-4` → `LEARNING_RATE = 5e-5`

**This should resolve 80% of your slot attention issues.**

