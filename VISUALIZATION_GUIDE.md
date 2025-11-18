# CNN & Slot Attention Visualization Guide

## Scripts Available

### 1. `visualize_cnn_features.py` - Diagnose CNN Feature Extraction
**What it shows:**
- CNN feature maps at each layer
- Channel activation statistics  
- Spatial diversity analysis
- **Automatic diagnosis** of problems
- Activation sparsity detection

**How to use:**
```bash
conda activate comm
python visualize_cnn_features.py
```

**Customize by editing the `main()` function:**
```python
def main():
    checkpoint_path = 'checkpoints/slot_attention.pth'  # Your checkpoint
    output_dir = 'cnn_feature_visualizations'           # Output folder
    num_grids = 10                                      # Number of examples
    output_format = 'png'  # Options: 'png' or 'pdf'   # Output format
```

**Output:**
- Individual files: `cnn_features_grid_XXXX.png` (or `.pdf`)
- Automatic diagnosis printed to console

---

### 2. `visualize_slot_attention.py` - See How Slots Segment Objects
**What it shows:**
- Original input grid
- Reconstructed output
- Each slot's attention mask (what it focuses on)
- Each slot's individual reconstruction
- Pixel accuracy

**How to use:**
```bash
conda activate comm
python visualize_slot_attention.py
```

**Customize by editing the `main()` function:**
```python
def main():
    checkpoint_path = 'checkpoints/slot_attention.pth'
    output_dir = 'slot_attention_visualizations'
    num_grids = 10
```

**Output:**
- Individual files: `grid_XXXX.png`
- Summary file: `summary.png` (shows multiple examples at once)

---

## Switching Between PNG and PDF

### For CNN Features:
Edit `visualize_cnn_features.py`, line ~584:
```python
output_format = 'pdf'  # Changed from 'png'
```

### Why use PDF?
- **Better text rendering** - No pixelation
- **Vector graphics** - Scales perfectly when zoomed
- **Smaller file sizes** for complex figures
- **Better for presentations** and papers

### Why use PNG?
- **Easier to view** in most image viewers
- **Works in more contexts** (web, Slack, etc.)
- **Faster to generate** slightly

---

## Understanding the Outputs

### CNN Feature Visualization Layout:

```
┌────────────────────────────────────────────────────────────┐
│ Row 0: Original | Embedding | Statistics Summary          │
├────────────────────────────────────────────────────────────┤
│ Row 1: CONV1 Avg | CONV2 Avg | CONV3 Avg | Spatial Avg   │
├────────────────────────────────────────────────────────────┤
│ Row 2: CONV1 Chs | CONV2 Chs | CONV3 Chs | Spatial Chs   │
├────────────────────────────────────────────────────────────┤
│ Row 3: (empty)   | (empty)   | (empty)   | Diagnosis Box  │
├────────────────────────────────────────────────────────────┤
│ Row 4: Channel Activation Bar Chart | Spatial Similarity  │
└────────────────────────────────────────────────────────────┘
```

**Key elements:**
- **Statistics Summary** (top right): Numerical metrics for each layer
- **Average maps**: Average activation across all channels
- **Channel grids**: Individual channels (16 shown in 4×4 grid)
- **Diagnosis Box**: Quick visual summary (✅ or ⚠️)
- **Bar chart**: Shows which channels are active/dead
- **Similarity matrix**: Shows if spatial locations have diverse features

### Slot Attention Visualization Layout:

```
┌──────────────────────────────────────────────────────────────┐
│ Row 0: Original | Reconstruction | Stats | Info             │
├──────────────────────────────────────────────────────────────┤
│ Row 1: Slot 1 Attention | Slot 2 | ... | Slot N            │
├──────────────────────────────────────────────────────────────┤
│ Row 2: Slot 1 Output | Slot 2 Output | ... | Slot N Output │
└──────────────────────────────────────────────────────────────┘
```

**Key elements:**
- **Attention masks**: Bright = high attention, dark = low attention
- **Slot outputs**: What each slot reconstructs
- **Average attention**: Shows if slot is being used (should be > 0.01)

---

## What to Look For

### In CNN Features:

#### ✅ **Good Signs:**
- Many bright channels in feature grids (not all dark)
- Spatial diversity > 0.5
- Activation sparsity < 0.4
- Channel entropy > 2.5
- Bar chart shows varied heights (not mostly zero)

#### ⚠️ **Bad Signs:**
- Most channels are dark/purple (dead neurons)
- High activation sparsity (> 0.6)
- Low spatial diversity (< 0.3)
- Bar chart is mostly flat/near zero
- Diagnosis box shows red warnings

### In Slot Attention:

#### ✅ **Good Signs:**
- 3-5 slots have bright attention masks
- Each slot focuses on different regions
- Slot outputs show meaningful object parts
- Reconstruction accuracy > 85%

#### ⚠️ **Bad Signs:**
- Only 1-2 slots have non-zero attention
- All attention masks look similar
- Most slots are completely black
- Reconstruction accuracy < 80%

---

## Quick Diagnostic Workflow

1. **Run CNN feature analysis first:**
   ```bash
   python visualize_cnn_features.py
   ```

2. **Check the console output** for automatic diagnosis:
   - Look for "⚠️ HIGH ACTIVATION SPARSITY"
   - Check the aggregate statistics

3. **Look at 2-3 visualizations:**
   - Open a small grid (3×3) and a large grid (18×19)
   - Compare the "Feature Channels" grids
   - Count how many channels have visible patterns

4. **If CNN looks bad, fix it first** before worrying about slots:
   - See `QUICK_FIX.md` for solutions
   - Retrain the model
   - Re-run visualizations

5. **Once CNN looks good, check slot attention:**
   ```bash
   python visualize_slot_attention.py
   ```

6. **Look at summary.png** to see patterns across multiple examples

---

## Customizing Visualizations

### Change Dataset:
Both scripts read from `config.py`:
```python
DATASET_VERSION = 'V1'  # or 'V2'
DATASET_SPLIT = 'training'  # or 'evaluation'
```

### Filter by Grid Size:
In `config.py`:
```python
FILTER_GRID_SIZE = (10, 10)  # Only 10×10 grids
# or
FILTER_GRID_SIZE = None  # All sizes
```

### Visualize More/Fewer Examples:
In the script's `main()` function:
```python
num_grids = 20  # Increase from 10
```

### Change Output Directory:
```python
output_dir = 'my_custom_folder'
```

---

## Generating a Full Report (PDF)

For a high-quality report with all visualizations:

1. **Edit `visualize_cnn_features.py`:**
   ```python
   output_format = 'pdf'
   num_grids = 20
   ```

2. **Run it:**
   ```bash
   python visualize_cnn_features.py
   ```

3. **Combine PDFs** (if needed):
   ```bash
   # On Mac:
   "/System/Library/Automator/Combine PDF Pages.action/Contents/Resources/join.py" \
     -o cnn_analysis_report.pdf cnn_feature_visualizations/*.pdf
   ```

---

## Troubleshooting

### "Checkpoint not found"
- Make sure `checkpoints/slot_attention.pth` exists
- Or update `checkpoint_path` in the script

### "No module named 'torch'"
- Make sure you activated the environment: `conda activate comm`

### Text Still Overlapping
- The script now uses better spacing (22×14 figure size, 5 rows)
- If still an issue, increase figure size in line ~328:
  ```python
  fig = plt.figure(figsize=(24, 16))  # Increase from (22, 14)
  ```

### PDF Too Large
- Reduce `num_grids` (fewer files)
- Or stick with PNG (slightly smaller)

### Want Interactive Viewing
- After running script, use:
  ```bash
  open slot_attention_visualizations/summary.png
  open cnn_feature_visualizations/cnn_features_grid_0022.png
  ```

---

## Next Steps After Visualization

1. **If CNN sparsity is high:**
   - Follow `QUICK_FIX.md`
   - Change ReLU → LeakyReLU
   - Lower learning rate
   - Retrain

2. **If slots aren't segmenting well:**
   - First fix CNN (above)
   - Then consider adjusting:
     - `NUM_SLOTS` (try 5, 7, or 10)
     - `SLOT_ITERATIONS` (try 5 or 7)
     - `SLOT_DIM` (try 128)

3. **Compare before/after:**
   - Save visualizations to different folders
   - Compare side-by-side

---

## File Summary

| File | Purpose | Output |
|------|---------|--------|
| `visualize_cnn_features.py` | Diagnose CNN feature extraction | PNG/PDF of feature maps + diagnosis |
| `visualize_slot_attention.py` | See slot segmentation | PNG of slot masks + reconstructions |
| `CNN_FEATURE_ANALYSIS.md` | Detailed analysis & recommendations | Documentation |
| `QUICK_FIX.md` | Minimal code changes to fix issues | Instructions |
| `VISUALIZATION_GUIDE.md` | How to use visualization scripts | This file |

---

**Quick Reference:**
```bash
# Diagnose CNN
python visualize_cnn_features.py

# Check slots
python visualize_slot_attention.py

# Generate PDFs instead of PNGs
# → Edit script to set: output_format = 'pdf'
```

