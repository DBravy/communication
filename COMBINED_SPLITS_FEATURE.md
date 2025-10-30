# Combined Splits Training Feature

## Overview
Added the ability to train on both training and evaluation dataset splits simultaneously. When enabled, the system will load and combine both splits, giving you access to all available data for training.

## Changes Made

### 1. Configuration (`config.py`)
- Added new config option: `USE_COMBINED_SPLITS = False`
- When set to `True`, both training and evaluation splits are loaded and combined
- The `DATASET_SPLIT` setting is ignored when `USE_COMBINED_SPLITS = True`

### 2. Training Script (`train.py`)
- Added `load_dataset_with_splits()` helper function that:
  - Loads a single split when `use_combined_splits=False` (default behavior)
  - Loads both training and evaluation splits and combines them using `ConcatDataset` when `use_combined_splits=True`
  - Merges `puzzle_id_map` dictionaries when using puzzle classification task
  - Prints informative messages about dataset loading
- Updated `main()` function to use the new helper

### 3. Web Application (`app.py`)
- Added `use_combined_splits` to `training_state` dictionary
- Added `load_dataset_with_splits()` helper function (similar to train.py)
- Updated `pretrain_worker()` to use the new dataset loading function
- Updated `train_worker()` to use the new dataset loading function
- Updated `/task_config` GET endpoint to return `use_combined_splits` status
- Updated `/task_config` POST endpoint to accept and save `use_combined_splits` setting

### 4. Web Interface (`templates/index.html`)
- Added checkbox control: "Use Combined Splits (Train + Eval)" in the Data Configuration section
- Updated JavaScript to:
  - Load the `use_combined_splits` value from the server
  - Save the `use_combined_splits` value to the server
  - Mark configuration as dirty when the checkbox changes

## Usage

### Command Line (train.py)
Edit `config.py`:
```python
USE_COMBINED_SPLITS = True  # Enable combined splits
DATASET_VERSION = 'V2'      # Which dataset to use
```

Then run:
```bash
python train.py
```

### Web Interface
1. Open the web app in your browser
2. Navigate to the "Data Configuration" section
3. Check the box "Use Combined Splits (Train + Eval)"
4. Click "Save Configuration"
5. Start training (either pretraining or main training)

## Benefits
- **More Training Data**: Access to the full dataset (training + evaluation)
- **Better Generalization**: Model sees more diverse examples during training
- **Useful for Small Datasets**: Particularly beneficial for V1 (400+400=800 total)

## Notes
- When combined splits are enabled, the dataset split selector (Training/Evaluation) is ignored
- For V2: Combines 1000 training + 120 evaluation = 1120 total grids
- For V1: Combines 400 training + 400 evaluation = 800 total grids
- The `max_grids` setting applies PER split (so you could get up to 2x max_grids if set)
- Puzzle classification tasks properly merge puzzle ID mappings from both splits

## Example Output
When enabled, you'll see messages like:
```
[Dataset] Loading combined splits: training + evaluation from V2
[Dataset] Combined 1000 training grids + 120 evaluation grids
```

Or for puzzle classification:
```
[Dataset] Combined 1000 training grids + 120 evaluation grids
[Dataset] Total unique puzzles: 120
```

