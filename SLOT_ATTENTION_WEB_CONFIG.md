# Slot Attention Web Configuration - Summary

This document summarizes the changes made to enable slot-attention bottleneck configuration through the web app.

## Changes Made

### 1. Backend Configuration (app.py)

#### Added Slot-Attention Parameters to Training State (lines 280-285)
- `num_slots`: Number of object slots (default: 7)
- `slot_dim`: Dimension of each slot (default: 64)
- `slot_iterations`: Number of attention iterations (default: 3)
- `slot_hidden_dim`: Hidden dimension for slot MLP (default: 128)
- `slot_eps`: Small epsilon for numerical stability (default: 1e-8)

#### Added to GET Endpoint `/task_config` (lines 2725-2730)
Returns current slot-attention configuration to the frontend.

#### Added to POST Endpoint `/task_config` (lines 2834-2844)
Accepts slot-attention configuration updates from the frontend.

#### Added to Checkpoint Saving (lines 220-225)
Saves slot-attention parameters in checkpoints for persistence.

#### Updated Model Instantiation (5 locations)
All places where `ARCAutoencoder` is instantiated now pass slot-attention parameters:
- Main training worker (lines 1363-1369)
- Batch test worker (lines 1952-1958)
- Finetune worker (lines 2320-2326)
- Solve puzzle route (lines 3481-3487)

All model instantiations include:
```python
use_beta_vae=training_state.get('use_beta_vae', False),
beta=training_state.get('beta_vae_beta', 4.0),
num_slots=training_state.get('num_slots', 7),
slot_dim=training_state.get('slot_dim', 64),
slot_iterations=training_state.get('slot_iterations', 3),
slot_hidden_dim=training_state.get('slot_hidden_dim', 128),
slot_eps=training_state.get('slot_eps', 1e-8)
```

### 2. Frontend Configuration (templates/index.html)

#### Added Bottleneck Type Option (line 584)
Added "Slot Attention" as a third option in the bottleneck type dropdown:
```html
<option value="slot_attention">Slot Attention</option>
```

#### Added Slot-Attention Configuration Section (lines 702-732)
New UI section with controls for:
- Number of Slots (1-20)
- Slot Dimension (16-512)
- Attention Iterations (1-10)
- MLP Hidden Dimension (16-512)

The section includes helpful tooltips explaining each parameter's purpose.

#### Updated JavaScript Functions

**updateBottleneckVisibility() (lines 1141-1153)**
- Shows slot-attention section when bottleneck_type='slot_attention'
- Hides it for other bottleneck types
- Called automatically when bottleneck type changes

**loadConfig() (lines 1077-1080)**
- Loads slot-attention parameters from server

**saveConfig() (lines 1314-1317 and 1372-1375)**
- Extracts slot-attention values from form
- Sends them to server in configuration update

### 3. Configuration Flow

1. **Page Load**: 
   - Frontend fetches current config from `/task_config`
   - Slot-attention values populate form fields
   - `updateBottleneckVisibility()` shows/hides section based on bottleneck_type

2. **User Changes Config**:
   - User selects "Slot Attention" from dropdown
   - Slot-attention section becomes visible
   - User adjusts parameters
   - Clicks "Save Configuration"
   - Frontend POSTs to `/task_config`
   - Backend updates training_state

3. **Training Starts**:
   - Backend reads slot-attention params from training_state
   - Passes them to `ARCAutoencoder` constructor
   - Model is created with slot-attention bottleneck
   - Parameters are saved in checkpoints

4. **Checkpoint Loading**:
   - Slot-attention params saved in checkpoint
   - Can be loaded for finetuning or solving

## How to Use

1. Open the web app at http://localhost:5002
2. In the "Task Configuration" section:
   - Set "Bottleneck Type" to "Slot Attention"
3. The "🎰 Slot Attention Configuration" section will appear
4. Adjust parameters:
   - **Number of Slots**: How many object slots to use (K in paper)
   - **Slot Dimension**: Size of each slot representation
   - **Attention Iterations**: How many refinement steps (T in paper)
   - **MLP Hidden Dim**: Hidden layer size in slot update MLP
5. Click "Save Configuration"
6. Start training with "Start Main Training"

## Parameters Guide

### Number of Slots (K)
- Controls how many objects the model can represent
- Recommended: 5-10 for ARC grids
- Higher values = more capacity but slower training

### Slot Dimension
- Size of each slot's representation
- Recommended: 64-128
- Should be compatible with encoder's feature dimension

### Attention Iterations (T)
- Number of iterative refinement steps
- Recommended: 3-5
- More iterations = better slot assignment but slower

### MLP Hidden Dim
- Hidden layer size in slot update MLP
- Recommended: 128-256
- Larger = more expressive but more parameters

## Testing

All changes have been verified:
- ✅ No linter errors in app.py
- ✅ No linter errors in index.html
- ✅ Parameters flow from config.py → training_state → model
- ✅ UI shows/hides appropriately based on bottleneck type
- ✅ Checkpoint saving includes slot-attention params

## Related Files

- `config.py`: Default slot-attention configuration values
- `model.py`: ARCAutoencoder accepts slot-attention parameters
- `app.py`: Backend API and training logic
- `templates/index.html`: Frontend UI and JavaScript

## Notes

- Slot-attention bottleneck only supports reconstruction task currently
- Selection and puzzle_classification tasks will show NotImplementedError
- The section only displays when bottleneck_type='slot_attention'
- All parameters have sensible defaults from config.py
- Epsilon parameter (slot_eps) is not exposed in UI due to advanced nature

