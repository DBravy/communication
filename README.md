# Grid Communication via Emergent Languages

A PyTorch implementation exploring emergent communication protocols for abstract visual reasoning tasks. This system successfully learns discrete symbolic languages (25-50 symbols, sequences of 5-10 tokens) that allow agents to communicate about complex grid patterns and solve selection tasks with 50-100 distractors.

## Inspiration

This work is inspired by the paper [**Emergence of Language with Multi-agent Games: Learning to Communicate with Sequences of Symbols**](https://arxiv.org/abs/1705.11192) by Havrylov & Titov (NIPS 2017). The paper demonstrates how agents can learn compositional communication protocols from scratch using differentiable relaxations (Gumbel-softmax).

## Overview

The system trains two neural agents (sender and receiver) to develop a communication protocol for describing abstract grid patterns. The sender views a grid and generates a discrete symbolic message. The receiver must use this message to identify the correct grid from a large set of distractors.

### Key Features

- **Emergent Discrete Languages**: Learns vocabularies of 25-50 discrete symbols with sequence lengths of 5-10 tokens
- **Selection Task**: Receiver identifies correct grid from 50-100 distractors based solely on sender's message
- **Generalization**: Learned protocols successfully generalize to unseen grid patterns
- **Variable-Length Messages**: Support for stop tokens enabling variable-length communication
- **Multiple Bottleneck Types**: Experiments with autoencoding, discrete communication, and slot attention
- **Training Stability**: Uses Gumbel-softmax straight-through estimator for gradient flow through discrete decisions

## Architecture

### Sender Agent
```
Grid Input → CNN Encoder → LSTM → Discrete Message Generation
```
- Encodes 30×30 grids with 10 possible colors
- Uses LSTM to generate sequential discrete symbols
- Employs Gumbel-softmax for differentiable discrete sampling
- Optional variable-length messages via stop tokens

### Receiver Agent
```
Discrete Message → LSTM → Grid Reconstruction/Selection
```
- Processes discrete symbolic sequences
- For selection: encodes candidate grids and computes similarity scores
- For reconstruction: decodes message back to grid representation
- Supports shared common ground (encoder's latent representation)

### Communication Protocol
The system uses a **communication bottleneck** that forces information compression through discrete symbols:

1. **Sender** observes a grid pattern
2. **Sender** generates a sequence of discrete symbols (message)
3. **Receiver** processes only the message (no direct access to grid)
4. **Receiver** must identify/reconstruct the correct grid

This bottleneck forces the agents to develop an efficient symbolic protocol.

## Configuration

Key parameters in `config.py`:

### Communication Protocol
```python
BOTTLENECK_TYPE = 'communication'  # 'autoencoder', 'communication', 'slot_attention'
VOCAB_SIZE = 25                    # Vocabulary size (25-50 typical)
MAX_MESSAGE_LENGTH = 5             # Sequence length (5-10 typical)
TEMPERATURE = 1.0                  # Gumbel-softmax temperature
USE_STOP_TOKEN = True              # Enable variable-length messages
```

### Task Configuration
```python
TASK_TYPE = 'selection'            # 'reconstruction' or 'selection'
NUM_DISTRACTORS = 99               # Number of distractor grids (50-100 typical)
RECEIVER_GETS_INPUT_PUZZLE = False # Shared common ground (encoder's latent)
```

### Network Architecture
```python
HIDDEN_DIM = 128                   # Hidden dimension for networks
LATENT_DIM = 128                   # Encoder latent dimension
NUM_CONV_LAYERS = 3                # Convolutional layers in encoder
```

## Usage

### Training a Communication System

```bash
# Configure parameters in config.py, then:
python train.py
```

### Web Interface

Launch the Flask-based training interface with live monitoring:

```bash
python app.py
```

Navigate to `http://localhost:5000` to:
- Start/stop training
- Monitor loss curves in real-time
- View message statistics and vocabulary usage
- Test generalization on unseen grids
- Visualize learned representations

### Key Training Parameters

Set these in `config.py` before training:

```python
# Core communication settings
BOTTLENECK_TYPE = 'communication'
VOCAB_SIZE = 25              # 25-50 for good results
MAX_MESSAGE_LENGTH = 5       # 5-10 for good results
TASK_TYPE = 'selection'
NUM_DISTRACTORS = 99         # 50-100 for challenging selection

# Dataset
DATASET_VERSION = 'V1'       # 'V1' or 'V2'
DATASET_SPLIT = 'training'
USE_ALL_DATASETS = False

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100000
```

## Experimental Results

The system achieves:

- **Vocabulary**: Successfully learns languages with **25-50 discrete symbols**
- **Message Length**: Effective communication with **5-10 token sequences**
- **Selection Accuracy**: High accuracy selecting correct grid from **50-100 distractors**
- **Generalization**: Learned protocols transfer to **unseen grid patterns**
- **Protocol Properties**: Emergent languages show compositionality and variability

### Example Training Dynamics

Typical training progression:
1. **Epochs 1-100**: Random baseline, ~1% accuracy (random chance)
2. **Epochs 100-500**: Rapid learning, accuracy rises to 20-40%
3. **Epochs 500-2000**: Protocol refinement, accuracy reaches 70-90%
4. **Epochs 2000+**: Fine-tuning, maintaining high accuracy on train and unseen test sets

## Bottleneck Types

The codebase supports three different information bottlenecks:

### 1. Communication (Discrete Messages)
```python
BOTTLENECK_TYPE = 'communication'
```
Forces discrete symbolic communication between sender and receiver. This is the primary focus for emergent language research.

**Key features**:
- Discrete vocabulary with configurable size
- Sequential message generation via LSTM
- Gumbel-softmax for differentiable training
- Optional variable-length messages

### 2. Autoencoder (Continuous)
```python
BOTTLENECK_TYPE = 'autoencoder'
```
Standard continuous autoencoder bottleneck for comparison.

**Key features**:
- Continuous latent representations
- Optional β-VAE with KL divergence regularization
- Direct reconstruction or selection

### 3. Slot Attention (Object-Centric)
```python
BOTTLENECK_TYPE = 'slot_attention'
```
Object-centric representations using slot attention mechanism.

**Key features**:
- Learns to decompose grids into object slots
- Iterative attention mechanism
- Each slot represents an object/pattern component

## Dataset

Uses the [ARC (Abstraction and Reasoning Corpus)](https://github.com/fchollet/ARC) dataset:
- **V1**: 800 puzzles (400 training, 400 evaluation)
- **V2**: 1120 puzzles (1000 training, 120 evaluation)
- Grid patterns with 10 colors (0-9)
- Variable grid sizes (3×3 to 30×30)

Each puzzle contains multiple input/output grid pairs demonstrating abstract reasoning transformations.

## Key Implementation Details

### Gumbel-Softmax Straight-Through Estimator

For differentiable discrete communication:

```python
# Gumbel-softmax with temperature
gumbel_logits = (logits + gumbel_noise) / temperature
soft_token = F.softmax(gumbel_logits, dim=-1)

# Straight-through: forward pass uses discrete, backward uses continuous
hard_token = F.one_hot(soft_token.argmax(dim=-1), num_classes=vocab_size)
token = hard_token.detach() - soft_token.detach() + soft_token
```

This allows gradient flow while maintaining discrete symbols in the forward pass.

### Gradient Flow Fix

The implementation includes fixes for gradient cancellation issues in multi-candidate selection:

- Uses **similarity-based scoring** (dot product) instead of concatenation
- Prevents gradient cancellation when receiver compares message against multiple candidates
- Enables stable learning of discriminative messages

### Variable-Length Messages

Stop token mechanism allows agents to learn optimal message lengths:

```python
USE_STOP_TOKEN = True
STOP_TOKEN_ID = VOCAB_SIZE  # Stop token appended to vocabulary
```

Agents can emit stop token to terminate message early, learning to balance informativeness with communication cost.

## File Structure

```
.
├── config.py                 # Main configuration file
├── model.py                  # Neural network architectures (encoder, sender, receiver)
├── dataset.py                # ARC dataset loader
├── train.py                  # Training script
├── app.py                    # Flask web interface
├── pretrain.py              # Encoder pretraining
│
├── templates/               # Web interface templates
│   ├── index.html          # Main training UI
│   └── ...
│
├── V1/                      # ARC dataset version 1
│   └── data/
│       ├── training/       # 400 training puzzles
│       └── evaluation/     # 400 evaluation puzzles
│
└── V2/                      # ARC dataset version 2
    └── data/
        ├── training/       # 1000 training puzzles
        └── evaluation/     # 120 evaluation puzzles
```

## Key Modules

### `model.py`
- `ARCEncoder`: CNN-based grid encoder
- `SenderAgent`: Generates discrete symbolic messages
- `ReceiverAgent`: Reconstructs grids from messages
- `ReceiverSelector`: Selects correct grid from candidates
- `SlotAttention`: Object-centric representation learning

### `dataset.py`
- `ARCDataset`: Loads and preprocesses ARC grids
- Supports both reconstruction and selection tasks
- Handles variable grid sizes with padding

### `train.py`
- Main training loop
- Loss computation and optimization
- Checkpoint saving and loading
- Generalization testing

### `app.py`
- Flask web server for training control
- Real-time monitoring and visualization
- Interactive training configuration

## Advanced Features

### β-VAE Support
```python
USE_BETA_VAE = True
BETA_VAE_BETA = 4.0  # β parameter for KL weighting
```

Adds KL divergence regularization to continuous bottlenecks for learning disentangled representations.

### Shared Common Ground
```python
RECEIVER_GETS_INPUT_PUZZLE = True
```

Optionally provides receiver with encoder's latent representation alongside message, modeling shared context between communicating agents.

### Generalization Testing
```python
GENERALIZATION_TEST_ENABLED = True
GENERALIZATION_TEST_DATASET_VERSION = 'V2'  # Test on V2 when training on V1
GENERALIZATION_TEST_INTERVAL = 20
```

Periodically evaluates protocol on completely unseen dataset to measure generalization.

### Similarity Testing
```python
SIMILARITY_TEST_ENABLED = True
SIMILARITY_TEST_INTERVAL = 20
```

Tests whether similar grids produce similar messages/encodings, measuring protocol consistency.

## Dependencies

```
torch >= 1.9.0
numpy
flask
matplotlib
tqdm
```

Install with:
```bash
pip install torch numpy flask matplotlib tqdm
```

## Citation

If you build upon this work, please cite the original paper that inspired it:

```bibtex
@inproceedings{havrylov2017emergence,
  title={Emergence of Language with Multi-agent Games: Learning to Communicate with Sequences of Symbols},
  author={Havrylov, Serhii and Titov, Ivan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
```

## License

This implementation is for research purposes. The ARC dataset has its own license (see `V1/LICENSE` and `V2/LICENSE`).

## Acknowledgments

- Original emergent communication framework: Havrylov & Titov (2017)
- ARC dataset: François Chollet
- Slot Attention: Locatello et al. (2020)

