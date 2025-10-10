"""Configuration for ARC communication model."""

# Data
DATA_PATH = 'arc-agi_test_challenges.json'
NUM_COLORS = 10
MIN_GRID_SIZE = 3
FILTER_GRID_SIZE = None  # Set to (height, width) to only train on specific grid size, or None for all grids
MAX_GRIDS = None # Maximum number of grids to load for training (None = load all grids)

# Model Architecture
EMBEDDING_DIM = 10  # One-hot encoding dimension (matches NUM_COLORS)
HIDDEN_DIM = 128
LATENT_DIM = 128
MAX_GRID_SIZE = 30  # Maximum grid size for reconstruction
NUM_CONV_LAYERS = 3  # Number of convolutional layers in encoder (1-3 recommended)

# Bottleneck Type
BOTTLENECK_TYPE = 'communication'  # 'communication' or 'autoencoder'

# Task Type
TASK_TYPE = 'selection'  # 'reconstruction', 'selection', or 'puzzle_classification'
NUM_DISTRACTORS = 1  # Number of distractor grids for selection task (actual number will be min(NUM_DISTRACTORS, total_grids-1))

# Communication Protocol (only used if BOTTLENECK_TYPE == 'communication')
VOCAB_SIZE = 100  # Size of discrete symbol vocabulary
MAX_MESSAGE_LENGTH = 3  # Maximum length of message sequences
TEMPERATURE = 1.0  # Gumbel-softmax temperature

# Loss weights
SIZE_LOSS_WEIGHT = 0.01  # Weight for size prediction loss

# Pretraining
PRETRAIN_EPOCHS = 1000000 # Number of epochs for encoder pretraining
PRETRAIN_LEARNING_RATE = 1e-4  # Learning rate for pretraining
PRETRAIN_TASK_TYPE = 'puzzle_classification'  # Options:
                                               # - 'binary': distinguish real ARC grids from noise
                                               # - 'selection': select correct grid from distractors
                                               # - 'puzzle_classification': classify grids by puzzle ID (inputs/outputs separated)
USE_PRETRAINED = True  # Whether to load pretrained encoder weights
FREEZE_ENCODER = True  # Whether to freeze encoder weights during main training (recommended when using pretrained encoder)
PRETRAINED_BINARY_PATH = 'checkpoints/pretrained_encoder_binary.pth'  # Path to binary pretrained encoder
PRETRAINED_SELECTION_PATH = 'checkpoints/pretrained_encoder_selection.pth'  # Path to selection pretrained encoder
LOAD_PRETRAINED_BEFORE_PRETRAIN = None  # Path to pretrained encoder to load before starting pretraining (None = train from scratch)

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10000
DEVICE = 'cpu'  # or 'cuda'

# Logging
LOG_INTERVAL = 10
SAVE_DIR = 'checkpoints'

# Memory Management (to prevent issues during long training sessions)
MAX_PLOT_POINTS = 10000  # Maximum data points kept in memory for plotting (older data is downsampled)
MAX_EPOCH_MARKERS = 50   # Maximum epoch markers displayed on plots (older ones are removed)
KEEP_LAST_CHECKPOINTS = 3  # Number of most recent checkpoint files to keep (older ones are deleted)