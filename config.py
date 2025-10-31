"""Configuration for ARC communication model."""

# Data
DATA_PATH = 'arc-agi_test_challenges.json'  # Legacy single-file format (deprecated)
DATASET_VERSION = 'V1'  # 'V1' or 'V2' - which dataset version to use
DATASET_SPLIT = 'training'  # 'training' or 'evaluation' - which split to use
USE_COMBINED_SPLITS = False  # If True, combine both training and evaluation splits for training
USE_ALL_DATASETS = False  # If True, use ALL datasets (V1+V2, train+eval) with 100 grids held out for generalization
HOLDOUT_GRIDS_PER_CATEGORY = 25  # Number of grids to hold out from each category (V1/train, V1/eval, V2/train, V2/eval)
HOLDOUT_SEED = 42  # Random seed for consistent holdout selection across runs
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
TASK_TYPE = 'reconstruction'  # 'reconstruction', 'selection', 'puzzle_classification', 'puzzle_solving'
NUM_DISTRACTORS = 1  # Number of distractor grids for selection task (actual number will be min(NUM_DISTRACTORS, total_grids-1))
USE_INPUT_OUTPUT_PAIRS = False  # If True, train on input→output transformations (requires reconstruction or selection task)

RULE_DIM = 256  # Dimension of rule representation
PAIR_COMBINATION = 'concat'  # How to combine input/output: 'concat' or 'delta'
MAX_PUZZLES = None  # Limit number of puzzles (None = all)
MAX_TRAIN_EXAMPLES_PER_PUZZLE = None  # Limit training examples per puzzle (None = all)

# Communication Protocol (only used if BOTTLENECK_TYPE == 'communication')
VOCAB_SIZE = 25  # Size of discrete symbol vocabulary (actual vocab, stop token is separate)
MAX_MESSAGE_LENGTH = 5  # Maximum length of message sequences
TEMPERATURE = 1.0  # Gumbel-softmax temperature
RECEIVER_GETS_INPUT_PUZZLE = False # If True, receiver also gets the encoder's latent (shared common ground)
USE_STOP_TOKEN = True  # If True, agents can emit a stop token to terminate messages early (variable length)
STOP_TOKEN_ID = VOCAB_SIZE  # ID for the stop token (should be VOCAB_SIZE, as it's appended to the vocabulary)

# Loss weights
SIZE_LOSS_WEIGHT = 0.01  # Weight for size prediction loss

# Pretraining
PRETRAIN_EPOCHS = 1000000 # Number of epochs for encoder pretraining
PRETRAIN_LEARNING_RATE = 1e-4  # Learning rate for pretraining
PRETRAIN_TASK_TYPE = 'puzzle_classification'  # Options:
                                               # - 'binary': distinguish real ARC grids from noise
                                               # - 'selection': select correct grid from distractors
                                               # - 'puzzle_classification': classify grids by puzzle ID (inputs/outputs separated)
USE_PRETRAINED = False  # Whether to load pretrained encoder weights
FREEZE_ENCODER = False  # Whether to freeze encoder weights during main training (recommended when using pretrained encoder)
PRETRAINED_BINARY_PATH = 'checkpoints/pretrained_encoder_binary.pth'  # Path to binary pretrained encoder
PRETRAINED_SELECTION_PATH = 'checkpoints/pretrained_encoder_selection.pth'  # Path to selection pretrained encoder
LOAD_PRETRAINED_BEFORE_PRETRAIN = None  # Path to pretrained encoder to load before starting pretraining (None = train from scratch)

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100000
DEVICE = 'cuda'  # or 'cuda'

# Logging
LOG_INTERVAL = 10
SAVE_DIR = 'checkpoints'

# Generalization Testing (test on unseen dataset during training)
GENERALIZATION_TEST_ENABLED = True  # Whether to run generalization tests on unseen dataset
GENERALIZATION_TEST_DATASET_VERSION = 'V2'  # Dataset version to test on (e.g., 'V2' when training on 'V1')
GENERALIZATION_TEST_DATASET_SPLIT = 'training'  # Split to use for generalization testing
GENERALIZATION_TEST_INTERVAL = 20  # Run generalization test every N epochs
GENERALIZATION_TEST_MAX_GRIDS = 100  # Maximum number of grids to test (None = all grids)

# Similarity Testing (test encoding consistency on similar/dissimilar pairs)
SIMILARITY_TEST_ENABLED = True  # Whether to run similarity tests on encoding consistency
SIMILARITY_TEST_INTERVAL = 20  # Run similarity test every N epochs
SIMILARITY_TEST_NUM_PAIRS = 50  # Number of similar/dissimilar pairs to test

# Memory Management (to prevent issues during long training sessions)
MAX_PLOT_POINTS = 10000  # Maximum data points kept in memory for plotting (older data is downsampled)
MAX_EPOCH_MARKERS = 50   # Maximum epoch markers displayed on plots (older ones are removed)
KEEP_LAST_CHECKPOINTS = 3  # Number of most recent checkpoint files to keep (older ones are deleted)