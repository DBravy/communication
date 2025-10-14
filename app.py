"""Flask web server for ARC training control and monitoring with pretraining support."""

from flask import Flask, render_template, jsonify, Response, request
import threading
import queue
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

import config
from dataset import ARCDataset, collate_fn, collate_fn_puzzle_classification
from model import ARCEncoder, ARCAutoencoder

app = Flask(__name__)

# Utility function for getting data path based on dataset version and split
def get_data_path(dataset_version='V2', dataset_split='training'):
    """Get the path to the dataset based on version and split."""
    if dataset_version in ['V1', 'V2']:
        # New directory-based format
        return os.path.join(dataset_version, 'data', dataset_split)
    else:
        # Legacy single-file format
        return config.DATA_PATH

# Utility function for saving checkpoints with full config
def save_full_checkpoint(model, optimizer, epoch, batch, checkpoint_name, training_state, val_loss=None, val_acc=None):
    """Save a complete checkpoint with model, optimizer, and all configuration."""
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # Sanitize checkpoint name (remove invalid characters)
    safe_name = "".join(c for c in checkpoint_name if c.isalnum() or c in (' ', '-', '_')).strip()
    if not safe_name:
        safe_name = f"checkpoint_{epoch}_{batch}"
    
    checkpoint_path = os.path.join(config.SAVE_DIR, f'{safe_name}.pth')
    
    # Build comprehensive checkpoint data
    checkpoint_data = {
        # Training state
        'epoch': epoch,
        'batch': batch,
        'val_loss': val_loss,
        'val_acc': val_acc,
        
        # Model and optimizer
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': model.encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        
        # Task configuration
        'task_type': training_state['task_type'],
        'num_distractors': training_state['num_distractors'],
        'bottleneck_type': training_state['bottleneck_type'],
        'use_input_output_pairs': training_state['use_input_output_pairs'],
        'receiver_gets_input_puzzle': training_state['receiver_gets_input_puzzle'],
        
        # Data configuration
        'max_grids': training_state['max_grids'],
        'filter_grid_size': training_state['filter_grid_size'],
        
        # Model architecture
        'hidden_dim': training_state['hidden_dim'],
        'latent_dim': training_state['latent_dim'],
        'num_conv_layers': training_state['num_conv_layers'],
        
        # Communication protocol
        'vocab_size': training_state['vocab_size'],
        'max_message_length': training_state['max_message_length'],
        'temperature': training_state['temperature'],
        'use_stop_token': training_state['use_stop_token'],
        
        # Training hyperparameters
        'batch_size': training_state['batch_size'],
        'learning_rate': training_state['learning_rate'],
        'pretrain_learning_rate': training_state['pretrain_learning_rate'],
        'num_epochs': training_state['num_epochs'],
        'pretrain_epochs': training_state['pretrain_epochs'],
        
        # Pretraining configuration
        'pretrain_task_type': training_state['pretrain_task_type'],
        'use_pretrained': training_state['use_pretrained'],
        'freeze_encoder': training_state['freeze_encoder'],
        
        # Static config values
        'num_colors': config.NUM_COLORS,
        'embedding_dim': config.EMBEDDING_DIM,
        'max_grid_size': config.MAX_GRID_SIZE,
    }
    
    torch.save(checkpoint_data, checkpoint_path)
    return checkpoint_path

# Global state
training_state = {
    'running': False,
    'mode': None,  # 'pretrain' or 'train'
    # Task configuration
    'task_type': getattr(config, 'TASK_TYPE', 'reconstruction'),
    'num_distractors': getattr(config, 'NUM_DISTRACTORS', 3),
    'bottleneck_type': getattr(config, 'BOTTLENECK_TYPE', 'communication'),
    'use_input_output_pairs': getattr(config, 'USE_INPUT_OUTPUT_PAIRS', False),
    'receiver_gets_input_puzzle': getattr(config, 'RECEIVER_GETS_INPUT_PUZZLE', False),
    # Data configuration
    'dataset_version': getattr(config, 'DATASET_VERSION', 'V2'),  # 'V1' or 'V2'
    'dataset_split': getattr(config, 'DATASET_SPLIT', 'training'),  # 'training' or 'evaluation'
    'max_grids': getattr(config, 'MAX_GRIDS', None),
    'filter_grid_size': getattr(config, 'FILTER_GRID_SIZE', None),
    # Model architecture
    'hidden_dim': getattr(config, 'HIDDEN_DIM', 128),
    'latent_dim': getattr(config, 'LATENT_DIM', 128),
    'num_conv_layers': getattr(config, 'NUM_CONV_LAYERS', 3),
    # Communication protocol
    'vocab_size': getattr(config, 'VOCAB_SIZE', 100),
    'max_message_length': getattr(config, 'MAX_MESSAGE_LENGTH', 3),
    'temperature': getattr(config, 'TEMPERATURE', 1.0),
    'use_stop_token': getattr(config, 'USE_STOP_TOKEN', False),
    # Training hyperparameters
    'batch_size': getattr(config, 'BATCH_SIZE', 32),
    'learning_rate': getattr(config, 'LEARNING_RATE', 1e-5),
    'pretrain_learning_rate': getattr(config, 'PRETRAIN_LEARNING_RATE', 1e-4),
    'num_epochs': getattr(config, 'NUM_EPOCHS', 10000),
    'pretrain_epochs': getattr(config, 'PRETRAIN_EPOCHS', 1000000),
    # STEP 1: Pretraining configuration (affects pretraining process)
    'pretrain_task_type': getattr(config, 'PRETRAIN_TASK_TYPE', 'binary'),  # Which pretraining task to use
    'load_pretrained_before_pretrain': None,  # Path to encoder checkpoint to CONTINUE pretraining from
    # STEP 2: Main training configuration (affects main training process)
    'use_pretrained': getattr(config, 'USE_PRETRAINED', True),  # Whether to LOAD pretrained encoder for main training
    'freeze_encoder': getattr(config, 'FREEZE_ENCODER', False),  # Whether to FREEZE encoder during main training
    # Training state
    'epoch': 0,
    'batch': 0,
    'metrics': {
        'loss': 0,
        'accuracy': 0
    },
    'viz_sample_idx': 0  # Counter for rotating through different samples in visualization
}

training_thread = None
metrics_queue = queue.Queue()
reconstructions_queue = queue.Queue()
stop_flag = threading.Event()
status_queue = queue.Queue()
save_checkpoint_queue = queue.Queue()  # Queue for checkpoint save requests

# Finetuning state and queues
finetuning_thread = None
finetuning_progress_queue = queue.Queue()
finetuning_state = {
    'running': False,
    'stop_requested': False,
    'puzzle_id': None,
    'epoch': 0,
    'total_epochs': 0,
    'loss': 0,
    'accuracy': 0
}

# Batch testing state and queues
batch_test_thread = None
batch_test_progress_queue = queue.Queue()
batch_test_state = {
    'running': False,
    'stop_requested': False,
    'current_puzzle_index': 0,
    'total_puzzles': 0,
    'current_puzzle_id': None,
    'results': []
}

# Store historical metrics for persistence across page reloads
# Limit to 5000 points to avoid excessive memory usage
MAX_HISTORY_POINTS = 5000
metrics_history = []  # List of {'batch_num': int, 'loss': float, 'accuracy': float}
history_lock = threading.Lock()

def add_metrics_to_history(epoch, batch, loss, accuracy):
    """Add metrics to history with automatic pruning to stay under MAX_HISTORY_POINTS."""
    global metrics_history
    with history_lock:
        # Calculate a global batch number for consistent x-axis
        batch_num = (epoch - 1) * 1000 + batch  # Rough estimate
        
        metrics_history.append({
            'batch_num': batch_num,
            'loss': float(loss),
            'accuracy': float(accuracy)
        })
        
        # Prune old data if we exceed the limit
        # Keep more recent data and decimate older data
        if len(metrics_history) > MAX_HISTORY_POINTS:
            # Keep the most recent half at full resolution
            keep_recent = MAX_HISTORY_POINTS // 2
            recent_data = metrics_history[-keep_recent:]
            
            # Decimate the older half (keep every other point)
            older_data = metrics_history[:-keep_recent:2]
            
            metrics_history = older_data + recent_data

class NoiseGridDataset(Dataset):
    """Generates random noise grids."""
    def __init__(self, num_samples=10000, min_size=3, max_size=30, num_colors=10):
        self.num_samples = num_samples
        self.min_size = min_size
        self.max_size = max_size
        self.num_colors = num_colors
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        h = np.random.randint(self.min_size, self.max_size + 1)
        w = np.random.randint(self.min_size, self.max_size + 1)
        grid = np.random.randint(0, self.num_colors, size=(h, w), dtype=np.int64)
        
        # Always pad to 30x30
        pad_h = max(0, 30 - h)
        pad_w = max(0, 30 - w)
        grid = np.pad(grid, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        
        grid_tensor = torch.from_numpy(grid).long()
        return grid_tensor, (h, w)


class BinaryARCDataset(Dataset):
    """Combines real ARC grids (label=1) and noise grids (label=0)."""
    def __init__(self, arc_dataset, noise_dataset):
        self.arc_dataset = arc_dataset
        self.noise_dataset = noise_dataset
        
    def __len__(self):
        return len(self.arc_dataset) + len(self.noise_dataset)
    
    def __getitem__(self, idx):
        if idx < len(self.arc_dataset):
            grid, size = self.arc_dataset[idx]
            label = 1
        else:
            noise_idx = idx - len(self.arc_dataset)
            grid, size = self.noise_dataset[noise_idx]
            label = 0
        return grid, size, label


def collate_fn_with_labels(batch):
    """Custom collate function that handles labels (all grids already 30x30)."""
    grids, sizes, labels = zip(*batch)
    
    # All grids are already 30x30, just stack them
    batch_grids = torch.stack(grids)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return batch_grids, sizes, labels_tensor


class EncoderClassifier(nn.Module):
    """Binary classifier for pretraining."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def forward(self, grids, sizes=None):
        latent = self.encoder(grids, sizes=sizes)
        logits = self.classifier(latent)
        return logits


class EncoderPuzzleClassifier(nn.Module):
    """Multi-class puzzle classifier for pretraining."""
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(encoder.latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, grids, sizes=None):
        latent = self.encoder(grids, sizes=sizes)
        logits = self.classifier(latent)
        return logits


class EncoderSelector(nn.Module):
    """Selection classifier for pretraining with selection task."""
    def __init__(self, encoder, num_colors, embedding_dim, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.num_colors = num_colors
        self.hidden_dim = hidden_dim
        
        # Encoder for candidate grids
        self.color_embed = nn.Embedding(num_colors, embedding_dim)
        self.conv1 = nn.Conv2d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.grid_fc = nn.Linear(hidden_dim * 4 * 4, encoder.latent_dim)
        
        # Scoring
        self.score_fc = nn.Linear(encoder.latent_dim * 2, 1)
        self.relu = nn.ReLU()
        
    def encode_candidates(self, candidates, candidate_sizes=None):
        """Encode candidate grids."""
        N, H, W = candidates.shape
        x = self.color_embed(candidates)
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Pool only the actual content region, not the padding!
        if candidate_sizes is not None:
            # Adaptive pool per candidate based on actual size
            pooled_features = []
            for i in range(N):
                h, w = candidate_sizes[i]
                # Extract only the actual content region
                content = x[i:i+1, :, :h, :w]  # [1, hidden_dim, h, w]
                # Pool this to 4x4
                pooled = self.adaptive_pool(content)  # [1, hidden_dim, 4, 4]
                pooled_features.append(pooled)
            x = torch.cat(pooled_features, dim=0)  # [N, hidden_dim, 4, 4]
        else:
            # Fallback to regular pooling
            x = self.adaptive_pool(x)
        
        x = x.reshape(N, -1)
        x = self.relu(self.grid_fc(x))
        return x
    
    def forward(self, target_grids, sizes, candidates_list, candidates_sizes_list=None):
        """Forward pass for selection pretraining."""
        target_latents = self.encoder(target_grids, sizes=sizes)
        logits_list = []
        for i, candidates in enumerate(candidates_list):
            target_latent = target_latents[i:i+1]
            candidate_sizes = candidates_sizes_list[i] if candidates_sizes_list is not None else None
            cand_repr = self.encode_candidates(candidates, candidate_sizes=candidate_sizes)
            num_candidates = candidates.shape[0]
            target_latent_expanded = target_latent.expand(num_candidates, -1)
            combined = torch.cat([target_latent_expanded, cand_repr], dim=-1)
            logits = self.score_fc(combined).squeeze(-1)
            logits_list.append(logits)
        return logits_list


def get_reconstructions(model, grids, sizes, device, num_samples=1):
    """Get reconstruction data for visualization - rotates through different samples."""
    global training_state
    model.eval()
    reconstructions = []
    
    with torch.no_grad():
        # Rotate through samples in the batch
        batch_size = grids.shape[0]
        i = training_state['viz_sample_idx'] % batch_size
        training_state['viz_sample_idx'] += 1
        
        grid = grids[i]
        actual_h, actual_w = sizes[i]
        input_grid = grid[:actual_h, :actual_w].cpu().numpy()
        
        single_grid = grid.unsqueeze(0)
        model_output = model(single_grid, [(actual_h, actual_w)], temperature=1.0)
        
        # Unpack based on number of return values (handle stop tokens)
        message_lengths = None
        if len(model_output) == 4:  # reconstruction with message_lengths
            logits_list, _, messages, message_lengths = model_output
        else:  # autoencoder mode
            logits_list, _, messages = model_output
        
        recon = logits_list[0].argmax(dim=1).squeeze(0).cpu().numpy()
        
        # Only get message if in communication mode
        if training_state['bottleneck_type'] == 'communication' and messages is not None:
            msg = messages[0].cpu().tolist()
            # Truncate to actual length if stop tokens are enabled
            if message_lengths is not None:
                actual_length = message_lengths[0].item()
                msg = msg[:actual_length]  # Only include up to stop token
        else:
            msg = None  # No message in autoencoder mode
        
        min_h = min(actual_h, recon.shape[0])
        min_w = min(actual_w, recon.shape[1])
        correct_pixels = (input_grid[:min_h, :min_w] == recon[:min_h, :min_w]).sum()
        total_pixels = min_h * min_w
        accuracy = 100.0 * correct_pixels / total_pixels if total_pixels > 0 else 0.0
        
        reconstructions.append({
            'task_type': 'reconstruction',
            'input': input_grid.tolist(),
            'output': recon.tolist(),
            'message': msg,
            'actual_size': [int(actual_h), int(actual_w)],
            'accuracy': float(accuracy)
        })
    
    model.train()
    return reconstructions

def get_classification_preview(model, grids, sizes, device):
    """Get classification preview data for visualization - rotates through different samples."""
    global training_state
    model.eval()
    
    with torch.no_grad():
        # Rotate through samples in the batch
        batch_size = grids.shape[0]
        i = training_state['viz_sample_idx'] % batch_size
        training_state['viz_sample_idx'] += 1
        
        grid = grids[i]
        actual_h, actual_w = sizes[i]
        input_grid = grid[:actual_h, :actual_w].cpu().numpy()
        
        single_grid = grid.unsqueeze(0)
        model_output = model(single_grid, [(actual_h, actual_w)], temperature=1.0)
        
        # Unpack based on number of return values (handle stop tokens)
        if len(model_output) == 4:  # puzzle_classification with message_lengths
            classification_logits, _, messages, message_lengths = model_output
        else:
            classification_logits, _, messages = model_output
            message_lengths = None
        
        # Get probabilities and prediction
        probs = torch.softmax(classification_logits[0], dim=0).cpu().numpy()
        pred_class = int(probs.argmax())
        
        # Get message
        if training_state['bottleneck_type'] == 'communication':
            sender_output = model.sender(single_grid, sizes=[(actual_h, actual_w)], temperature=1.0)
            if len(sender_output) == 3:  # with message_lengths
                message, _, msg_lengths = sender_output
                msg = message[0].cpu().tolist()
                # Truncate to actual length
                actual_length = msg_lengths[0].item()
                msg = msg[:actual_length]
            else:
                message, _ = sender_output
                msg = message[0].cpu().tolist()
        else:
            msg = None  # No message in autoencoder mode
            
        preview = [{
            'task_type': 'puzzle_classification',
            'input': input_grid.tolist(),
            'pred_class': pred_class,
            'top_probs': probs.tolist()[:10],
            'message': msg,
            'actual_size': [int(actual_h), int(actual_w)]
        }]
        
    model.train()
    return preview

def get_selections(model, batch_data, device, task_type):
    """Get selection data for visualization - rotates through different samples."""
    global training_state
    model.eval()
    selections = []
    
    with torch.no_grad():
        # Unpack based on whether we're using input-output pairs
        use_input_output_pairs = training_state.get('use_input_output_pairs', False)
        
        if use_input_output_pairs:
            # 7 elements: input and output grids separate
            input_grids, input_sizes, output_grids, output_sizes, candidates_list, candidates_sizes_list, target_indices = batch_data
            grids = input_grids  # Use input grids for sender encoding
            sizes = input_sizes
        else:
            # 5 elements: original format (self-supervised)
            grids, sizes, candidates_list, candidates_sizes_list, target_indices = batch_data
        
        grids = grids.to(device)
        candidates_list = [c.to(device) for c in candidates_list]
        target_indices = target_indices.to(device)
        
        # Rotate through samples in the batch
        batch_size = grids.shape[0]
        i = training_state['viz_sample_idx'] % batch_size
        training_state['viz_sample_idx'] += 1
        
        grid = grids[i]
        actual_h, actual_w = sizes[i]
        target_grid = grid[:actual_h, :actual_w].cpu().numpy()
        
        single_grid = grid.unsqueeze(0)
        candidates = candidates_list[i]
        candidate_sizes = candidates_sizes_list[i]
        target_idx = target_indices[i]
        
        # Forward pass
        model_output = model(
            single_grid, [(actual_h, actual_w)], temperature=1.0,
            candidates_list=[candidates],
            candidates_sizes_list=[candidate_sizes],
            target_indices=target_idx.unsqueeze(0)
        )
        
        # Unpack based on number of return values (handle stop tokens)
        if len(model_output) == 5:  # selection with message_lengths
            selection_logits_list, reconstruction_logits_list, _, messages, message_lengths = model_output
        else:  # autoencoder mode
            selection_logits_list, reconstruction_logits_list, _, messages = model_output
            message_lengths = None
        
        sel_logits = selection_logits_list[0]
        probs = torch.softmax(sel_logits, dim=0).cpu().numpy()
        pred_idx = sel_logits.argmax().item()
        
        if training_state['bottleneck_type'] == 'communication':
            sender_output = model.sender(single_grid, sizes=[(actual_h, actual_w)], temperature=1.0)
            if len(sender_output) == 3:  # with message_lengths
                message, _, msg_lengths = sender_output
                msg = message[0].cpu().tolist()
                # Truncate to actual length
                actual_length = msg_lengths[0].item()
                msg = msg[:actual_length]
            else:
                message, _ = sender_output
                msg = message[0].cpu().tolist()
        else:
            msg = None  # No message in autoencoder mode
        
        # Get all candidate grids
        candidates_data = []
        for c_idx in range(len(candidates)):
            # Get candidate size - it's the actual size of the candidate grid
            c_h, c_w = candidate_sizes[c_idx]
            cand_grid = candidates[c_idx][:c_h, :c_w].cpu().numpy()
            candidates_data.append({
                'grid': cand_grid.tolist(),
                'probability': float(probs[c_idx]),
                'is_target': c_idx == target_idx.item(),
                'is_selected': c_idx == pred_idx
            })
        
        correct = (pred_idx == target_idx.item())
        confidence = float(probs[pred_idx])
        
        selections.append({
            'task_type': 'selection',
            'target': target_grid.tolist(),
            'candidates': candidates_data,
            'message': msg,
            'actual_size': [int(actual_h), int(actual_w)],
            'correct': correct,
            'confidence': confidence * 100,
            'num_candidates': len(candidates_data)
        })
    
    model.train()
    return selections

def pretrain_worker():
    """Background pretraining worker."""
    global training_state
    
    try:
        device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        
        # Get pretraining task type from training_state
        pretrain_task_type = training_state['pretrain_task_type']
        num_epochs = training_state['pretrain_epochs']
        
        # Determine num_distractors and track_puzzle_ids for dataset
        if pretrain_task_type == 'selection':
            num_distractors_for_dataset = training_state['num_distractors']
            track_puzzle_ids = False
        elif pretrain_task_type == 'puzzle_classification':
            num_distractors_for_dataset = 0
            track_puzzle_ids = True
        else:
            num_distractors_for_dataset = 0
            track_puzzle_ids = False
        
        # Get dataset path based on version and split
        data_path = get_data_path(
            dataset_version=training_state['dataset_version'],
            dataset_split=training_state['dataset_split']
        )
        
        # Load ARC dataset
        arc_dataset = ARCDataset(
            data_path, 
            min_size=config.MIN_GRID_SIZE,
            filter_size=training_state['filter_grid_size'],  # ✅ Read from training_state
            max_grids=training_state['max_grids'],           # ✅ Read from training_state
            num_distractors=num_distractors_for_dataset,     # (or num_distractors in train_worker)
            track_puzzle_ids=track_puzzle_ids
        )
        
        # Create encoder
        encoder = ARCEncoder(
            num_colors=config.NUM_COLORS,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=training_state['hidden_dim'],
            latent_dim=training_state['latent_dim'],
            num_conv_layers=training_state['num_conv_layers']
        )
        
        # PRETRAINING STEP 1: Optionally load a checkpoint to CONTINUE pretraining
        # (This is different from main training's use_pretrained - this continues pretraining from a previous run)
        load_from_path = training_state.get('load_pretrained_before_pretrain')
        if load_from_path and os.path.exists(load_from_path):
            print(f'[PRETRAIN] Loading checkpoint from {load_from_path} to continue pretraining...')
            checkpoint = torch.load(load_from_path, map_location='cpu')
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print(f"[PRETRAIN] ✓ Loaded checkpoint - continuing pretraining from epoch {checkpoint.get('epoch', 0)}")
        elif load_from_path:
            print(f'[PRETRAIN] Warning: Checkpoint path specified but not found: {load_from_path}')
            print('[PRETRAIN] Starting pretraining from scratch')
        else:
            print('[PRETRAIN] Starting pretraining from scratch (no checkpoint specified)')
        
        best_val_acc = 0.0
        
        # Determine save path based on pretraining task type
        if pretrain_task_type == 'selection':
            save_path = os.path.join(config.SAVE_DIR, 'pretrained_encoder_selection.pth')
        elif pretrain_task_type == 'puzzle_classification':
            save_path = os.path.join(config.SAVE_DIR, 'pretrained_encoder_puzzle.pth')
        else:  # binary
            save_path = os.path.join(config.SAVE_DIR, 'pretrained_encoder_binary.pth')
        
        if pretrain_task_type == 'puzzle_classification':
            # Puzzle classification task pretraining
            num_puzzles = len(arc_dataset.puzzle_id_map)
            num_classes = num_puzzles * 2
            
            print(f'Setting up puzzle classification task...')
            print(f'Number of puzzles: {num_puzzles}')
            print(f'Number of classes (inputs + outputs): {num_classes}')
            
            train_size = int(0.8 * len(arc_dataset))
            val_size = len(arc_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                arc_dataset, [train_size, val_size]
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=training_state['batch_size'],
                shuffle=True,
                collate_fn=collate_fn_puzzle_classification,
                num_workers=0
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=training_state['batch_size'],
                shuffle=False,
                collate_fn=collate_fn_puzzle_classification,
                num_workers=0
            )
            
            model = EncoderPuzzleClassifier(encoder, num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=training_state['pretrain_learning_rate'])
            
            # Training loop for puzzle classification
            for epoch in range(num_epochs):
                if stop_flag.is_set():
                    break
                
                model.train()
                training_state['epoch'] = epoch + 1
                
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_idx, (grids, sizes, labels) in enumerate(train_loader):
                    if stop_flag.is_set():
                        break
                    
                    grids = grids.to(device)
                    labels = labels.to(device)
                    training_state['batch'] = batch_idx + 1
                    
                    optimizer.zero_grad()
                    logits = model(grids, sizes=sizes)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    pred = logits.argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total += labels.size(0)
                    
                    avg_loss = total_loss / (batch_idx + 1)
                    accuracy = 100. * correct / total if total > 0 else 0.0
                    
                    training_state['metrics'] = {
                        'loss': avg_loss,
                        'accuracy': accuracy
                    }
                    
                    # Add to persistent history
                    add_metrics_to_history(epoch + 1, batch_idx + 1, avg_loss, accuracy)
                    
                    metrics_queue.put({
                        'epoch': epoch + 1,
                        'batch': batch_idx + 1,
                        'metrics': training_state['metrics']
                    })
                    
                    time.sleep(0.01)
                
                # Validation
                model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for grids, sizes, labels in val_loader:
                        grids = grids.to(device)
                        labels = labels.to(device)
                        logits = model(grids, sizes=sizes)
                        pred = logits.argmax(dim=1)
                        val_correct += (pred == labels).sum().item()
                        val_total += labels.size(0)
                
                val_acc = 100. * val_correct / val_total if val_total > 0 else 0.0
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    os.makedirs(config.SAVE_DIR, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'encoder_state_dict': encoder.state_dict(),
                        'classifier_state_dict': model.classifier.state_dict(),
                        'val_acc': val_acc,
                        'num_classes': num_classes,
                        'pretrain_task_type': 'puzzle_classification',
                    }, save_path)
                    print(f'Saved puzzle classification pretrained encoder to {save_path}')
        
        elif pretrain_task_type == 'selection':
            # Selection task pretraining
            num_distractors = training_state['num_distractors']
            
            train_size = int(0.8 * len(arc_dataset))
            val_size = len(arc_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                arc_dataset, [train_size, val_size]
            )
            
            from functools import partial
            collate_fn_with_distractors = partial(collate_fn, num_distractors=num_distractors)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=training_state['batch_size'],
                shuffle=True,
                collate_fn=collate_fn_with_distractors,
                num_workers=0
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=training_state['batch_size'],
                shuffle=False,
                collate_fn=collate_fn_with_distractors,
                num_workers=0
            )
            
            model = EncoderSelector(
                encoder,
                num_colors=config.NUM_COLORS,
                embedding_dim=config.EMBEDDING_DIM,
                hidden_dim=training_state['hidden_dim']
            ).to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=training_state['pretrain_learning_rate'])
            
            # Training loop for selection task
            for epoch in range(num_epochs):
                if stop_flag.is_set():
                    break
                
                model.train()
                training_state['epoch'] = epoch + 1
                
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_idx, batch_data in enumerate(train_loader):
                    if stop_flag.is_set():
                        break
                    
                    grids, sizes, candidates_list, candidates_sizes_list, target_indices = batch_data
                    grids = grids.to(device)
                    candidates_list = [c.to(device) for c in candidates_list]
                    target_indices = target_indices.to(device)
                    training_state['batch'] = batch_idx + 1
                    
                    optimizer.zero_grad()
                    logits_list = model(grids, sizes, candidates_list, candidates_sizes_list)
                    
                    batch_loss = 0
                    for i, logits in enumerate(logits_list):
                        sample_loss = criterion(logits.unsqueeze(0), target_indices[i].unsqueeze(0))
                        batch_loss += sample_loss
                        pred_idx = logits.argmax()
                        correct += (pred_idx == target_indices[i]).item()
                        total += 1
                    
                    loss = batch_loss / len(logits_list)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    avg_loss = total_loss / (batch_idx + 1)
                    accuracy = 100. * correct / total if total > 0 else 0.0
                    
                    training_state['metrics'] = {
                        'loss': avg_loss,
                        'accuracy': accuracy
                    }
                    
                    # Add to persistent history
                    add_metrics_to_history(epoch + 1, batch_idx + 1, avg_loss, accuracy)
                    
                    metrics_queue.put({
                        'epoch': epoch + 1,
                        'batch': batch_idx + 1,
                        'metrics': training_state['metrics']
                    })
                    
                    time.sleep(0.01)
                
                # Validation
                model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_data in val_loader:
                        grids, sizes, candidates_list, candidates_sizes_list, target_indices = batch_data
                        grids = grids.to(device)
                        candidates_list = [c.to(device) for c in candidates_list]
                        target_indices = target_indices.to(device)
                        
                        logits_list = model(grids, sizes, candidates_list, candidates_sizes_list)
                        for i, logits in enumerate(logits_list):
                            pred_idx = logits.argmax()
                            val_correct += (pred_idx == target_indices[i]).item()
                            val_total += 1
                
                val_acc = 100. * val_correct / val_total if val_total > 0 else 0.0
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    os.makedirs(config.SAVE_DIR, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'encoder_state_dict': encoder.state_dict(),
                        'selector_state_dict': model.score_fc.state_dict(),
                        'val_acc': val_acc,
                        'pretrain_task_type': 'selection',
                    }, save_path)
                    print(f'Saved selection pretrained encoder to {save_path}')
        
        else:  # binary classification
            noise_dataset = NoiseGridDataset(
                num_samples=len(arc_dataset),
                min_size=config.MIN_GRID_SIZE,
                max_size=config.MAX_GRID_SIZE,
                num_colors=config.NUM_COLORS
            )
            
            binary_dataset = BinaryARCDataset(arc_dataset, noise_dataset)
            train_size = int(0.8 * len(binary_dataset))
            val_size = len(binary_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                binary_dataset, [train_size, val_size]
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=training_state['batch_size'],
                shuffle=True,
                collate_fn=collate_fn_with_labels,
                num_workers=0
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=training_state['batch_size'],
                shuffle=False,
                collate_fn=collate_fn_with_labels,
                num_workers=0
            )
            
            model = EncoderClassifier(encoder).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=training_state['pretrain_learning_rate'])
            
            # Training loop for binary classification
            for epoch in range(num_epochs):
                if stop_flag.is_set():
                    break
                
                model.train()
                training_state['epoch'] = epoch + 1
                
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_idx, (grids, sizes, labels) in enumerate(train_loader):
                    if stop_flag.is_set():
                        break
                    
                    grids = grids.to(device)
                    labels = labels.to(device)
                    training_state['batch'] = batch_idx + 1
                    
                    optimizer.zero_grad()
                    logits = model(grids, sizes=sizes)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    pred = logits.argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total += labels.size(0)
                    
                    avg_loss = total_loss / (batch_idx + 1)
                    accuracy = 100. * correct / total if total > 0 else 0.0
                    
                    training_state['metrics'] = {
                        'loss': avg_loss,
                        'accuracy': accuracy
                    }
                    
                    # Add to persistent history
                    add_metrics_to_history(epoch + 1, batch_idx + 1, avg_loss, accuracy)
                    
                    metrics_queue.put({
                        'epoch': epoch + 1,
                        'batch': batch_idx + 1,
                        'metrics': training_state['metrics']
                    })
                    
                    time.sleep(0.01)
                
                # Validation
                model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for grids, sizes, labels in val_loader:
                        grids = grids.to(device)
                        labels = labels.to(device)
                        logits = model(grids, sizes=sizes)
                        pred = logits.argmax(dim=1)
                        val_correct += (pred == labels).sum().item()
                        val_total += labels.size(0)
                
                val_acc = 100. * val_correct / val_total if val_total > 0 else 0.0
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    os.makedirs(config.SAVE_DIR, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'encoder_state_dict': encoder.state_dict(),
                        'classifier_state_dict': model.classifier.state_dict(),
                        'val_acc': val_acc,
                        'pretrain_task_type': 'binary',
                    }, save_path)
                    print(f'Saved binary pretrained encoder to {save_path}')
        
        training_state['running'] = False
        training_state['mode'] = None
        metrics_queue.put({'status': 'completed', 'best_val_acc': best_val_acc})
        
    except Exception as e:
        training_state['running'] = False
        training_state['mode'] = None
        metrics_queue.put({'status': 'error', 'message': str(e)})


def train_worker():
    """Background main training worker."""
    global training_state
    
    try:
        device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        
        # Get task settings from training_state
        task_type = training_state['task_type']
        num_distractors = training_state['num_distractors'] if task_type == 'selection' else 0
        track_puzzle_ids = task_type == 'puzzle_classification'
        use_input_output_pairs = training_state.get('use_input_output_pairs', False)
        
        # Get dataset path based on version and split
        data_path = get_data_path(
            dataset_version=training_state['dataset_version'],
            dataset_split=training_state['dataset_split']
        )
        
        dataset = ARCDataset(
            data_path, 
            min_size=config.MIN_GRID_SIZE,
            filter_size=training_state['filter_grid_size'],
            max_grids=training_state['max_grids'],
            num_distractors=num_distractors,
            track_puzzle_ids=track_puzzle_ids,
            use_input_output_pairs=use_input_output_pairs
        )
        
        # For puzzle classification, get number of classes
        num_classes = None
        if task_type == 'puzzle_classification':
            num_puzzles = len(dataset.puzzle_id_map)
            num_classes = num_puzzles * 2
            print(f'Puzzle classification setup:')
            print(f'  - Number of puzzles: {num_puzzles}')
            print(f'  - Number of classes (inputs + outputs): {num_classes}')
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create collate function based on task type
        from functools import partial
        if task_type == 'puzzle_classification':
            collate_fn_for_task = collate_fn_puzzle_classification
        else:
            collate_fn_for_task = partial(collate_fn, num_distractors=num_distractors, use_input_output_pairs=use_input_output_pairs)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_state['batch_size'],
            shuffle=True,
            collate_fn=collate_fn_for_task,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_state['batch_size'],
            shuffle=False,
            collate_fn=collate_fn_for_task,
            num_workers=0
        )
        
        encoder = ARCEncoder(
            num_colors=config.NUM_COLORS,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=training_state['hidden_dim'],
            latent_dim=training_state['latent_dim'],
            num_conv_layers=training_state['num_conv_layers']
        )
        
        # MAIN TRAINING STEP 1: Optionally load pretrained encoder from Step 1 (pretraining)
        use_pretrained = training_state['use_pretrained']
        
        # Determine which pretrained encoder to load based on pretrain_task_type
        pretrain_task_type = training_state['pretrain_task_type']
        if pretrain_task_type == 'selection':
            pretrained_path = os.path.join(config.SAVE_DIR, 'pretrained_encoder_selection.pth')
        elif pretrain_task_type == 'puzzle_classification':
            pretrained_path = os.path.join(config.SAVE_DIR, 'pretrained_encoder_puzzle.pth')
        else:  # binary
            pretrained_path = os.path.join(config.SAVE_DIR, 'pretrained_encoder_binary.pth')
        
        if use_pretrained and os.path.exists(pretrained_path):
            print(f'[MAIN TRAIN] Loading pretrained encoder from {pretrained_path}...')
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print(f"[MAIN TRAIN] ✓ Loaded pretrained encoder (task: {pretrain_task_type}, acc: {checkpoint.get('val_acc', 0):.2f}%)")
        elif use_pretrained:
            print(f'[MAIN TRAIN] Warning: Pretrained encoder requested but not found at {pretrained_path}')
            print('[MAIN TRAIN] Training from scratch instead')
        else:
            print('[MAIN TRAIN] Training from scratch (pretrained encoder disabled)')
        
        receiver_gets_input_puzzle = training_state.get('receiver_gets_input_puzzle', False)
        use_stop_token = training_state.get('use_stop_token', False)
        stop_token_id = training_state['vocab_size'] if use_stop_token else None
        model = ARCAutoencoder(
            encoder=encoder,
            vocab_size=training_state['vocab_size'],
            max_length=training_state['max_message_length'],
            num_colors=config.NUM_COLORS,
            embedding_dim=getattr(config, 'EMBEDDING_DIM', 16),
            hidden_dim=training_state['hidden_dim'],
            max_grid_size=config.MAX_GRID_SIZE,
            bottleneck_type=training_state['bottleneck_type'],
            task_type=task_type,
            num_conv_layers=training_state['num_conv_layers'],
            num_classes=num_classes,
            receiver_gets_input_puzzle=receiver_gets_input_puzzle,
            use_stop_token=use_stop_token,
            stop_token_id=stop_token_id
        ).to(device)
        
        # MAIN TRAINING STEP 2: Optionally freeze encoder weights during main training
        freeze_encoder = training_state.get('freeze_encoder', getattr(config, 'FREEZE_ENCODER', False))
        if freeze_encoder:
            print('[MAIN TRAIN] 🔒 Freezing encoder weights (encoder will NOT be updated during training)')
            for param in model.encoder.parameters():
                param.requires_grad = False
        else:
            print('[MAIN TRAIN] 🔓 Encoder weights will be updated during training')
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=training_state['learning_rate'])
        
        for epoch in range(training_state['num_epochs']):
            if stop_flag.is_set():
                break
            
            model.train()
            training_state['epoch'] = epoch + 1
            
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, batch_data in enumerate(train_loader):
                if stop_flag.is_set():
                    break
                
                # Unpack batch based on task type and use_input_output_pairs
                if use_input_output_pairs:
                    if task_type == 'selection':
                        input_grids, input_sizes, output_grids, output_sizes, candidates_list, candidates_sizes_list, target_indices = batch_data
                        input_grids = input_grids.to(device)
                        output_grids = output_grids.to(device)
                        candidates_list = [c.to(device) for c in candidates_list]
                        target_indices = target_indices.to(device)
                    else:  # reconstruction
                        input_grids, input_sizes, output_grids, output_sizes = batch_data
                        input_grids = input_grids.to(device)
                        output_grids = output_grids.to(device)
                        candidates_list = None
                        target_indices = None
                else:
                    # Original behavior (self-supervised)
                    if task_type == 'selection':
                        grids, sizes, candidates_list, candidates_sizes_list, target_indices = batch_data
                        grids = grids.to(device)
                        candidates_list = [c.to(device) for c in candidates_list]
                        target_indices = target_indices.to(device)
                        input_grids = grids
                        input_sizes = sizes
                    elif task_type == 'puzzle_classification':
                        grids, sizes, labels = batch_data
                        grids = grids.to(device)
                        labels = labels.to(device)
                        input_grids = grids
                        input_sizes = sizes
                    else:
                        grids, sizes = batch_data
                        grids = grids.to(device)
                        candidates_list = None
                        target_indices = None
                        input_grids = grids
                        input_sizes = sizes
                
                training_state['batch'] = batch_idx + 1
                
                optimizer.zero_grad()
                
                if task_type == 'puzzle_classification':
                    # Puzzle classification task (no I/O pairs)
                    model_output = model(input_grids, input_sizes, temperature=training_state['temperature'], labels=labels)
                    if len(model_output) == 4:  # with message_lengths
                        classification_logits, _, messages, message_lengths = model_output
                    else:
                        classification_logits, _, messages = model_output
                    
                    # Compute classification loss
                    loss = criterion(classification_logits, labels)
                    
                    # Calculate accuracy
                    pred = classification_logits.argmax(dim=1)
                    batch_correct = (pred == labels).sum().item()
                    batch_total = labels.size(0)
                elif task_type == 'selection':
                    # Selection task (now also returns reconstruction logits)
                    model_output = model(
                        input_grids, input_sizes, temperature=training_state['temperature'],
                        candidates_list=candidates_list, 
                        candidates_sizes_list=candidates_sizes_list,
                        target_indices=target_indices
                    )
                    if len(model_output) == 5:  # with message_lengths
                        selection_logits_list, reconstruction_logits_list, actual_sizes, messages, message_lengths = model_output
                    else:  # autoencoder mode
                        selection_logits_list, reconstruction_logits_list, actual_sizes, messages = model_output
                    
                    batch_loss = 0
                    batch_correct = 0
                    batch_total = 0
                    
                    for sample_idx, sel_logits in enumerate(selection_logits_list):
                        target_idx = target_indices[sample_idx]
                        sample_loss = criterion(sel_logits.unsqueeze(0), target_idx.unsqueeze(0))
                        batch_loss += sample_loss
                        
                        pred_idx = sel_logits.argmax()
                        batch_correct += (pred_idx == target_idx).item()
                        batch_total += 1
                    
                    loss = batch_loss / len(selection_logits_list)
                else:
                    # Reconstruction task
                    if use_input_output_pairs:
                        # Use input to generate message, reconstruct output
                        model_output = model(input_grids, input_sizes, temperature=training_state['temperature'])
                        if len(model_output) == 4:  # with message_lengths
                            logits_list, actual_sizes, messages, message_lengths = model_output
                        else:  # autoencoder mode
                            logits_list, actual_sizes, messages = model_output
                        
                        # Compute reconstruction loss for each sample (target is OUTPUT)
                        batch_loss = 0
                        batch_correct = 0
                        batch_total = 0
                        
                        for sample_idx, (logits, (actual_h, actual_w)) in enumerate(zip(logits_list, actual_sizes)):
                            actual_h, actual_w = output_sizes[sample_idx]  # Use OUTPUT size
                            H, W = logits.shape[2], logits.shape[3]
                            
                            target_grid = output_grids[sample_idx:sample_idx+1, :H, :W]  # Use OUTPUT grid
                            
                            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
                            targets_flat = target_grid.reshape(-1)
                            
                            sample_loss = criterion(logits_flat, targets_flat)
                            batch_loss += sample_loss
                            
                            pred = logits.argmax(dim=1).squeeze(0)
                            target = target_grid.squeeze(0)
                            batch_correct += (pred == target).sum().item()
                            batch_total += target.numel()
                        
                        loss = batch_loss / len(logits_list)
                    else:
                        # Original: reconstruct the same grid
                        model_output = model(input_grids, input_sizes, temperature=training_state['temperature'])
                        if len(model_output) == 4:  # with message_lengths
                            logits_list, actual_sizes, messages, message_lengths = model_output
                        else:  # autoencoder mode
                            logits_list, actual_sizes, messages = model_output
                        
                        # Compute reconstruction loss for each sample
                        batch_loss = 0
                        batch_correct = 0
                        batch_total = 0
                        
                        for sample_idx, (logits, (actual_h, actual_w)) in enumerate(zip(logits_list, actual_sizes)):
                            actual_h, actual_w = input_sizes[sample_idx]
                            H, W = logits.shape[2], logits.shape[3]
                            
                            target_grid = input_grids[sample_idx:sample_idx+1, :H, :W]
                            
                            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
                            targets_flat = target_grid.reshape(-1)
                            
                            sample_loss = criterion(logits_flat, targets_flat)
                            batch_loss += sample_loss
                            
                            pred = logits.argmax(dim=1).squeeze(0)
                            target = target_grid.squeeze(0)
                            batch_correct += (pred == target).sum().item()
                            batch_total += target.numel()
                        
                        loss = batch_loss / len(logits_list)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                correct += batch_correct
                total += batch_total
                
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100. * correct / total if total > 0 else 0.0
                
                training_state['metrics'] = {
                    'loss': avg_loss,
                    'accuracy': accuracy
                }
                
                metrics_queue.put({
                    'epoch': epoch + 1,
                    'batch': batch_idx + 1,
                    'metrics': training_state['metrics']
                })
                
                # Check for checkpoint save requests
                try:
                    while not save_checkpoint_queue.empty():
                        checkpoint_request = save_checkpoint_queue.get_nowait()
                        checkpoint_name = checkpoint_request.get('name', f'manual_epoch_{epoch+1}_batch_{batch_idx+1}')
                        try:
                            saved_path = save_full_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                epoch=epoch + 1,
                                batch=batch_idx + 1,
                                checkpoint_name=checkpoint_name,
                                training_state=training_state,
                                val_loss=None,  # Could compute validation loss if needed
                                val_acc=None
                            )
                            status_queue.put({
                                'status': 'checkpoint_saved',
                                'checkpoint_name': checkpoint_name,
                                'path': saved_path,
                                'epoch': epoch + 1,
                                'batch': batch_idx + 1
                            })
                            print(f'\n✓ Saved checkpoint: {checkpoint_name}')
                        except Exception as e:
                            status_queue.put({
                                'status': 'checkpoint_error',
                                'message': str(e),
                                'checkpoint_name': checkpoint_name
                            })
                            print(f'\n✗ Failed to save checkpoint {checkpoint_name}: {e}')
                except queue.Empty:
                    pass
                
                time.sleep(0.01)
                
                # Visualization at epoch level
                if epoch % 2 == 0:  # Visualize every 2 epochs
                    # Training batch visualization
                    if task_type == 'selection':
                        results = get_selections(model, batch_data, device, task_type)
                    elif task_type == 'puzzle_classification':
                        results = get_classification_preview(model, input_grids, input_sizes, device)
                    else:
                        results = get_reconstructions(model, input_grids, input_sizes, device, num_samples=1)
                    reconstructions_queue.put(results)
                    
                    # Validation batch visualization
                    for val_batch in val_loader:
                        if task_type == 'selection':
                            results = get_selections(model, val_batch, device, task_type)
                        elif task_type == 'puzzle_classification':
                            if use_input_output_pairs:
                                val_grids = val_batch[0].to(device)
                                val_sizes = val_batch[1]
                            else:
                                val_grids, val_sizes, val_labels = val_batch
                                val_grids = val_grids.to(device)
                            results = get_classification_preview(model, val_grids, val_sizes, device)
                        else:
                            if use_input_output_pairs:
                                val_input_grids = val_batch[0].to(device)
                                val_input_sizes = val_batch[1]
                            else:
                                val_input_grids, val_input_sizes = val_batch
                                val_input_grids = val_input_grids.to(device)
                            results = get_reconstructions(model, val_input_grids, val_input_sizes, device, num_samples=1)
                        reconstructions_queue.put(results)
                        break
                        
        training_state['running'] = False
        training_state['mode'] = None
        metrics_queue.put({'status': 'completed'})
        
    except Exception as e:
        training_state['running'] = False
        training_state['mode'] = None
        metrics_queue.put({'status': 'error', 'message': str(e)})


def batch_test_worker(puzzle_ids, checkpoint_path, dataset_version, dataset_split, epochs, lr, batch_size, early_stop_threshold, receiver_lr=None):
    """Background batch testing worker - finetunes and solves multiple puzzles."""
    global batch_test_state
    
    try:
        from puzzle_dataset import ARCSinglePuzzleDataset, collate_fn_puzzle
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        batch_test_state['running'] = True
        batch_test_state['stop_requested'] = False
        batch_test_state['total_puzzles'] = len(puzzle_ids)
        batch_test_state['current_puzzle_index'] = 0
        batch_test_state['results'] = []
        
        # Send start message
        batch_test_progress_queue.put({
            'status': 'started',
            'total_puzzles': len(puzzle_ids),
            'puzzle_ids': puzzle_ids
        })
        
        # Get data path
        data_path = get_data_path(dataset_version, dataset_split)
        
        for idx, puzzle_id in enumerate(puzzle_ids):
            # Check if stop was requested before starting next puzzle
            if batch_test_state['stop_requested']:
                batch_test_progress_queue.put({
                    'status': 'stopped',
                    'message': 'Batch test stopped by user',
                    'processed_puzzles': idx,
                    'total_puzzles': len(puzzle_ids)
                })
                batch_test_state['running'] = False
                batch_test_state['stop_requested'] = False
                return
            
            batch_test_state['current_puzzle_index'] = idx
            batch_test_state['current_puzzle_id'] = puzzle_id
            
            # Send puzzle start message
            batch_test_progress_queue.put({
                'status': 'puzzle_started',
                'puzzle_id': puzzle_id,
                'puzzle_index': idx + 1,
                'total_puzzles': len(puzzle_ids)
            })
            
            try:
                # STEP 1: Finetune on this puzzle
                batch_test_progress_queue.put({
                    'status': 'finetuning',
                    'puzzle_id': puzzle_id,
                    'puzzle_index': idx + 1
                })
                
                # Load dataset
                train_dataset = ARCSinglePuzzleDataset(data_path, puzzle_id, split='train')
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate_fn_puzzle,
                    num_workers=0
                )
                
                # Create model
                receiver_gets_input_puzzle = getattr(config, 'RECEIVER_GETS_INPUT_PUZZLE', False)
                
                encoder = ARCEncoder(
                    num_colors=config.NUM_COLORS,
                    embedding_dim=config.EMBEDDING_DIM,
                    hidden_dim=config.HIDDEN_DIM,
                    latent_dim=config.LATENT_DIM,
                    num_conv_layers=getattr(config, 'NUM_CONV_LAYERS', 3)
                )
                
                model = ARCAutoencoder(
                    encoder=encoder,
                    vocab_size=config.VOCAB_SIZE if config.BOTTLENECK_TYPE == 'communication' else None,
                    max_length=config.MAX_MESSAGE_LENGTH if config.BOTTLENECK_TYPE == 'communication' else None,
                    num_colors=config.NUM_COLORS,
                    embedding_dim=config.EMBEDDING_DIM,
                    hidden_dim=config.HIDDEN_DIM,
                    max_grid_size=config.MAX_GRID_SIZE,
                    bottleneck_type=config.BOTTLENECK_TYPE,
                    task_type='reconstruction',
                    num_conv_layers=getattr(config, 'NUM_CONV_LAYERS', 3),
                    receiver_gets_input_puzzle=receiver_gets_input_puzzle,
                    use_stop_token=getattr(config, 'USE_STOP_TOKEN', False),
                    stop_token_id=getattr(config, 'STOP_TOKEN_ID', None)
                ).to(device)
                
                # Load checkpoint if provided
                if checkpoint_path and os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    state_dict = checkpoint.get('model_state_dict', checkpoint)
                    
                    if 'receiver_reconstructor.symbol_embed.weight' in state_dict or \
                       'decoder_reconstructor.fc_decode.weight' in state_dict:
                        new_state_dict = {}
                        for k, v in state_dict.items():
                            if k.startswith('receiver_reconstructor.'):
                                new_key = k.replace('receiver_reconstructor.', 'receiver.')
                                new_state_dict[new_key] = v
                            elif k.startswith('decoder_reconstructor.'):
                                new_key = k.replace('decoder_reconstructor.', 'decoder.')
                                new_state_dict[new_key] = v
                            elif k.startswith('encoder.') or k.startswith('sender.'):
                                new_state_dict[k] = v
                        model.load_state_dict(new_state_dict, strict=False)
                    else:
                        if 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'])
                        elif 'encoder_state_dict' in checkpoint:
                            encoder.load_state_dict(checkpoint['encoder_state_dict'])
                
                criterion = nn.CrossEntropyLoss()
                
                # Use separate learning rate for receiver if specified
                if receiver_lr is not None and hasattr(model, 'receiver'):
                    # Separate receiver parameters from other parameters
                    receiver_params = []
                    other_params = []
                    
                    for name, param in model.named_parameters():
                        if 'receiver' in name:
                            receiver_params.append(param)
                        else:
                            other_params.append(param)
                    
                    # Create optimizer with parameter groups
                    optimizer = optim.Adam([
                        {'params': other_params, 'lr': lr},
                        {'params': receiver_params, 'lr': receiver_lr}
                    ])
                else:
                    # Standard optimizer with single learning rate
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                
                best_loss = float('inf')
                best_acc = 0.0
                save_dir = 'puzzle_checkpoints'
                os.makedirs(save_dir, exist_ok=True)
                
                # Finetune for specified epochs
                for epoch in range(epochs):
                    model.train()
                    total_loss = 0
                    correct = 0
                    total = 0
                    
                    for input_grids, input_sizes, output_grids, output_sizes in train_loader:
                        input_grids = input_grids.to(device)
                        output_grids = output_grids.to(device)
                        
                        optimizer.zero_grad()
                        logits_list, _, _, _ = model(input_grids, input_sizes, temperature=1.0)
                        
                        batch_loss = 0
                        batch_correct = 0
                        batch_total = 0
                        
                        for i, logits in enumerate(logits_list):
                            output_h, output_w = output_sizes[i]
                            H, W = logits.shape[2], logits.shape[3]
                            target_grid = output_grids[i:i+1, :H, :W]
                            
                            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
                            targets_flat = target_grid.reshape(-1)
                            
                            sample_loss = criterion(logits_flat, targets_flat)
                            batch_loss += sample_loss
                            
                            pred = logits.argmax(dim=1).squeeze(0)
                            target = target_grid.squeeze(0)
                            sample_correct = (pred[:output_h, :output_w] == target[:output_h, :output_w]).sum().item()
                            sample_total = output_h * output_w
                            
                            batch_correct += sample_correct
                            batch_total += sample_total
                        
                        loss = batch_loss / len(logits_list)
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                        correct += batch_correct
                        total += batch_total
                    
                    avg_loss = total_loss / len(train_loader)
                    accuracy = 100.0 * correct / total if total > 0 else 0.0
                    
                    # Check if stop was requested
                    if batch_test_state['stop_requested']:
                        batch_test_progress_queue.put({
                            'status': 'stopped',
                            'message': 'Batch test stopped by user',
                            'puzzle_id': puzzle_id,
                            'puzzle_index': idx + 1,
                            'processed_puzzles': idx + 1,
                            'total_puzzles': len(puzzle_ids)
                        })
                        batch_test_state['running'] = False
                        batch_test_state['stop_requested'] = False
                        return
                    
                    # Send epoch progress update (every 10 epochs or if early stopping)
                    if (epoch + 1) % 10 == 0 or epoch == 0 or accuracy >= early_stop_threshold:
                        batch_test_progress_queue.put({
                            'status': 'finetune_progress',
                            'puzzle_id': puzzle_id,
                            'puzzle_index': idx + 1,
                            'epoch': epoch + 1,
                            'total_epochs': epochs,
                            'loss': avg_loss,
                            'accuracy': accuracy
                        })
                    
                    # Save best model
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_acc = accuracy
                        save_path = os.path.join(save_dir, f'{puzzle_id}_best.pth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': avg_loss,
                            'train_acc': accuracy,
                            'puzzle_id': puzzle_id,
                            'bottleneck_type': config.BOTTLENECK_TYPE,
                        }, save_path)
                    
                    # Early stopping check
                    if accuracy >= early_stop_threshold:
                        break
                
                finetune_checkpoint = os.path.join(save_dir, f'{puzzle_id}_best.pth')
                
                # STEP 2: Solve using the finetuned model
                batch_test_progress_queue.put({
                    'status': 'solving',
                    'puzzle_id': puzzle_id,
                    'puzzle_index': idx + 1,
                    'finetune_loss': best_loss,
                    'finetune_acc': best_acc
                })
                
                # Load test dataset
                test_dataset = ARCSinglePuzzleDataset(data_path, puzzle_id, split='test')
                
                model.eval()
                num_correct = 0
                total_pixel_acc = 0
                last_example_grids = None  # Store the last test example for display
                
                with torch.no_grad():
                    for i in range(len(test_dataset)):
                        input_grid, input_size, output_grid, output_size = test_dataset[i]
                        
                        input_h, input_w = input_size
                        output_h, output_w = output_size
                        
                        input_actual = input_grid[:input_h, :input_w].numpy()
                        output_actual = output_grid[:output_h, :output_w].numpy()
                        
                        input_batch = input_grid.unsqueeze(0).to(device)
                        logits_list, _, messages, message_lengths = model(input_batch, [input_size], temperature=1.0)
                        
                        logits = logits_list[0]
                        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
                        predicted = pred[:output_h, :output_w]
                        
                        # Capture messages if available (communication mode)
                        message_data = None
                        if messages is not None:
                            message_seq = messages[0].cpu().tolist()
                            message_data = {
                                'symbols': message_seq,
                                'length': int(message_lengths[0]) if message_lengths is not None else len(message_seq)
                            }
                        
                        # Calculate accuracy
                        if predicted.shape == output_actual.shape:
                            exact_match = np.array_equal(predicted, output_actual)
                            correct_pixels = (predicted == output_actual).sum()
                            total_pixels = predicted.size
                        else:
                            exact_match = False
                            min_h = min(predicted.shape[0], output_actual.shape[0])
                            min_w = min(predicted.shape[1], output_actual.shape[1])
                            correct_pixels = (predicted[:min_h, :min_w] == output_actual[:min_h, :min_w]).sum() if min_h > 0 and min_w > 0 else 0
                            total_pixels = output_actual.size
                        
                        pixel_accuracy = 100.0 * correct_pixels / total_pixels if total_pixels > 0 else 0.0
                        
                        if exact_match:
                            num_correct += 1
                        total_pixel_acc += pixel_accuracy
                        
                        # Store the last example for visualization
                        last_example_grids = {
                            'input': input_actual.tolist(),
                            'target': output_actual.tolist(),
                            'predicted': predicted.tolist(),
                            'input_size': [int(input_h), int(input_w)],
                            'output_size': [int(output_h), int(output_w)],
                            'exact_match': bool(exact_match),
                            'pixel_accuracy': float(pixel_accuracy),
                            'message': message_data  # Include message sequence
                        }
                
                avg_pixel_acc = total_pixel_acc / len(test_dataset) if len(test_dataset) > 0 else 0.0
                
                # Store result
                result = {
                    'puzzle_id': puzzle_id,
                    'num_test_examples': len(test_dataset),
                    'num_correct': num_correct,
                    'avg_pixel_accuracy': avg_pixel_acc,
                    'finetune_loss': best_loss,
                    'finetune_acc': best_acc,
                    'success': True
                }
                
                batch_test_state['results'].append(result)
                
                # Send puzzle completion with grid visualization
                batch_test_progress_queue.put({
                    'status': 'puzzle_completed',
                    'puzzle_id': puzzle_id,
                    'puzzle_index': idx + 1,
                    'result': result,
                    'last_example': last_example_grids  # Include grid data for visualization
                })
                
            except Exception as e:
                # Record error for this puzzle
                error_result = {
                    'puzzle_id': puzzle_id,
                    'error': str(e),
                    'success': False
                }
                batch_test_state['results'].append(error_result)
                
                batch_test_progress_queue.put({
                    'status': 'puzzle_error',
                    'puzzle_id': puzzle_id,
                    'puzzle_index': idx + 1,
                    'error': str(e)
                })
        
        # Send completion message
        batch_test_progress_queue.put({
            'status': 'completed',
            'total_puzzles': len(puzzle_ids),
            'results': batch_test_state['results']
        })
        
        batch_test_state['running'] = False
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        batch_test_progress_queue.put({
            'status': 'error',
            'message': str(e)
        })
        batch_test_state['running'] = False
        batch_test_state['stop_requested'] = False


def finetune_worker(puzzle_id, checkpoint_path, dataset_version, dataset_split, epochs, lr, batch_size, early_stop_threshold=99.0, receiver_lr=None):
    """Background finetuning worker with progress reporting."""
    global finetuning_state
    
    try:
        from puzzle_dataset import ARCSinglePuzzleDataset, collate_fn_puzzle
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        finetuning_state['running'] = True
        finetuning_state['stop_requested'] = False
        finetuning_state['puzzle_id'] = puzzle_id
        finetuning_state['total_epochs'] = epochs
        finetuning_state['epoch'] = 0
        
        # Send start message
        finetuning_progress_queue.put({
            'status': 'started',
            'puzzle_id': puzzle_id,
            'total_epochs': epochs,
            'early_stop_threshold': early_stop_threshold
        })
        
        # Get data path
        data_path = get_data_path(dataset_version, dataset_split)
        
        # Load dataset
        train_dataset = ARCSinglePuzzleDataset(data_path, puzzle_id, split='train')
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_puzzle,
            num_workers=0
        )
        
        # Create model
        receiver_gets_input_puzzle = getattr(config, 'RECEIVER_GETS_INPUT_PUZZLE', False)
        
        encoder = ARCEncoder(
            num_colors=config.NUM_COLORS,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            latent_dim=config.LATENT_DIM,
            num_conv_layers=getattr(config, 'NUM_CONV_LAYERS', 3)
        )
        
        model = ARCAutoencoder(
            encoder=encoder,
            vocab_size=config.VOCAB_SIZE if config.BOTTLENECK_TYPE == 'communication' else None,
            max_length=config.MAX_MESSAGE_LENGTH if config.BOTTLENECK_TYPE == 'communication' else None,
            num_colors=config.NUM_COLORS,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            max_grid_size=config.MAX_GRID_SIZE,
            bottleneck_type=config.BOTTLENECK_TYPE,
            task_type='reconstruction',
            num_conv_layers=getattr(config, 'NUM_CONV_LAYERS', 3),
            receiver_gets_input_puzzle=receiver_gets_input_puzzle,
            use_stop_token=getattr(config, 'USE_STOP_TOKEN', False),
            stop_token_id=getattr(config, 'STOP_TOKEN_ID', None)
        ).to(device)
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Handle selection task checkpoint mapping
            if 'receiver_reconstructor.symbol_embed.weight' in state_dict or \
               'decoder_reconstructor.fc_decode.weight' in state_dict:
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('receiver_reconstructor.'):
                        new_key = k.replace('receiver_reconstructor.', 'receiver.')
                        new_state_dict[new_key] = v
                    elif k.startswith('decoder_reconstructor.'):
                        new_key = k.replace('decoder_reconstructor.', 'decoder.')
                        new_state_dict[new_key] = v
                    elif k.startswith('encoder.') or k.startswith('sender.'):
                        new_state_dict[k] = v
                
                model.load_state_dict(new_state_dict, strict=False)
            else:
                # Standard checkpoint
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'encoder_state_dict' in checkpoint:
                    encoder.load_state_dict(checkpoint['encoder_state_dict'])
        
        criterion = nn.CrossEntropyLoss()
        
        # Use separate learning rate for receiver if specified
        if receiver_lr is not None and hasattr(model, 'receiver'):
            # Separate receiver parameters from other parameters
            receiver_params = []
            other_params = []
            
            for name, param in model.named_parameters():
                if 'receiver' in name:
                    receiver_params.append(param)
                else:
                    other_params.append(param)
            
            # Create optimizer with parameter groups
            optimizer = optim.Adam([
                {'params': other_params, 'lr': lr},
                {'params': receiver_params, 'lr': receiver_lr}
            ])
        else:
            # Standard optimizer with single learning rate
            optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_loss = float('inf')
        best_acc = 0.0
        save_dir = 'puzzle_checkpoints'
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for input_grids, input_sizes, output_grids, output_sizes in train_loader:
                input_grids = input_grids.to(device)
                output_grids = output_grids.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                logits_list, _, _, _ = model(input_grids, input_sizes, temperature=1.0)
                
                # Compute loss
                batch_loss = 0
                batch_correct = 0
                batch_total = 0
                
                for i, logits in enumerate(logits_list):
                    output_h, output_w = output_sizes[i]
                    H, W = logits.shape[2], logits.shape[3]
                    
                    target_grid = output_grids[i:i+1, :H, :W]
                    
                    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
                    targets_flat = target_grid.reshape(-1)
                    
                    sample_loss = criterion(logits_flat, targets_flat)
                    batch_loss += sample_loss
                    
                    # Compute accuracy
                    pred = logits.argmax(dim=1).squeeze(0)
                    target = target_grid.squeeze(0)
                    sample_correct = (pred[:output_h, :output_w] == target[:output_h, :output_w]).sum().item()
                    sample_total = output_h * output_w
                    
                    batch_correct += sample_correct
                    batch_total += sample_total
                
                loss = batch_loss / len(logits_list)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                correct += batch_correct
                total += batch_total
            
            avg_loss = total_loss / len(train_loader)
            accuracy = 100.0 * correct / total if total > 0 else 0.0
            
            # Update state
            finetuning_state['epoch'] = epoch + 1
            finetuning_state['loss'] = avg_loss
            finetuning_state['accuracy'] = accuracy
            
            # Check if stop was requested
            if finetuning_state['stop_requested']:
                finetuning_progress_queue.put({
                    'status': 'stopped',
                    'message': 'Finetuning stopped by user',
                    'epoch': epoch + 1,
                    'total_epochs': epochs,
                    'train_loss': avg_loss,
                    'train_acc': accuracy,
                    'puzzle_id': puzzle_id
                })
                finetuning_state['running'] = False
                finetuning_state['stop_requested'] = False
                return
            
            # Send progress update
            finetuning_progress_queue.put({
                'status': 'progress',
                'epoch': epoch + 1,
                'total_epochs': epochs,
                'loss': avg_loss,
                'accuracy': accuracy
            })
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_acc = accuracy
                save_path = os.path.join(save_dir, f'{puzzle_id}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_loss,
                    'train_acc': accuracy,
                    'puzzle_id': puzzle_id,
                    'bottleneck_type': config.BOTTLENECK_TYPE,
                }, save_path)
            
            # Early stopping check
            if accuracy >= early_stop_threshold:
                checkpoint_file = os.path.join(save_dir, f'{puzzle_id}_best.pth')
                finetuning_progress_queue.put({
                    'status': 'early_stopped',
                    'epoch': epoch + 1,
                    'total_epochs': epochs,
                    'loss': avg_loss,
                    'accuracy': accuracy,
                    'message': f'Early stopping: reached {accuracy:.2f}% accuracy',
                    'puzzle_id': puzzle_id,
                    'checkpoint_path': checkpoint_file,
                    'train_loss': best_loss,
                    'train_acc': best_acc
                })
                break
        
        # Send completion message (only if not early stopped)
        checkpoint_file = os.path.join(save_dir, f'{puzzle_id}_best.pth')
        finetuning_progress_queue.put({
            'status': 'completed',
            'puzzle_id': puzzle_id,
            'checkpoint_path': checkpoint_file,
            'train_loss': best_loss,
            'train_acc': best_acc
        })
        
        finetuning_state['running'] = False
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        finetuning_progress_queue.put({
            'status': 'error',
            'message': str(e)
        })
        finetuning_state['running'] = False
        finetuning_state['stop_requested'] = False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/puzzle_solver')
def puzzle_solver():
    return render_template('puzzle_solver.html')


@app.route('/list_puzzles', methods=['GET'])
def list_puzzles():
    """List all available puzzle IDs from the dataset."""
    try:
        from puzzle_dataset import load_all_puzzle_ids
        
        # Get dataset version and split from query parameters
        dataset_version = request.args.get('dataset_version', 'V2')
        dataset_split = request.args.get('dataset_split', 'evaluation')
        
        # Get data path
        data_path = get_data_path(dataset_version, dataset_split)
        
        puzzle_ids = load_all_puzzle_ids(data_path)
        
        return jsonify({
            'puzzle_ids': puzzle_ids,
            'count': len(puzzle_ids),
            'dataset_version': dataset_version,
            'dataset_split': dataset_split
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/start_pretrain', methods=['POST'])
def start_pretrain():
    global training_thread, training_state
    
    if training_state['running']:
        return jsonify({'error': 'Training already running'}), 400
    
    # Wait for previous thread to finish (with timeout)
    if training_thread is not None and training_thread.is_alive():
        training_thread.join(timeout=2.0)
    
    stop_flag.clear()
    training_state['running'] = True
    training_state['mode'] = 'pretrain'
    training_state['epoch'] = 0
    training_state['batch'] = 0
    training_state['viz_sample_idx'] = 0
    
    # Clear queues and history
    while not metrics_queue.empty():
        metrics_queue.get()
    while not reconstructions_queue.empty():
        reconstructions_queue.get()
    while not status_queue.empty():  # NEW
        status_queue.get()
    while not save_checkpoint_queue.empty():
        save_checkpoint_queue.get()
    
    # Clear metrics history for new training run
    global metrics_history
    with history_lock:
        metrics_history = []
    
    training_thread = threading.Thread(target=pretrain_worker)
    training_thread.start()
    
    # NEW: Send explicit start notification
    status_queue.put({'status': 'started', 'mode': 'pretrain'})
    
    return jsonify({'status': 'started', 'mode': 'pretrain'})

@app.route('/start_train', methods=['POST'])
def start_train():
    global training_thread, training_state
    
    if training_state['running']:
        return jsonify({'error': 'Training already running'}), 400
    
    # Wait for previous thread to finish (with timeout)
    if training_thread is not None and training_thread.is_alive():
        training_thread.join(timeout=2.0)
    
    stop_flag.clear()
    training_state['running'] = True
    training_state['mode'] = 'train'
    training_state['epoch'] = 0
    training_state['batch'] = 0
    training_state['viz_sample_idx'] = 0
    
    # Clear queues and history
    while not metrics_queue.empty():
        metrics_queue.get()
    while not reconstructions_queue.empty():
        reconstructions_queue.get()
    while not status_queue.empty():  # NEW
        status_queue.get()
    while not save_checkpoint_queue.empty():
        save_checkpoint_queue.get()
    
    # Clear metrics history for new training run
    global metrics_history
    with history_lock:
        metrics_history = []
    
    training_thread = threading.Thread(target=train_worker)
    training_thread.start()
    
    # NEW: Send explicit start notification
    status_queue.put({'status': 'started', 'mode': 'train'})
    
    return jsonify({'status': 'started', 'mode': 'train'})



@app.route('/stop', methods=['POST'])
def stop_training():
    global training_state
    
    if not training_state['running']:
        return jsonify({'error': 'Training not running'}), 400
    
    stop_flag.set()
    training_state['running'] = False
    
    # NEW: Send explicit stop notification
    status_queue.put({'status': 'stopped'})
    
    return jsonify({'status': 'stopped'})



@app.route('/status')
def get_status():
    return jsonify(training_state)


@app.route('/metrics_history')
def get_metrics_history():
    """Fetch historical metrics for populating charts on page load."""
    with history_lock:
        return jsonify({
            'history': metrics_history,
            'running': training_state['running'],
            'mode': training_state['mode']
        })


@app.route('/task_config', methods=['GET'])
def get_task_config():
    return jsonify({
        # Task configuration
        'task_type': training_state['task_type'],
        'num_distractors': training_state['num_distractors'],
        'bottleneck_type': training_state['bottleneck_type'],
        'use_input_output_pairs': training_state['use_input_output_pairs'],
        'receiver_gets_input_puzzle': training_state['receiver_gets_input_puzzle'],
        # Data configuration
        'dataset_version': training_state['dataset_version'],
        'dataset_split': training_state['dataset_split'],
        'max_grids': training_state['max_grids'],
        'filter_grid_size': training_state['filter_grid_size'],
        # Model architecture
        'hidden_dim': training_state['hidden_dim'],
        'latent_dim': training_state['latent_dim'],
        'num_conv_layers': training_state['num_conv_layers'],
        # Communication protocol
        'vocab_size': training_state['vocab_size'],
        'max_message_length': training_state['max_message_length'],
        'temperature': training_state['temperature'],
        'use_stop_token': training_state['use_stop_token'],
        # Training hyperparameters
        'batch_size': training_state['batch_size'],
        'learning_rate': training_state['learning_rate'],
        'pretrain_learning_rate': training_state['pretrain_learning_rate'],
        'num_epochs': training_state['num_epochs'],
        'pretrain_epochs': training_state['pretrain_epochs'],
        # Pretraining configuration
        'pretrain_task_type': training_state['pretrain_task_type'],
        'use_pretrained': training_state['use_pretrained'],
        'freeze_encoder': training_state['freeze_encoder'],
        'load_pretrained_before_pretrain': training_state['load_pretrained_before_pretrain']
    })


@app.route('/task_config', methods=['POST'])
def set_task_config():
    global training_state
    if training_state['running']:
        return jsonify({'status': 'error', 'message': 'Cannot change config while training'})
    
    data = json.loads(request.data)
    
    # Task configuration
    if 'task_type' in data:
        training_state['task_type'] = data['task_type']
    if 'num_distractors' in data:
        training_state['num_distractors'] = int(data['num_distractors'])
    if 'bottleneck_type' in data:
        training_state['bottleneck_type'] = data['bottleneck_type']
    if 'use_input_output_pairs' in data:
        training_state['use_input_output_pairs'] = bool(data['use_input_output_pairs'])
    if 'receiver_gets_input_puzzle' in data:
        training_state['receiver_gets_input_puzzle'] = bool(data['receiver_gets_input_puzzle'])
    
    # Data configuration
    if 'dataset_version' in data:
        training_state['dataset_version'] = data['dataset_version']
    if 'dataset_split' in data:
        training_state['dataset_split'] = data['dataset_split']
    if 'max_grids' in data:
        training_state['max_grids'] = int(data['max_grids']) if data['max_grids'] else None
    if 'filter_grid_size' in data:
        training_state['filter_grid_size'] = data['filter_grid_size']  # Can be None or [height, width]
    
    # Model architecture
    if 'hidden_dim' in data:
        training_state['hidden_dim'] = int(data['hidden_dim'])
    if 'latent_dim' in data:
        training_state['latent_dim'] = int(data['latent_dim'])
    if 'num_conv_layers' in data:
        training_state['num_conv_layers'] = int(data['num_conv_layers'])
    
    # Communication protocol
    if 'vocab_size' in data:
        training_state['vocab_size'] = int(data['vocab_size'])
    if 'max_message_length' in data:
        training_state['max_message_length'] = int(data['max_message_length'])
    if 'temperature' in data:
        training_state['temperature'] = float(data['temperature'])
    if 'use_stop_token' in data:
        training_state['use_stop_token'] = bool(data['use_stop_token'])
    
    # Training hyperparameters
    if 'batch_size' in data:
        training_state['batch_size'] = int(data['batch_size'])
    if 'learning_rate' in data:
        training_state['learning_rate'] = float(data['learning_rate'])
    if 'pretrain_learning_rate' in data:
        training_state['pretrain_learning_rate'] = float(data['pretrain_learning_rate'])
    if 'num_epochs' in data:
        training_state['num_epochs'] = int(data['num_epochs'])
    if 'pretrain_epochs' in data:
        training_state['pretrain_epochs'] = int(data['pretrain_epochs'])
    
    # Pretraining configuration
    if 'pretrain_task_type' in data:
        training_state['pretrain_task_type'] = data['pretrain_task_type']
    if 'use_pretrained' in data:
        training_state['use_pretrained'] = bool(data['use_pretrained'])
    if 'freeze_encoder' in data:
        training_state['freeze_encoder'] = bool(data['freeze_encoder'])
    if 'load_pretrained_before_pretrain' in data:
        training_state['load_pretrained_before_pretrain'] = data['load_pretrained_before_pretrain']
    
    return jsonify({
        'status': 'success',
        'task_type': training_state['task_type'],
        'num_distractors': training_state['num_distractors'],
        'bottleneck_type': training_state['bottleneck_type'],
        'use_input_output_pairs': training_state['use_input_output_pairs'],
        'receiver_gets_input_puzzle': training_state['receiver_gets_input_puzzle'],
        'dataset_version': training_state['dataset_version'],
        'dataset_split': training_state['dataset_split'],
        'max_grids': training_state['max_grids'],
        'filter_grid_size': training_state['filter_grid_size'],
        'hidden_dim': training_state['hidden_dim'],
        'latent_dim': training_state['latent_dim'],
        'num_conv_layers': training_state['num_conv_layers'],
        'vocab_size': training_state['vocab_size'],
        'max_message_length': training_state['max_message_length'],
        'temperature': training_state['temperature'],
        'use_stop_token': training_state['use_stop_token'],
        'batch_size': training_state['batch_size'],
        'learning_rate': training_state['learning_rate'],
        'pretrain_learning_rate': training_state['pretrain_learning_rate'],
        'num_epochs': training_state['num_epochs'],
        'pretrain_epochs': training_state['pretrain_epochs'],
        'pretrain_task_type': training_state['pretrain_task_type'],
        'use_pretrained': training_state['use_pretrained'],
        'freeze_encoder': training_state['freeze_encoder'],
        'load_pretrained_before_pretrain': training_state['load_pretrained_before_pretrain']
    })


@app.route('/pretrained_encoders', methods=['GET'])
def list_pretrained_encoders():
    """List all available pretrained encoder files."""
    encoders = []
    checkpoint_dir = config.SAVE_DIR
    
    if os.path.exists(checkpoint_dir):
        # Look for pretrained encoder files
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith('pretrained_encoder') and filename.endswith('.pth'):
                filepath = os.path.join(checkpoint_dir, filename)
                try:
                    # Load checkpoint to get metadata
                    checkpoint = torch.load(filepath, map_location='cpu')
                    task_type = checkpoint.get('pretrain_task_type', 'unknown')
                    val_acc = checkpoint.get('val_acc', 0.0)
                    epoch = checkpoint.get('epoch', 0)
                    
                    encoders.append({
                        'filename': filename,
                        'path': filepath,
                        'task_type': task_type,
                        'val_acc': float(val_acc),
                        'epoch': int(epoch)
                    })
                except Exception as e:
                    # If we can't load it, still list it but with limited info
                    encoders.append({
                        'filename': filename,
                        'path': filepath,
                        'task_type': 'unknown',
                        'val_acc': 0.0,
                        'epoch': 0,
                        'error': str(e)
                    })
    
    return jsonify({'encoders': encoders})


@app.route('/save_checkpoint', methods=['POST'])
def request_checkpoint_save():
    """Request a checkpoint save during training."""
    global training_state
    
    if not training_state['running']:
        return jsonify({'error': 'Training is not running'}), 400
    
    if training_state['mode'] != 'train':
        return jsonify({'error': 'Checkpoint saving is only available during main training (not pretraining)'}), 400
    
    data = json.loads(request.data) if request.data else {}
    checkpoint_name = data.get('name', f"manual_checkpoint_epoch_{training_state['epoch']}")
    
    # Add request to the queue
    save_checkpoint_queue.put({'name': checkpoint_name})
    
    return jsonify({
        'status': 'requested',
        'checkpoint_name': checkpoint_name,
        'message': f'Checkpoint save requested: {checkpoint_name}'
    })


@app.route('/list_checkpoints', methods=['GET'])
def list_all_checkpoints():
    """List all saved checkpoints (not just pretrained encoders)."""
    checkpoints = []
    checkpoint_dir = config.SAVE_DIR
    
    if os.path.exists(checkpoint_dir):
        for filename in os.listdir(checkpoint_dir):
            if filename.endswith('.pth'):
                filepath = os.path.join(checkpoint_dir, filename)
                try:
                    # Get file stats
                    file_stat = os.stat(filepath)
                    file_size_mb = file_stat.st_size / (1024 * 1024)
                    modified_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                                  time.localtime(file_stat.st_mtime))
                    
                    # Try to load checkpoint to get metadata
                    checkpoint = torch.load(filepath, map_location='cpu')
                    
                    checkpoint_info = {
                        'filename': filename,
                        'path': filepath,
                        'size_mb': round(file_size_mb, 2),
                        'modified': modified_time,
                        'epoch': checkpoint.get('epoch', 0),
                        'batch': checkpoint.get('batch', 0),
                        'task_type': checkpoint.get('task_type', checkpoint.get('pretrain_task_type', 'unknown')),
                        'bottleneck_type': checkpoint.get('bottleneck_type', 'unknown'),
                        'val_loss': checkpoint.get('val_loss'),
                        'val_acc': checkpoint.get('val_acc'),
                    }
                    
                    # Add additional metadata if available
                    if 'hidden_dim' in checkpoint:
                        checkpoint_info['hidden_dim'] = checkpoint['hidden_dim']
                    if 'latent_dim' in checkpoint:
                        checkpoint_info['latent_dim'] = checkpoint['latent_dim']
                    if 'vocab_size' in checkpoint:
                        checkpoint_info['vocab_size'] = checkpoint['vocab_size']
                    
                    checkpoints.append(checkpoint_info)
                except Exception as e:
                    # If we can't load it, still list it with basic info
                    checkpoints.append({
                        'filename': filename,
                        'path': filepath,
                        'size_mb': round(file_size_mb, 2),
                        'modified': modified_time,
                        'error': f'Could not load checkpoint: {str(e)}'
                    })
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x.get('modified', ''), reverse=True)
    
    return jsonify({'checkpoints': checkpoints})


@app.route('/stream')
def stream():
    def generate():
        last_status_check = time.time()
        
        while True:
            # Check for status updates
            try:
                status_update = status_queue.get_nowait()
                yield f"data: {json.dumps({'type': 'status', 'data': status_update})}\n\n"
            except queue.Empty:
                pass
            
            # Check for reconstructions
            try:
                recons = reconstructions_queue.get_nowait()
                yield f"data: {json.dumps({'type': 'reconstructions', 'data': recons})}\n\n"
            except queue.Empty:
                pass
            
            # Check for metrics
            try:
                metrics = metrics_queue.get(timeout=0.1)
                yield f"data: {json.dumps({'type': 'metrics', 'data': metrics})}\n\n"
            except queue.Empty:
                # Send periodic status updates (every 2 seconds)
                current_time = time.time()
                if current_time - last_status_check > 2.0:
                    yield f"data: {json.dumps({'type': 'status', 'data': {'running': training_state['running'], 'mode': training_state['mode']}})}\n\n"
                    last_status_check = current_time
                else:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            
            if not training_state['running']:
                time.sleep(0.5)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/finetune_puzzle', methods=['POST'])
def finetune_puzzle_route():
    """Start background finetuning on a specific puzzle."""
    global finetuning_thread, finetuning_state
    
    if finetuning_state['running']:
        return jsonify({'error': 'Finetuning already in progress'}), 400
    
    data = json.loads(request.data)
    puzzle_id = data.get('puzzle_id')
    checkpoint_path = data.get('checkpoint')
    epochs = data.get('epochs', 500)
    lr = data.get('lr', 1e-4)
    receiver_lr = data.get('receiver_lr', None)
    batch_size = data.get('batch_size', 8)
    early_stop_threshold = data.get('early_stop_threshold', 99.0)
    dataset_version = data.get('dataset_version', 'V2')
    dataset_split = data.get('dataset_split', 'training')
    
    if not puzzle_id:
        return jsonify({'error': 'puzzle_id is required'}), 400
    
    try:
        # Clear the progress queue
        while not finetuning_progress_queue.empty():
            finetuning_progress_queue.get()
        
        # Start finetuning in background thread
        finetuning_thread = threading.Thread(
            target=finetune_worker,
            args=(puzzle_id, checkpoint_path, dataset_version, dataset_split, epochs, lr, batch_size, early_stop_threshold, receiver_lr)
        )
        finetuning_thread.start()
        
        return jsonify({
            'status': 'started',
            'puzzle_id': puzzle_id,
            'epochs': epochs,
            'early_stop_threshold': early_stop_threshold
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/finetune_progress')
def finetune_progress_stream():
    """Stream finetuning progress updates."""
    def generate():
        while True:
            try:
                progress = finetuning_progress_queue.get(timeout=1.0)
                yield f"data: {json.dumps(progress)}\n\n"
                
                # Stop streaming if completed, errored, or stopped
                if progress.get('status') in ['completed', 'error', 'early_stopped', 'stopped']:
                    break
            except queue.Empty:
                # Send heartbeat
                yield f"data: {json.dumps({'status': 'heartbeat', 'running': finetuning_state['running']})}\n\n"
            
            if not finetuning_state['running']:
                time.sleep(0.5)
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/stop_finetuning', methods=['POST'])
def stop_finetuning():
    """Request to stop the current finetuning process."""
    global finetuning_state
    
    if not finetuning_state['running']:
        return jsonify({'error': 'No finetuning in progress'}), 400
    
    finetuning_state['stop_requested'] = True
    
    return jsonify({
        'status': 'stop_requested',
        'message': 'Finetuning will stop after the current epoch'
    })


@app.route('/batch_test', methods=['POST'])
def batch_test_route():
    """Start batch testing on multiple puzzles."""
    global batch_test_thread, batch_test_state
    
    if batch_test_state['running']:
        return jsonify({'error': 'Batch test already in progress'}), 400
    
    data = json.loads(request.data)
    puzzle_ids = data.get('puzzle_ids', [])
    checkpoint_path = data.get('checkpoint')
    epochs = data.get('epochs', 500)
    lr = data.get('lr', 1e-4)
    receiver_lr = data.get('receiver_lr', None)
    batch_size = data.get('batch_size', 8)
    early_stop_threshold = data.get('early_stop_threshold', 99.0)
    dataset_version = data.get('dataset_version', 'V2')
    dataset_split = data.get('dataset_split', 'evaluation')
    
    if not puzzle_ids or len(puzzle_ids) == 0:
        return jsonify({'error': 'puzzle_ids list is required'}), 400
    
    try:
        # Clear the progress queue
        while not batch_test_progress_queue.empty():
            batch_test_progress_queue.get()
        
        # Start batch test in background thread
        batch_test_thread = threading.Thread(
            target=batch_test_worker,
            args=(puzzle_ids, checkpoint_path, dataset_version, dataset_split, epochs, lr, batch_size, early_stop_threshold, receiver_lr)
        )
        batch_test_thread.start()
        
        return jsonify({
            'status': 'started',
            'total_puzzles': len(puzzle_ids),
            'puzzle_ids': puzzle_ids
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_test_progress')
def batch_test_progress_stream():
    """Stream batch test progress updates."""
    def generate():
        while True:
            try:
                progress = batch_test_progress_queue.get(timeout=1.0)
                yield f"data: {json.dumps(progress)}\n\n"
                
                # Stop streaming if completed, errored, or stopped
                if progress.get('status') in ['completed', 'error', 'stopped']:
                    break
            except queue.Empty:
                # Send heartbeat
                yield f"data: {json.dumps({'status': 'heartbeat', 'running': batch_test_state['running']})}\n\n"
            
            if not batch_test_state['running']:
                time.sleep(0.5)
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/stop_batch_test', methods=['POST'])
def stop_batch_test():
    """Request to stop the current batch test process."""
    global batch_test_state
    
    if not batch_test_state['running']:
        return jsonify({'error': 'No batch test in progress'}), 400
    
    batch_test_state['stop_requested'] = True
    
    return jsonify({
        'status': 'stop_requested',
        'message': 'Batch test will stop after the current epoch or puzzle'
    })


@app.route('/batch_test_status', methods=['GET'])
def batch_test_status():
    """Get current batch test status."""
    return jsonify({
        'running': batch_test_state['running'],
        'stop_requested': batch_test_state['stop_requested'],
        'current_puzzle_index': batch_test_state.get('current_puzzle_index', 0),
        'total_puzzles': batch_test_state.get('total_puzzles', 0),
        'current_puzzle_id': batch_test_state.get('current_puzzle_id', None)
    })


@app.route('/reset_batch_test', methods=['POST'])
def reset_batch_test():
    """Force reset batch test state (use when stuck)."""
    global batch_test_state, batch_test_thread
    
    # Force reset the state
    was_running = batch_test_state['running']
    batch_test_state['running'] = False
    batch_test_state['stop_requested'] = False
    batch_test_state['current_puzzle_index'] = 0
    batch_test_state['total_puzzles'] = 0
    batch_test_state['current_puzzle_id'] = None
    batch_test_state['results'] = []
    
    # Clear the queue
    while not batch_test_progress_queue.empty():
        try:
            batch_test_progress_queue.get_nowait()
        except:
            break
    
    return jsonify({
        'status': 'reset',
        'message': 'Batch test state has been reset',
        'was_running': was_running
    })


@app.route('/solve_puzzle', methods=['POST'])
def solve_puzzle_route():
    """Solve a puzzle using a finetuned model."""
    data = json.loads(request.data)
    puzzle_id = data.get('puzzle_id')
    checkpoint_path = data.get('checkpoint')
    dataset_version = data.get('dataset_version', 'V2')
    dataset_split = data.get('dataset_split', 'evaluation')
    
    if not puzzle_id:
        return jsonify({'error': 'puzzle_id is required'}), 400
    
    if not checkpoint_path:
        return jsonify({'error': 'checkpoint is required'}), 400
    
    try:
        # Import puzzle solving functions
        from puzzle_dataset import ARCSinglePuzzleDataset
        from model import ARCEncoder, ARCAutoencoder
        
        # Get data path
        data_path = get_data_path(dataset_version, dataset_split)
        
        # Load test dataset
        test_dataset = ARCSinglePuzzleDataset(data_path, puzzle_id, split='test')
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        encoder = ARCEncoder(
            num_colors=config.NUM_COLORS,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            latent_dim=config.LATENT_DIM,
            num_conv_layers=getattr(config, 'NUM_CONV_LAYERS', 3)
        )
        
        if not os.path.exists(checkpoint_path):
            return jsonify({'error': f'Checkpoint not found: {checkpoint_path}'}), 404
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load receiver_gets_input_puzzle and use_stop_token from checkpoint if available, otherwise use config default
        receiver_gets_input_puzzle = checkpoint.get('receiver_gets_input_puzzle', getattr(config, 'RECEIVER_GETS_INPUT_PUZZLE', False))
        use_stop_token = checkpoint.get('use_stop_token', getattr(config, 'USE_STOP_TOKEN', False))
        # Get vocab_size from checkpoint or config, then calculate stop_token_id
        checkpoint_vocab_size = checkpoint.get('vocab_size', config.VOCAB_SIZE if config.BOTTLENECK_TYPE == 'communication' else None)
        stop_token_id = checkpoint_vocab_size if use_stop_token else None
        
        model = ARCAutoencoder(
            encoder=encoder,
            vocab_size=config.VOCAB_SIZE if config.BOTTLENECK_TYPE == 'communication' else None,
            max_length=config.MAX_MESSAGE_LENGTH if config.BOTTLENECK_TYPE == 'communication' else None,
            num_colors=config.NUM_COLORS,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            max_grid_size=config.MAX_GRID_SIZE,
            bottleneck_type=config.BOTTLENECK_TYPE,
            task_type='reconstruction',
            num_conv_layers=getattr(config, 'NUM_CONV_LAYERS', 3),
            receiver_gets_input_puzzle=receiver_gets_input_puzzle,
            use_stop_token=use_stop_token,
            stop_token_id=stop_token_id
        ).to(device)
        
        # Load checkpoint with mapping for selection task checkpoints
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        if 'receiver_reconstructor.symbol_embed.weight' in state_dict or \
           'decoder_reconstructor.fc_decode.weight' in state_dict:
            print('Detected selection task checkpoint - mapping background reconstruction weights...')
            
            # Map receiver_reconstructor -> receiver OR decoder_reconstructor -> decoder
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('receiver_reconstructor.'):
                    # Map background reconstruction receiver to main receiver
                    new_key = k.replace('receiver_reconstructor.', 'receiver.')
                    new_state_dict[new_key] = v
                elif k.startswith('decoder_reconstructor.'):
                    # Map background reconstruction decoder to main decoder
                    new_key = k.replace('decoder_reconstructor.', 'decoder.')
                    new_state_dict[new_key] = v
                elif k.startswith('encoder.'):
                    # Keep encoder weights
                    new_state_dict[k] = v
                elif k.startswith('sender.'):
                    # Keep sender weights (for communication mode)
                    new_state_dict[k] = v
                # Skip receiver.* and decoder.* keys (those are for selection, not reconstruction)
            
            # Load the mapped weights
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                print(f'⚠️  Missing keys (will be randomly initialized): {missing_keys[:5]}...')
            if unexpected_keys:
                print(f'⚠️  Unexpected keys (ignored): {unexpected_keys[:5]}...')
            
            print('✓ Loaded and mapped weights from selection checkpoint')
        else:
            # Standard reconstruction checkpoint
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        
        # Solve all test examples
        results = {
            'puzzle_id': puzzle_id,
            'num_test_examples': len(test_dataset),
            'predictions': []
        }
        
        for i in range(len(test_dataset)):
            input_grid, input_size, output_grid, output_size = test_dataset[i]
            
            # Get actual grids (without padding)
            input_h, input_w = input_size
            output_h, output_w = output_size
            
            input_actual = input_grid[:input_h, :input_w].numpy()
            output_actual = output_grid[:output_h, :output_w].numpy()
            
            # Generate prediction
            with torch.no_grad():
                input_batch = input_grid.unsqueeze(0).to(device)
                model_output = model(input_batch, [input_size], temperature=1.0)
                
                # Unpack based on number of return values
                if len(model_output) == 4:  # with message_lengths
                    logits_list, _, messages, message_lengths = model_output
                else:  # autoencoder mode
                    logits_list, _, messages = model_output
                
                logits = logits_list[0]
                pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
                predicted = pred[:output_h, :output_w]
            
            # Evaluate
            # Handle case where predicted and output_actual might have different shapes
            if predicted.shape != output_actual.shape:
                # If shapes don't match, they can't be equal
                exact_match = False
                # Calculate overlap area for pixel accuracy
                min_h = min(predicted.shape[0], output_actual.shape[0])
                min_w = min(predicted.shape[1], output_actual.shape[1])
                if min_h > 0 and min_w > 0:
                    correct_pixels = (predicted[:min_h, :min_w] == output_actual[:min_h, :min_w]).sum()
                else:
                    correct_pixels = 0
                total_pixels = output_actual.size
            else:
                exact_match = np.array_equal(predicted, output_actual)
                correct_pixels = (predicted == output_actual).sum()
                total_pixels = predicted.size
            
            pixel_accuracy = 100.0 * correct_pixels / total_pixels if total_pixels > 0 else 0.0
            
            # Store results
            prediction_result = {
                'example_id': i,
                'input_size': list(input_size),
                'output_size': list(output_size),
                'exact_match': bool(exact_match),
                'pixel_accuracy': float(pixel_accuracy),
                'input': input_actual.tolist(),
                'target': output_actual.tolist(),
                'predicted': predicted.tolist(),
                'messages': messages[0].cpu().tolist() if messages is not None else None
            }
            results['predictions'].append(prediction_result)
        
        # Calculate summary stats
        results['num_correct'] = sum(1 for p in results['predictions'] if p['exact_match'])
        pixel_accuracies = [p['pixel_accuracy'] for p in results['predictions']]
        if len(pixel_accuracies) > 0:
            avg_acc = float(np.mean(pixel_accuracies))
            results['avg_pixel_accuracy'] = 0.0 if np.isnan(avg_acc) or np.isinf(avg_acc) else avg_acc
        else:
            results['avg_pixel_accuracy'] = 0.0
        
        return jsonify(results)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=True, port=5002)