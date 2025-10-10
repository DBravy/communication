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

# Global state
training_state = {
    'running': False,
    'mode': None,  # 'pretrain' or 'train'
    # Task configuration
    'task_type': getattr(config, 'TASK_TYPE', 'reconstruction'),
    'num_distractors': getattr(config, 'NUM_DISTRACTORS', 3),
    'bottleneck_type': getattr(config, 'BOTTLENECK_TYPE', 'communication'),
    # Data configuration
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
    # Training hyperparameters
    'batch_size': getattr(config, 'BATCH_SIZE', 32),
    'learning_rate': getattr(config, 'LEARNING_RATE', 1e-5),
    'pretrain_learning_rate': getattr(config, 'PRETRAIN_LEARNING_RATE', 1e-4),
    # Pretraining configuration
    'pretrain_task_type': getattr(config, 'PRETRAIN_TASK_TYPE', 'binary'),
    'use_pretrained': getattr(config, 'USE_PRETRAINED', True),
    'freeze_encoder': getattr(config, 'FREEZE_ENCODER', False),
    'load_pretrained_before_pretrain': None,  # Path to encoder to load before pretraining
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
        logits_list, _, messages = model(single_grid, [(actual_h, actual_w)], temperature=1.0)
        recon = logits_list[0].argmax(dim=1).squeeze(0).cpu().numpy()
        
        message, _ = model.sender(single_grid, sizes=[(actual_h, actual_w)], temperature=1.0)
        msg = message[0].cpu().tolist()
        
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
        classification_logits, _, messages = model(single_grid, [(actual_h, actual_w)], temperature=1.0)
        
        # Get probabilities and prediction
        probs = torch.softmax(classification_logits[0], dim=0).cpu().numpy()
        pred_class = int(probs.argmax())
        
        # Get message
        message, _ = model.sender(single_grid, sizes=[(actual_h, actual_w)], temperature=1.0)

        msg = message[0].cpu().tolist()
        
        preview = [{
            'task_type': 'puzzle_classification',
            'input': input_grid.tolist(),
            'pred_class': pred_class,
            'top_probs': probs.tolist()[:10],  # Show top 10 probabilities
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
        selection_logits_list, _, messages = model(
            single_grid, [(actual_h, actual_w)], temperature=1.0,
            candidates_list=[candidates],
            candidates_sizes_list=[candidate_sizes],
            target_indices=target_idx.unsqueeze(0)
        )
        
        sel_logits = selection_logits_list[0]
        probs = torch.softmax(sel_logits, dim=0).cpu().numpy()
        pred_idx = sel_logits.argmax().item()
        
        message, _ = model.sender(single_grid, sizes=[(actual_h, actual_w)], temperature=1.0)

        msg = message[0].cpu().tolist()
        
        # Get all candidate grids
        candidates_data = []
        for c_idx in range(len(candidates)):
            cand_grid = candidates[c_idx][:actual_h, :actual_w].cpu().numpy()
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
        num_epochs = getattr(config, 'PRETRAIN_EPOCHS', 20)
        
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
        
        # Load ARC dataset
        arc_dataset = ARCDataset(  # or 'dataset' in train_worker
            config.DATA_PATH, 
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
            hidden_dim=config.HIDDEN_DIM,
            latent_dim=config.LATENT_DIM,
            num_conv_layers=config.NUM_CONV_LAYERS if hasattr(config, 'NUM_CONV_LAYERS') else 3
        )
        
        # Load pretrained encoder if specified
        load_from_path = training_state.get('load_pretrained_before_pretrain')
        if load_from_path and os.path.exists(load_from_path):
            print(f'Loading pretrained encoder from {load_from_path} before starting pretraining...')
            checkpoint = torch.load(load_from_path, map_location='cpu')
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print(f"✓ Loaded pretrained encoder (continuing from checkpoint)")
        elif load_from_path:
            print(f'Warning: Pretrained encoder path specified but not found: {load_from_path}')
            print('Starting pretraining from scratch')
        else:
            print('Starting pretraining from scratch (no pretrained encoder specified)')
        
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
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                collate_fn=collate_fn_puzzle_classification,
                num_workers=0
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=collate_fn_puzzle_classification,
                num_workers=0
            )
            
            model = EncoderPuzzleClassifier(encoder, num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=getattr(config, 'PRETRAIN_LEARNING_RATE', 1e-3))
            
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
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                collate_fn=collate_fn_with_distractors,
                num_workers=0
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=collate_fn_with_distractors,
                num_workers=0
            )
            
            model = EncoderSelector(
                encoder,
                num_colors=config.NUM_COLORS,
                embedding_dim=config.EMBEDDING_DIM,
                hidden_dim=config.HIDDEN_DIM
            ).to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=getattr(config, 'PRETRAIN_LEARNING_RATE', 1e-3))
            
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
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                collate_fn=collate_fn_with_labels,
                num_workers=0
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=collate_fn_with_labels,
                num_workers=0
            )
            
            model = EncoderClassifier(encoder).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=getattr(config, 'PRETRAIN_LEARNING_RATE', 1e-3))
            
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
        
        dataset = ARCDataset(  # or 'dataset' in train_worker
            config.DATA_PATH, 
            min_size=config.MIN_GRID_SIZE,
            filter_size=training_state['filter_grid_size'],  # ✅ Read from training_state
            max_grids=training_state['max_grids'],           # ✅ Read from training_state
            num_distractors=num_distractors,     # (or num_distractors in train_worker)
            track_puzzle_ids=track_puzzle_ids
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
            collate_fn_for_task = partial(collate_fn, num_distractors=num_distractors)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn_for_task,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn_for_task,
            num_workers=0
        )
        
        encoder = ARCEncoder(
            num_colors=config.NUM_COLORS,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            latent_dim=config.LATENT_DIM,
            num_conv_layers=config.NUM_CONV_LAYERS if hasattr(config, 'NUM_CONV_LAYERS') else 3
        )
        
        # Load pretrained encoder if available and enabled
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
            print(f'Loading pretrained encoder from {pretrained_path}...')
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print(f"✓ Loaded pretrained encoder (task type: {pretrain_task_type})")
        elif use_pretrained:
            print(f'Warning: Pretrained encoder requested but not found at {pretrained_path}')
            print('Training from scratch')
        else:
            print('Training from scratch (pretrained encoder disabled)')
        
        model = ARCAutoencoder(
            encoder=encoder,
            vocab_size=config.VOCAB_SIZE,
            max_length=config.MAX_MESSAGE_LENGTH,
            num_colors=config.NUM_COLORS,
            embedding_dim=getattr(config, 'EMBEDDING_DIM', 16),
            hidden_dim=config.HIDDEN_DIM,
            max_grid_size=config.MAX_GRID_SIZE,
            bottleneck_type='communication',
            task_type=task_type,
            num_conv_layers=config.NUM_CONV_LAYERS if hasattr(config, 'NUM_CONV_LAYERS') else 2,
            num_classes=num_classes  # For puzzle_classification task
        ).to(device)
        
        # Freeze encoder if configured
        freeze_encoder = training_state.get('freeze_encoder', getattr(config, 'FREEZE_ENCODER', False))
        if freeze_encoder:
            print('🔒 Freezing encoder weights')
            for param in model.encoder.parameters():
                param.requires_grad = False
        else:
            print('🔓 Encoder weights will be updated')
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        for epoch in range(config.NUM_EPOCHS):
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
                
                # Unpack based on task type
                if task_type == 'selection':
                    grids, sizes, candidates_list, candidates_sizes_list, target_indices = batch_data
                    grids = grids.to(device)
                    candidates_list = [c.to(device) for c in candidates_list]
                    target_indices = target_indices.to(device)
                elif task_type == 'puzzle_classification':
                    grids, sizes, labels = batch_data
                    grids = grids.to(device)
                    labels = labels.to(device)
                else:
                    grids, sizes = batch_data
                    grids = grids.to(device)
                    candidates_list = None
                    target_indices = None
                
                training_state['batch'] = batch_idx + 1
                
                optimizer.zero_grad()
                
                if task_type == 'puzzle_classification':
                    # Puzzle classification task
                    classification_logits, _, messages = model(grids, sizes, temperature=config.TEMPERATURE, labels=labels)
                    
                    # Compute classification loss
                    loss = criterion(classification_logits, labels)
                    
                    # Calculate accuracy
                    pred = classification_logits.argmax(dim=1)
                    batch_correct = (pred == labels).sum().item()
                    batch_total = labels.size(0)
                elif task_type == 'selection':
                    # Selection task
                    selection_logits_list, actual_sizes, messages = model(
                        grids, sizes, temperature=config.TEMPERATURE,
                        candidates_list=candidates_list, 
                        candidates_sizes_list=candidates_sizes_list,  # Add this parameter
                        target_indices=target_indices
                    )
                    
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
                    logits_list, actual_sizes, messages = model(grids, sizes, temperature=config.TEMPERATURE)
                    
                    batch_loss = 0
                    batch_correct = 0
                    batch_total = 0
                    
                    for sample_idx, (logits, (actual_h, actual_w)) in enumerate(zip(logits_list, actual_sizes)):
                        actual_h, actual_w = sizes[sample_idx]
                        H, W = logits.shape[2], logits.shape[3]
                        
                        target_grid = grids[sample_idx:sample_idx+1, :H, :W]
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
                
                if batch_idx % 10 == 0:
                    if task_type == 'selection':
                        results = get_selections(model, batch_data, device, task_type)
                    elif task_type == 'puzzle_classification':
                        results = get_classification_preview(model, grids, sizes, device)
                    else:
                        results = get_reconstructions(model, grids, sizes, device, num_samples=1)
                    reconstructions_queue.put(results)
                
                time.sleep(0.01)
                            
                for val_batch in val_loader:
                    if task_type == 'selection':
                        val_grids = val_batch[0].to(device)
                        results = get_selections(model, val_batch, device, task_type)
                    elif task_type == 'puzzle_classification':
                        val_grids, val_sizes, val_labels = val_batch
                        val_grids = val_grids.to(device)
                        results = get_classification_preview(model, val_grids, val_sizes, device)
                    else:
                        val_grids, val_sizes = val_batch
                        val_grids = val_grids.to(device)
                        results = get_reconstructions(model, val_grids, val_sizes, device, num_samples=1)
                    reconstructions_queue.put(results)
                    break
                        
        training_state['running'] = False
        training_state['mode'] = None
        metrics_queue.put({'status': 'completed'})
        
    except Exception as e:
        training_state['running'] = False
        training_state['mode'] = None
        metrics_queue.put({'status': 'error', 'message': str(e)})



@app.route('/')
def index():
    return render_template('index.html')


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
    
    # Clear queues
    while not metrics_queue.empty():
        metrics_queue.get()
    while not reconstructions_queue.empty():
        reconstructions_queue.get()
    while not status_queue.empty():  # NEW
        status_queue.get()
    
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
    
    # Clear queues
    while not metrics_queue.empty():
        metrics_queue.get()
    while not reconstructions_queue.empty():
        reconstructions_queue.get()
    while not status_queue.empty():  # NEW
        status_queue.get()
    
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


@app.route('/task_config', methods=['GET'])
def get_task_config():
    return jsonify({
        # Task configuration
        'task_type': training_state['task_type'],
        'num_distractors': training_state['num_distractors'],
        'bottleneck_type': training_state['bottleneck_type'],
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
        # Training hyperparameters
        'batch_size': training_state['batch_size'],
        'learning_rate': training_state['learning_rate'],
        'pretrain_learning_rate': training_state['pretrain_learning_rate'],
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
    
    # Data configuration
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
    
    # Training hyperparameters
    if 'batch_size' in data:
        training_state['batch_size'] = int(data['batch_size'])
    if 'learning_rate' in data:
        training_state['learning_rate'] = float(data['learning_rate'])
    if 'pretrain_learning_rate' in data:
        training_state['pretrain_learning_rate'] = float(data['pretrain_learning_rate'])
    
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
        'max_grids': training_state['max_grids'],
        'filter_grid_size': training_state['filter_grid_size'],
        'hidden_dim': training_state['hidden_dim'],
        'latent_dim': training_state['latent_dim'],
        'num_conv_layers': training_state['num_conv_layers'],
        'vocab_size': training_state['vocab_size'],
        'max_message_length': training_state['max_message_length'],
        'temperature': training_state['temperature'],
        'batch_size': training_state['batch_size'],
        'learning_rate': training_state['learning_rate'],
        'pretrain_learning_rate': training_state['pretrain_learning_rate'],
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=True, port=5002)