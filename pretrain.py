"""Pretraining script for ARC encoder - learns to distinguish real grids from noise."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from tqdm import tqdm

import config
from dataset import ARCDataset, collate_fn, collate_fn_puzzle_classification
from model import ARCEncoder


class NoiseGridDataset(Dataset):
    """Generates random noise grids that mimic ARC grid dimensions."""
    def __init__(self, num_samples=10000, min_size=3, max_size=30, num_colors=10):
        self.num_samples = num_samples
        self.min_size = min_size
        self.max_size = max_size
        self.num_colors = num_colors
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random grid size
        h = np.random.randint(self.min_size, self.max_size + 1)
        w = np.random.randint(self.min_size, self.max_size + 1)
        
        # Random noise grid
        grid = np.random.randint(0, self.num_colors, size=(h, w), dtype=np.int64)
        grid_tensor = torch.from_numpy(grid).long()
        
        return grid_tensor, (h, w)


class BinaryARCDataset(Dataset):
    """Wrapper that combines real ARC grids (label=1) and noise grids (label=0)."""
    def __init__(self, arc_dataset, noise_dataset):
        self.arc_dataset = arc_dataset
        self.noise_dataset = noise_dataset
        
    def __len__(self):
        return len(self.arc_dataset) + len(self.noise_dataset)
    
    def __getitem__(self, idx):
        if idx < len(self.arc_dataset):
            # Real ARC grid
            grid, size = self.arc_dataset[idx]
            label = 1
        else:
            # Noise grid
            noise_idx = idx - len(self.arc_dataset)
            grid, size = self.noise_dataset[noise_idx]
            label = 0
        
        return grid, size, label


def collate_fn_with_labels(batch):
    """Custom collate function that handles labels."""
    grids, sizes, labels = zip(*batch)
    
    # Find max dimensions in batch
    max_h = max(g.shape[0] for g in grids)
    max_w = max(g.shape[1] for g in grids)
    
    # Pad all grids to same size
    padded_grids = []
    for grid in grids:
        h, w = grid.shape
        pad_h = max_h - h
        pad_w = max_w - w
        padded = torch.nn.functional.pad(grid, (0, pad_w, 0, pad_h), value=0)
        padded_grids.append(padded)
    
    batch_grids = torch.stack(padded_grids)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return batch_grids, sizes, labels_tensor


class EncoderClassifier(nn.Module):
    """Binary classifier on top of encoder for pretraining."""
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
            nn.Linear(128, 2)  # Binary classification
        )
    
    def forward(self, grids, sizes=None):
        latent = self.encoder(grids, sizes=sizes)
        logits = self.classifier(latent)
        return logits


class EncoderPuzzleClassifier(nn.Module):
    """Multi-class puzzle classifier on top of encoder for pretraining.
    
    Learns to classify grids into puzzle categories, with separate classes
    for inputs and outputs of each puzzle.
    """
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
    """Selection classifier on top of encoder for pretraining with selection task."""
    def __init__(self, encoder, num_colors, embedding_dim, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.num_colors = num_colors
        self.hidden_dim = hidden_dim
        
        # Scoring: compare target representation with candidate representations
        # Both now use the same encoder, so they produce the same latent_dim
        self.score_fc = nn.Linear(encoder.latent_dim * 2, 1)
        self.relu = nn.ReLU()
        
    def encode_candidates(self, candidates, candidate_sizes=None):
        """
        Encode candidate grids into representations using the same encoder as target.
        Args:
            candidates: [num_candidates, H, W]
            candidate_sizes: list of (H, W) tuples - actual sizes for each candidate (optional)
        Returns:
            representations: [num_candidates, latent_dim]
        """
        # Use the same encoder as for the target grids
        return self.encoder(candidates, sizes=candidate_sizes)  # [num_candidates, latent_dim]
    
    def forward(self, target_grids, sizes, candidates_list, candidates_sizes_list=None):
        """
        Args:
            target_grids: [batch, H, W] - target grids
            sizes: list of (H, W) tuples - actual sizes for each target grid
            candidates_list: list of [num_candidates, H, W] tensors
            candidates_sizes_list: list of lists of (H, W) tuples for each candidate (optional)
        Returns:
            logits_list: list of [num_candidates] tensors - selection logits for each sample
        """
        # Encode target grids
        target_latents = self.encoder(target_grids, sizes=sizes)  # [batch, latent_dim]
        
        # Process each sample's candidates
        logits_list = []
        for i, candidates in enumerate(candidates_list):
            target_latent = target_latents[i:i+1]  # [1, latent_dim]
            candidate_sizes = candidates_sizes_list[i] if candidates_sizes_list is not None else None
            
            # Encode all candidates
            cand_repr = self.encode_candidates(candidates, candidate_sizes=candidate_sizes)  # [num_candidates, latent_dim]
            
            # Compute similarity scores
            num_candidates = candidates.shape[0]
            target_latent_expanded = target_latent.expand(num_candidates, -1)
            
            # Concatenate and score
            combined = torch.cat([target_latent_expanded, cand_repr], dim=-1)
            logits = self.score_fc(combined).squeeze(-1)  # [num_candidates]
            
            logits_list.append(logits)
        
        return logits_list


def pretrain_encoder(encoder, train_loader, val_loader, device, num_epochs=20):
    """Pretrain encoder with binary classification task."""
    model = EncoderClassifier(encoder).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    best_val_acc = 0.0
    
    print(f'\nPretraining encoder with {sum(p.numel() for p in model.parameters()):,} parameters')
    print('Task: Distinguish real ARC grids from random noise\n')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for grids, sizes, labels in pbar:
            grids = grids.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(grids, sizes=sizes)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{train_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for grids, sizes, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                grids = grids.to(device)
                labels = labels.to(device)
                
                logits = model(grids, sizes=sizes)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                pred = logits.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'classifier_state_dict': model.classifier.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(config.SAVE_DIR, 'pretrained_encoder.pth'))
            print(f'  ✓ Saved best pretrained encoder (Val Acc: {val_acc:.2f}%)')
        
        print()
    
    print(f'Pretraining complete! Best validation accuracy: {best_val_acc:.2f}%')
    return encoder


def pretrain_encoder_selection(encoder, train_loader, val_loader, device, num_epochs=20, num_distractors=3):
    """Pretrain encoder with selection task."""
    model = EncoderSelector(
        encoder,
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    best_val_acc = 0.0
    
    print(f'\nPretraining encoder with {sum(p.numel() for p in model.parameters()):,} parameters')
    print(f'Task: Select correct grid from {num_distractors + 1} candidates\n')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_data in pbar:
            grids, sizes, candidates_list, candidates_sizes_list, target_indices = batch_data
            grids = grids.to(device)
            candidates_list = [c.to(device) for c in candidates_list]
            target_indices = target_indices.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - get logits for each sample
            logits_list = model(grids, sizes, candidates_list, candidates_sizes_list)
            
            # Compute loss for each sample
            batch_loss = 0
            for i, logits in enumerate(logits_list):
                sample_loss = criterion(logits.unsqueeze(0), target_indices[i].unsqueeze(0))
                batch_loss += sample_loss
                
                # Track accuracy
                pred_idx = logits.argmax()
                train_correct += (pred_idx == target_indices[i]).item()
                train_total += 1
            
            loss = batch_loss / len(logits_list)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f'{train_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                grids, sizes, candidates_list, candidates_sizes_list, target_indices = batch_data
                grids = grids.to(device)
                candidates_list = [c.to(device) for c in candidates_list]
                target_indices = target_indices.to(device)
                
                logits_list = model(grids, sizes, candidates_list, candidates_sizes_list)
                
                batch_loss = 0
                for i, logits in enumerate(logits_list):
                    sample_loss = criterion(logits.unsqueeze(0), target_indices[i].unsqueeze(0))
                    batch_loss += sample_loss
                    
                    pred_idx = logits.argmax()
                    val_correct += (pred_idx == target_indices[i]).item()
                    val_total += 1
                
                loss = batch_loss / len(logits_list)
                val_loss += loss.item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  (Random chance: {100.0 / (num_distractors + 1):.2f}%)')
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'selector_state_dict': model.score_fc.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(config.SAVE_DIR, 'pretrained_encoder.pth'))
            print(f'  ✓ Saved best pretrained encoder (Val Acc: {val_acc:.2f}%)')
        
        print()
    
    print(f'Pretraining complete! Best validation accuracy: {best_val_acc:.2f}%')
    print(f'Random chance: {100.0 / (num_distractors + 1):.2f}%')
    return encoder


def pretrain_encoder_puzzle_classification(encoder, train_loader, val_loader, device, num_epochs=20, num_classes=None):
    """Pretrain encoder with puzzle classification task.
    
    Each puzzle has two classes: one for inputs and one for outputs.
    The encoder learns to classify grids into the correct puzzle category.
    """
    model = EncoderPuzzleClassifier(encoder, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.PRETRAIN_LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    best_val_acc = 0.0
    
    print(f'\nPretraining encoder with {sum(p.numel() for p in model.parameters()):,} parameters')
    print(f'Task: Classify grids into {num_classes} puzzle categories (inputs/outputs per puzzle)\n')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for grids, sizes, labels in pbar:
            grids = grids.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(grids, sizes=sizes)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{train_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for grids, sizes, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                grids = grids.to(device)
                labels = labels.to(device)
                
                logits = model(grids, sizes=sizes)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                pred = logits.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  (Random chance: {100.0 / num_classes:.2f}%)')
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'classifier_state_dict': model.classifier.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
            }, os.path.join(config.SAVE_DIR, 'pretrained_encoder.pth'))
            print(f'  ✓ Saved best pretrained encoder (Val Acc: {val_acc:.2f}%)')
        
        print()
    
    print(f'Pretraining complete! Best validation accuracy: {best_val_acc:.2f}%')
    print(f'Random chance: {100.0 / num_classes:.2f}%')
    return encoder


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # Get pretraining task type
    pretrain_task_type = getattr(config, 'PRETRAIN_TASK_TYPE', 'binary')
    num_epochs = getattr(config, 'PRETRAIN_EPOCHS', 20)
    
    print(f'\n{"="*60}')
    print(f'Pretraining Task Type: {pretrain_task_type}')
    print(f'{"="*60}\n')
    
    # Load real ARC grids
    print('Loading ARC dataset...')
    
    # Determine num_distractors and track_puzzle_ids for dataset
    if pretrain_task_type == 'selection':
        num_distractors = getattr(config, 'NUM_DISTRACTORS', 3)
        track_puzzle_ids = False
    elif pretrain_task_type == 'puzzle_classification':
        num_distractors = 0
        track_puzzle_ids = True
    else:
        num_distractors = 0
        track_puzzle_ids = False
    
    arc_dataset = ARCDataset(
        config.DATA_PATH, 
        min_size=config.MIN_GRID_SIZE,
        filter_size=getattr(config, 'FILTER_GRID_SIZE', None),
        max_grids=getattr(config, 'MAX_GRIDS', None),
        num_distractors=num_distractors,
        track_puzzle_ids=track_puzzle_ids
    )
    
    # Create encoder
    print('\nCreating encoder...')
    encoder = ARCEncoder(
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_conv_layers=config.NUM_CONV_LAYERS if hasattr(config, 'NUM_CONV_LAYERS') else 3
    )
    
    if pretrain_task_type == 'puzzle_classification':
        # Puzzle classification task pretraining
        print(f'Setting up puzzle classification task...')
        
        # Calculate number of classes: num_puzzles * 2 (for inputs and outputs)
        num_puzzles = len(arc_dataset.puzzle_id_map)
        num_classes = num_puzzles * 2
        print(f'Number of puzzles: {num_puzzles}')
        print(f'Number of classes (inputs + outputs): {num_classes}')
        
        # Split dataset
        train_size = int(0.8 * len(arc_dataset))
        val_size = len(arc_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            arc_dataset, [train_size, val_size]
        )
        
        # Create dataloaders with puzzle classification collate function
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn_puzzle_classification,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn_puzzle_classification,
            num_workers=4
        )
        
        print(f'Total dataset size: {len(arc_dataset)} samples')
        print(f'  - Train: {train_size}')
        print(f'  - Val: {val_size}')
        
        # Pretrain with puzzle classification task
        encoder = pretrain_encoder_puzzle_classification(
            encoder,
            train_loader,
            val_loader,
            device,
            num_epochs=num_epochs,
            num_classes=num_classes
        )
        
    elif pretrain_task_type == 'selection':
        # Selection task pretraining
        print(f'Setting up selection task with {num_distractors} distractors...')
        
        # Split dataset
        train_size = int(0.8 * len(arc_dataset))
        val_size = len(arc_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            arc_dataset, [train_size, val_size]
        )
        
        # Create dataloaders with distractors
        from functools import partial
        collate_fn_with_distractors = partial(collate_fn, num_distractors=num_distractors)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn_with_distractors,
            num_workers=0  # Set to 0 to avoid issues with random sampling in multiprocessing
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn_with_distractors,
            num_workers=0  # Set to 0 to avoid issues with random sampling in multiprocessing
        )
        
        print(f'Total dataset size: {len(arc_dataset)} samples')
        print(f'  - Train: {train_size}')
        print(f'  - Val: {val_size}')
        print(f'  - Selection from {num_distractors + 1} candidates')
        
        # Pretrain with selection task
        encoder = pretrain_encoder_selection(
            encoder, 
            train_loader, 
            val_loader, 
            device,
            num_epochs=num_epochs,
            num_distractors=num_distractors
        )
        
    else:  # binary classification
        # Create noise dataset with same number of samples
        print('Generating noise dataset...')
        noise_dataset = NoiseGridDataset(
            num_samples=len(arc_dataset),
            min_size=config.MIN_GRID_SIZE,
            max_size=config.MAX_GRID_SIZE,
            num_colors=config.NUM_COLORS
        )
        
        # Combine into binary classification dataset
        binary_dataset = BinaryARCDataset(arc_dataset, noise_dataset)
        print(f'Total dataset size: {len(binary_dataset)} samples')
        print(f'  - Real ARC grids: {len(arc_dataset)}')
        print(f'  - Noise grids: {len(noise_dataset)}')
        
        # Split into train/val
        train_size = int(0.8 * len(binary_dataset))
        val_size = len(binary_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            binary_dataset, [train_size, val_size]
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn_with_labels,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn_with_labels,
            num_workers=4
        )
        
        # Pretrain with binary classification
        encoder = pretrain_encoder(
            encoder, 
            train_loader, 
            val_loader, 
            device,
            num_epochs=num_epochs
        )
    
    print('\nPretrained encoder saved to:', 
          os.path.join(config.SAVE_DIR, 'pretrained_encoder.pth'))
    print('Use this encoder in train.py by loading these weights!')


if __name__ == '__main__':
    main()