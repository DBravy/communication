"""Dataset loader for single ARC puzzles (for finetuning and solving)."""

import json
import torch
from torch.utils.data import Dataset
import numpy as np


class ARCSinglePuzzleDataset(Dataset):
    """Dataset for a single ARC puzzle with train/test split."""
    
    def __init__(self, json_path, puzzle_id, split='train'):
        """
        Args:
            json_path: Path to ARC JSON file
            puzzle_id: ID of the puzzle to load
            split: 'train' or 'test'
        """
        self.puzzle_id = puzzle_id
        self.split = split
        self.input_output_pairs = []
        
        # Load puzzle data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if puzzle_id not in data:
            raise ValueError(f"Puzzle {puzzle_id} not found in {json_path}")
        
        puzzle_data = data[puzzle_id]
        
        # Get examples from the specified split
        examples = puzzle_data.get(split, [])
        
        if len(examples) == 0:
            raise ValueError(f"No {split} examples found for puzzle {puzzle_id}")
        
        # Store input-output pairs
        for example in examples:
            if 'input' not in example or 'output' not in example:
                continue
            
            input_grid = np.array(example['input'], dtype=np.int64)
            output_grid = np.array(example['output'], dtype=np.int64)
            
            self.input_output_pairs.append((input_grid, output_grid))
        
        print(f"Loaded puzzle {puzzle_id} ({split}): {len(self.input_output_pairs)} examples")
        
        # Print grid size info
        if len(self.input_output_pairs) > 0:
            input_sizes = [pair[0].shape for pair in self.input_output_pairs]
            output_sizes = [pair[1].shape for pair in self.input_output_pairs]
            print(f"  Input sizes: {input_sizes}")
            print(f"  Output sizes: {output_sizes}")
    
    def __len__(self):
        return len(self.input_output_pairs)
    
    def __getitem__(self, idx):
        input_grid, output_grid = self.input_output_pairs[idx]
        
        # Get original sizes
        input_H, input_W = input_grid.shape
        output_H, output_W = output_grid.shape
        
        # Pad to 30x30
        input_pad_h = max(0, 30 - input_H)
        input_pad_w = max(0, 30 - input_W)
        input_grid_padded = np.pad(input_grid, ((0, input_pad_h), (0, input_pad_w)), 
                                   mode='constant', constant_values=0)
        
        output_pad_h = max(0, 30 - output_H)
        output_pad_w = max(0, 30 - output_W)
        output_grid_padded = np.pad(output_grid, ((0, output_pad_h), (0, output_pad_w)), 
                                    mode='constant', constant_values=0)
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_grid_padded).long()
        output_tensor = torch.from_numpy(output_grid_padded).long()
        
        return input_tensor, (input_H, input_W), output_tensor, (output_H, output_W)


def collate_fn_puzzle(batch):
    """Collate function for puzzle dataset."""
    input_grids, input_sizes, output_grids, output_sizes = zip(*batch)
    
    batch_input_grids = torch.stack(input_grids)
    batch_output_grids = torch.stack(output_grids)
    
    return batch_input_grids, input_sizes, batch_output_grids, output_sizes


def load_all_puzzle_ids(json_path):
    """Get list of all puzzle IDs in a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return list(data.keys())