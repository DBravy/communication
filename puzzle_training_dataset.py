"""Dataset for training puzzle solver on multiple puzzles."""

import json
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import random


class ARCPuzzleTrainingDataset(Dataset):
    """
    Dataset for training on multiple ARC puzzles.
    Each sample is a complete puzzle with training and test examples.
    """
    
    def __init__(self, json_path, max_puzzles=None, max_train_examples=None, max_test_examples=1):
        """
        Args:
            json_path: Path to ARC JSON file or directory
            max_puzzles: Maximum number of puzzles to load (None = all)
            max_train_examples: Max training examples per puzzle (None = all)
            max_test_examples: Max test examples per puzzle (default: 1)
        """
        self.max_train_examples = max_train_examples
        self.max_test_examples = max_test_examples
        self.puzzles = []
        
        # Load puzzles
        if os.path.isfile(json_path):
            # Single JSON file
            with open(json_path, 'r') as f:
                all_data = json.load(f)
            puzzle_ids = list(all_data.keys())
            
            # Load each puzzle
            for puzzle_id in puzzle_ids:
                puzzle_data = all_data[puzzle_id]
                self._process_puzzle(puzzle_id, puzzle_data)
                
        elif os.path.isdir(json_path):
            # Directory format
            json_files = glob.glob(os.path.join(json_path, '*.json'))
            puzzle_ids = [os.path.splitext(os.path.basename(f))[0] for f in json_files]
            
            # Load each puzzle from separate file
            for puzzle_id in puzzle_ids:
                puzzle_file = os.path.join(json_path, f"{puzzle_id}.json")
                try:
                    with open(puzzle_file, 'r') as f:
                        puzzle_data = json.load(f)
                    self._process_puzzle(puzzle_id, puzzle_data)
                except Exception as e:
                    print(f"Warning: Could not load puzzle {puzzle_id}: {e}")
                    continue
        else:
            raise ValueError(f"Path {json_path} is neither file nor directory")
        
        # Limit number of puzzles if specified
        if max_puzzles is not None and len(self.puzzles) > max_puzzles:
            self.puzzles = self.puzzles[:max_puzzles]
        
        print(f"Loaded {len(self.puzzles)} puzzles from {json_path}")
        if len(self.puzzles) == 0:
            raise ValueError("No valid puzzles loaded! Check your dataset path and format.")
    
    def _process_puzzle(self, puzzle_id, puzzle_data):
        """Process and validate a single puzzle."""
        # Get training examples
        train_examples = puzzle_data.get('train', [])
        if len(train_examples) == 0:
            return  # Skip puzzles with no training data
        
        # Validate training examples have both input and output
        valid_train = []
        for ex in train_examples:
            if 'input' in ex and 'output' in ex:
                valid_train.append(ex)
        
        if len(valid_train) == 0:
            return  # Skip if no valid training examples
        
        # Get test examples
        test_examples = puzzle_data.get('test', [])
        if len(test_examples) == 0:
            return  # Skip puzzles with no test data
        
        # Validate test examples have both input and output
        # (we need outputs for training the solver)
        valid_test = []
        for ex in test_examples:
            if 'input' in ex and 'output' in ex:
                valid_test.append(ex)
        
        if len(valid_test) == 0:
            # If test examples don't have outputs, this is likely evaluation set
            # Skip for training purposes
            return
        
        # Store puzzle
        self.puzzles.append({
            'id': puzzle_id,
            'train': valid_train,
            'test': valid_test
        })
    
    def __len__(self):
        return len(self.puzzles)
    
    def __getitem__(self, idx):
        """
        Returns a single puzzle with its training and test examples.
        
        Returns:
            train_inputs: List of padded input grids [30, 30]
            train_input_sizes: List of (h, w) tuples
            train_outputs: List of padded output grids [30, 30]
            train_output_sizes: List of (h, w) tuples
            test_inputs: List of padded test input grids [30, 30]
            test_input_sizes: List of (h, w) tuples
            test_outputs: List of padded test output grids [30, 30]
            test_output_sizes: List of (h, w) tuples
            puzzle_id: String identifier
        """
        puzzle = self.puzzles[idx]
        
        # Process training examples
        train_examples = puzzle['train']
        if self.max_train_examples is not None:
            train_examples = train_examples[:self.max_train_examples]
        
        train_inputs = []
        train_input_sizes = []
        train_outputs = []
        train_output_sizes = []
        
        for example in train_examples:
            inp = np.array(example['input'], dtype=np.int64)
            out = np.array(example['output'], dtype=np.int64)
            
            inp_h, inp_w = inp.shape
            out_h, out_w = out.shape
            
            # Pad to 30x30
            inp_padded = np.pad(inp, ((0, max(0, 30 - inp_h)), (0, max(0, 30 - inp_w))), 
                               constant_values=0)
            out_padded = np.pad(out, ((0, max(0, 30 - out_h)), (0, max(0, 30 - out_w))), 
                               constant_values=0)
            
            train_inputs.append(torch.from_numpy(inp_padded).long())
            train_input_sizes.append((inp_h, inp_w))
            train_outputs.append(torch.from_numpy(out_padded).long())
            train_output_sizes.append((out_h, out_w))
        
        # Process test examples
        test_examples = puzzle['test']
        if self.max_test_examples is not None:
            test_examples = test_examples[:self.max_test_examples]
        
        test_inputs = []
        test_input_sizes = []
        test_outputs = []
        test_output_sizes = []
        
        for example in test_examples:
            inp = np.array(example['input'], dtype=np.int64)
            out = np.array(example['output'], dtype=np.int64)
            
            inp_h, inp_w = inp.shape
            out_h, out_w = out.shape
            
            # Pad to 30x30
            inp_padded = np.pad(inp, ((0, max(0, 30 - inp_h)), (0, max(0, 30 - inp_w))), 
                               constant_values=0)
            out_padded = np.pad(out, ((0, max(0, 30 - out_h)), (0, max(0, 30 - out_w))), 
                               constant_values=0)
            
            test_inputs.append(torch.from_numpy(inp_padded).long())
            test_input_sizes.append((inp_h, inp_w))
            test_outputs.append(torch.from_numpy(out_padded).long())
            test_output_sizes.append((out_h, out_w))
        
        return (train_inputs, train_input_sizes, train_outputs, train_output_sizes,
                test_inputs, test_input_sizes, test_outputs, test_output_sizes,
                puzzle['id'])


def collate_fn_puzzle_training(batch):
    """
    Collate function for puzzle training.
    
    Note: batch_size should be 1 for puzzle training since each puzzle
    has variable numbers of training/test examples.
    """
    if len(batch) != 1:
        raise ValueError("Puzzle training only supports batch_size=1")
    
    return batch[0]