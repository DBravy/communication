"""Dataset loader for ARC puzzles."""

import json
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob


def load_arc_data(path):
    """
    Load ARC data from either a single JSON file or a directory of JSON files.
    
    Args:
        path: Either a path to a JSON file or a directory containing JSON files
    
    Returns:
        Dictionary mapping task_id to task data
    """
    if os.path.isfile(path):
        # Single JSON file format (legacy)
        with open(path, 'r') as f:
            return json.load(f)
    elif os.path.isdir(path):
        # Directory format (V1/V2)
        data = {}
        json_files = glob.glob(os.path.join(path, '*.json'))
        for json_file in json_files:
            # Task ID is the filename without extension
            task_id = os.path.splitext(os.path.basename(json_file))[0]
            with open(json_file, 'r') as f:
                data[task_id] = json.load(f)
        return data
    else:
        raise ValueError(f"Path {path} is neither a file nor a directory")


class ARCDataset(Dataset):
    def __init__(self, json_path, min_size=3, filter_size=None, max_grids=None, num_distractors=0, track_puzzle_ids=False, use_input_output_pairs=False):
        """
        Args:
            json_path: Path to ARC JSON file
            min_size: Minimum grid size (pad smaller grids)
            filter_size: If provided, only load grids of this exact size (height, width)
            max_grids: Maximum number of grids to load (None = load all)
            num_distractors: Number of distractor grids to include for selection task (0 = reconstruction)
            track_puzzle_ids: If True, track which puzzle each grid belongs to and whether it's input/output
            use_input_output_pairs: If True, use input-output pairs (input as source, output as target)
        """
        self.min_size = min_size
        self.filter_size = filter_size
        self.max_grids = max_grids
        self.num_distractors = num_distractors
        self.track_puzzle_ids = track_puzzle_ids
        self.use_input_output_pairs = use_input_output_pairs
        self.grids = []
        self.puzzle_ids = []  # Which puzzle this grid belongs to
        self.is_input = []  # Whether this grid is an input (True) or output (False)
        self.puzzle_id_map = {}  # Map from task_id to integer puzzle ID
        self.input_output_pairs = []  # List of (input_idx, output_idx) pairs
        
        # Load all grids from JSON (supports both file and directory formats)
        data = load_arc_data(json_path)
        
        total_grids_before_filter = 0
        
        for task_id, task_data in data.items():
            # Process examples
            for example in task_data.get('train', []) + task_data.get('test', []):
                if 'input' not in example or 'output' not in example:
                    continue
                    
                input_grid = np.array(example['input'], dtype=np.int64)
                output_grid = np.array(example['output'], dtype=np.int64)
                total_grids_before_filter += 2
                
                # Check if both grids should be included
                input_valid = self._should_include_grid(input_grid)
                output_valid = self._should_include_grid(output_grid)
                
                if use_input_output_pairs:
                    # Only add pairs where both input and output are valid
                    if input_valid and output_valid:
                        # ASSIGN PUZZLE ID ONLY WHEN ACTUALLY ADDING GRIDS
                        if track_puzzle_ids and task_id not in self.puzzle_id_map:
                            self.puzzle_id_map[task_id] = len(self.puzzle_id_map)
                        
                        input_idx = len(self.grids)
                        self.grids.append(input_grid)
                        if track_puzzle_ids:
                            self.puzzle_ids.append(self.puzzle_id_map[task_id])
                            self.is_input.append(True)
                        
                        output_idx = len(self.grids)
                        self.grids.append(output_grid)
                        if track_puzzle_ids:
                            self.puzzle_ids.append(self.puzzle_id_map[task_id])
                            self.is_input.append(False)
                        
                        # Store the pair
                        self.input_output_pairs.append((input_idx, output_idx))
                        
                        if self.max_grids and len(self.input_output_pairs) >= self.max_grids:
                            break
                else:
                    # Original behavior: add grids independently
                    if input_valid:
                        # ASSIGN PUZZLE ID ONLY WHEN ACTUALLY ADDING A GRID
                        if track_puzzle_ids and task_id not in self.puzzle_id_map:
                            self.puzzle_id_map[task_id] = len(self.puzzle_id_map)
                        
                        self.grids.append(input_grid)
                        if track_puzzle_ids:
                            self.puzzle_ids.append(self.puzzle_id_map[task_id])
                            self.is_input.append(True)
                        if self.max_grids and len(self.grids) >= self.max_grids:
                            break
                    
                    if output_valid and (not self.max_grids or len(self.grids) < self.max_grids):
                        # ASSIGN PUZZLE ID ONLY WHEN ACTUALLY ADDING A GRID
                        if track_puzzle_ids and task_id not in self.puzzle_id_map:
                            self.puzzle_id_map[task_id] = len(self.puzzle_id_map)
                        
                        self.grids.append(output_grid)
                        if track_puzzle_ids:
                            self.puzzle_ids.append(self.puzzle_id_map[task_id])
                            self.is_input.append(False)
                        if self.max_grids and len(self.grids) >= self.max_grids:
                            break
            
            # Break if we've reached max_grids
            if use_input_output_pairs and self.max_grids and len(self.input_output_pairs) >= self.max_grids:
                break
            if not use_input_output_pairs and self.max_grids and len(self.grids) >= self.max_grids:
                break
        
        # Print loading summary
        if use_input_output_pairs:
            if self.filter_size:
                print(f"Loaded {len(self.input_output_pairs)} input-output pairs of size {self.filter_size[0]}x{self.filter_size[1]} from {total_grids_before_filter} total grids in {len(data)} tasks")
            else:
                print(f"Loaded {len(self.input_output_pairs)} input-output pairs from {len(data)} tasks")
        else:
            if self.filter_size and self.max_grids:
                print(f"Loaded {len(self.grids)} grids of size {self.filter_size[0]}x{self.filter_size[1]} (max: {self.max_grids}) from {total_grids_before_filter} total grids")
            elif self.filter_size:
                print(f"Loaded {len(self.grids)} grids of size {self.filter_size[0]}x{self.filter_size[1]} from {total_grids_before_filter} total grids in {len(data)} tasks")
            elif self.max_grids:
                print(f"Loaded {len(self.grids)} grids (max: {self.max_grids}) from {len(data)} tasks")
            else:
                print(f"Loaded {len(self.grids)} grids from {len(data)} tasks")
        
        if track_puzzle_ids:
            print(f"Tracking {len(self.puzzle_id_map)} unique puzzles with input/output labels")
            # Count inputs vs outputs
            num_inputs = sum(self.is_input)
            num_outputs = len(self.is_input) - num_inputs
            print(f"  - Inputs: {num_inputs}, Outputs: {num_outputs}")
    
    def _should_include_grid(self, grid):
        """Check if grid should be included based on filter_size."""
        if self.filter_size is None:
            return True
        h, w = grid.shape
        return h == self.filter_size[0] and w == self.filter_size[1]
    
    def __len__(self):
        if self.use_input_output_pairs:
            return len(self.input_output_pairs)
        return len(self.grids)
    
    def __getitem__(self, idx):
        # For input-output pairs
        if self.use_input_output_pairs:
            input_idx, output_idx = self.input_output_pairs[idx]
            
            # Get input grid (source)
            input_grid = self.grids[input_idx].copy()
            input_H, input_W = input_grid.shape
            
            # Get output grid (target)
            output_grid = self.grids[output_idx].copy()
            output_H, output_W = output_grid.shape
            
            # Pad input grid to 30x30
            input_pad_h = max(0, 30 - input_H)
            input_pad_w = max(0, 30 - input_W)
            input_grid_padded = np.pad(input_grid, ((0, input_pad_h), (0, input_pad_w)), 
                                       mode='constant', constant_values=0)
            input_tensor = torch.from_numpy(input_grid_padded).long()
            
            # Pad output grid to 30x30
            output_pad_h = max(0, 30 - output_H)
            output_pad_w = max(0, 30 - output_W)
            output_grid_padded = np.pad(output_grid, ((0, output_pad_h), (0, output_pad_w)), 
                                        mode='constant', constant_values=0)
            output_tensor = torch.from_numpy(output_grid_padded).long()
            
            # For selection task with input-output pairs
            if self.num_distractors > 0:
                # Sample distractor OUTPUT grids (excluding the current output)
                available_pairs = list(range(len(self.input_output_pairs)))
                available_pairs.remove(idx)
                
                # Limit to actual available distractors
                actual_num_distractors = min(self.num_distractors, len(available_pairs))
                
                # Randomly sample distractor pair indices
                if actual_num_distractors > 0:
                    distractor_pair_indices = np.random.choice(available_pairs, size=actual_num_distractors, replace=False).tolist()
                else:
                    distractor_pair_indices = []
                
                # Create candidate set: correct output + distractor outputs
                # Randomly position the correct answer
                num_candidates = len(distractor_pair_indices) + 1
                target_idx = np.random.randint(0, num_candidates)
                
                candidates = []
                candidate_sizes = []
                for j in range(num_candidates):
                    if j == target_idx:
                        # Add the target OUTPUT grid
                        candidates.append(output_tensor)
                        candidate_sizes.append((output_H, output_W))
                    else:
                        # Add a distractor OUTPUT
                        dist_pos = j if j < target_idx else j - 1
                        dist_pair_idx = distractor_pair_indices[dist_pos]
                        _, dist_output_idx = self.input_output_pairs[dist_pair_idx]
                        
                        # Load and pad distractor output
                        dist_grid = self.grids[dist_output_idx].copy()
                        dist_H, dist_W = dist_grid.shape
                        dist_pad_h = max(0, 30 - dist_H)
                        dist_pad_w = max(0, 30 - dist_W)
                        dist_grid_padded = np.pad(dist_grid, ((0, dist_pad_h), (0, dist_pad_w)), 
                                                 mode='constant', constant_values=0)
                        dist_tensor = torch.from_numpy(dist_grid_padded).long()
                        
                        candidates.append(dist_tensor)
                        candidate_sizes.append((dist_H, dist_W))
                
                candidates_tensor = torch.stack(candidates)
                
                # Return: input grid, input size, output grid, output size, candidates, candidate_sizes, target_idx
                return input_tensor, (input_H, input_W), output_tensor, (output_H, output_W), candidates_tensor, candidate_sizes, target_idx
            else:
                # Reconstruction task with input-output pairs
                # Return: input grid, input size, output grid, output size
                return input_tensor, (input_H, input_W), output_tensor, (output_H, output_W)
        
        # Original behavior (self-supervised)
        grid = self.grids[idx].copy()
        H, W = grid.shape
        
        # Always pad to 30x30
        pad_h = max(0, 30 - H)
        pad_w = max(0, 30 - W)
        grid = np.pad(grid, ((0, pad_h), (0, pad_w)), 
                     mode='constant', constant_values=0)
        
        # Convert to tensor
        grid_tensor = torch.from_numpy(grid).long()
        
        # If tracking puzzle IDs, return with puzzle classification info
        if self.track_puzzle_ids and self.num_distractors == 0:
            puzzle_id = self.puzzle_ids[idx]
            is_input = self.is_input[idx]
            # Create combined label: puzzle_id * 2 + (0 if input else 1)
            # This creates separate classes for inputs and outputs of each puzzle
            combined_label = puzzle_id * 2 + (0 if is_input else 1)
            return grid_tensor, (H, W), combined_label
        
        # If no distractors and not tracking puzzle IDs, return original format
        if self.num_distractors == 0:
            return grid_tensor, (H, W)
        
        # For selection task, sample distractors from the dataset
        # Sample distractor indices (all dataset items except current one)
        available_indices = list(range(len(self.grids)))
        available_indices.remove(idx)
        
        # Limit to actual available distractors
        actual_num_distractors = min(self.num_distractors, len(available_indices))
        
        # Randomly sample distractor indices
        if actual_num_distractors > 0:
            distractor_indices = np.random.choice(available_indices, size=actual_num_distractors, replace=False).tolist()
        else:
            distractor_indices = []
        
        # Create candidate set: correct grid + distractors
        # Randomly position the correct answer
        num_candidates = len(distractor_indices) + 1
        target_idx = np.random.randint(0, num_candidates)
        
        candidates = []
        candidate_sizes = []
        for j in range(num_candidates):
            if j == target_idx:
                # Add the target grid
                candidates.append(grid_tensor)
                candidate_sizes.append((H, W))
            else:
                # Add a distractor
                dist_pos = j if j < target_idx else j - 1
                dist_idx = distractor_indices[dist_pos]
                
                # Load and pad distractor
                dist_grid = self.grids[dist_idx].copy()
                dist_H, dist_W = dist_grid.shape
                dist_pad_h = max(0, 30 - dist_H)
                dist_pad_w = max(0, 30 - dist_W)
                dist_grid = np.pad(dist_grid, ((0, dist_pad_h), (0, dist_pad_w)), 
                                 mode='constant', constant_values=0)
                dist_tensor = torch.from_numpy(dist_grid).long()
                
                candidates.append(dist_tensor)
                candidate_sizes.append((dist_H, dist_W))
        
        candidates_tensor = torch.stack(candidates)
        
        return grid_tensor, (H, W), candidates_tensor, candidate_sizes, target_idx

def collate_fn(batch, num_distractors=0, use_input_output_pairs=False):
    """
    Custom collate to handle grids (all padded to 30x30 already).
    
    Args:
        batch: List of dataset outputs
               - If use_input_output_pairs=False and num_distractors == 0: list of (grid, size) tuples
               - If use_input_output_pairs=False and num_distractors > 0: list of (grid, size, candidates, candidate_sizes, target_idx) tuples
               - If use_input_output_pairs=True and num_distractors == 0: list of (input_grid, input_size, output_grid, output_size) tuples
               - If use_input_output_pairs=True and num_distractors > 0: list of (input_grid, input_size, output_grid, output_size, candidates, candidate_sizes, target_idx) tuples
        num_distractors: Number of distractor grids (0 for reconstruction task)
        use_input_output_pairs: Whether using input-output pairs
    
    Returns:
        If use_input_output_pairs=False and num_distractors == 0 (reconstruction):
            batch_grids: [batch_size, 30, 30]
            sizes: list of (H, W) tuples
        
        If use_input_output_pairs=False and num_distractors > 0 (selection):
            batch_grids: [batch_size, 30, 30]  # target grids
            sizes: list of (H, W) tuples
            candidates_list: list of [num_candidates, 30, 30] tensors
            candidates_sizes_list: list of lists of (H, W) tuples for each candidate
            target_indices: [batch_size] indices of correct grid in candidates
        
        If use_input_output_pairs=True and num_distractors == 0 (reconstruction with I/O):
            batch_input_grids: [batch_size, 30, 30]
            input_sizes: list of (H, W) tuples
            batch_output_grids: [batch_size, 30, 30]
            output_sizes: list of (H, W) tuples
        
        If use_input_output_pairs=True and num_distractors > 0 (selection with I/O):
            batch_input_grids: [batch_size, 30, 30]
            input_sizes: list of (H, W) tuples
            batch_output_grids: [batch_size, 30, 30]
            output_sizes: list of (H, W) tuples
            candidates_list: list of [num_candidates, 30, 30] tensors
            candidates_sizes_list: list of lists of (H, W) tuples for each candidate
            target_indices: [batch_size] indices of correct grid in candidates
    """
    # Handle input-output pairs
    if use_input_output_pairs:
        if num_distractors == 0:
            # Reconstruction with input-output pairs
            input_grids, input_sizes, output_grids, output_sizes = zip(*batch)
            batch_input_grids = torch.stack(input_grids)
            batch_output_grids = torch.stack(output_grids)
            return batch_input_grids, input_sizes, batch_output_grids, output_sizes
        else:
            # Selection with input-output pairs
            input_grids, input_sizes, output_grids, output_sizes, candidates_list, candidates_sizes_list, target_indices = zip(*batch)
            batch_input_grids = torch.stack(input_grids)
            batch_output_grids = torch.stack(output_grids)
            candidates_list = list(candidates_list)
            candidates_sizes_list = list(candidates_sizes_list)
            target_indices = torch.tensor(target_indices, dtype=torch.long)
            return batch_input_grids, input_sizes, batch_output_grids, output_sizes, candidates_list, candidates_sizes_list, target_indices
    
    # Original behavior (self-supervised)
    # If no distractors, batch is list of (grid, size) tuples
    if num_distractors == 0:
        grids, sizes = zip(*batch)
        batch_grids = torch.stack(grids)
        return batch_grids, sizes
    
    # For selection task, batch is list of (grid, size, candidates, candidate_sizes, target_idx) tuples
    grids, sizes, candidates_list, candidates_sizes_list, target_indices = zip(*batch)
    
    # Stack grids
    batch_grids = torch.stack(grids)
    
    # Convert candidates_list and target_indices to lists (they're tuples from zip)
    candidates_list = list(candidates_list)
    candidates_sizes_list = list(candidates_sizes_list)
    target_indices = torch.tensor(target_indices, dtype=torch.long)
    
    return batch_grids, sizes, candidates_list, candidates_sizes_list, target_indices


def collate_fn_puzzle_classification(batch):
    """
    Custom collate function for puzzle classification task.
    
    Args:
        batch: List of (grid, size, label) tuples
    
    Returns:
        batch_grids: [batch_size, 30, 30]
        sizes: list of (H, W) tuples
        labels: [batch_size] tensor of puzzle classification labels
    """
    grids, sizes, labels = zip(*batch)
    batch_grids = torch.stack(grids)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return batch_grids, sizes, labels_tensor