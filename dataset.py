"""Dataset loader for ARC puzzles."""

import json
import torch
from torch.utils.data import Dataset
import numpy as np

class ARCDataset(Dataset):
    def __init__(self, json_path, min_size=3, filter_size=None, max_grids=None, num_distractors=0, track_puzzle_ids=False):
        """
        Args:
            json_path: Path to ARC JSON file
            min_size: Minimum grid size (pad smaller grids)
            filter_size: If provided, only load grids of this exact size (height, width)
            max_grids: Maximum number of grids to load (None = load all)
            num_distractors: Number of distractor grids to include for selection task (0 = reconstruction)
            track_puzzle_ids: If True, track which puzzle each grid belongs to and whether it's input/output
        """
        self.min_size = min_size
        self.filter_size = filter_size
        self.max_grids = max_grids
        self.num_distractors = num_distractors
        self.track_puzzle_ids = track_puzzle_ids
        self.grids = []
        self.puzzle_ids = []  # Which puzzle this grid belongs to
        self.is_input = []  # Whether this grid is an input (True) or output (False)
        self.puzzle_id_map = {}  # Map from task_id to integer puzzle ID
        
        # Load all grids from JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        total_grids_before_filter = 0
        
        for task_id, task_data in data.items():
            # DON'T assign puzzle ID here anymore!
            # We'll assign it only when we actually add a grid from this puzzle
            
            # Process training examples
            for example in task_data.get('train', []):
                if 'input' in example:
                    grid = np.array(example['input'], dtype=np.int64)
                    total_grids_before_filter += 1
                    if self._should_include_grid(grid):
                        # ASSIGN PUZZLE ID ONLY WHEN ACTUALLY ADDING A GRID
                        if track_puzzle_ids and task_id not in self.puzzle_id_map:
                            self.puzzle_id_map[task_id] = len(self.puzzle_id_map)
                        
                        self.grids.append(grid)
                        if track_puzzle_ids:
                            self.puzzle_ids.append(self.puzzle_id_map[task_id])
                            self.is_input.append(True)
                        if self.max_grids and len(self.grids) >= self.max_grids:
                            break
                if 'output' in example and (not self.max_grids or len(self.grids) < self.max_grids):
                    grid = np.array(example['output'], dtype=np.int64)
                    total_grids_before_filter += 1
                    if self._should_include_grid(grid):
                        # ASSIGN PUZZLE ID ONLY WHEN ACTUALLY ADDING A GRID
                        if track_puzzle_ids and task_id not in self.puzzle_id_map:
                            self.puzzle_id_map[task_id] = len(self.puzzle_id_map)
                        
                        self.grids.append(grid)
                        if track_puzzle_ids:
                            self.puzzle_ids.append(self.puzzle_id_map[task_id])
                            self.is_input.append(False)
                        if self.max_grids and len(self.grids) >= self.max_grids:
                            break
            
            # Break if we've reached max_grids
            if self.max_grids and len(self.grids) >= self.max_grids:
                break
            
            # Process test examples
            for example in task_data.get('test', []):
                if 'input' in example:
                    grid = np.array(example['input'], dtype=np.int64)
                    total_grids_before_filter += 1
                    if self._should_include_grid(grid):
                        # ASSIGN PUZZLE ID ONLY WHEN ACTUALLY ADDING A GRID
                        if track_puzzle_ids and task_id not in self.puzzle_id_map:
                            self.puzzle_id_map[task_id] = len(self.puzzle_id_map)
                        
                        self.grids.append(grid)
                        if track_puzzle_ids:
                            self.puzzle_ids.append(self.puzzle_id_map[task_id])
                            self.is_input.append(True)
                        if self.max_grids and len(self.grids) >= self.max_grids:
                            break
                # Note: test outputs usually not available
                if 'output' in example and (not self.max_grids or len(self.grids) < self.max_grids):
                    grid = np.array(example['output'], dtype=np.int64)
                    total_grids_before_filter += 1
                    if self._should_include_grid(grid):
                        # ASSIGN PUZZLE ID ONLY WHEN ACTUALLY ADDING A GRID
                        if track_puzzle_ids and task_id not in self.puzzle_id_map:
                            self.puzzle_id_map[task_id] = len(self.puzzle_id_map)
                        
                        self.grids.append(grid)
                        if track_puzzle_ids:
                            self.puzzle_ids.append(self.puzzle_id_map[task_id])
                            self.is_input.append(False)
                        if self.max_grids and len(self.grids) >= self.max_grids:
                            break
            
            # Break if we've reached max_grids
            if self.max_grids and len(self.grids) >= self.max_grids:
                break
        
        # Print loading summary
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
        return len(self.grids)
    
    def __getitem__(self, idx):
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

def collate_fn(batch, num_distractors=0):
    """
    Custom collate to handle grids (all padded to 30x30 already).
    
    Args:
        batch: List of dataset outputs
               - If num_distractors == 0: list of (grid, size) tuples
               - If num_distractors > 0: list of (grid, size, candidates, candidate_sizes, target_idx) tuples
        num_distractors: Number of distractor grids (0 for reconstruction task)
    
    Returns:
        If num_distractors == 0 (reconstruction):
            batch_grids: [batch_size, 30, 30]
            sizes: list of (H, W) tuples
        
        If num_distractors > 0 (selection):
            batch_grids: [batch_size, 30, 30]  # target grids
            sizes: list of (H, W) tuples
            candidates_list: list of [num_candidates, 30, 30] tensors
            candidates_sizes_list: list of lists of (H, W) tuples for each candidate
            target_indices: [batch_size] indices of correct grid in candidates
    """
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