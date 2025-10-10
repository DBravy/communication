"""Test if duplicate grids in batches are breaking the selection task."""

import torch
from torch.utils.data import DataLoader
import numpy as np

import config
from dataset import ARCDataset, collate_fn
from functools import partial


def test_for_duplicates():
    """Check if batches contain duplicate grids when MAX_GRIDS is small."""
    
    print("="*80)
    print("Testing for duplicate grids in selection task")
    print("="*80)
    
    # Load dataset with current config
    dataset = ARCDataset(
        config.DATA_PATH,
        min_size=config.MIN_GRID_SIZE,
        filter_size=getattr(config, 'FILTER_GRID_SIZE', None),
        max_grids=getattr(config, 'MAX_GRIDS', None),
        num_distractors=0  # Test file doesn't need selection task
    )
    
    print(f"\nDataset has {len(dataset)} total samples")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Number of distractors: {config.NUM_DISTRACTORS}")
    
    # Check how many unique grids there are
    unique_grids = set()
    for i in range(len(dataset)):
        grid, size = dataset[i]
        # Convert to tuple for hashing
        h, w = size
        grid_tuple = tuple(grid[:h, :w].flatten().tolist())
        unique_grids.add(grid_tuple)
    
    print(f"Number of UNIQUE grids in dataset: {len(unique_grids)}")
    
    if len(unique_grids) < config.BATCH_SIZE:
        print("\n⚠️  WARNING: You have fewer unique grids than batch size!")
        print(f"   Unique grids: {len(unique_grids)}")
        print(f"   Batch size: {config.BATCH_SIZE}")
        print(f"   This means batches WILL contain duplicates!")
    
    # Create dataloader
    collate_fn_with_distractors = partial(collate_fn, num_distractors=config.NUM_DISTRACTORS)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_with_distractors,
        num_workers=0
    )
    
    # Check a few batches
    print("\n" + "="*80)
    print("Analyzing first batch...")
    print("="*80)
    
    batch = next(iter(loader))
    
    # Handle different batch formats based on actual returned values
    print(f"\nBatch contains {len(batch)} elements")
    print(f"Element types: {[type(x).__name__ for x in batch]}")
    print(f"Element shapes/lengths: {[x.shape if hasattr(x, 'shape') else len(x) if hasattr(x, '__len__') else 'N/A' for x in batch]}")
    
    if len(batch) == 5:
        print("\n⚠️  Unexpected: batch has 5 elements!")
        print("This might mean collate_fn is returning something extra.")
        print("Let me try to unpack it anyway...")
        try:
            # Maybe it's: grids, sizes, candidates_list, target_indices, something_extra?
            grids = batch[0]
            sizes = batch[1]
            candidates_list = batch[2]
            target_indices = batch[3]
            extra = batch[4]
            print(f"\nElement 0 (grids): {grids.shape}")
            print(f"Element 1 (sizes): {sizes}")
            print(f"Element 2 (candidates_list): {type(candidates_list)}, len={len(candidates_list)}")
            print(f"Element 3 (target_indices): {target_indices}")
            print(f"Element 4 (extra???): {type(extra)}, {extra}")
            batch_size = len(grids)
        except Exception as e:
            print(f"Error unpacking: {e}")
            return False
    elif len(batch) == 4:
        grids, sizes, candidates_list, target_indices = batch
        batch_size = len(grids)
        print(f"Batch size (with selection task): {batch_size}")
    elif len(batch) == 2:
        grids, sizes = batch
        batch_size = len(grids)
        print(f"Batch size (reconstruction task): {batch_size}")
        print("\n⚠️  No selection task data - skipping candidate analysis")
        return True
    else:
        print(f"\n❌ Unexpected batch format with {len(batch)} elements")
        return False
    
    # Check for duplicates in the batch itself
    batch_grids_list = []
    for i in range(batch_size):
        h, w = sizes[i]
        grid_data = grids[i, :h, :w]
        batch_grids_list.append(grid_data)
    
    print(f"\nBatch has {batch_size} samples")
    
    # Count duplicates
    duplicate_count = 0
    for i in range(batch_size):
        for j in range(i+1, batch_size):
            h1, w1 = sizes[i]
            h2, w2 = sizes[j]
            if h1 == h2 and w1 == w2:
                grid1 = grids[i, :h1, :w1]
                grid2 = grids[j, :h2, :w2]
                if torch.equal(grid1, grid2):
                    duplicate_count += 1
    
    print(f"Number of duplicate pairs in batch: {duplicate_count}")
    
    # Check if any candidate set contains duplicates
    print("\n" + "="*80)
    print("Checking candidate sets for duplicates...")
    print("="*80)
    
    impossible_count = 0
    
    for i in range(min(10, batch_size)):  # Check first 10 samples
        h, w = sizes[i]
        target_grid = grids[i, :h, :w]
        candidates = candidates_list[i]
        target_idx = target_indices[i].item()
        
        num_candidates = len(candidates)
        
        # Check if target appears multiple times in candidates
        target_appearances = 0
        for c_idx in range(num_candidates):
            cand_grid = candidates[c_idx, :h, :w]
            if torch.equal(target_grid, cand_grid):
                target_appearances += 1
        
        if target_appearances > 1:
            impossible_count += 1
            print(f"\nSample {i}: ❌ IMPOSSIBLE TASK!")
            print(f"  Target grid appears {target_appearances} times in candidates")
            print(f"  Target should be at index {target_idx}")
            print(f"  But there are {target_appearances} identical grids!")
            print(f"  The model cannot solve this - it's fundamentally ambiguous!")
    
    if impossible_count > 0:
        print("\n" + "="*80)
        print(f"🔴 FOUND THE PROBLEM!")
        print("="*80)
        print(f"\n{impossible_count}/{min(10, batch_size)} samples have IMPOSSIBLE tasks!")
        print("\nThe target grid appears multiple times in the candidate set,")
        print("making it impossible for the model to learn which one is 'correct'.")
        print("\nThis is why your accuracy is stuck at 50% - the task is often")
        print("fundamentally ambiguous due to duplicate grids in batches!")
        
        print("\n" + "="*80)
        print("SOLUTIONS:")
        print("="*80)
        print("\n1. Increase MAX_GRIDS to have more unique grids")
        print(f"   Current: MAX_GRIDS = {config.MAX_GRIDS}")
        print(f"   Recommended: MAX_GRIDS >= {config.BATCH_SIZE * 2}")
        
        print("\n2. OR: Reduce BATCH_SIZE to be less than unique grids")
        print(f"   Current: BATCH_SIZE = {config.BATCH_SIZE}")
        print(f"   Recommended: BATCH_SIZE <= {len(unique_grids)}")
        
        print("\n3. OR: Modify collate_fn to ensure distractors are DIFFERENT grids")
        print("   (Check actual grid content, not just batch position)")
        
        return False
    else:
        print("\n✓ No impossible tasks found in checked samples")
        return True


if __name__ == '__main__':
    success = test_for_duplicates()
    
    if not success:
        print("\n" + "="*80)
        print("RECOMMENDATION: Fix the config before training!")
        print("="*80)