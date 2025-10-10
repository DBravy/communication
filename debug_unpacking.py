"""Debug script to identify batch unpacking issues."""

import torch
from torch.utils.data import DataLoader
from functools import partial
import config
from dataset import ARCDataset, collate_fn

def test_batch_structure():
    """Test the structure of batches with different configurations."""
    
    print("="*80)
    print("BATCH STRUCTURE DEBUGGING")
    print("="*80)
    
    # Test configuration from config.py
    task_type = getattr(config, 'TASK_TYPE', 'selection')
    num_distractors = getattr(config, 'NUM_DISTRACTORS', 1)
    use_input_output_pairs = getattr(config, 'USE_INPUT_OUTPUT_PAIRS', True)
    
    print(f"\nConfiguration:")
    print(f"  - task_type: {task_type}")
    print(f"  - num_distractors: {num_distractors}")
    print(f"  - use_input_output_pairs: {use_input_output_pairs}")
    
    # Load dataset
    dataset = ARCDataset(
        config.DATA_PATH, 
        min_size=config.MIN_GRID_SIZE,
        filter_size=getattr(config, 'FILTER_GRID_SIZE', None),
        max_grids=10,  # Small number for quick testing
        num_distractors=num_distractors,
        track_puzzle_ids=False,
        use_input_output_pairs=use_input_output_pairs
    )
    
    # Create collate function
    collate_fn_for_task = partial(
        collate_fn, 
        num_distractors=num_distractors, 
        use_input_output_pairs=use_input_output_pairs
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn_for_task,
        num_workers=0
    )
    
    # Get first batch
    print("\n" + "-"*80)
    print("INSPECTING FIRST BATCH")
    print("-"*80)
    
    for batch_data in dataloader:
        print(f"\nBatch is a {type(batch_data)}")
        print(f"Number of elements in batch: {len(batch_data)}")
        
        for i, element in enumerate(batch_data):
            if isinstance(element, torch.Tensor):
                print(f"  Element {i}: Tensor with shape {element.shape}")
            elif isinstance(element, list):
                print(f"  Element {i}: List with {len(element)} items")
                if len(element) > 0:
                    if isinstance(element[0], torch.Tensor):
                        print(f"    - First item is Tensor with shape {element[0].shape}")
                    elif isinstance(element[0], tuple):
                        print(f"    - First item is tuple: {element[0]}")
                    else:
                        print(f"    - First item type: {type(element[0])}")
            elif isinstance(element, tuple):
                print(f"  Element {i}: Tuple with {len(element)} items")
                if len(element) > 0:
                    print(f"    - First item type: {type(element[0])}")
            else:
                print(f"  Element {i}: {type(element)}")
        
        # Try unpacking with different expectations
        print("\n" + "-"*80)
        print("UNPACKING TESTS")
        print("-"*80)
        
        # Test 5-element unpacking (original selection format)
        print("\nTest 1: Unpacking as 5 elements (selection without I/O pairs)")
        try:
            grids, sizes, candidates_list, candidates_sizes_list, target_indices = batch_data
            print("  ✓ SUCCESS - Batch has 5 elements")
        except ValueError as e:
            print(f"  ✗ FAILED - {e}")
        
        # Test 7-element unpacking (selection with I/O pairs)
        print("\nTest 2: Unpacking as 7 elements (selection with I/O pairs)")
        try:
            input_grids, input_sizes, output_grids, output_sizes, candidates_list, candidates_sizes_list, target_indices = batch_data
            print("  ✓ SUCCESS - Batch has 7 elements")
            print(f"    - input_grids: {input_grids.shape}")
            print(f"    - output_grids: {output_grids.shape}")
            print(f"    - candidates_list length: {len(candidates_list)}")
        except ValueError as e:
            print(f"  ✗ FAILED - {e}")
        
        # Show what get_selections expects
        print("\n" + "-"*80)
        print("CURRENT get_selections() EXPECTATION")
        print("-"*80)
        print("\nThe get_selections() function in app.py currently expects:")
        print("  grids, sizes, candidates_list, candidates_sizes_list, target_indices = batch_data")
        print("  (5 elements)")
        
        print("\nBut when use_input_output_pairs=True, batch_data actually has 7 elements!")
        print("\nFIX: get_selections() needs to check use_input_output_pairs and unpack accordingly")
        
        break  # Only check first batch
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    test_batch_structure()