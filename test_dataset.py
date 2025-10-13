"""Test script to validate ARC dataset loading and identify issues."""

import json
import numpy as np
import torch
from collections import defaultdict
import argparse

import config
from dataset import ARCDataset, collate_fn
from puzzle_dataset import ARCSinglePuzzleDataset, load_all_puzzle_ids


def visualize_grid(grid, title="Grid"):
    """Print a simple ASCII representation of a grid."""
    colors = ['.', '#', 'o', 'O', '*', '+', 'x', 'X', '@', '%']
    
    print(f"\n{title} ({grid.shape[0]}x{grid.shape[1]}):")
    print("+" + "-" * (grid.shape[1] * 2) + "+")
    
    for row in grid:
        print("|", end="")
        for cell in row:
            print(colors[int(cell) % len(colors)] + " ", end="")
        print("|")
    
    print("+" + "-" * (grid.shape[1] * 2) + "+")


def check_grid_properties(grid, name="Grid"):
    """Check various properties of a grid."""
    issues = []
    
    # Check if grid is empty (all zeros)
    if np.all(grid == 0):
        issues.append(f"{name} is completely blank (all zeros)")
    
    # Check if grid has only one unique value
    unique_values = np.unique(grid)
    if len(unique_values) == 1:
        issues.append(f"{name} has only one unique value: {unique_values[0]}")
    
    # Check grid size
    h, w = grid.shape
    if h == 0 or w == 0:
        issues.append(f"{name} has zero dimension: {h}x{w}")
    
    if h > 30 or w > 30:
        issues.append(f"{name} exceeds max size: {h}x{w}")
    
    # Check for invalid color values
    if np.any(grid < 0) or np.any(grid >= config.NUM_COLORS):
        issues.append(f"{name} has invalid color values (outside 0-{config.NUM_COLORS-1})")
    
    return issues, {
        'shape': (h, w),
        'unique_values': len(unique_values),
        'all_zeros': np.all(grid == 0),
        'min_value': int(np.min(grid)),
        'max_value': int(np.max(grid))
    }


def test_json_file_structure(json_path):
    """Test the structure of the JSON file."""
    print(f"\n{'='*80}")
    print(f"TESTING JSON FILE STRUCTURE: {json_path}")
    print(f"{'='*80}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"\nTotal puzzles in file: {len(data)}")
    
    issues_by_puzzle = {}
    stats = {
        'total_puzzles': len(data),
        'puzzles_with_train': 0,
        'puzzles_with_test': 0,
        'puzzles_with_no_test': 0,  # NEW
        'puzzles_with_no_train': 0,  # NEW
        'total_train_examples': 0,
        'total_test_examples': 0,
        'blank_grids': 0,
        'problematic_puzzles': []
    }
    
    for puzzle_id, puzzle_data in data.items():
        puzzle_issues = []
        
        # Check for train/test splits
        has_train = 'train' in puzzle_data and len(puzzle_data.get('train', [])) > 0
        has_test = 'test' in puzzle_data and len(puzzle_data.get('test', [])) > 0
        
        if has_train:
            stats['puzzles_with_train'] += 1
            stats['total_train_examples'] += len(puzzle_data['train'])
        else:
            stats['puzzles_with_no_train'] += 1
            puzzle_issues.append("No training examples")
        
        if has_test:
            stats['puzzles_with_test'] += 1
            stats['total_test_examples'] += len(puzzle_data['test'])
        else:
            stats['puzzles_with_no_test'] += 1
            puzzle_issues.append("No test examples (CANNOT BE SOLVED)")
        
        # Check each example in the puzzle
        for split in ['train', 'test']:
            if split not in puzzle_data:
                continue
            
            for idx, example in enumerate(puzzle_data[split]):
                if 'input' not in example:
                    puzzle_issues.append(f"{split}[{idx}]: Missing input")
                    continue
                
                if 'output' not in example:
                    puzzle_issues.append(f"{split}[{idx}]: Missing output")
                    continue
                
                # Check input grid
                input_grid = np.array(example['input'], dtype=np.int64)
                input_issues, input_stats = check_grid_properties(
                    input_grid, f"{split}[{idx}] input"
                )
                
                if input_issues:
                    puzzle_issues.extend(input_issues)
                
                if input_stats['all_zeros']:
                    stats['blank_grids'] += 1
                
                # Check output grid
                output_grid = np.array(example['output'], dtype=np.int64)
                output_issues, output_stats = check_grid_properties(
                    output_grid, f"{split}[{idx}] output"
                )
                
                if output_issues:
                    puzzle_issues.extend(output_issues)
                
                if output_stats['all_zeros']:
                    stats['blank_grids'] += 1
        
        if puzzle_issues:
            issues_by_puzzle[puzzle_id] = puzzle_issues
            stats['problematic_puzzles'].append(puzzle_id)
    
    # Print statistics
    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}")
    print(f"Total puzzles: {stats['total_puzzles']}")
    print(f"Puzzles with training data: {stats['puzzles_with_train']}")
    print(f"Puzzles with test data: {stats['puzzles_with_test']}")
    print(f"Puzzles WITHOUT training data: {stats['puzzles_with_no_train']}")
    print(f"Puzzles WITHOUT test data (unsolvable): {stats['puzzles_with_no_test']}")
    print(f"Total training examples: {stats['total_train_examples']}")
    print(f"Total test examples: {stats['total_test_examples']}")
    print(f"Blank grids found: {stats['blank_grids']}")
    print(f"Problematic puzzles: {len(stats['problematic_puzzles'])}")
    
    # Print issues
    if issues_by_puzzle:
        print(f"\n{'='*80}")
        print("ISSUES FOUND")
        print(f"{'='*80}")
        
        for puzzle_id, issues in list(issues_by_puzzle.items())[:10]:  # Show first 10
            print(f"\nPuzzle {puzzle_id}:")
            for issue in issues:
                print(f"  - {issue}")
        
        if len(issues_by_puzzle) > 10:
            print(f"\n... and {len(issues_by_puzzle) - 10} more puzzles with issues")
    else:
        print(f"\n[OK] No issues found!")
    
    return stats, issues_by_puzzle


def test_arc_dataset(json_path, filter_size=None, max_grids=None, 
                     num_distractors=0, use_input_output_pairs=False):
    """Test the ARCDataset class."""
    print(f"\n{'='*80}")
    print(f"TESTING ARCDataset")
    print(f"{'='*80}")
    print(f"Filter size: {filter_size}")
    print(f"Max grids: {max_grids}")
    print(f"Num distractors: {num_distractors}")
    print(f"Use input-output pairs: {use_input_output_pairs}")
    
    try:
        dataset = ARCDataset(
            json_path,
            min_size=config.MIN_GRID_SIZE,
            filter_size=filter_size,
            max_grids=max_grids,
            num_distractors=num_distractors,
            track_puzzle_ids=False,
            use_input_output_pairs=use_input_output_pairs
        )
        
        print(f"\n[OK] Dataset loaded successfully")
        print(f"Total items: {len(dataset)}")
        
        if len(dataset) == 0:
            print("[WARNING] Dataset is empty!")
            return
        
        # Test a few samples
        print(f"\nTesting first 5 samples...")
        
        issues_found = []
        size_distribution = defaultdict(int)
        
        for i in range(min(5, len(dataset))):
            print(f"\nSample {i}:")
            
            try:
                item = dataset[i]
                
                if use_input_output_pairs:
                    if num_distractors > 0:
                        input_grid, input_size, output_grid, output_size, candidates, candidate_sizes, target_idx = item
                        print(f"  Input size: {input_size}, Output size: {output_size}")
                        print(f"  Num candidates: {len(candidates)}, Target index: {target_idx}")
                        
                        # Visualize
                        input_np = input_grid[:input_size[0], :input_size[1]].numpy()
                        output_np = output_grid[:output_size[0], :output_size[1]].numpy()
                        visualize_grid(input_np, f"Sample {i} - Input")
                        visualize_grid(output_np, f"Sample {i} - Output (Target)")
                    else:
                        input_grid, input_size, output_grid, output_size = item
                        print(f"  Input size: {input_size}, Output size: {output_size}")
                        
                        # Visualize
                        input_np = input_grid[:input_size[0], :input_size[1]].numpy()
                        output_np = output_grid[:output_size[0], :output_size[1]].numpy()
                        visualize_grid(input_np, f"Sample {i} - Input")
                        visualize_grid(output_np, f"Sample {i} - Output")
                else:
                    if num_distractors > 0:
                        grid, size, candidates, candidate_sizes, target_idx = item
                        print(f"  Grid size: {size}")
                        print(f"  Num candidates: {len(candidates)}, Target index: {target_idx}")
                    else:
                        grid, size = item
                        print(f"  Grid size: {size}")
                        
                        # Visualize
                        grid_np = grid[:size[0], :size[1]].numpy()
                        visualize_grid(grid_np, f"Sample {i}")
                        
                        # Check for issues
                        grid_issues, _ = check_grid_properties(grid_np)
                        if grid_issues:
                            issues_found.extend([(i, issue) for issue in grid_issues])
                
                size_distribution[size if not use_input_output_pairs else input_size] += 1
                
            except Exception as e:
                print(f"  [ERROR] Error loading sample {i}: {e}")
                import traceback
                traceback.print_exc()
        
        # Print size distribution
        print(f"\n{'='*80}")
        print("SIZE DISTRIBUTION (first 100 samples)")
        print(f"{'='*80}")
        
        size_counts = defaultdict(int)
        for i in range(min(100, len(dataset))):
            try:
                item = dataset[i]
                if use_input_output_pairs:
                    if num_distractors > 0:
                        _, input_size, _, output_size, _, _, _ = item
                    else:
                        _, input_size, _, output_size = item
                    size_counts[input_size] += 1
                else:
                    if num_distractors > 0:
                        _, size, _, _, _ = item
                    else:
                        _, size = item
                    size_counts[size] += 1
            except Exception as e:
                print(f"Error on sample {i}: {e}")
        
        for size, count in sorted(size_counts.items()):
            print(f"  {size[0]:2d}x{size[1]:2d}: {count:3d} grids")
        
        # Print issues
        if issues_found:
            print(f"\n{'='*80}")
            print(f"ISSUES FOUND IN DATASET")
            print(f"{'='*80}")
            for sample_idx, issue in issues_found[:20]:  # Show first 20
                print(f"  Sample {sample_idx}: {issue}")
            
            if len(issues_found) > 20:
                print(f"\n  ... and {len(issues_found) - 20} more issues")
        else:
            print(f"\n[OK] No issues found in sampled data!")
        
    except Exception as e:
        print(f"[ERROR] Error creating dataset: {e}")
        import traceback
        traceback.print_exc()


def test_single_puzzle_dataset(json_path, num_puzzles_to_test=5):
    """Test the ARCSinglePuzzleDataset class."""
    print(f"\n{'='*80}")
    print(f"TESTING ARCSinglePuzzleDataset")
    print(f"{'='*80}")
    
    # Get all puzzle IDs
    puzzle_ids = load_all_puzzle_ids(json_path)
    print(f"Total puzzles available: {len(puzzle_ids)}")
    
    # Test a few puzzles
    puzzles_to_test = puzzle_ids[:num_puzzles_to_test]
    
    for puzzle_id in puzzles_to_test:
        print(f"\n{'='*80}")
        print(f"Testing puzzle: {puzzle_id}")
        print(f"{'='*80}")
        
        # Test train split
        try:
            train_dataset = ARCSinglePuzzleDataset(json_path, puzzle_id, split='train')
            print(f"[OK] Train split loaded: {len(train_dataset)} examples")
            
            # Show first example
            if len(train_dataset) > 0:
                input_grid, input_size, output_grid, output_size = train_dataset[0]
                print(f"  First example - Input: {input_size}, Output: {output_size}")
                
                # Visualize
                input_np = input_grid[:input_size[0], :input_size[1]].numpy()
                output_np = output_grid[:output_size[0], :output_size[1]].numpy()
                visualize_grid(input_np, "Train Input")
                visualize_grid(output_np, "Train Output")
                
                # Check for issues
                input_issues, _ = check_grid_properties(input_np, "Train input")
                output_issues, _ = check_grid_properties(output_np, "Train output")
                
                if input_issues or output_issues:
                    print("  Issues found:")
                    for issue in input_issues + output_issues:
                        print(f"    - {issue}")
        
        except Exception as e:
            print(f"[ERROR] Error loading train split: {e}")
        
        # Test test split
        try:
            test_dataset = ARCSinglePuzzleDataset(json_path, puzzle_id, split='test')
            print(f"[OK] Test split loaded: {len(test_dataset)} examples")
            
            if len(test_dataset) == 0:
                print(f"  [WARNING] This puzzle has NO TEST EXAMPLES - cannot be solved!")
            
            # Show first example
            if len(test_dataset) > 0:
                input_grid, input_size, output_grid, output_size = test_dataset[0]
                print(f"  First example - Input: {input_size}, Output: {output_size}")
        
        except Exception as e:
            print(f"[ERROR] Error loading test split: {e}")


def get_solvable_puzzles(json_path):
    """Get list of puzzles that have test data and can be solved."""
    print(f"\n{'='*80}")
    print(f"IDENTIFYING SOLVABLE PUZZLES")
    print(f"{'='*80}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    solvable = []
    unsolvable = []
    
    for puzzle_id, puzzle_data in data.items():
        has_test = 'test' in puzzle_data and len(puzzle_data.get('test', [])) > 0
        
        if has_test:
            solvable.append(puzzle_id)
        else:
            unsolvable.append(puzzle_id)
    
    print(f"\nTotal puzzles: {len(data)}")
    print(f"Solvable (have test data): {len(solvable)}")
    print(f"Unsolvable (no test data): {len(unsolvable)}")
    
    if unsolvable:
        print(f"\nUnsolvable puzzle IDs:")
        for pid in unsolvable[:20]:  # Show first 20
            print(f"  - {pid}")
        if len(unsolvable) > 20:
            print(f"  ... and {len(unsolvable) - 20} more")
    
    return solvable, unsolvable


def main():
    parser = argparse.ArgumentParser(description='Test ARC dataset loading')
    parser.add_argument('--data_path', type=str, default='arc-agi_test_challenges.json',
                       help='Path to ARC JSON file')
    parser.add_argument('--filter_size', type=int, nargs=2, default=None,
                       help='Filter to specific grid size (height width)')
    parser.add_argument('--max_grids', type=int, default=None,
                       help='Maximum number of grids to load')
    parser.add_argument('--num_distractors', type=int, default=0,
                       help='Number of distractors for selection task')
    parser.add_argument('--use_input_output_pairs', action='store_true',
                       help='Test with input-output pairs')
    parser.add_argument('--skip_structure_test', action='store_true',
                       help='Skip JSON structure test (faster)')
    parser.add_argument('--skip_dataset_test', action='store_true',
                       help='Skip ARCDataset test')
    parser.add_argument('--skip_puzzle_test', action='store_true',
                       help='Skip single puzzle test')
    parser.add_argument('--list_solvable', action='store_true',
                       help='List all solvable puzzles (have test data)')
    
    args = parser.parse_args()
    
    # Convert filter_size to tuple if provided
    filter_size = tuple(args.filter_size) if args.filter_size else None
    
    print(f"\n{'#'*80}")
    print(f"ARC DATASET VALIDATION TEST")
    print(f"{'#'*80}")
    print(f"Data path: {args.data_path}")
    
    # Test 1: JSON file structure
    if not args.skip_structure_test:
        stats, issues = test_json_file_structure(args.data_path)
    
    # List solvable puzzles
    if args.list_solvable:
        solvable, unsolvable = get_solvable_puzzles(args.data_path)
    
    # Test 2: ARCDataset class
    if not args.skip_dataset_test:
        test_arc_dataset(
            args.data_path,
            filter_size=filter_size,
            max_grids=args.max_grids,
            num_distractors=args.num_distractors,
            use_input_output_pairs=args.use_input_output_pairs
        )
    
    # Test 3: ARCSinglePuzzleDataset class
    if not args.skip_puzzle_test:
        test_single_puzzle_dataset(args.data_path, num_puzzles_to_test=5)
    
    print(f"\n{'#'*80}")
    print(f"TEST COMPLETE")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()