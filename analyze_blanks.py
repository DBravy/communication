"""Analyze the percentage of blank squares in ARC puzzles."""

import json
import numpy as np

def analyze_arc_blanks(json_path):
    """
    Calculate statistics about blank (0-valued) squares in ARC grids.
    
    Args:
        json_path: Path to ARC JSON file
    
    Returns:
        Dictionary with statistics
    """
    blank_percentages = []
    grid_sizes = []
    
    # Load ARC data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Process all grids
    for task_id, task_data in data.items():
        # Process training examples
        for example in task_data.get('train', []):
            if 'input' in example:
                grid = np.array(example['input'], dtype=np.int64)
                analyze_grid(grid, blank_percentages, grid_sizes)
            if 'output' in example:
                grid = np.array(example['output'], dtype=np.int64)
                analyze_grid(grid, blank_percentages, grid_sizes)
        
        # Process test examples
        for example in task_data.get('test', []):
            if 'input' in example:
                grid = np.array(example['input'], dtype=np.int64)
                analyze_grid(grid, blank_percentages, grid_sizes)
            if 'output' in example:
                grid = np.array(example['output'], dtype=np.int64)
                analyze_grid(grid, blank_percentages, grid_sizes)
    
    # Calculate statistics
    blank_percentages = np.array(blank_percentages)
    grid_sizes = np.array(grid_sizes)
    
    stats = {
        'num_grids': len(blank_percentages),
        'avg_blank_percentage': np.mean(blank_percentages),
        'median_blank_percentage': np.median(blank_percentages),
        'min_blank_percentage': np.min(blank_percentages),
        'max_blank_percentage': np.max(blank_percentages),
        'std_blank_percentage': np.std(blank_percentages),
        'avg_grid_size': np.mean(grid_sizes),
        'num_fully_filled': np.sum(blank_percentages == 0),
        'num_mostly_blank': np.sum(blank_percentages > 50),
    }
    
    return stats, blank_percentages, grid_sizes

def analyze_grid(grid, blank_percentages, grid_sizes):
    """Helper function to analyze a single grid."""
    h, w = grid.shape
    total_cells = h * w
    blank_cells = np.sum(grid == 0)
    blank_pct = (blank_cells / total_cells) * 100
    
    blank_percentages.append(blank_pct)
    grid_sizes.append(total_cells)

def print_statistics(stats):
    """Print statistics in a readable format."""
    print("=" * 60)
    print("ARC BLANK SQUARE ANALYSIS")
    print("=" * 60)
    print(f"\nTotal grids analyzed: {stats['num_grids']}")
    print(f"Average grid size: {stats['avg_grid_size']:.1f} cells")
    print(f"\nBLANK SQUARE STATISTICS:")
    print(f"  Average blank percentage: {stats['avg_blank_percentage']:.2f}%")
    print(f"  Median blank percentage:  {stats['median_blank_percentage']:.2f}%")
    print(f"  Std deviation:            {stats['std_blank_percentage']:.2f}%")
    print(f"  Min blank percentage:     {stats['min_blank_percentage']:.2f}%")
    print(f"  Max blank percentage:     {stats['max_blank_percentage']:.2f}%")
    print(f"\nDISTRIBUTION:")
    print(f"  Fully filled grids (0% blank):   {stats['num_fully_filled']}")
    print(f"  Mostly blank grids (>50% blank): {stats['num_mostly_blank']}")
    print("=" * 60)

def plot_distribution(blank_percentages):
    """Optional: plot histogram of blank percentages."""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.hist(blank_percentages, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Percentage of Blank Squares (%)')
        plt.ylabel('Number of Grids')
        plt.title('Distribution of Blank Squares in ARC Puzzles')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('arc_blank_distribution.png', dpi=300, bbox_inches='tight')
        print("\nHistogram saved as 'arc_blank_distribution.png'")
        plt.show()
    except ImportError:
        print("\nMatplotlib not available - skipping histogram plot")

if __name__ == '__main__':
    # Path to your ARC JSON file
    json_path = 'arc-agi_test_challenges.json'
    
    try:
        # Analyze the data
        stats, blank_percentages, grid_sizes = analyze_arc_blanks(json_path)
        
        # Print results
        print_statistics(stats)
        
        # Optional: create histogram
        # Uncomment the line below if you want to see a plot
        # plot_distribution(blank_percentages)
        
    except FileNotFoundError:
        print(f"Error: Could not find '{json_path}'")
        print("Please update the json_path variable with the correct path to your ARC data file.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()