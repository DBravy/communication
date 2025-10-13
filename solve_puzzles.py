"""Unified script to solve single or multiple ARC puzzles."""

import torch
import numpy as np
import json
import argparse
import os
import subprocess
from tqdm import tqdm
from colorama import init, Fore, Style

import config
from puzzle_dataset import ARCSinglePuzzleDataset, load_all_puzzle_ids
from model import ARCEncoder, ARCAutoencoder

# Initialize colorama for colored terminal output
init(autoreset=True)


def visualize_grid(grid, title="Grid"):
    """Print a colored grid."""
    colors = {
        0: Fore.BLACK + '██',
        1: Fore.BLUE + '██',
        2: Fore.RED + '██',
        3: Fore.GREEN + '██',
        4: Fore.YELLOW + '██',
        5: Fore.WHITE + '██',
        6: Fore.MAGENTA + '██',
        7: Fore.CYAN + '██',
        8: Fore.LIGHTBLACK_EX + '██',
        9: Fore.LIGHTRED_EX + '██',
    }
    
    print(f"\n{title} ({grid.shape[0]}x{grid.shape[1]}):")
    print("┌" + "─" * (grid.shape[1] * 2) + "┐")
    
    for row in grid:
        print("│", end="")
        for cell in row:
            print(colors.get(int(cell), '  '), end="")
        print(Style.RESET_ALL + "│")
    
    print("└" + "─" * (grid.shape[1] * 2) + "┘")


def predict_output(model, input_grid, input_size, device, output_size=None):
    """
    Generate output prediction for a given input.
    
    Args:
        model: Finetuned model
        input_grid: Input grid tensor [30, 30]
        input_size: Tuple (H, W) of actual input size
        device: torch device
        output_size: Optional tuple (H, W) for expected output size
    
    Returns:
        predicted_grid: numpy array of predicted output
        messages: messages tensor (or None)
    """
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension
        input_batch = input_grid.unsqueeze(0).to(device)
        
        # If output_size not provided, use input_size
        if output_size is None:
            output_size = input_size
        
        # Forward pass
        logits_list, _, messages = model(input_batch, [input_size], temperature=1.0)
        
        # Get prediction
        logits = logits_list[0]
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        
        # Extract actual prediction (remove padding)
        output_h, output_w = output_size
        predicted_grid = pred[:output_h, :output_w]
        
        return predicted_grid, messages


def evaluate_prediction(predicted, target):
    """
    Evaluate prediction against target.
    
    Returns:
        exact_match: bool, whether grids are exactly the same
        pixel_accuracy: float, percentage of correct pixels
    """
    # Check if sizes match
    if predicted.shape != target.shape:
        return False, 0.0
    
    # Exact match
    exact_match = np.array_equal(predicted, target)
    
    # Pixel accuracy
    correct_pixels = (predicted == target).sum()
    total_pixels = predicted.size
    pixel_accuracy = 100.0 * correct_pixels / total_pixels
    
    return exact_match, pixel_accuracy


def solve_single_puzzle(model, puzzle_dataset, device, puzzle_id, visualize=True):
    """
    Solve all test examples in a puzzle.
    
    Returns:
        results: Dict with prediction results
    """
    results = {
        'puzzle_id': puzzle_id,
        'num_test_examples': len(puzzle_dataset),
        'predictions': []
    }
    
    for i in range(len(puzzle_dataset)):
        input_grid, input_size, output_grid, output_size = puzzle_dataset[i]
        
        # Get actual grids (without padding)
        input_h, input_w = input_size
        output_h, output_w = output_size
        
        input_actual = input_grid[:input_h, :input_w].numpy()
        output_actual = output_grid[:output_h, :output_w].numpy()
        
        # Generate prediction
        predicted, messages = predict_output(model, input_grid, input_size, device, output_size)
        
        # Evaluate
        exact_match, pixel_acc = evaluate_prediction(predicted, output_actual)
        
        # Store results
        prediction_result = {
            'example_id': i,
            'input_size': input_size,
            'output_size': output_size,
            'exact_match': exact_match,
            'pixel_accuracy': pixel_acc,
            'input': input_actual.tolist(),
            'target': output_actual.tolist(),
            'predicted': predicted.tolist(),
            'messages': messages[0].cpu().tolist() if messages is not None else None
        }
        results['predictions'].append(prediction_result)
        
        # Visualize
        if visualize:
            print(f"\n{'='*80}")
            print(f"TEST EXAMPLE {i+1}")
            print(f"{'='*80}")
            
            visualize_grid(input_actual, f"INPUT")
            visualize_grid(output_actual, f"TARGET OUTPUT")
            visualize_grid(predicted, f"PREDICTED OUTPUT")
            
            if messages is not None:
                print(f"\n📨 MESSAGE: {prediction_result['messages']}")
            
            # Show metrics
            status = "✓ CORRECT" if exact_match else "✗ INCORRECT"
            color = Fore.GREEN if exact_match else Fore.RED
            print(f"\n{color}📊 METRICS:")
            print(f"   Exact Match: {status}")
            print(f"   Pixel Accuracy: {pixel_acc:.2f}%")
    
    # Calculate summary stats
    results['num_correct'] = sum(1 for p in results['predictions'] if p['exact_match'])
    results['avg_pixel_accuracy'] = np.mean([p['pixel_accuracy'] for p in results['predictions']])
    
    return results


def load_model(checkpoint_path, device):
    """Load a finetuned model from checkpoint."""
    encoder = ARCEncoder(
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_conv_layers=getattr(config, 'NUM_CONV_LAYERS', 3)
    )
    
    model = ARCAutoencoder(
        encoder=encoder,
        vocab_size=config.VOCAB_SIZE if config.BOTTLENECK_TYPE == 'communication' else None,
        max_length=config.MAX_MESSAGE_LENGTH if config.BOTTLENECK_TYPE == 'communication' else None,
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        max_grid_size=config.MAX_GRID_SIZE,
        bottleneck_type=config.BOTTLENECK_TYPE,
        task_type='reconstruction',
        num_conv_layers=getattr(config, 'NUM_CONV_LAYERS', 3)
    ).to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint


def finetune_puzzle(puzzle_id, data_path, pretrained_checkpoint, save_dir, 
                    epochs, lr, batch_size, verbose=True):
    """Run finetuning on a single puzzle."""
    checkpoint_arg = f"--checkpoint {pretrained_checkpoint}" if pretrained_checkpoint else ""
    
    finetune_cmd = (
        f"python finetune_puzzle.py "
        f"--puzzle_id {puzzle_id} "
        f"--data_path {data_path} "
        f"{checkpoint_arg} "
        f"--epochs {epochs} "
        f"--lr {lr} "
        f"--batch_size {batch_size} "
        f"--save_dir {save_dir}"
    )
    
    if verbose:
        print(f"\nFinetuning {puzzle_id}...")
    
    result = subprocess.run(finetune_cmd, shell=True, capture_output=not verbose, text=True)
    
    if result.returncode != 0:
        if not verbose:
            print(f"Error finetuning {puzzle_id}: {result.stderr}")
        return False
    
    return True


def solve_puzzle_with_finetuning(puzzle_id, data_path, pretrained_checkpoint, 
                                 finetune_save_dir, results_dir, epochs, lr, 
                                 batch_size, visualize, device):
    """Finetune and solve a single puzzle."""
    
    # Step 1: Finetune
    if not finetune_puzzle(puzzle_id, data_path, pretrained_checkpoint, 
                          finetune_save_dir, epochs, lr, batch_size, 
                          verbose=visualize):
        return None
    
    # Step 2: Load finetuned model
    finetuned_checkpoint = os.path.join(finetune_save_dir, f"{puzzle_id}_best.pth")
    
    try:
        model, checkpoint = load_model(finetuned_checkpoint, device)
        
        if visualize:
            print(f"\n✓ Loaded checkpoint")
            print(f"  Puzzle ID: {checkpoint.get('puzzle_id', 'unknown')}")
            print(f"  Training loss: {checkpoint.get('train_loss', 'N/A')}")
            print(f"  Training accuracy: {checkpoint.get('train_acc', 'N/A')}")
    except Exception as e:
        print(f"Error loading model for {puzzle_id}: {e}")
        return None
    
    # Step 3: Load test dataset
    try:
        test_dataset = ARCSinglePuzzleDataset(data_path, puzzle_id, split='test')
    except Exception as e:
        print(f"Error loading test data for {puzzle_id}: {e}")
        return None
    
    # Step 4: Solve
    if visualize:
        print(f'\n{"="*80}')
        print(f"SOLVING PUZZLE: {puzzle_id}")
        print(f'{"="*80}')
    
    results = solve_single_puzzle(model, test_dataset, device, puzzle_id, visualize=visualize)
    
    # Step 5: Save predictions
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        predictions_path = os.path.join(results_dir, f"{puzzle_id}_predictions.json")
        results['checkpoint'] = finetuned_checkpoint
        
        with open(predictions_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def batch_solve_puzzles(puzzle_ids, data_path, pretrained_checkpoint, 
                        finetune_save_dir, results_dir, epochs, lr, batch_size):
    """Finetune and solve multiple puzzles."""
    os.makedirs(finetune_save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = []
    
    for puzzle_id in tqdm(puzzle_ids, desc="Processing puzzles"):
        print(f"\n{'='*80}")
        print(f"PUZZLE: {puzzle_id}")
        print(f"{'='*80}")
        
        result = solve_puzzle_with_finetuning(
            puzzle_id=puzzle_id,
            data_path=data_path,
            pretrained_checkpoint=pretrained_checkpoint,
            finetune_save_dir=finetune_save_dir,
            results_dir=results_dir,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            visualize=False,  # Don't visualize in batch mode
            device=device
        )
        
        if result:
            all_results.append(result)
    
    # Save summary
    summary_path = os.path.join(results_dir, "summary.json")
    summary = {
        'total_puzzles': len(puzzle_ids),
        'processed_puzzles': len(all_results),
        'results': all_results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("BATCH SOLVING SUMMARY")
    print(f"{'='*80}")
    
    total_correct = sum(r['num_correct'] for r in all_results)
    total_examples = sum(r['num_test_examples'] for r in all_results)
    avg_accuracy = sum(r['avg_pixel_accuracy'] for r in all_results) / len(all_results) if all_results else 0
    
    print(f"Puzzles processed: {len(all_results)}/{len(puzzle_ids)}")
    print(f"Total test examples: {total_examples}")
    print(f"Exact matches: {total_correct}/{total_examples}")
    print(f"Average pixel accuracy: {avg_accuracy:.2f}%")
    print(f"\nResults saved to: {results_dir}")
    print(f"Summary saved to: {summary_path}")
    
    # Per-puzzle breakdown
    print(f"\n{'='*80}")
    print("PER-PUZZLE RESULTS")
    print(f"{'='*80}")
    
    for result in all_results:
        status = "✓" if result['num_correct'] == result['num_test_examples'] else "✗"
        print(f"{status} {result['puzzle_id']}: {result['num_correct']}/{result['num_test_examples']} correct "
              f"({result['avg_pixel_accuracy']:.1f}% pixel acc)")


def main():
    parser = argparse.ArgumentParser(
        description='Solve ARC puzzles (single or batch)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve a single puzzle (must already be finetuned)
  python solve_puzzles.py --puzzle_id 007bbfb7 --checkpoint puzzle_checkpoints/007bbfb7_best.pth
  
  # Finetune and solve a single puzzle
  python solve_puzzles.py --puzzle_id 007bbfb7 --finetune --epochs 1000
  
  # Batch process multiple puzzles (with finetuning)
  python solve_puzzles.py --batch --num_puzzles 10 --epochs 500
  
  # Batch process specific puzzles
  python solve_puzzles.py --batch --puzzle_ids 007bbfb7 00d62c1b --epochs 500
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--puzzle_id', type=str, help='Single puzzle ID to solve')
    mode_group.add_argument('--batch', action='store_true', help='Batch process multiple puzzles')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='arc-agi_training_challenges.json',
                       help='Path to ARC data')
    
    # Single puzzle mode arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to finetuned checkpoint (single mode, if not finetuning)')
    parser.add_argument('--finetune', action='store_true',
                       help='Finetune before solving (single mode)')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Disable visualization (single mode)')
    
    # Batch mode arguments
    parser.add_argument('--puzzle_ids', type=str, nargs='+', default=None,
                       help='Specific puzzle IDs for batch mode (default: all)')
    parser.add_argument('--num_puzzles', type=int, default=None,
                       help='Number of puzzles for batch mode (default: all)')
    
    # Finetuning arguments (used in both modes if finetuning)
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                       help='Path to pretrained checkpoint to start finetuning from')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of finetuning epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate for finetuning')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for finetuning')
    
    # Output arguments
    parser.add_argument('--finetune_save_dir', type=str, default='puzzle_checkpoints',
                       help='Directory to save finetuned models')
    parser.add_argument('--results_dir', type=str, default='puzzle_results',
                       help='Directory to save results')
    parser.add_argument('--save_predictions', type=str, default=None,
                       help='Path to save predictions JSON (single mode)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # ========== SINGLE PUZZLE MODE ==========
    if args.puzzle_id:
        # If finetuning, do that first
        if args.finetune:
            result = solve_puzzle_with_finetuning(
                puzzle_id=args.puzzle_id,
                data_path=args.data_path,
                pretrained_checkpoint=args.pretrained_checkpoint,
                finetune_save_dir=args.finetune_save_dir,
                results_dir=args.results_dir if args.save_predictions is None else None,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                visualize=not args.no_visualize,
                device=device
            )
            
            if result is None:
                print("Error: Failed to finetune and solve puzzle")
                return
            
            # Save predictions if requested
            if args.save_predictions:
                with open(args.save_predictions, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n✓ Predictions saved to {args.save_predictions}")
        
        # Otherwise, solve using existing checkpoint
        else:
            if args.checkpoint is None:
                print("Error: Must provide --checkpoint or use --finetune")
                return
            
            # Load test dataset
            print(f'\nLoading puzzle {args.puzzle_id} (test split)...')
            try:
                test_dataset = ARCSinglePuzzleDataset(args.data_path, args.puzzle_id, split='test')
            except ValueError as e:
                print(f"Error: {e}")
                return
            
            # Load model
            print('\nLoading model...')
            try:
                model, checkpoint = load_model(args.checkpoint, device)
                
                print(f"\n✓ Loaded checkpoint")
                print(f"  Puzzle ID: {checkpoint.get('puzzle_id', 'unknown')}")
                print(f"  Training loss: {checkpoint.get('train_loss', 'N/A')}")
                print(f"  Training accuracy: {checkpoint.get('train_acc', 'N/A')}")
            except Exception as e:
                print(f"Error: {e}")
                return
            
            # Solve
            print(f'\n{"="*80}')
            print(f"SOLVING PUZZLE: {args.puzzle_id}")
            print(f'{"="*80}')
            
            results = solve_single_puzzle(model, test_dataset, device, args.puzzle_id, 
                                         visualize=not args.no_visualize)
            
            # Print summary
            print(f"\n{'='*80}")
            print("SUMMARY")
            print(f"{'='*80}")
            
            print(f"Total test examples: {results['num_test_examples']}")
            print(f"Exact matches: {results['num_correct']}/{results['num_test_examples']}")
            print(f"Average pixel accuracy: {results['avg_pixel_accuracy']:.2f}%")
            
            if results['num_correct'] == results['num_test_examples']:
                print(f"\n{Fore.GREEN}🎉 PERFECT SCORE! All test examples solved correctly!")
            elif results['num_correct'] > 0:
                print(f"\n{Fore.YELLOW}⚠️  Partial success: {results['num_correct']}/{results['num_test_examples']} correct")
            else:
                print(f"\n{Fore.RED}❌ No test examples solved correctly")
            
            # Save predictions if requested
            if args.save_predictions:
                results['checkpoint'] = args.checkpoint
                with open(args.save_predictions, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n✓ Predictions saved to {args.save_predictions}")
    
    # ========== BATCH MODE ==========
    else:
        # Get puzzle IDs
        if args.puzzle_ids:
            puzzle_ids = args.puzzle_ids
        else:
            print(f"Loading all puzzle IDs from {args.data_path}...")
            puzzle_ids = load_all_puzzle_ids(args.data_path)
            
            if args.num_puzzles:
                puzzle_ids = puzzle_ids[:args.num_puzzles]
        
        print(f"\nWill process {len(puzzle_ids)} puzzles")
        
        # Run batch solving
        batch_solve_puzzles(
            puzzle_ids=puzzle_ids,
            data_path=args.data_path,
            pretrained_checkpoint=args.pretrained_checkpoint,
            finetune_save_dir=args.finetune_save_dir,
            results_dir=args.results_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size
        )


if __name__ == '__main__':
    main()