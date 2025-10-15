"""
Diagnostic script to identify accuracy calculation issues in ARC training.

This script will:
1. Load a checkpoint
2. Test on a few puzzles with detailed logging
3. Compare training-style vs. solving-style accuracy calculations
4. Identify size mismatches and calculation inconsistencies
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add parent directory to path to import modules
sys.path.append('.')

import config
from model import ARCEncoder, ARCAutoencoder
from puzzle_dataset import ARCSinglePuzzleDataset, load_all_puzzle_ids


def diagnose_single_puzzle(model, puzzle_id, data_path, device):
    """Diagnose a single puzzle with detailed logging."""
    print(f"\n{'='*80}")
    print(f"DIAGNOSING PUZZLE: {puzzle_id}")
    print(f"{'='*80}")
    
    # Load puzzle
    try:
        dataset = ARCSinglePuzzleDataset(data_path, puzzle_id, split='test')
    except Exception as e:
        print(f"❌ Could not load puzzle: {e}")
        return None
    
    if len(dataset) == 0:
        print("❌ No test examples found")
        return None
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            input_grid, input_size, output_grid, output_size = dataset[i]
            
            print(f"\n--- Example {i+1}/{len(dataset)} ---")
            print(f"Input size: {input_size}")
            print(f"Output size: {output_size}")
            
            input_h, input_w = input_size
            output_h, output_w = output_size
            
            input_actual = input_grid[:input_h, :input_w].cpu().numpy()
            output_actual = output_grid[:output_h, :output_w].cpu().numpy()
            
            input_batch = input_grid.unsqueeze(0).to(device)
            
            # TEST 1: Model output WITH output_sizes (correct way)
            print("\nTEST 1: Passing output_sizes to model (CORRECT)")
            logits_list, _, messages, message_lengths = model(
                input_batch, [input_size], temperature=1.0,
                output_sizes=[output_size]
            )
            logits = logits_list[0]
            pred_correct = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            
            print(f"  Model output shape: {pred_correct.shape}")
            print(f"  Expected output shape: {output_actual.shape}")
            
            # Calculate accuracy (correct way)
            if pred_correct.shape == output_actual.shape:
                correct_pixels = (pred_correct == output_actual).sum()
                total_pixels = output_actual.size
                acc_correct = 100.0 * correct_pixels / total_pixels
                print(f"  ✓ Shapes match!")
            else:
                min_h = min(pred_correct.shape[0], output_actual.shape[0])
                min_w = min(pred_correct.shape[1], output_actual.shape[1])
                correct_pixels = (pred_correct[:min_h, :min_w] == output_actual[:min_h, :min_w]).sum()
                total_pixels = output_actual.size
                acc_correct = 100.0 * correct_pixels / total_pixels
                print(f"  ⚠ Shapes mismatch! Predicted {pred_correct.shape} vs Expected {output_actual.shape}")
            
            print(f"  Accuracy: {acc_correct:.2f}%")
            print(f"  Correct pixels: {correct_pixels}/{total_pixels}")
            
            # TEST 2: Model output WITHOUT output_sizes (wrong way - simulates training viz bug)
            print("\nTEST 2: NOT passing output_sizes to model (BUG - simulates training viz)")
            logits_list_wrong, _, _, _ = model(
                input_batch, [input_size], temperature=1.0
                # Note: NOT passing output_sizes!
            )
            logits_wrong = logits_list_wrong[0]
            pred_wrong = logits_wrong.argmax(dim=1).squeeze(0).cpu().numpy()
            
            print(f"  Model output shape: {pred_wrong.shape}")
            print(f"  Expected output shape: {output_actual.shape}")
            
            # Calculate accuracy comparing to INPUT (training viz bug)
            min_h = min(input_h, pred_wrong.shape[0])
            min_w = min(input_w, pred_wrong.shape[1])
            correct_pixels_vs_input = (input_actual[:min_h, :min_w] == pred_wrong[:min_h, :min_w]).sum()
            total_pixels_wrong = min_h * min_w
            acc_vs_input = 100.0 * correct_pixels_vs_input / total_pixels_wrong
            
            print(f"  ⚠ Comparing against INPUT (WRONG!)")
            print(f"  Accuracy vs INPUT: {acc_vs_input:.2f}%")
            
            # Calculate accuracy comparing to OUTPUT (what we should do)
            min_h_out = min(pred_wrong.shape[0], output_actual.shape[0])
            min_w_out = min(pred_wrong.shape[1], output_actual.shape[1])
            correct_pixels_vs_output = (pred_wrong[:min_h_out, :min_w_out] == output_actual[:min_h_out, :min_w_out]).sum()
            total_pixels_out = output_actual.size
            acc_vs_output = 100.0 * correct_pixels_vs_output / total_pixels_out
            
            print(f"  Accuracy vs OUTPUT: {acc_vs_output:.2f}%")
            
            # Show message if available
            if messages is not None:
                msg = messages[0].cpu().tolist()
                if message_lengths is not None:
                    actual_length = message_lengths[0].item()
                    msg = msg[:actual_length]
                print(f"  Message: {msg[:10]}... (length: {len(msg)})")
            
            results.append({
                'example': i,
                'input_size': input_size,
                'output_size': output_size,
                'acc_correct': acc_correct,
                'acc_vs_input': acc_vs_input,
                'acc_vs_output': acc_vs_output,
                'shape_mismatch': pred_correct.shape != output_actual.shape
            })
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY FOR {puzzle_id}")
    print(f"{'='*80}")
    
    avg_correct = np.mean([r['acc_correct'] for r in results])
    avg_vs_input = np.mean([r['acc_vs_input'] for r in results])
    avg_vs_output = np.mean([r['acc_vs_output'] for r in results])
    
    print(f"Average accuracy (CORRECT method): {avg_correct:.2f}%")
    print(f"Average accuracy vs INPUT (BUG): {avg_vs_input:.2f}%")
    print(f"Average accuracy vs OUTPUT (without output_sizes): {avg_vs_output:.2f}%")
    
    if abs(avg_correct - avg_vs_input) > 10:
        print(f"\n⚠️  LARGE DISCREPANCY DETECTED!")
        print(f"   Difference: {abs(avg_correct - avg_vs_input):.2f}%")
        print(f"   This suggests the training visualization is comparing against INPUT instead of OUTPUT!")
    
    return results


def main():
    print("ARC Accuracy Diagnostic Script")
    print("="*80)
    
    # Get checkpoint path from command line or use default
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = os.path.join(config.SAVE_DIR, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"Usage: python diagnose_accuracy.py [checkpoint_path]")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get architecture params
    vocab_size = checkpoint.get('vocab_size', config.VOCAB_SIZE if config.BOTTLENECK_TYPE == 'communication' else None)
    max_length = checkpoint.get('max_message_length', config.MAX_MESSAGE_LENGTH if config.BOTTLENECK_TYPE == 'communication' else None)
    receiver_gets_input = checkpoint.get('receiver_gets_input_puzzle', getattr(config, 'RECEIVER_GETS_INPUT_PUZZLE', False))
    use_stop_token = checkpoint.get('use_stop_token', getattr(config, 'USE_STOP_TOKEN', False))
    stop_token_id = vocab_size if use_stop_token else None
    
    print(f"Bottleneck type: {checkpoint.get('bottleneck_type', 'unknown')}")
    print(f"Task type: {checkpoint.get('task_type', 'unknown')}")
    if vocab_size:
        print(f"Vocab size: {vocab_size}")
        print(f"Max message length: {max_length}")
        print(f"Use stop token: {use_stop_token}")
    
    # Create model
    encoder = ARCEncoder(
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_conv_layers=getattr(config, 'NUM_CONV_LAYERS', 3)
    )
    
    model = ARCAutoencoder(
        encoder=encoder,
        vocab_size=vocab_size,
        max_length=max_length,
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        max_grid_size=config.MAX_GRID_SIZE,
        bottleneck_type=checkpoint.get('bottleneck_type', config.BOTTLENECK_TYPE),
        task_type='reconstruction',
        num_conv_layers=getattr(config, 'NUM_CONV_LAYERS', 3),
        receiver_gets_input_puzzle=receiver_gets_input,
        use_stop_token=use_stop_token,
        stop_token_id=stop_token_id,
        lstm_hidden_dim=getattr(config, 'LSTM_HIDDEN_DIM', None)
    ).to(device)
    
    # Load weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get data path
    dataset_version = checkpoint.get('dataset_version', 'V2')
    dataset_split = checkpoint.get('dataset_split', 'evaluation')
    
    if dataset_version in ['V1', 'V2']:
        data_path = os.path.join(dataset_version, 'data', dataset_split)
    else:
        data_path = config.DATA_PATH
    
    print(f"Data path: {data_path}")
    
    # Get puzzle IDs
    try:
        puzzle_ids = load_all_puzzle_ids(data_path)
        print(f"Found {len(puzzle_ids)} puzzles")
    except Exception as e:
        print(f"❌ Error loading puzzles: {e}")
        return
    
    # Test on first 3 puzzles
    num_puzzles_to_test = min(3, len(puzzle_ids))
    print(f"\nTesting on first {num_puzzles_to_test} puzzles...")
    
    all_results = {}
    for puzzle_id in puzzle_ids[:num_puzzles_to_test]:
        results = diagnose_single_puzzle(model, puzzle_id, data_path, device)
        if results:
            all_results[puzzle_id] = results
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    if all_results:
        all_correct = []
        all_vs_input = []
        all_vs_output = []
        
        for puzzle_id, results in all_results.items():
            for r in results:
                all_correct.append(r['acc_correct'])
                all_vs_input.append(r['acc_vs_input'])
                all_vs_output.append(r['acc_vs_output'])
        
        print(f"Overall average accuracy (CORRECT method): {np.mean(all_correct):.2f}%")
        print(f"Overall average accuracy vs INPUT (BUG): {np.mean(all_vs_input):.2f}%")
        print(f"Overall average accuracy vs OUTPUT (bug): {np.mean(all_vs_output):.2f}%")
        
        if abs(np.mean(all_correct) - np.mean(all_vs_input)) > 10:
            print(f"\n🔴 BUG CONFIRMED!")
            print(f"   The training visualization is likely comparing against INPUT grids")
            print(f"   instead of OUTPUT grids, causing incorrect accuracy reporting.")
            print(f"\n   FIX: Update get_reconstructions() in app.py to:")
            print(f"   1. Accept output_grids and output_sizes parameters")
            print(f"   2. Pass output_sizes to model.forward()")
            print(f"   3. Compare reconstruction against OUTPUT grid, not INPUT")
        else:
            print(f"\n✓ No major discrepancy detected")
    
    print("\nDiagnostic complete!")


if __name__ == '__main__':
    main()