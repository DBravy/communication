"""
Systematic Hyperparameter Testing for Slot Attention Model
Tests combinations of:
- Number of Slots: 3, 7, 14
- Slot Dimension: 32, 64, 128
- Attention Iterations: 1, 3, 5

Each experiment trains for 300 epochs on 200 grids from V1 (100 from training + 100 from eval),
testing generalization on V2 training set.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import json
from tqdm import tqdm
import itertools
from functools import partial

# Import from the training scripts
import config
from dataset import ARCDataset, collate_fn
from model import ARCEncoder, ARCAutoencoder
from train import validate

def run_single_experiment(num_slots, slot_dim, slot_iterations, experiment_id, total_experiments):
    """Run a single experiment with given hyperparameters."""
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {experiment_id}/{total_experiments}")
    print(f"{'='*80}")
    print(f"Num Slots: {num_slots}")
    print(f"Slot Dimension: {slot_dim}")
    print(f"Attention Iterations: {slot_iterations}")
    print(f"{'='*80}\n")
    
    # Update config for this experiment
    config.NUM_SLOTS = num_slots
    config.SLOT_DIM = slot_dim
    config.SLOT_ITERATIONS = slot_iterations
    config.BOTTLENECK_TYPE = 'slot_attention'
    config.MAX_GRIDS = 200  # 100 from training + 100 from eval
    config.DATASET_VERSION = 'V1'
    config.DATASET_SPLIT = 'training'  # Note: we actually use both training and eval
    config.GENERALIZATION_TEST_ENABLED = True
    config.GENERALIZATION_TEST_DATASET_VERSION = 'V2'
    config.GENERALIZATION_TEST_DATASET_SPLIT = 'training'
    config.GENERALIZATION_TEST_MAX_GRIDS = None  # Use all V2 grids
    config.NUM_EPOCHS = 300
    config.BATCH_SIZE = 32
    config.LEARNING_RATE = 1e-4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load V1 training dataset (100 grids from training + 100 from eval)
    train_data_path = os.path.join('V1', 'data', 'training')
    eval_data_path = os.path.join('V1', 'data', 'evaluation')
    
    train_dataset_part = ARCDataset(
        train_data_path,
        min_size=config.MIN_GRID_SIZE,
        filter_size=config.FILTER_GRID_SIZE,
        max_grids=100,
        num_distractors=0,
        track_puzzle_ids=False,
        use_input_output_pairs=False
    )
    
    eval_dataset_part = ARCDataset(
        eval_data_path,
        min_size=config.MIN_GRID_SIZE,
        filter_size=config.FILTER_GRID_SIZE,
        max_grids=100,
        num_distractors=0,
        track_puzzle_ids=False,
        use_input_output_pairs=False
    )
    
    # Combine both datasets
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset([train_dataset_part, eval_dataset_part])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, num_distractors=0, use_input_output_pairs=False),
        num_workers=0
    )
    
    # Load V2 training dataset for generalization testing
    gen_data_path = os.path.join('V2', 'data', 'training')
    gen_dataset = ARCDataset(
        gen_data_path,
        min_size=config.MIN_GRID_SIZE,
        filter_size=config.FILTER_GRID_SIZE,
        max_grids=None,  # Use all grids
        num_distractors=0,
        track_puzzle_ids=False,
        use_input_output_pairs=False
    )
    
    gen_loader = DataLoader(
        gen_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, num_distractors=0, use_input_output_pairs=False),
        num_workers=0
    )
    
    print(f"Training grids (V1): {len(train_dataset)}")
    print(f"Generalization grids (V2): {len(gen_dataset)}")
    
    # Create model
    encoder = ARCEncoder(
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_conv_layers=config.NUM_CONV_LAYERS,
        use_beta_vae=config.USE_BETA_VAE
    )
    
    model = ARCAutoencoder(
        encoder=encoder,
        vocab_size=config.VOCAB_SIZE,
        max_length=config.MAX_MESSAGE_LENGTH,
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        max_grid_size=config.MAX_GRID_SIZE,
        bottleneck_type=config.BOTTLENECK_TYPE,
        task_type='reconstruction',
        num_conv_layers=config.NUM_CONV_LAYERS,
        receiver_gets_input_puzzle=config.RECEIVER_GETS_INPUT_PUZZLE,
        use_stop_token=config.USE_STOP_TOKEN,
        stop_token_id=config.STOP_TOKEN_ID,
        lstm_hidden_dim=config.HIDDEN_DIM,
        use_beta_vae=config.USE_BETA_VAE,
        beta=config.BETA_VAE_BETA,
        num_slots=config.NUM_SLOTS,
        slot_dim=config.SLOT_DIM,
        slot_iterations=config.SLOT_ITERATIONS,
        slot_hidden_dim=config.SLOT_HIDDEN_DIM,
        slot_eps=config.SLOT_EPS
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Training metrics storage
    train_losses = []
    train_accs = []
    gen_losses = []
    gen_accs = []
    epoch_times = []
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        
        total_loss = 0
        total_correct = 0
        total_pixels = 0
        
        # Training loop
        for batch_data in train_loader:
            # Unpack tuple (not dict) - for reconstruction task without input_output_pairs
            grids, sizes = batch_data
            grids = grids.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            model_output = model(grids, sizes, temperature=config.TEMPERATURE)
            
            # Unpack output based on bottleneck type
            if config.BOTTLENECK_TYPE == 'communication':
                logits_list, actual_sizes, messages, message_lengths = model_output
            else:  # autoencoder or slot_attention
                logits_list, actual_sizes, _ = model_output
            
            # Compute loss
            batch_loss = 0
            batch_correct = 0
            batch_pixels = 0
            
            for i, logits in enumerate(logits_list):
                h, w = actual_sizes[i]
                target = grids[i, :h, :w]  # [h, w]
                
                # Handle logits shape - remove batch dimension if present
                if logits.dim() == 4:  # [1, num_colors, H, W]
                    logits = logits.squeeze(0)  # [num_colors, H, W]
                
                # Extract actual region (logits might be padded to max_grid_size)
                logits = logits[:, :h, :w]  # [num_colors, h, w]
                
                # Reshape for CrossEntropyLoss: expects (N, C) input and (N) target
                # where N is number of pixels, C is number of classes
                logits_reshaped = logits.permute(1, 2, 0).reshape(h * w, config.NUM_COLORS)  # [h*w, num_colors]
                target_reshaped = target.reshape(h * w)  # [h*w]
                
                sample_loss = criterion(logits_reshaped, target_reshaped)
                batch_loss += sample_loss
                
                # Compute accuracy
                pred = logits.argmax(dim=0)  # [h, w]
                correct = (pred == target).sum().item()
                batch_correct += correct
                batch_pixels += h * w
            
            loss = batch_loss / len(logits_list)
            
            # Add KL divergence if using β-VAE
            if config.USE_BETA_VAE and hasattr(encoder, 'last_mu') and encoder.last_mu is not None:
                kl_loss = -0.5 * torch.sum(1 + encoder.last_logvar - encoder.last_mu.pow(2) - encoder.last_logvar.exp())
                kl_loss = kl_loss / len(logits_list)
                loss = loss + config.BETA_VAE_BETA * kl_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_correct += batch_correct
            total_pixels += batch_pixels
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100 * total_correct / total_pixels
        epoch_time = time.time() - epoch_start
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        epoch_times.append(epoch_time)
        
        # Validation on generalization set every 20 epochs
        if (epoch + 1) % 20 == 0:
            model.eval()
            gen_loss, gen_acc = validate(model, gen_loader, criterion, device, 
                                        task_type='reconstruction', use_input_output_pairs=False)
            gen_losses.append(gen_loss)
            gen_accs.append(gen_acc)
            
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Gen Loss: {gen_loss:.4f}, Gen Acc: {gen_acc:.2f}% - "
                  f"Time: {epoch_time:.2f}s")
        elif (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Time: {epoch_time:.2f}s")
    
    # Final evaluation
    model.eval()
    final_gen_loss, final_gen_acc = validate(model, gen_loader, criterion, device,
                                             task_type='reconstruction', use_input_output_pairs=False)
    
    print(f"\nFinal Results:")
    print(f"Final Training Accuracy: {train_accs[-1]:.2f}%")
    print(f"Final Generalization Accuracy: {final_gen_acc:.2f}%")
    
    # Return experiment results
    return {
        'num_slots': num_slots,
        'slot_dim': slot_dim,
        'slot_iterations': slot_iterations,
        'final_train_loss': float(train_losses[-1]),
        'final_train_acc': float(train_accs[-1]),
        'final_gen_loss': float(final_gen_loss),
        'final_gen_acc': float(final_gen_acc),
        'avg_epoch_time': float(sum(epoch_times) / len(epoch_times)),
        'total_train_time': float(sum(epoch_times)),
        'gen_acc_history': [float(x) for x in gen_accs],  # Gen acc every 20 epochs
        'train_acc_history': [float(train_accs[i]) for i in range(19, 300, 20)]  # Train acc at epochs 20, 40, ..., 300
    }


def main():
    """Run all experiments and save results."""
    
    # Define hyperparameter combinations
    num_slots_values = [3, 7, 14]
    slot_dim_values = [32, 64, 128]
    slot_iterations_values = [1, 3, 5]
    
    # Generate all combinations
    experiments = list(itertools.product(num_slots_values, slot_dim_values, slot_iterations_values))
    total_experiments = len(experiments)
    
    print(f"{'='*80}")
    print(f"SLOT ATTENTION HYPERPARAMETER TESTING")
    print(f"{'='*80}")
    print(f"Total experiments: {total_experiments}")
    print(f"Experiments per slot count: {len(slot_dim_values) * len(slot_iterations_values)}")
    print(f"\nHyperparameter ranges:")
    print(f"  - Number of Slots: {num_slots_values}")
    print(f"  - Slot Dimension: {slot_dim_values}")
    print(f"  - Attention Iterations: {slot_iterations_values}")
    print(f"\nTraining configuration:")
    print(f"  - Training set: V1/training (100 grids) + V1/evaluation (100 grids) = 200 total")
    print(f"  - Generalization set: V2/training (all grids)")
    print(f"  - Epochs per experiment: 300")
    print(f"  - Batch size: 32")
    print(f"  - Learning rate: 1e-4")
    print(f"{'='*80}\n")
    
    # Run all experiments
    results = []
    start_time = time.time()
    
    for exp_id, (num_slots, slot_dim, slot_iterations) in enumerate(experiments, 1):
        try:
            result = run_single_experiment(num_slots, slot_dim, slot_iterations, exp_id, total_experiments)
            results.append(result)
            
            # Save intermediate results after each experiment
            with open('slot_attention_experiment_results.json', 'w') as f:
                json.dump({
                    'completed_experiments': len(results),
                    'total_experiments': total_experiments,
                    'results': results,
                    'timestamp': time.time()
                }, f, indent=2)
                
        except Exception as e:
            print(f"\n!!! ERROR in experiment {exp_id}: {e} !!!\n")
            results.append({
                'num_slots': num_slots,
                'slot_dim': slot_dim,
                'slot_iterations': slot_iterations,
                'error': str(e)
            })
    
    total_time = time.time() - start_time
    
    # Save final results
    final_results = {
        'completed_experiments': len([r for r in results if 'error' not in r]),
        'total_experiments': total_experiments,
        'total_time_hours': total_time / 3600,
        'results': results,
        'timestamp': time.time()
    }
    
    with open('slot_attention_experiment_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Generate summary report
    generate_summary_report(results, total_time)
    
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED!")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Results saved to: slot_attention_experiment_results.json")
    print(f"Summary saved to: slot_attention_summary.txt")
    print(f"{'='*80}\n")


def generate_summary_report(results, total_time):
    """Generate a concise summary report of all experiments."""
    
    successful_results = [r for r in results if 'error' not in r]
    
    with open('slot_attention_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("SLOT ATTENTION HYPERPARAMETER TESTING - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Experiments: {len(results)}\n")
        f.write(f"Successful: {len(successful_results)}\n")
        f.write(f"Failed: {len(results) - len(successful_results)}\n")
        f.write(f"Total Runtime: {total_time/3600:.2f} hours\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESULTS BY CONFIGURATION\n")
        f.write("="*80 + "\n\n")
        
        # Sort results by final generalization accuracy
        sorted_results = sorted(successful_results, key=lambda x: x['final_gen_acc'], reverse=True)
        
        for rank, result in enumerate(sorted_results, 1):
            f.write(f"Rank #{rank}\n")
            f.write(f"  Configuration:\n")
            f.write(f"    - Num Slots: {result['num_slots']}\n")
            f.write(f"    - Slot Dim: {result['slot_dim']}\n")
            f.write(f"    - Attention Iterations: {result['slot_iterations']}\n")
            f.write(f"  Performance:\n")
            f.write(f"    - Final Train Accuracy: {result['final_train_acc']:.2f}%\n")
            f.write(f"    - Final Gen Accuracy: {result['final_gen_acc']:.2f}%\n")
            f.write(f"    - Final Train Loss: {result['final_train_loss']:.4f}\n")
            f.write(f"    - Final Gen Loss: {result['final_gen_loss']:.4f}\n")
            f.write(f"    - Avg Epoch Time: {result['avg_epoch_time']:.2f}s\n")
            f.write(f"    - Total Train Time: {result['total_train_time']/60:.2f} min\n")
            f.write("\n")
        
        # Analysis by parameter
        f.write("="*80 + "\n")
        f.write("ANALYSIS BY PARAMETER\n")
        f.write("="*80 + "\n\n")
        
        # Group by number of slots
        f.write("By Number of Slots:\n")
        f.write("-" * 40 + "\n")
        for num_slots in [3, 7, 14]:
            slot_results = [r for r in successful_results if r['num_slots'] == num_slots]
            if slot_results:
                avg_gen_acc = sum(r['final_gen_acc'] for r in slot_results) / len(slot_results)
                best_gen_acc = max(r['final_gen_acc'] for r in slot_results)
                f.write(f"  {num_slots} slots: Avg Gen Acc = {avg_gen_acc:.2f}%, Best = {best_gen_acc:.2f}%\n")
        f.write("\n")
        
        # Group by slot dimension
        f.write("By Slot Dimension:\n")
        f.write("-" * 40 + "\n")
        for slot_dim in [32, 64, 128]:
            dim_results = [r for r in successful_results if r['slot_dim'] == slot_dim]
            if dim_results:
                avg_gen_acc = sum(r['final_gen_acc'] for r in dim_results) / len(dim_results)
                best_gen_acc = max(r['final_gen_acc'] for r in dim_results)
                f.write(f"  {slot_dim} dim: Avg Gen Acc = {avg_gen_acc:.2f}%, Best = {best_gen_acc:.2f}%\n")
        f.write("\n")
        
        # Group by attention iterations
        f.write("By Attention Iterations:\n")
        f.write("-" * 40 + "\n")
        for iterations in [1, 3, 5]:
            iter_results = [r for r in successful_results if r['slot_iterations'] == iterations]
            if iter_results:
                avg_gen_acc = sum(r['final_gen_acc'] for r in iter_results) / len(iter_results)
                best_gen_acc = max(r['final_gen_acc'] for r in iter_results)
                f.write(f"  {iterations} iterations: Avg Gen Acc = {avg_gen_acc:.2f}%, Best = {best_gen_acc:.2f}%\n")
        f.write("\n")
        
        # Best configuration
        f.write("="*80 + "\n")
        f.write("BEST CONFIGURATION\n")
        f.write("="*80 + "\n\n")
        
        if sorted_results:
            best = sorted_results[0]
            f.write(f"Configuration:\n")
            f.write(f"  - Num Slots: {best['num_slots']}\n")
            f.write(f"  - Slot Dim: {best['slot_dim']}\n")
            f.write(f"  - Attention Iterations: {best['slot_iterations']}\n\n")
            f.write(f"Performance:\n")
            f.write(f"  - Final Train Accuracy: {best['final_train_acc']:.2f}%\n")
            f.write(f"  - Final Gen Accuracy: {best['final_gen_acc']:.2f}%\n")
            f.write(f"  - Improvement over training: {best['final_gen_acc'] - best['final_train_acc']:.2f}%\n\n")
            
            # Show generalization progress
            f.write(f"Generalization Accuracy Progress (every 20 epochs):\n")
            for i, acc in enumerate(best['gen_acc_history'], 1):
                f.write(f"  Epoch {i*20}: {acc:.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()