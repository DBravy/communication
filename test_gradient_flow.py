"""Comprehensive gradient flow testing script for all task types and bottleneck types.

Tests gradient flow through the entire model architecture in:
- 3 task types: reconstruction, selection, puzzle_classification
- 2 bottleneck types: communication, autoencoder
- Total: 6 different configurations

For each configuration, checks:
1. Which model components receive gradients
2. Gradient magnitude at each layer
3. Identifies any gradient flow blockages
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import sys

# Import model components
from model import (
    ARCEncoder, ARCDecoder, SenderAgent, ReceiverAgent, 
    ReceiverSelector, ReceiverPuzzleClassifier, DecoderSelector,
    ARCAutoencoder
)
from dataset import ARCDataset, collate_fn, collate_fn_puzzle_classification
from torch.utils.data import DataLoader
import config

class GradientFlowTester:
    """Comprehensive gradient flow testing for all configurations."""
    
    def __init__(self):
        self.device = 'cpu'  # Use CPU for easier debugging
        self.results = {}
        
    def create_model(self, bottleneck_type, task_type, num_classes=None):
        """Create model for given configuration."""
        encoder = ARCEncoder(
            num_colors=config.NUM_COLORS,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            latent_dim=config.LATENT_DIM,
            num_conv_layers=config.NUM_CONV_LAYERS
        ).to(self.device)
        
        model = ARCAutoencoder(
            encoder=encoder,
            vocab_size=config.VOCAB_SIZE,
            max_length=config.MAX_MESSAGE_LENGTH,
            num_colors=config.NUM_COLORS,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            max_grid_size=config.MAX_GRID_SIZE,
            bottleneck_type=bottleneck_type,
            task_type=task_type,
            num_conv_layers=config.NUM_CONV_LAYERS,
            num_classes=num_classes
        ).to(self.device)
        
        return model
    
    def create_dummy_data(self, task_type, batch_size=2):
        """Create dummy data for testing."""
        # Create simple grids
        grids = torch.randint(0, config.NUM_COLORS, (batch_size, 30, 30)).to(self.device)
        sizes = [(10, 10), (15, 12)]  # Different sizes for variety
        
        if task_type == 'reconstruction':
            return grids, sizes, None, None, None, None
            
        elif task_type == 'selection':
            # Create candidate grids (including correct one)
            candidates_list = []
            candidates_sizes_list = []
            target_indices = torch.tensor([0, 1])  # Targets for each batch item
            
            for i in range(batch_size):
                num_candidates = 3
                candidates = torch.randint(0, config.NUM_COLORS, 
                                          (num_candidates, 30, 30)).to(self.device)
                cand_sizes = [(8, 9), (10, 10), (12, 11)]
                candidates_list.append(candidates)
                candidates_sizes_list.append(cand_sizes)
            
            return grids, sizes, candidates_list, candidates_sizes_list, target_indices, None
            
        elif task_type == 'puzzle_classification':
            # Create labels (puzzle IDs)
            labels = torch.tensor([0, 1]).to(self.device)
            return grids, sizes, None, None, None, labels
            
    def get_all_parameters_with_names(self, model):
        """Get all parameters with their full names."""
        params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                params[name] = param
        return params
    
    def compute_loss(self, model, outputs, task_type, target_indices=None, labels=None, 
                     grids=None, sizes=None):
        """Compute loss based on task type."""
        
        if task_type == 'reconstruction':
            logits_list, actual_sizes, messages = outputs
            total_loss = 0.0
            
            for i, (logits, (H, W)) in enumerate(zip(logits_list, actual_sizes)):
                target_grid = grids[i, :H, :W]
                # logits shape: [1, num_colors, H', W'] where H', W' >= H, W
                pred_grid = logits.squeeze(0)[:, :H, :W]  # [num_colors, H, W]
                # Reshape: [num_colors, H, W] -> [H, W, num_colors] -> [H*W, num_colors]
                pred_flat = pred_grid.permute(1, 2, 0).reshape(-1, config.NUM_COLORS)
                target_flat = target_grid.reshape(-1)
                loss = F.cross_entropy(pred_flat, target_flat)
                total_loss += loss
            
            return total_loss / len(logits_list)
            
        elif task_type == 'selection':
            selection_logits_list, actual_sizes, messages = outputs
            total_loss = 0.0
            
            for i, sel_logits in enumerate(selection_logits_list):
                target = target_indices[i:i+1]
                loss = F.cross_entropy(sel_logits.unsqueeze(0), target)
                total_loss += loss
            
            return total_loss / len(selection_logits_list)
            
        elif task_type == 'puzzle_classification':
            classification_logits, _, messages = outputs
            loss = F.cross_entropy(classification_logits, labels)
            return loss
    
    def analyze_gradients(self, params):
        """Analyze gradient flow through parameters."""
        results = {
            'has_grad': [],
            'no_grad': [],
            'zero_grad': [],
            'grad_stats': {}
        }
        
        for name, param in params.items():
            if param.grad is None:
                results['no_grad'].append(name)
            else:
                grad_norm = param.grad.norm().item()
                grad_max = param.grad.abs().max().item()
                grad_mean = param.grad.abs().mean().item()
                
                if grad_norm < 1e-10:
                    results['zero_grad'].append(name)
                else:
                    results['has_grad'].append(name)
                
                results['grad_stats'][name] = {
                    'norm': grad_norm,
                    'max': grad_max,
                    'mean': grad_mean,
                    'shape': tuple(param.grad.shape)
                }
        
        return results
    
    def test_configuration(self, bottleneck_type, task_type):
        """Test a specific configuration."""
        print("\n" + "="*80)
        print(f"Testing: {bottleneck_type.upper()} + {task_type.upper()}")
        print("="*80)
        
        # Determine number of classes for puzzle classification
        num_classes = 10 if task_type == 'puzzle_classification' else None
        
        # Create model
        model = self.create_model(bottleneck_type, task_type, num_classes)
        model.train()
        
        # Get all parameters
        params = self.get_all_parameters_with_names(model)
        print(f"\nTotal trainable parameters: {len(params)}")
        
        # Create dummy data
        grids, sizes, candidates_list, candidates_sizes_list, target_indices, labels = \
            self.create_dummy_data(task_type)
        
        # Forward pass
        print(f"\nRunning forward pass...")
        if task_type == 'reconstruction':
            outputs = model(grids, sizes, temperature=config.TEMPERATURE)
        elif task_type == 'selection':
            outputs = model(grids, sizes, temperature=config.TEMPERATURE,
                          candidates_list=candidates_list,
                          candidates_sizes_list=candidates_sizes_list,
                          target_indices=target_indices)
        elif task_type == 'puzzle_classification':
            outputs = model(grids, sizes, temperature=config.TEMPERATURE,
                          labels=labels)
        
        # Compute loss
        loss = self.compute_loss(model, outputs, task_type, target_indices, labels, grids, sizes)
        print(f"Loss: {loss.item():.6f}")
        
        # Backward pass
        print(f"Running backward pass...")
        model.zero_grad()
        loss.backward()
        
        # Analyze gradients
        grad_results = self.analyze_gradients(params)
        
        # Print results
        print(f"\n{'='*80}")
        print(f"GRADIENT FLOW ANALYSIS")
        print(f"{'='*80}")
        
        print(f"\n✓ Parameters WITH gradients ({len(grad_results['has_grad'])}):")
        for name in sorted(grad_results['has_grad']):
            stats = grad_results['grad_stats'][name]
            print(f"  {name:60s} | norm: {stats['norm']:>12.6e} | max: {stats['max']:>12.6e}")
        
        if grad_results['zero_grad']:
            print(f"\n⚠ Parameters with ZERO gradients ({len(grad_results['zero_grad'])}):")
            for name in sorted(grad_results['zero_grad']):
                stats = grad_results['grad_stats'][name]
                print(f"  {name:60s} | norm: {stats['norm']:>12.6e}")
        
        if grad_results['no_grad']:
            print(f"\n✗ Parameters with NO gradients ({len(grad_results['no_grad'])}):")
            for name in sorted(grad_results['no_grad']):
                print(f"  {name}")
        
        # Group analysis by component
        print(f"\n{'='*80}")
        print(f"COMPONENT-WISE ANALYSIS")
        print(f"{'='*80}")
        
        components = self.group_by_component(grad_results['grad_stats'])
        for comp_name, comp_stats in sorted(components.items()):
            total_norm = sum(s['norm'] for s in comp_stats.values())
            avg_norm = total_norm / len(comp_stats) if comp_stats else 0
            print(f"\n{comp_name}:")
            print(f"  Layers: {len(comp_stats)}")
            print(f"  Total gradient norm: {total_norm:.6e}")
            print(f"  Average gradient norm: {avg_norm:.6e}")
            
            # Show top 3 layers by gradient magnitude
            sorted_layers = sorted(comp_stats.items(), 
                                 key=lambda x: x[1]['norm'], reverse=True)[:3]
            if sorted_layers:
                print(f"  Top gradients:")
                for layer_name, stats in sorted_layers:
                    print(f"    {layer_name}: {stats['norm']:.6e}")
        
        # Store results
        config_key = f"{bottleneck_type}_{task_type}"
        self.results[config_key] = {
            'has_grad': grad_results['has_grad'],
            'zero_grad': grad_results['zero_grad'],
            'no_grad': grad_results['no_grad'],
            'grad_stats': grad_results['grad_stats'],
            'components': components
        }
        
        # Identify potential issues
        self.diagnose_issues(bottleneck_type, task_type, grad_results, components)
    
    def group_by_component(self, grad_stats):
        """Group parameters by model component."""
        components = defaultdict(dict)
        
        for name, stats in grad_stats.items():
            # Determine component
            if name.startswith('encoder.'):
                comp = 'encoder'
            elif name.startswith('decoder.'):
                comp = 'decoder'
            elif name.startswith('sender.'):
                comp = 'sender'
            elif name.startswith('receiver.'):
                comp = 'receiver'
            else:
                comp = 'other'
            
            # Further subdivide
            if 'conv' in name:
                comp += '.conv'
            elif 'fc' in name or 'linear' in name.lower():
                comp += '.fc'
            elif 'lstm' in name:
                comp += '.lstm'
            elif 'embed' in name:
                comp += '.embedding'
            elif 'bn' in name or 'norm' in name:
                comp += '.norm'
            
            components[comp][name] = stats
        
        return dict(components)
    
    def diagnose_issues(self, bottleneck_type, task_type, grad_results, components):
        """Diagnose potential gradient flow issues."""
        print(f"\n{'='*80}")
        print(f"DIAGNOSIS")
        print(f"{'='*80}")
        
        issues = []
        
        # Check for components with no/zero gradients
        for comp_name, comp_stats in components.items():
            if not comp_stats:
                continue
            
            zero_count = sum(1 for s in comp_stats.values() if s['norm'] < 1e-10)
            total_count = len(comp_stats)
            
            if zero_count == total_count:
                issues.append(f"⚠ {comp_name}: ALL gradients are zero!")
            elif zero_count > total_count * 0.5:
                issues.append(f"⚠ {comp_name}: {zero_count}/{total_count} gradients are zero")
        
        # Check for very small gradients
        for comp_name, comp_stats in components.items():
            avg_norm = sum(s['norm'] for s in comp_stats.values()) / len(comp_stats) if comp_stats else 0
            if avg_norm < 1e-8 and avg_norm > 0:
                issues.append(f"⚠ {comp_name}: Very small gradients (avg: {avg_norm:.2e})")
        
        # Specific checks for communication bottleneck
        if bottleneck_type == 'communication':
            sender_norms = [s['norm'] for name, s in grad_results['grad_stats'].items() 
                          if name.startswith('sender.')]
            receiver_norms = [s['norm'] for name, s in grad_results['grad_stats'].items() 
                            if name.startswith('receiver.')]
            
            if sender_norms and max(sender_norms) < 1e-8:
                issues.append("⚠ CRITICAL: Sender receives almost no gradients!")
                issues.append("  → This means the communication channel is not learning")
            
            if receiver_norms and max(receiver_norms) < 1e-8:
                issues.append("⚠ CRITICAL: Receiver receives almost no gradients!")
        
        # Print diagnosis
        if issues:
            print("\n🔴 ISSUES FOUND:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("\n✅ No gradient flow issues detected!")
            print("  All major components are receiving gradients.")
        
        # Additional context
        print(f"\nNOTE: For '{task_type}' task with '{bottleneck_type}' bottleneck:")
        if task_type == 'reconstruction':
            print("  - Expect gradients through encoder → bottleneck → decoder")
        elif task_type == 'selection':
            print("  - Expect gradients through encoder → bottleneck → selector")
            if bottleneck_type == 'communication':
                print("  - Check if message representation gradients cancel in selector")
        elif task_type == 'puzzle_classification':
            print("  - Expect gradients through encoder → bottleneck → classifier")
    
    def run_all_tests(self):
        """Run tests for all configurations."""
        print("\n" + "="*80)
        print("COMPREHENSIVE GRADIENT FLOW TESTING")
        print("="*80)
        print("\nTesting all combinations of:")
        print("  - Bottleneck types: communication, autoencoder")
        print("  - Task types: reconstruction, selection, puzzle_classification")
        print("  - Total: 6 configurations")
        
        configurations = [
            ('communication', 'reconstruction'),
            ('communication', 'selection'),
            ('communication', 'puzzle_classification'),
            ('autoencoder', 'reconstruction'),
            ('autoencoder', 'selection'),
            # Note: autoencoder + puzzle_classification not implemented
        ]
        
        for bottleneck_type, task_type in configurations:
            try:
                self.test_configuration(bottleneck_type, task_type)
            except Exception as e:
                print(f"\n❌ ERROR in {bottleneck_type} + {task_type}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary comparison across all configurations."""
        print("\n" + "="*80)
        print("SUMMARY: GRADIENT FLOW COMPARISON")
        print("="*80)
        
        print(f"\n{'Configuration':<40} {'Has Grad':<12} {'Zero Grad':<12} {'No Grad':<12} {'Status'}")
        print("-" * 90)
        
        for config_key, results in sorted(self.results.items()):
            has_grad = len(results['has_grad'])
            zero_grad = len(results['zero_grad'])
            no_grad = len(results['no_grad'])
            
            # Determine status
            if zero_grad == 0 and no_grad == 0:
                status = "✅ Good"
            elif zero_grad > 0 or no_grad > 0:
                status = "⚠ Issues"
            else:
                status = "❓ Unknown"
            
            print(f"{config_key:<40} {has_grad:<12} {zero_grad:<12} {no_grad:<12} {status}")
        
        # Component comparison
        print(f"\n{'='*80}")
        print("COMPONENT GRADIENT NORMS BY CONFIGURATION")
        print("="*80)
        
        # Collect all component names
        all_components = set()
        for results in self.results.values():
            all_components.update(results['components'].keys())
        
        for comp in sorted(all_components):
            print(f"\n{comp}:")
            for config_key, results in sorted(self.results.items()):
                if comp in results['components']:
                    comp_stats = results['components'][comp]
                    total_norm = sum(s['norm'] for s in comp_stats.values())
                    print(f"  {config_key:<40} {total_norm:>12.6e}")
                else:
                    print(f"  {config_key:<40} {'N/A':>12}")

def main():
    """Main entry point."""
    print(f"\nUsing configuration:")
    print(f"  NUM_COLORS: {config.NUM_COLORS}")
    print(f"  HIDDEN_DIM: {config.HIDDEN_DIM}")
    print(f"  LATENT_DIM: {config.LATENT_DIM}")
    print(f"  VOCAB_SIZE: {config.VOCAB_SIZE}")
    print(f"  MAX_MESSAGE_LENGTH: {config.MAX_MESSAGE_LENGTH}")
    print(f"  NUM_CONV_LAYERS: {config.NUM_CONV_LAYERS}")
    
    tester = GradientFlowTester()
    tester.run_all_tests()
    
    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)

if __name__ == "__main__":
    main()

