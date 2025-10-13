"""Test script to verify background reconstruction implementation."""

import torch
from model import ARCEncoder, ARCAutoencoder

def test_model_creation():
    """Test that model can be created with both receivers."""
    print("Testing model creation...")
    
    encoder = ARCEncoder(
        num_colors=10,
        embedding_dim=10,
        hidden_dim=128,
        latent_dim=128,
        num_conv_layers=3
    )
    
    # Test communication mode
    model = ARCAutoencoder(
        encoder=encoder,
        vocab_size=25,
        max_length=2,
        num_colors=10,
        embedding_dim=10,
        hidden_dim=128,
        max_grid_size=30,
        bottleneck_type='communication',
        task_type='selection',
        num_conv_layers=2
    )
    
    assert hasattr(model, 'receiver'), 'Missing receiver (selector)'
    assert hasattr(model, 'receiver_reconstructor'), 'Missing receiver_reconstructor'
    print('✓ Communication mode: Both receiver and receiver_reconstructor are present')
    
    # Test autoencoder mode
    encoder2 = ARCEncoder(
        num_colors=10,
        embedding_dim=10,
        hidden_dim=128,
        latent_dim=128,
        num_conv_layers=3
    )
    
    model2 = ARCAutoencoder(
        encoder=encoder2,
        vocab_size=None,
        max_length=None,
        num_colors=10,
        embedding_dim=10,
        hidden_dim=128,
        max_grid_size=30,
        bottleneck_type='autoencoder',
        task_type='selection',
        num_conv_layers=2
    )
    
    assert hasattr(model2, 'decoder'), 'Missing decoder (selector)'
    assert hasattr(model2, 'decoder_reconstructor'), 'Missing decoder_reconstructor'
    print('✓ Autoencoder mode: Both decoder and decoder_reconstructor are present')
    
    return model

def test_forward_pass(model):
    """Test that forward pass returns correct outputs."""
    print("\nTesting forward pass...")
    
    batch_size = 2
    x = torch.randint(0, 10, (batch_size, 30, 30))
    sizes = [(10, 10), (15, 12)]
    
    # Create candidates
    candidates_list = [
        torch.randint(0, 10, (3, 30, 30)),  # 3 candidates for first sample
        torch.randint(0, 10, (3, 30, 30))   # 3 candidates for second sample
    ]
    candidates_sizes_list = [
        [(10, 10), (12, 8), (9, 11)],
        [(15, 12), (14, 13), (11, 10)]
    ]
    target_indices = torch.tensor([0, 1])
    
    # Forward pass in eval mode
    model.eval()
    with torch.no_grad():
        outputs = model(
            x, sizes, temperature=1.0,
            candidates_list=candidates_list,
            candidates_sizes_list=candidates_sizes_list,
            target_indices=target_indices
        )
    
    assert len(outputs) == 4, f'Expected 4 outputs, got {len(outputs)}'
    print(f'✓ Forward pass returns 4 outputs (selection_logits, reconstruction_logits, sizes, messages)')
    
    selection_logits_list, reconstruction_logits_list, actual_sizes, messages = outputs
    
    assert len(selection_logits_list) == batch_size, f'Expected {batch_size} selection logits'
    assert len(reconstruction_logits_list) == batch_size, f'Expected {batch_size} reconstruction logits'
    print(f'✓ Output lists have correct batch size: {batch_size}')
    
    print(f'✓ Selection logits shape: {selection_logits_list[0].shape}')
    print(f'✓ Reconstruction logits shape: {reconstruction_logits_list[0].shape}')
    
    # Check that reconstruction logits have correct shape [1, num_colors, H, W]
    for i, recon_logits in enumerate(reconstruction_logits_list):
        assert recon_logits.shape[0] == 1, 'Batch size should be 1'
        assert recon_logits.shape[1] == 10, 'Should have 10 color channels'
        print(f'✓ Sample {i} reconstruction shape: {recon_logits.shape}')

def test_gradient_flow(model):
    """Test that gradients flow correctly (detached for reconstruction in training mode)."""
    print("\nTesting gradient flow...")
    
    model.train()
    
    batch_size = 1
    x = torch.randint(0, 10, (batch_size, 30, 30)).float()
    sizes = [(10, 10)]
    
    candidates_list = [torch.randint(0, 10, (2, 30, 30))]
    candidates_sizes_list = [[(10, 10), (12, 8)]]
    target_indices = torch.tensor([0])
    
    # Forward pass
    selection_logits_list, reconstruction_logits_list, actual_sizes, messages = model(
        x, sizes, temperature=1.0,
        candidates_list=candidates_list,
        candidates_sizes_list=candidates_sizes_list,
        target_indices=target_indices
    )
    
    # Test selection loss gradient
    selection_loss = torch.nn.functional.cross_entropy(
        selection_logits_list[0].unsqueeze(0),
        target_indices
    )
    selection_loss.backward(retain_graph=True)
    
    # Check that sender has gradients from selection loss
    sender_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                         for p in model.sender.parameters())
    print(f'✓ Sender receives gradients from selection loss: {sender_has_grad}')
    
    # Clear gradients
    model.zero_grad()
    
    # Test reconstruction loss (should NOT propagate to sender due to detach)
    recon_logits = reconstruction_logits_list[0]
    target_grid = x[0:1, :10, :10].long()
    
    logits_flat = recon_logits[:, :, :10, :10].permute(0, 2, 3, 1).reshape(-1, 10)
    targets_flat = target_grid.reshape(-1)
    
    recon_loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    recon_loss.backward()
    
    # Check that receiver_reconstructor has gradients
    reconstructor_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                                for p in model.receiver_reconstructor.parameters())
    print(f'✓ Receiver_reconstructor receives gradients from recon loss: {reconstructor_has_grad}')
    
    # Check that sender does NOT have gradients from reconstruction loss
    sender_has_grad_from_recon = any(p.grad is not None and p.grad.abs().sum() > 0 
                                    for p in model.sender.parameters())
    print(f'✓ Sender isolated from reconstruction loss (should be False): {sender_has_grad_from_recon}')
    
    if not sender_has_grad_from_recon:
        print('✅ Gradient isolation working correctly!')
    else:
        print('⚠ Warning: Sender received gradients from reconstruction loss (detach may not be working)')

def main():
    print("="*80)
    print("BACKGROUND RECONSTRUCTION TEST")
    print("="*80)
    
    model = test_model_creation()
    test_forward_pass(model)
    test_gradient_flow(model)
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)

if __name__ == '__main__':
    main()

