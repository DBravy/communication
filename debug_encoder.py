"""Debug script to figure out why the encoder can't learn simple classification tasks."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from model import ARCEncoder


def create_toy_patterns():
    """Create 3 very distinct 4x4 patterns that should be easy to distinguish."""
    # Pattern 1: Horizontal stripes
    pattern1 = np.array([
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0]
    ], dtype=np.int64)
    
    # Pattern 2: Vertical stripes
    pattern2 = np.array([
        [1, 0, 1, 0],
        [1, 0, 1, 0],
        [1, 0, 1, 0],
        [1, 0, 1, 0]
    ], dtype=np.int64)
    
    # Pattern 3: Checkerboard
    pattern3 = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ], dtype=np.int64)
    
    return [pattern1, pattern2, pattern3]


def pad_to_30x30(grid):
    """Pad a grid to 30x30."""
    h, w = grid.shape
    padded = np.zeros((30, 30), dtype=np.int64)
    padded[:h, :w] = grid
    return padded


def test_encoder_capacity():
    """Test if the encoder can learn to distinguish 3 distinct patterns."""
    print("="*80)
    print("TEST 1: Can the encoder distinguish 3 distinct 4x4 patterns?")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create patterns
    patterns = create_toy_patterns()
    print("\nPatterns to distinguish:")
    for i, p in enumerate(patterns):
        print(f"\nPattern {i}:")
        print(p)
    
    # Create dataset
    num_samples_per_class = 100
    X = []
    y = []
    
    for label, pattern in enumerate(patterns):
        for _ in range(num_samples_per_class):
            padded = pad_to_30x30(pattern)
            X.append(torch.from_numpy(padded).long())
            y.append(label)
    
    X = torch.stack(X)
    y = torch.tensor(y, dtype=torch.long)
    
    # Shuffle
    perm = torch.randperm(len(X))
    X = X[perm]
    y = y[perm]
    
    # Split train/val
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"\nDataset: {len(X_train)} train, {len(X_val)} val samples")
    print(f"Classes: {len(patterns)}")
    print(f"Random chance accuracy: {100.0/len(patterns):.2f}%")
    
    # Create encoder + classifier
    encoder = ARCEncoder(
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_conv_layers=config.NUM_CONV_LAYERS if hasattr(config, 'NUM_CONV_LAYERS') else 3
    ).to(device)
    
    classifier = nn.Sequential(
        nn.Linear(config.LATENT_DIM, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, len(patterns))
    ).to(device)
    
    model = nn.Sequential(encoder, classifier)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    batch_size = 32
    num_epochs = 50
    
    train_accs = []
    val_accs = []
    train_losses = []
    
    print("\nTraining...")
    for epoch in range(num_epochs):
        model.train()
        
        # Shuffle training data
        perm = torch.randperm(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]
        
        epoch_loss = 0
        correct = 0
        total = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += len(batch_y)
        
        train_acc = 100.0 * correct / total
        avg_loss = epoch_loss / (len(X_train) // batch_size)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            val_pred = val_logits.argmax(dim=1)
            val_acc = 100.0 * (val_pred == y_val.to(device)).sum().item() / len(y_val)
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
    
    print(f"\nFinal Val Accuracy: {val_accs[-1]:.2f}%")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    ax2.plot(train_accs, label='Train')
    ax2.plot(val_accs, label='Val')
    ax2.axhline(y=100.0/len(patterns), color='r', linestyle='--', label='Random')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Over Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('debug_test1_results.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot to debug_test1_results.png")
    
    return val_accs[-1] > 90.0  # Success if >90% accuracy


def test_feature_extraction():
    """Test what features the encoder extracts from the patterns."""
    print("\n" + "="*80)
    print("TEST 2: What features does the encoder extract?")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    patterns = create_toy_patterns()
    
    encoder = ARCEncoder(
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_conv_layers=config.NUM_CONV_LAYERS if hasattr(config, 'NUM_CONV_LAYERS') else 3
    ).to(device)
    
    # Extract features
    encoder.eval()
    with torch.no_grad():
        features = []
        for pattern in patterns:
            padded = pad_to_30x30(pattern)
            x = torch.from_numpy(padded).long().unsqueeze(0).to(device)
            feat = encoder(x)
            features.append(feat.cpu().numpy().flatten())
    
    features = np.array(features)
    
    print(f"\nFeature shape: {features.shape}")
    print(f"Feature mean: {features.mean():.4f}")
    print(f"Feature std: {features.std():.4f}")
    print(f"Feature min: {features.min():.4f}")
    print(f"Feature max: {features.max():.4f}")
    
    # Compute pairwise distances
    print("\nPairwise distances between pattern features:")
    for i in range(len(patterns)):
        for j in range(i+1, len(patterns)):
            dist = np.linalg.norm(features[i] - features[j])
            print(f"  Pattern {i} <-> Pattern {j}: {dist:.4f}")
    
    # Check if features are actually different
    same_features = True
    for i in range(len(patterns)):
        for j in range(i+1, len(patterns)):
            if not np.allclose(features[i], features[j], rtol=1e-3):
                same_features = False
                break
    
    if same_features:
        print("\n⚠️  WARNING: All patterns produce nearly identical features!")
        print("This means the encoder is not learning meaningful representations.")
    else:
        print("\n✓ Features are different between patterns (good!)")
    
    return not same_features


def test_gradient_flow():
    """Test if gradients are flowing through the network."""
    print("\n" + "="*80)
    print("TEST 3: Are gradients flowing through the encoder?")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = ARCEncoder(
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_conv_layers=config.NUM_CONV_LAYERS if hasattr(config, 'NUM_CONV_LAYERS') else 3
    ).to(device)
    
    classifier = nn.Linear(config.LATENT_DIM, 3).to(device)
    
    # Create a batch
    patterns = create_toy_patterns()
    X = torch.stack([torch.from_numpy(pad_to_30x30(p)).long() for p in patterns]).to(device)
    y = torch.tensor([0, 1, 2], dtype=torch.long).to(device)
    
    # Forward + backward
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=1e-3)
    optimizer.zero_grad()
    
    features = encoder(X)
    logits = classifier(features)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    
    print(f"\nLoss: {loss.item():.4f}")
    
    # Check gradients
    print("\nGradient statistics by layer:")
    has_gradients = True
    for name, param in encoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            print(f"  {name:30s}: norm={grad_norm:.6f}, mean={grad_mean:.6f}, std={grad_std:.6f}")
            
            if grad_norm < 1e-7:
                print(f"    ⚠️  WARNING: Very small gradient!")
                has_gradients = False
        else:
            print(f"  {name:30s}: NO GRADIENT")
            has_gradients = False
    
    if has_gradients:
        print("\n✓ Gradients are flowing (good!)")
    else:
        print("\n⚠️  WARNING: Some layers have no or very small gradients!")
    
    return has_gradients


def test_batch_norm_issue():
    """Test if batch norm is causing issues with small batches."""
    print("\n" + "="*80)
    print("TEST 4: Is batch normalization causing issues?")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    patterns = create_toy_patterns()
    X = torch.stack([torch.from_numpy(pad_to_30x30(p)).long() for p in patterns]).to(device)
    
    encoder = ARCEncoder(
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_conv_layers=config.NUM_CONV_LAYERS if hasattr(config, 'NUM_CONV_LAYERS') else 3
    ).to(device)
    
    print("\nTesting with batch size 3 (small batch)...")
    encoder.train()
    with torch.no_grad():
        features_train = encoder(X)
    
    encoder.eval()
    with torch.no_grad():
        features_eval = encoder(X)
    
    diff = (features_train - features_eval).abs().mean().item()
    print(f"Mean difference between train and eval mode: {diff:.6f}")
    
    if diff > 0.1:
        print("⚠️  WARNING: Large difference between train/eval mode!")
        print("This suggests batch norm might be unstable with small batches.")
        print("Solution: Use larger batch sizes or replace BatchNorm with LayerNorm/GroupNorm")
        return False
    else:
        print("✓ Train/eval difference is small (good!)")
        return True


def test_pooling_issue():
    """Test if adaptive pooling is destroying the signal for small grids."""
    print("\n" + "="*80)
    print("TEST 5: Is adaptive pooling destroying the signal?")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    patterns = create_toy_patterns()
    
    encoder = ARCEncoder(
        num_colors=config.NUM_COLORS,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_conv_layers=config.NUM_CONV_LAYERS if hasattr(config, 'NUM_CONV_LAYERS') else 3
    ).to(device)
    
    encoder.eval()
    
    print("\nExtracting feature maps for each pattern...")
    
    for idx, pattern in enumerate(patterns):
        padded = pad_to_30x30(pattern)
        x = torch.from_numpy(padded).long().unsqueeze(0).to(device)
        
        # Get intermediate feature maps
        feats = encoder.extract_feature_maps(x)
        
        print(f"\nPattern {idx}:")
        print("  After conv layers (before pooling):")
        
        # Check the actual content region vs padding region
        conv3 = feats['conv3'][0]  # [hidden_dim, 30, 30]
        
        content_region = conv3[:, :4, :4]  # actual pattern
        padding_region = conv3[:, 10:20, 10:20]  # pure padding
        
        content_activation = content_region.abs().mean().item()
        padding_activation = padding_region.abs().mean().item()
        
        print(f"    Content region (0:4, 0:4) mean activation: {content_activation:.4f}")
        print(f"    Padding region (10:20, 10:20) mean activation: {padding_activation:.4f}")
        print(f"    Ratio (content/padding): {content_activation/max(padding_activation, 1e-6):.2f}x")
        
        # Check pooled features
        pooled = feats['pooled'][0]  # [hidden_dim, 4, 4]
        print(f"  After pooling to 4x4:")
        print(f"    Mean activation: {pooled.abs().mean().item():.4f}")
        print(f"    Std activation: {pooled.std().item():.4f}")
    
    print("\n" + "="*80)
    print("ANALYSIS:")
    print("If the content/padding ratio is close to 1.0, the network is not")
    print("distinguishing between actual pattern and padding - THIS IS BAD.")
    print("The ratio should be >>1.0 for the network to focus on actual content.")
    print("="*80)


def main():
    """Run all diagnostic tests."""
    print("\n" + "="*80)
    print("ENCODER DIAGNOSTIC SUITE")
    print("="*80)
    print("\nThis will run a series of tests to figure out why the encoder")
    print("can't learn to distinguish simple patterns.")
    
    # Run tests
    results = {}
    
    results['capacity'] = test_encoder_capacity()
    results['features'] = test_feature_extraction()
    results['gradients'] = test_gradient_flow()
    results['batch_norm'] = test_batch_norm_issue()
    test_pooling_issue()  # Just informational
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*80)
    
    if not results['capacity']:
        print("\n🔴 The encoder CANNOT learn to distinguish 3 simple patterns.")
        print("This is a fundamental problem that needs to be fixed.")
        
        if not results['features']:
            print("\n💡 Root cause: The encoder produces nearly identical features")
            print("   for different patterns. Possible causes:")
            print("   - Adaptive pooling is averaging away the signal")
            print("   - Network is too simple/not expressive enough")
            print("   - Padding is overwhelming the actual content")
        
        if not results['gradients']:
            print("\n💡 Root cause: Gradients are not flowing properly.")
            print("   Check for dead ReLUs or vanishing gradients.")
        
        if not results['batch_norm']:
            print("\n💡 Contributing issue: Batch normalization is unstable")
            print("   with small batches. Try using larger batches or")
            print("   replace BatchNorm with GroupNorm/LayerNorm.")
    else:
        print("\n🟢 The encoder CAN learn! Check the earlier test results")
        print("   to see what might be different in your pretraining setup.")


if __name__ == '__main__':
    main()