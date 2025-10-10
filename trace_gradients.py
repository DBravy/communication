"""Detailed gradient flow tracing to find the exact break point."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from model import ARCEncoder, ARCAutoencoder

device = torch.device('cpu')

print("Creating minimal test case...")

# Create minimal model
encoder = ARCEncoder(
    num_colors=10,
    embedding_dim=10,
    hidden_dim=128,
    latent_dim=128,
    num_conv_layers=3
)

model = ARCAutoencoder(
    encoder=encoder,
    vocab_size=100,
    max_length=3,
    num_colors=10,
    embedding_dim=10,
    hidden_dim=128,
    max_grid_size=30,
    bottleneck_type='communication',
    task_type='selection',
    num_conv_layers=2,
    num_classes=None
).to(device)

model.train()

# Create fake data
grids = torch.randint(0, 10, (2, 30, 30))
sizes = [(15, 15), (20, 20)]
candidates = torch.randint(0, 10, (2, 30, 30))
candidate_sizes = [(15, 15), (15, 15)]
target_idx = 0

print("\n" + "="*80)
print("STEP 1: Test sender in isolation")
print("="*80)

grids.requires_grad_(False)  # Grids are discrete, no gradients
single_grid = grids[0:1]
single_size = [sizes[0]]

# Forward through sender
messages, soft_messages = model.sender(single_grid, sizes=single_size, temperature=1.0)

print(f"Message shape: {messages.shape}")
print(f"Soft message shape: {soft_messages.shape}")
print(f"Message: {messages[0].tolist()}")
print(f"Soft message requires_grad: {soft_messages.requires_grad}")
print(f"Soft message grad_fn: {soft_messages.grad_fn}")

# Create a dummy loss from soft messages
dummy_loss = soft_messages.sum()
dummy_loss.backward()

# Check if sender got gradients
has_sender_grads = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in model.sender.parameters())
print(f"\nSender received gradients from direct loss: {has_sender_grads}")

if has_sender_grads:
    for name, param in model.sender.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                print(f"  {name}: {grad_norm:.6f}")
                break

model.zero_grad()

print("\n" + "="*80)
print("STEP 2: Test receiver with soft messages")
print("="*80)

# Forward through sender again (fresh)
messages, soft_messages = model.sender(single_grid, sizes=single_size, temperature=1.0)

print(f"Soft message requires_grad: {soft_messages.requires_grad}")
print(f"Soft message is_leaf: {soft_messages.is_leaf}")

# Forward through receiver with soft message
logits = model.receiver(
    messages,
    candidates,
    candidate_sizes=candidate_sizes,
    soft_message=soft_messages
)

print(f"\nReceiver output shape: {logits.shape}")
print(f"Receiver output requires_grad: {logits.requires_grad}")
print(f"Receiver output grad_fn: {logits.grad_fn}")

# Create loss from receiver output
criterion = nn.CrossEntropyLoss()
target = torch.tensor([target_idx])
loss = criterion(logits.unsqueeze(0), target)

print(f"\nLoss: {loss.item():.6f}")
print(f"Loss requires_grad: {loss.requires_grad}")
print(f"Loss grad_fn: {loss.grad_fn}")

# Backward
print("\nRunning backward pass...")
loss.backward()

# Check receiver gradients
print("\nReceiver gradients:")
for name, param in model.receiver.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 0:
            print(f"  {name}: {grad_norm:.6f}")
        if name == 'continuous_proj.weight':
            print(f"    continuous_proj.weight grad (CRITICAL): {grad_norm:.6f}")

# Check if soft_messages got gradients
if soft_messages.grad is not None:
    print(f"\nsoft_messages.grad: {soft_messages.grad.norm().item():.6f}")
else:
    print(f"\nsoft_messages.grad: None (PROBLEM!)")

# Check sender gradients
print("\nSender gradients:")
sender_grad_count = 0
for name, param in model.sender.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 0:
            sender_grad_count += 1
            if sender_grad_count <= 5:
                print(f"  {name}: {grad_norm:.6f}")

if sender_grad_count == 0:
    print("  NO GRADIENTS IN SENDER!")

model.zero_grad()

print("\n" + "="*80)
print("STEP 3: Test with manual soft message")
print("="*80)

# Create a manual soft message with requires_grad
manual_soft = torch.randn(1, 3, 100, requires_grad=True)
print(f"Manual soft message requires_grad: {manual_soft.requires_grad}")

# Get discrete version for receiver
manual_discrete = manual_soft.argmax(dim=-1)
print(f"Manual discrete message: {manual_discrete[0].tolist()}")

# Pass through receiver
logits2 = model.receiver(
    manual_discrete,
    candidates,
    candidate_sizes=candidate_sizes,
    soft_message=manual_soft
)

loss2 = criterion(logits2.unsqueeze(0), target)
print(f"\nLoss with manual soft message: {loss2.item():.6f}")

loss2.backward()

if manual_soft.grad is not None:
    print(f"Manual soft message grad: {manual_soft.grad.norm().item():.6f}")
    print("SUCCESS: Gradients flow to manually created soft message!")
else:
    print("FAIL: No gradients even to manual soft message!")

print("\n" + "="*80)
print("STEP 4: Check if sender's soft_messages computation graph is intact")
print("="*80)

model.zero_grad()

# Get soft messages again
messages3, soft_messages3 = model.sender(single_grid, sizes=single_size, temperature=1.0)

# Manually trace the computation graph
print(f"soft_messages3.requires_grad: {soft_messages3.requires_grad}")
print(f"soft_messages3.grad_fn: {soft_messages3.grad_fn}")

if soft_messages3.grad_fn is not None:
    print(f"soft_messages3.grad_fn.next_functions: {soft_messages3.grad_fn.next_functions[0]}")

# Try to backward directly from soft messages
test_loss = soft_messages3.sum()
test_loss.backward()

vocab_proj_grad = model.sender.vocab_proj.weight.grad
if vocab_proj_grad is not None:
    print(f"\nvocab_proj.weight grad from direct backward: {vocab_proj_grad.norm().item():.6f}")
else:
    print(f"\nvocab_proj.weight grad from direct backward: None")

print("\n" + "="*80)
print("STEP 5: Test full forward pass")
print("="*80)

model.zero_grad()

# Full forward pass through model
selection_logits_list, _, msg = model(
    grids[0:1],
    [sizes[0]],
    temperature=1.0,
    candidates_list=[candidates],
    candidates_sizes_list=[candidate_sizes],
    target_indices=torch.tensor([target_idx])
)

full_loss = criterion(selection_logits_list[0].unsqueeze(0), torch.tensor([target_idx]))
print(f"Full model loss: {full_loss.item():.6f}")

full_loss.backward()

print("\nGradients after full model backward:")
sender_has_grads = False
for name, param in model.sender.named_parameters():
    if param.grad is not None and param.grad.abs().sum() > 0:
        print(f"  {name}: {param.grad.norm().item():.6f}")
        sender_has_grads = True
        break

if not sender_has_grads:
    print("  NO SENDER GRADIENTS IN FULL MODEL!")
    
    # Check receiver
    print("\nBut receiver has gradients:")
    for name, param in model.receiver.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            print(f"  {name}: {param.grad.norm().item():.6f}")
            break

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

print("""
The gradient flow is breaking somewhere between the sender's soft_messages
output and the receiver's use of them. Likely causes:

1. The soft_messages tensor is being detached somewhere
2. The receiver isn't actually using soft_messages in training mode
3. There's a no-op operation creating a graph break
4. The batching loop is causing issues

Check the ARCAutoencoder.forward() method carefully.
""")