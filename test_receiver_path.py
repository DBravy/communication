"""Test which path the receiver actually takes."""

import torch
import torch.nn as nn
from model import ARCEncoder, ARCAutoencoder

device = torch.device('cpu')

print("="*80)
print("TEST: Which path does receiver take?")
print("="*80)

# Create model
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

# Set to training mode
model.train()
print(f"\nModel training mode: {model.training}")
print(f"Receiver training mode: {model.receiver.training}")

# Create test data
grids = torch.randint(0, 10, (1, 30, 30))
sizes = [(15, 15)]
candidates = torch.randint(0, 10, (2, 30, 30))
candidate_sizes = [(15, 15), (15, 15)]

# Get messages from sender
messages, soft_messages = model.sender(grids, sizes=sizes, temperature=1.0)

print(f"\nMessages shape: {messages.shape}")
print(f"Soft messages shape: {soft_messages.shape}")
print(f"Soft messages is None: {soft_messages is None}")
print(f"Soft messages requires_grad: {soft_messages.requires_grad}")

# Add hooks to track which layer is called
continuous_proj_called = [False]
symbol_embed_called = [False]

def continuous_proj_hook(module, input, output):
    continuous_proj_called[0] = True
    print("\n>>> continuous_proj was called! <<<")

def symbol_embed_hook(module, input, output):
    symbol_embed_called[0] = True
    print("\n>>> symbol_embed was called! <<<")

# Register hooks
cont_hook = model.receiver.continuous_proj.register_forward_hook(continuous_proj_hook)
emb_hook = model.receiver.symbol_embed.register_forward_hook(symbol_embed_hook)

# Call receiver
print("\n" + "="*80)
print("Calling receiver with soft_message...")
print("="*80)

logits = model.receiver(
    messages,
    candidates,
    candidate_sizes=candidate_sizes,
    soft_message=soft_messages
)

print(f"\nReceiver output shape: {logits.shape}")
print(f"\nWhich layer was called?")
print(f"  continuous_proj: {continuous_proj_called[0]}")
print(f"  symbol_embed: {symbol_embed_called[0]}")

# Remove hooks
cont_hook.remove()
emb_hook.remove()

# Now test the condition manually
print("\n" + "="*80)
print("Manual condition check")
print("="*80)

print(f"soft_message is not None: {soft_messages is not None}")
print(f"self.training: {model.receiver.training}")
print(f"Condition (soft_message is not None and self.training): {soft_messages is not None and model.receiver.training}")

# Test what happens in eval mode
print("\n" + "="*80)
print("Testing in EVAL mode")
print("="*80)

model.eval()
continuous_proj_called = [False]
symbol_embed_called = [False]

cont_hook = model.receiver.continuous_proj.register_forward_hook(continuous_proj_hook)
emb_hook = model.receiver.symbol_embed.register_forward_hook(symbol_embed_hook)

print(f"Model training mode: {model.training}")
print(f"Receiver training mode: {model.receiver.training}")

logits_eval = model.receiver(
    messages,
    candidates,
    candidate_sizes=candidate_sizes,
    soft_message=soft_messages
)

print(f"\nWhich layer was called in eval mode?")
print(f"  continuous_proj: {continuous_proj_called[0]}")
print(f"  symbol_embed: {symbol_embed_called[0]}")

cont_hook.remove()
emb_hook.remove()

# Now let's manually trace through the receiver code
print("\n" + "="*80)
print("Manual trace through receiver.forward()")
print("="*80)

model.train()

# Recreate the exact condition from the receiver
print("\nInside receiver.forward():")
print(f"  message: {messages}")
print(f"  soft_message: {soft_messages.shape if soft_messages is not None else None}")
print(f"  self.training: {model.receiver.training}")

if soft_messages is not None and model.receiver.training:
    print("\n  Taking CONTINUOUS_PROJ path")
    msg_emb = model.receiver.continuous_proj(soft_messages)
else:
    print("\n  Taking SYMBOL_EMBED path")
    msg_emb = model.receiver.symbol_embed(messages)

print(f"  msg_emb shape: {msg_emb.shape}")

# Test if the issue is with how we're passing soft_message in the full forward
print("\n" + "="*80)
print("Test full model forward")
print("="*80)

model.train()
print(f"Model training mode before forward: {model.training}")

continuous_proj_called = [False]
symbol_embed_called = [False]

cont_hook = model.receiver.continuous_proj.register_forward_hook(continuous_proj_hook)
emb_hook = model.receiver.symbol_embed.register_forward_hook(symbol_embed_hook)

# Full forward pass
selection_logits_list, _, msg = model(
    grids,
    sizes,
    temperature=1.0,
    candidates_list=[candidates],
    candidates_sizes_list=[candidate_sizes],
    target_indices=torch.tensor([0])
)

print(f"\nFull forward complete")
print(f"Which layer was called during full forward?")
print(f"  continuous_proj: {continuous_proj_called[0]}")
print(f"  symbol_embed: {symbol_embed_called[0]}")

cont_hook.remove()
emb_hook.remove()

# Let's check if model.receiver.training is True during the full forward
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if not continuous_proj_called[0]:
    print("""
PROBLEM FOUND: The receiver is NOT using continuous_proj!

This means one of these is happening:
1. soft_message is None when passed to receiver
2. model.receiver.training is False
3. The receiver.forward() method has a bug in the condition

Need to check the ARCAutoencoder.forward() method to see how it calls
the receiver.
    """)
else:
    print("continuous_proj IS being called. The problem is elsewhere.")