"""Test if the message actually affects the receiver's output."""

import torch
import torch.nn as nn
from model import ARCEncoder, ARCAutoencoder

device = torch.device('cpu')

print("="*80)
print("TEST: Does the message affect receiver output?")
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

model.train()

# Create test data
grids = torch.randint(0, 10, (1, 30, 30))
sizes = [(15, 15)]
candidates = torch.randint(0, 10, (2, 30, 30))
candidate_sizes = [(15, 15), (15, 15)]

print("\nTest 1: Do different messages produce different outputs?")
print("-" * 80)

# Create two very different messages
message1 = torch.tensor([[0, 0, 0]], dtype=torch.long)
soft_message1 = torch.zeros(1, 3, 100)
soft_message1[:, :, 0] = 1.0  # One-hot for symbol 0

message2 = torch.tensor([[99, 99, 99]], dtype=torch.long)
soft_message2 = torch.zeros(1, 3, 100)
soft_message2[:, :, 99] = 1.0  # One-hot for symbol 99

# Get outputs for both messages
logits1 = model.receiver(message1, candidates, candidate_sizes=candidate_sizes, soft_message=soft_message1)
logits2 = model.receiver(message2, candidates, candidate_sizes=candidate_sizes, soft_message=soft_message2)

print(f"Message 1: {message1[0].tolist()}")
print(f"  Logits: {logits1.detach().numpy()}")
print(f"\nMessage 2: {message2[0].tolist()}")
print(f"  Logits: {logits2.detach().numpy()}")

diff = (logits1 - logits2).abs().max().item()
print(f"\nMax difference in logits: {diff:.6f}")

if diff < 1e-5:
    print("WARNING: Messages produce nearly identical outputs!")
else:
    print("OK: Messages produce different outputs")

print("\n" + "="*80)
print("Test 2: Trace message through receiver")
print("="*80)

# Let's manually trace what happens to the message
soft_msg = torch.randn(1, 3, 100, requires_grad=True)
discrete_msg = soft_msg.argmax(dim=-1)

print(f"Soft message shape: {soft_msg.shape}")
print(f"Soft message requires_grad: {soft_msg.requires_grad}")

# Step 1: continuous_proj
msg_emb = model.receiver.continuous_proj(soft_msg)
print(f"\nAfter continuous_proj:")
print(f"  Shape: {msg_emb.shape}")
print(f"  Requires_grad: {msg_emb.requires_grad}")
print(f"  Mean: {msg_emb.mean().item():.6f}")
print(f"  Std: {msg_emb.std().item():.6f}")

# Step 2: LSTM
lstm_out, (h, c) = model.receiver.lstm(msg_emb)
print(f"\nAfter LSTM:")
print(f"  lstm_out shape: {lstm_out.shape}")
print(f"  h shape: {h.shape}")
print(f"  h requires_grad: {h.requires_grad}")
print(f"  h mean: {h.mean().item():.6f}")
print(f"  h std: {h.std().item():.6f}")

msg_repr = h.squeeze(0)
print(f"\nmsg_repr:")
print(f"  Shape: {msg_repr.shape}")
print(f"  Mean: {msg_repr.mean().item():.6f}")
print(f"  Std: {msg_repr.std().item():.6f}")

# Step 3: Encode candidates
cand_repr = model.receiver.encode_candidates(candidates, candidate_sizes=candidate_sizes)
print(f"\ncand_repr:")
print(f"  Shape: {cand_repr.shape}")
print(f"  Mean: {cand_repr.mean().item():.6f}")
print(f"  Std: {cand_repr.std().item():.6f}")

# Step 4: Combine
num_candidates = candidates.shape[0]
msg_repr_expanded = msg_repr.expand(num_candidates, -1)
print(f"\nmsg_repr_expanded:")
print(f"  Shape: {msg_repr_expanded.shape}")

combined = torch.cat([msg_repr_expanded, cand_repr], dim=-1)
print(f"\ncombined:")
print(f"  Shape: {combined.shape}")
print(f"  Requires_grad: {combined.requires_grad}")

# Step 5: Score
logits = model.receiver.score_fc(combined).squeeze(-1)
print(f"\nlogits:")
print(f"  Shape: {logits.shape}")
print(f"  Values: {logits.detach().numpy()}")
print(f"  Requires_grad: {logits.requires_grad}")

# Step 6: Loss and backward
criterion = nn.CrossEntropyLoss()
target = torch.tensor([0])
loss = criterion(logits.unsqueeze(0), target)
print(f"\nLoss: {loss.item():.6f}")

# Backward
loss.backward()

# Check gradients
print("\n" + "="*80)
print("Gradients:")
print("="*80)

print(f"soft_msg.grad: {soft_msg.grad.norm().item():.6f}")
print(f"msg_emb.grad: {msg_emb.grad.norm().item() if msg_emb.grad is not None else 'None'}")

print("\nReceiver layer gradients:")
for name, param in model.receiver.named_parameters():
    if param.grad is not None and param.grad.norm().item() > 0:
        print(f"  {name}: {param.grad.norm().item():.6f}")

print("\n" + "="*80)
print("Test 3: Check if LSTM is the problem")
print("="*80)

# Reset
model.zero_grad()

# Test if LSTM produces meaningful output
test_input = torch.randn(1, 3, 128, requires_grad=True)
lstm_output, (h_test, c_test) = model.receiver.lstm(test_input)
h_test = h_test.squeeze(0)

# Create a dummy loss from LSTM output
dummy_loss = h_test.sum()
dummy_loss.backward()

print(f"LSTM input requires_grad: {test_input.requires_grad}")
print(f"LSTM output h requires_grad: {h_test.requires_grad}")
print(f"LSTM input gradient norm: {test_input.grad.norm().item():.6f}")

lstm_has_grads = any(p.grad is not None and p.grad.norm().item() > 0 
                     for p in model.receiver.lstm.parameters())
print(f"LSTM parameters got gradients: {lstm_has_grads}")

print("\n" + "="*80)
print("Test 4: Check score_fc weights")
print("="*80)

score_fc_weight = model.receiver.score_fc.weight
print(f"score_fc input dim: {score_fc_weight.shape[1]}")
print(f"  First half (message): {score_fc_weight[:, :128].abs().mean().item():.6f}")
print(f"  Second half (candidate): {score_fc_weight[:, 128:].abs().mean().item():.6f}")

# Check if the message part of score_fc is being used
print("\nDoes score_fc use both parts equally?")
msg_part_norm = score_fc_weight[:, :128].norm().item()
cand_part_norm = score_fc_weight[:, 128:].norm().item()
print(f"  Message part norm: {msg_part_norm:.6f}")
print(f"  Candidate part norm: {cand_part_norm:.6f}")
print(f"  Ratio (msg/cand): {msg_part_norm / cand_part_norm if cand_part_norm > 0 else 'inf':.6f}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if diff < 1e-5:
    print("""
CRITICAL ISSUE: Different messages produce identical outputs!

This means the receiver is completely ignoring the message. Possible causes:
1. LSTM is outputting zeros or constants
2. score_fc has learned to ignore the message features
3. Message representation has collapsed to a single value
    """)
elif msg_part_norm / cand_part_norm < 0.01:
    print("""
ISSUE: score_fc weights heavily favor candidate features over message features.

The network has learned to ignore messages and only use candidate features.
This can happen when:
1. Messages don't provide useful information initially
2. Candidate features alone are sufficient for the task
3. Learning rate or initialization issues
    """)
else:
    print("The message does affect the output. Problem must be elsewhere.")