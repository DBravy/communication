"""Test if gradients are canceling due to shared message representation."""

import torch
import torch.nn as nn

print("="*80)
print("TEST: Gradient cancellation in selection task")
print("="*80)

# Simulate the selection task structure
print("\nSimulating the receiver's selection mechanism...")

# Create a shared message representation (like msg_repr in the real code)
msg_repr = torch.randn(1, 128, requires_grad=True)
print(f"msg_repr shape: {msg_repr.shape}")
print(f"msg_repr requires_grad: {msg_repr.requires_grad}")

# Create candidate representations (different for each candidate)
cand_repr = torch.randn(2, 128, requires_grad=False)  # 2 candidates
print(f"cand_repr shape: {cand_repr.shape}")

# Expand message to match number of candidates
num_candidates = 2
msg_repr_expanded = msg_repr.expand(num_candidates, -1)
print(f"msg_repr_expanded shape: {msg_repr_expanded.shape}")

# Combine message and candidate features
combined = torch.cat([msg_repr_expanded, cand_repr], dim=-1)
print(f"combined shape: {combined.shape}")

# Score each candidate
score_fc = nn.Linear(256, 1)
logits = score_fc(combined).squeeze(-1)
print(f"logits shape: {logits.shape}")
print(f"logits values: {logits.detach().numpy()}")

# Compute loss (target is first candidate)
criterion = nn.CrossEntropyLoss()
target = torch.tensor([0])
loss = criterion(logits.unsqueeze(0), target)
print(f"\nLoss: {loss.item():.6f}")

# Backward
loss.backward()

# Check gradients
print("\n" + "="*80)
print("Gradients after backward:")
print("="*80)

print(f"\nmsg_repr.grad:")
print(f"  Norm: {msg_repr.grad.norm().item():.6f}")
print(f"  Values: {msg_repr.grad[0, :5].numpy()}")  # Show first 5 values

print(f"\nscore_fc.weight.grad[:, :5] (message part):")
print(f"  {score_fc.weight.grad[:, :5].numpy()}")

print(f"\nscore_fc.weight.grad[:, 128:133] (candidate part):")
print(f"  {score_fc.weight.grad[:, 128:133].numpy()}")

# Now let's manually compute what the gradient SHOULD be
print("\n" + "="*80)
print("Manual gradient computation:")
print("="*80)

# Get the loss gradient w.r.t. logits
probs = torch.softmax(logits, dim=0)
print(f"\nSoftmax probabilities: {probs.detach().numpy()}")

# For CrossEntropyLoss with target=0:
# d_loss/d_logit[0] = prob[0] - 1
# d_loss/d_logit[1] = prob[1] - 0
d_loss_d_logits = probs.detach().clone()
d_loss_d_logits[0] -= 1.0
print(f"d_loss/d_logits: {d_loss_d_logits.numpy()}")

# Now, logit[i] = score_fc(combined[i])
# Since combined[i] = [msg_repr, cand_repr[i]]
# d_logit[i]/d_msg_repr = score_fc.weight[:, :128]

score_weights_msg = score_fc.weight.data[:, :128]
print(f"\nScore weights for message part (first 5): {score_weights_msg[:, :5].numpy()}")

# The gradient for msg_repr should be:
# d_loss/d_msg_repr = sum_i (d_loss/d_logit[i] * d_logit[i]/d_msg_repr)

expected_grad_contribution_0 = d_loss_d_logits[0].item() * score_weights_msg
expected_grad_contribution_1 = d_loss_d_logits[1].item() * score_weights_msg
expected_total_grad = expected_grad_contribution_0 + expected_grad_contribution_1

print(f"\nExpected gradient contributions:")
print(f"  From candidate 0: {expected_grad_contribution_0[0, :5].numpy()}")
print(f"  From candidate 1: {expected_grad_contribution_1[0, :5].numpy()}")
print(f"  Total: {expected_total_grad[0, :5].numpy()}")
print(f"  Total norm: {expected_total_grad.norm().item():.6f}")

# Compare with actual
actual_grad_norm = msg_repr.grad.norm().item()
expected_grad_norm = expected_total_grad.norm().item()

print(f"\nActual gradient norm: {actual_grad_norm:.6f}")
print(f"Expected gradient norm: {expected_grad_norm:.6f}")
print(f"Difference: {abs(actual_grad_norm - expected_grad_norm):.9f}")

print("\n" + "="*80)
print("Test with more candidates:")
print("="*80)

# Test with more candidates to see if gradients still cancel
msg_repr2 = torch.randn(1, 128, requires_grad=True)
cand_repr2 = torch.randn(5, 128, requires_grad=False)  # 5 candidates

msg_expanded2 = msg_repr2.expand(5, -1)
combined2 = torch.cat([msg_expanded2, cand_repr2], dim=-1)
logits2 = score_fc(combined2).squeeze(-1)

target2 = torch.tensor([2])  # Middle candidate
loss2 = criterion(logits2.unsqueeze(0), target2)
loss2.backward()

print(f"With 5 candidates, target=2:")
print(f"  msg_repr2.grad norm: {msg_repr2.grad.norm().item():.6f}")

probs2 = torch.softmax(logits2, dim=0)
print(f"  Probabilities: {probs2.detach().numpy()}")

# Calculate expected gradient cancellation
d_loss_d_logits2 = probs2.detach().clone()
d_loss_d_logits2[2] -= 1.0  # Target is index 2
print(f"  d_loss/d_logits: {d_loss_d_logits2.numpy()}")
print(f"  Sum of d_loss/d_logits: {d_loss_d_logits2.sum().item():.9f}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if actual_grad_norm < 0.01:
    print("""
CRITICAL BUG FOUND: Gradient cancellation in selection task!

The problem: When the same message representation is used for all candidates,
the gradients from CrossEntropyLoss partially cancel out.

Why this happens:
1. msg_repr is shared across all candidates via expand()
2. CrossEntropyLoss gives opposite-sign gradients to correct vs incorrect choices
3. Since all candidates use the same msg_repr, gradients accumulate:
   d_loss/d_msg_repr = sum of gradients from all candidates
   
If the task is balanced (roughly equal probability to each candidate),
the positive and negative gradients can nearly cancel out, leading to
very weak or zero gradients for the message path.

This is why:
- Receiver's candidate-encoding layers get strong gradients (unique per candidate)
- Receiver's message-processing layers get near-zero gradients (shared across all)
- Sender never learns (no gradient signal flows back)

SOLUTION: The message representation should NOT be shared via expand().
Instead, each candidate should get its own forward pass, or the architecture
needs to be redesigned for the selection task.
    """)
else:
    print(f"""
Gradients are not completely canceling (norm: {actual_grad_norm:.6f}).

The issue might be more subtle, such as:
1. Very small gradients that effectively don't train
2. Gradient flow blocked elsewhere in the network
3. Initialization or numerical stability issues
    """)