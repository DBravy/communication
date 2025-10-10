"""FIXED version of ReceiverSelector that avoids gradient cancellation.

The key fix: Instead of expanding msg_repr to all candidates and concatenating,
we compute a similarity score between the message and each candidate separately.
This prevents gradient cancellation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReceiverSelectorFixed(nn.Module):
    """
    FIXED: Receiver selector that avoids gradient cancellation.
    
    Key change: Uses dot-product similarity instead of concatenation,
    which prevents CrossEntropyLoss gradients from canceling out.
    """
    def __init__(self, vocab_size, num_colors, embedding_dim, hidden_dim, num_conv_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_colors = num_colors
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        
        assert embedding_dim == num_colors
        assert num_conv_layers >= 1
        
        self.symbol_embed = nn.Embedding(vocab_size, hidden_dim)
        self.continuous_proj = nn.Linear(vocab_size, hidden_dim)
        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Candidate encoder
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for i in range(num_conv_layers):
            in_channels = embedding_dim if i == 0 else hidden_dim
            self.conv_layers.append(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
            )
            self.bn_layers.append(nn.BatchNorm2d(hidden_dim))
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.grid_fc = nn.Linear(hidden_dim * 4 * 4, hidden_dim)
        
        # FIXED: Instead of concatenating and using a linear layer,
        # we'll use separate projections and compute similarity
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cand_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Optional: Add a learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1))
        
        self.relu = nn.ReLU()
        
    def encode_candidates(self, candidates, candidate_sizes=None):
        N, H, W = candidates.shape
        
        x = F.one_hot(candidates.long(), num_classes=self.num_colors).float()
        x = x.permute(0, 3, 1, 2)
        
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.relu(bn(conv(x)))
        
        if candidate_sizes is not None:
            pooled_features = []
            for i in range(N):
                h, w = candidate_sizes[i]
                content = x[i:i+1, :, :h, :w]
                pooled = self.adaptive_pool(content)
                pooled_features.append(pooled)
            x = torch.cat(pooled_features, dim=0)
        else:
            x = self.adaptive_pool(x)
        
        x = x.reshape(N, -1)
        x = self.relu(self.grid_fc(x))
        
        return x
    
    def forward(self, message, candidates, candidate_sizes=None, soft_message=None):
        """
        FIXED: Computes similarity scores instead of concatenating features.
        
        This prevents gradient cancellation because the message representation
        is not directly shared across all candidates.
        """
        if soft_message is not None and self.training:
            msg_emb = self.continuous_proj(soft_message)
        else:
            msg_emb = self.symbol_embed(message)
        
        _, (h, _) = self.lstm(msg_emb)
        msg_repr = h.squeeze(0)  # [1, hidden_dim]
        
        # Project message to comparison space
        msg_proj = self.msg_proj(msg_repr)  # [1, hidden_dim]
        
        # Encode and project candidates
        cand_repr = self.encode_candidates(candidates, candidate_sizes=candidate_sizes)  # [N, hidden_dim]
        cand_proj = self.cand_proj(cand_repr)  # [N, hidden_dim]
        
        # FIXED: Compute similarity using dot product
        # This prevents gradient cancellation because we're not using expand()
        logits = torch.matmul(cand_proj, msg_proj.squeeze(0)) / self.temperature
        
        return logits


class ReceiverSelectorBilinear(nn.Module):
    """
    Alternative fix using bilinear layer.
    
    This is more expressive than dot product but still avoids gradient cancellation.
    """
    def __init__(self, vocab_size, num_colors, embedding_dim, hidden_dim, num_conv_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_colors = num_colors
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        
        assert embedding_dim == num_colors
        assert num_conv_layers >= 1
        
        self.symbol_embed = nn.Embedding(vocab_size, hidden_dim)
        self.continuous_proj = nn.Linear(vocab_size, hidden_dim)
        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Candidate encoder
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for i in range(num_conv_layers):
            in_channels = embedding_dim if i == 0 else hidden_dim
            self.conv_layers.append(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
            )
            self.bn_layers.append(nn.BatchNorm2d(hidden_dim))
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.grid_fc = nn.Linear(hidden_dim * 4 * 4, hidden_dim)
        
        # FIXED: Use bilinear layer for scoring
        # This computes: score = candidate^T * W * message
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)
        
        self.relu = nn.ReLU()
        
    def encode_candidates(self, candidates, candidate_sizes=None):
        N, H, W = candidates.shape
        
        x = F.one_hot(candidates.long(), num_classes=self.num_colors).float()
        x = x.permute(0, 3, 1, 2)
        
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.relu(bn(conv(x)))
        
        if candidate_sizes is not None:
            pooled_features = []
            for i in range(N):
                h, w = candidate_sizes[i]
                content = x[i:i+1, :, :h, :w]
                pooled = self.adaptive_pool(content)
                pooled_features.append(pooled)
            x = torch.cat(pooled_features, dim=0)
        else:
            x = self.adaptive_pool(x)
        
        x = x.reshape(N, -1)
        x = self.relu(self.grid_fc(x))
        
        return x
    
    def forward(self, message, candidates, candidate_sizes=None, soft_message=None):
        """
        FIXED: Uses bilinear layer for scoring to avoid gradient cancellation.
        """
        if soft_message is not None and self.training:
            msg_emb = self.continuous_proj(soft_message)
        else:
            msg_emb = self.symbol_embed(message)
        
        _, (h, _) = self.lstm(msg_emb)
        msg_repr = h.squeeze(0)  # [1, hidden_dim]
        
        # Encode candidates
        cand_repr = self.encode_candidates(candidates, candidate_sizes=candidate_sizes)  # [N, hidden_dim]
        
        # FIXED: Compute scores using bilinear layer
        # Expand message to match number of candidates for bilinear input
        num_candidates = cand_repr.shape[0]
        msg_expanded = msg_repr.expand(num_candidates, -1)  # [N, hidden_dim]
        
        # Bilinear doesn't suffer from gradient cancellation because it computes:
        # score_i = cand_i^T * W * msg
        # The gradient w.r.t. msg is: sum_i (grad_score_i * cand_i^T * W)
        # This doesn't cancel out because cand_i is different for each i
        logits = self.bilinear(cand_repr, msg_expanded).squeeze(-1)
        
        return logits


class ReceiverSelectorAttention(nn.Module):
    """
    Alternative fix using attention mechanism.
    
    This treats the message as a query and candidates as keys/values.
    """
    def __init__(self, vocab_size, num_colors, embedding_dim, hidden_dim, num_conv_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_colors = num_colors
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        
        assert embedding_dim == num_colors
        assert num_conv_layers >= 1
        
        self.symbol_embed = nn.Embedding(vocab_size, hidden_dim)
        self.continuous_proj = nn.Linear(vocab_size, hidden_dim)
        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Candidate encoder
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for i in range(num_conv_layers):
            in_channels = embedding_dim if i == 0 else hidden_dim
            self.conv_layers.append(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
            )
            self.bn_layers.append(nn.BatchNorm2d(hidden_dim))
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.grid_fc = nn.Linear(hidden_dim * 4 * 4, hidden_dim)
        
        # FIXED: Attention mechanism
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.relu = nn.ReLU()
        
    def encode_candidates(self, candidates, candidate_sizes=None):
        N, H, W = candidates.shape
        
        x = F.one_hot(candidates.long(), num_classes=self.num_colors).float()
        x = x.permute(0, 3, 1, 2)
        
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.relu(bn(conv(x)))
        
        if candidate_sizes is not None:
            pooled_features = []
            for i in range(N):
                h, w = candidate_sizes[i]
                content = x[i:i+1, :, :h, :w]
                pooled = self.adaptive_pool(content)
                pooled_features.append(pooled)
            x = torch.cat(pooled_features, dim=0)
        else:
            x = self.adaptive_pool(x)
        
        x = x.reshape(N, -1)
        x = self.relu(self.grid_fc(x))
        
        return x
    
    def forward(self, message, candidates, candidate_sizes=None, soft_message=None):
        """
        FIXED: Uses attention mechanism to avoid gradient cancellation.
        """
        if soft_message is not None and self.training:
            msg_emb = self.continuous_proj(soft_message)
        else:
            msg_emb = self.symbol_embed(message)
        
        _, (h, _) = self.lstm(msg_emb)
        msg_repr = h.squeeze(0)  # [1, hidden_dim]
        
        # Encode candidates
        cand_repr = self.encode_candidates(candidates, candidate_sizes=candidate_sizes)  # [N, hidden_dim]
        
        # FIXED: Attention-based scoring
        query = self.query_proj(msg_repr)  # [1, hidden_dim]
        keys = self.key_proj(cand_repr)  # [N, hidden_dim]
        
        # Compute attention scores (scaled dot-product)
        logits = torch.matmul(keys, query.transpose(0, 1)).squeeze(-1) / (self.hidden_dim ** 0.5)
        
        return logits


# Test the fix
if __name__ == '__main__':
    print("="*80)
    print("Testing fixed ReceiverSelector")
    print("="*80)
    
    # Create fixed model
    receiver_fixed = ReceiverSelectorFixed(
        vocab_size=100,
        num_colors=10,
        embedding_dim=10,
        hidden_dim=128,
        num_conv_layers=2
    )
    
    receiver_fixed.train()
    
    # Create test data
    soft_msg = torch.randn(1, 3, 100, requires_grad=True)
    discrete_msg = soft_msg.argmax(dim=-1)
    candidates = torch.randint(0, 10, (2, 30, 30))
    candidate_sizes = [(15, 15), (15, 15)]
    
    # Forward pass
    logits = receiver_fixed(
        discrete_msg,
        candidates,
        candidate_sizes=candidate_sizes,
        soft_message=soft_msg
    )
    
    print(f"Logits: {logits.detach().numpy()}")
    
    # Loss and backward
    criterion = nn.CrossEntropyLoss()
    target = torch.tensor([0])
    loss = criterion(logits.unsqueeze(0), target)
    
    print(f"Loss: {loss.item():.6f}")
    
    loss.backward()
    
    # Check gradients
    print(f"\nsoft_msg.grad norm: {soft_msg.grad.norm().item():.6f}")
    
    print("\nReceiver parameter gradients:")
    for name, param in receiver_fixed.named_parameters():
        if param.grad is not None and param.grad.norm().item() > 0:
            print(f"  {name}: {param.grad.norm().item():.6f}")
    
    if soft_msg.grad.norm().item() > 0:
        print("\nSUCCESS: Gradients flow to soft_msg!")
    else:
        print("\nFAILED: Still no gradients")