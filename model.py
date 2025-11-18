"""Communication model for ARC grids - FIXED VERSION with gradient flow and no gradient cancellation.

Key fixes:
1. Sender returns both discrete messages AND soft (continuous) representations
2. Receivers accept soft_message parameter for training
3. ARCAutoencoder passes soft messages to receivers during training
4. Proper straight-through estimator implementation
5. FIXED: ReceiverSelector uses similarity-based scoring to avoid gradient cancellation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ARCEncoder(nn.Module):
    """Encodes 30x30 ARC grids into fixed-size representations.
    
    Supports β-VAE: when use_beta_vae=True, outputs parameters for a Gaussian distribution
    (mean and log variance) and uses the reparameterization trick.
    """
    def __init__(self, 
                 num_colors=10,
                 embedding_dim=10,
                 hidden_dim=128,
                 latent_dim=512,
                 num_conv_layers=3,
                 conv_channels=None,
                 use_beta_vae=False):
        super().__init__()
        
        self.num_colors = num_colors
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_conv_layers = num_conv_layers
        self.use_beta_vae = use_beta_vae
        
        # Configure per-layer output channels for conv stack
        if conv_channels is None:
            self.conv_channels = [hidden_dim] * num_conv_layers
        else:
            # Accept either a single int (apply to all) or a list of length num_conv_layers
            if isinstance(conv_channels, int):
                self.conv_channels = [conv_channels] * num_conv_layers
            else:
                assert isinstance(conv_channels, (list, tuple)), "conv_channels must be int or list/tuple of ints"
                assert len(conv_channels) == num_conv_layers, "conv_channels length must equal num_conv_layers"
                self.conv_channels = list(conv_channels)
        
        assert embedding_dim == num_colors, "embedding_dim must equal num_colors for one-hot encoding"
        assert num_conv_layers >= 1, "Must have at least 1 convolutional layer"
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for i in range(num_conv_layers):
            in_channels = embedding_dim if i == 0 else self.conv_channels[i-1]
            out_channels = self.conv_channels[i]
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            self.bn_layers.append(nn.BatchNorm2d(out_channels))
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # For β-VAE, we need separate layers for mean and log variance
        if use_beta_vae:
            self.fc_mu = nn.Linear(self.conv_channels[-1] * 4 * 4, latent_dim)
            self.fc_logvar = nn.Linear(self.conv_channels[-1] * 4 * 4, latent_dim)
        else:
            self.fc = nn.Linear(self.conv_channels[-1] * 4 * 4, latent_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Store last mu and logvar for KL divergence calculation
        self.last_mu = None
        self.last_logvar = None
        
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for β-VAE: z = μ + σ * ε, where ε ~ N(0, I)
        
        Args:
            mu: Mean of the latent Gaussian [B, latent_dim]
            logvar: Log variance of the latent Gaussian [B, latent_dim]
            
        Returns:
            z: Sampled latent vector [B, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, sizes=None):
        B = x.shape[0]
        
        # One-hot encode
        x = F.one_hot(x.long(), num_classes=self.num_colors).float()
        x = x.permute(0, 3, 1, 2)
        
        # Convolutional layers
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.relu(bn(conv(x)))
        
        # Pool
        if sizes is not None:
            pooled_features = []
            for i in range(B):
                h, w = sizes[i]
                content = x[i:i+1, :, :h, :w]
                pooled = self.pool(content)
                pooled_features.append(pooled)
            x = torch.cat(pooled_features, dim=0)
        else:
            x = self.pool(x)
        
        x = x.reshape(B, -1)
        x = self.dropout(x)
        
        if self.use_beta_vae:
            # β-VAE mode: output mean and log variance, then sample
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            
            # Store for KL divergence calculation
            self.last_mu = mu
            self.last_logvar = logvar
            
            # Sample using reparameterization trick
            if self.training:
                z = self.reparameterize(mu, logvar)
            else:
                # During inference, use the mean
                z = mu
            
            return z
        else:
            # Standard autoencoder mode
            latent = self.relu(self.fc(x))
            return latent
    
    def extract_feature_maps(self, x, sizes=None):
        """Returns intermediate CNN feature maps for visualization."""
        self.eval()
        with torch.no_grad():
            B = x.shape[0]
            emb = F.one_hot(x.long(), num_classes=self.num_colors).float()
            emb = emb.permute(0, 3, 1, 2)
            
            feats = {"embed": emb}
            x_current = emb
            for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
                x_current = self.relu(bn(conv(x_current)))
                feats[f"conv{i+1}"] = x_current
            
            if sizes is not None:
                pooled_features = []
                for i in range(B):
                    h, w = sizes[i]
                    content = x_current[i:i+1, :, :h, :w]
                    pooled_sample = self.pool(content)
                    pooled_features.append(pooled_sample)
                pooled = torch.cat(pooled_features, dim=0)
            else:
                pooled = self.pool(x_current)
            
            feats["pooled"] = pooled
            return feats
    
    def get_spatial_features(self, x, sizes=None):
        """
        Extract spatial feature maps before pooling for slot attention.
        
        Returns:
            features: [B, C, H, W] spatial features
        """
        B = x.shape[0]
        
        # One-hot encode
        x = F.one_hot(x.long(), num_classes=self.num_colors).float()
        x = x.permute(0, 3, 1, 2)
        
        # Convolutional layers
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.relu(bn(conv(x)))
        
        # If sizes are provided, mask out padding (NON-INPLACE)
        if sizes is not None:
            # Create a mask instead of modifying in-place
            _, C, H, W = x.shape
            mask = torch.ones_like(x)
            for i in range(B):
                h, w = sizes[i]
                # Zero out padding regions in the mask
                mask[i, :, h:, :] = 0
                mask[i, :, :, w:] = 0
            # Apply mask (non-inplace)
            x = x * mask
        
        return x



class ARCDecoder(nn.Module):
    """Decoder for autoencoder mode."""
    def __init__(self, latent_dim, num_colors, hidden_dim, max_grid_size=30):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_colors = num_colors
        self.hidden_dim = hidden_dim
        self.max_grid_size = max_grid_size
        
        self.fc_decode = nn.Linear(latent_dim, hidden_dim * 4 * 4)
        
        self.deconv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                         kernel_size=4, stride=2, padding=1)
        self.bn_d1 = nn.BatchNorm2d(hidden_dim)
        
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                         kernel_size=4, stride=2, padding=1)
        self.bn_d2 = nn.BatchNorm2d(hidden_dim)
        
        self.deconv3 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                         kernel_size=4, stride=2, padding=1)
        self.bn_d3 = nn.BatchNorm2d(hidden_dim)
        
        self.conv_out1 = nn.Conv2d(hidden_dim, hidden_dim,
                                   kernel_size=3, padding=1)
        self.bn_out1 = nn.BatchNorm2d(hidden_dim)
        
        self.conv_out = nn.Conv2d(hidden_dim, num_colors, kernel_size=1)
        
        self.relu = nn.ReLU()
    
    def forward(self, latent, target_size):
        H, W = target_size
        
        x_dec = self.relu(self.fc_decode(latent))
        x_dec = x_dec.reshape(1, self.hidden_dim, 4, 4)
        
        x_dec = self.relu(self.bn_d1(self.deconv1(x_dec)))
        x_dec = self.relu(self.bn_d2(self.deconv2(x_dec)))
        x_dec = self.relu(self.bn_d3(self.deconv3(x_dec)))
        
        x_dec = self.relu(self.bn_out1(self.conv_out1(x_dec)))
        
        logits = self.conv_out(x_dec)
        
        if H != logits.shape[2] or W != logits.shape[3]:
            logits = F.interpolate(logits, size=(H, W),
                                  mode='bilinear', align_corners=False)
        
        return logits


class SlotAttention(nn.Module):
    """
    Slot Attention module for object-centric learning.
    
    Based on "Object-Centric Learning with Slot Attention" (Locatello et al., 2020).
    Learns to decompose input features into a fixed number of slots through iterative attention.
    """
    def __init__(self, num_slots, slot_dim, feature_dim, num_iterations=3, hidden_dim=128, eps=1e-8, max_spatial_size=30):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.feature_dim = feature_dim
        self.num_iterations = num_iterations
        self.eps = eps
        self.max_spatial_size = max_spatial_size
        
        # Learned 2D spatial positional encodings
        # These will be added to input features to give spatial awareness
        # Shape: [1, max_spatial_size, max_spatial_size, feature_dim]
        self.spatial_pos_encoding = nn.Parameter(
            torch.randn(1, max_spatial_size, max_spatial_size, feature_dim) * 0.02
        )
        
        # Learned slot initialization parameters
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        
        # Layer norm for slots and inputs
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_inputs = nn.LayerNorm(feature_dim)
        
        # Linear maps for attention (Q, K, V)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(feature_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(feature_dim, slot_dim, bias=False)
        
        # Slot update functions
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim)
        )
        
    def forward(self, inputs, spatial_size=None):
        """
        Args:
            inputs: Feature maps [B, H*W, feature_dim] or [B, num_features, feature_dim]
            spatial_size: Optional tuple (H, W) indicating spatial dimensions of the features.
                         If None, assumes square grid and computes from input size.
            
        Returns:
            slots: [B, num_slots, slot_dim]
        """
        B, N, D = inputs.shape
        
        # Determine spatial dimensions
        if spatial_size is not None:
            H, W = spatial_size
        else:
            # Assume square grid
            H = W = int(N ** 0.5)
            assert H * W == N, f"Cannot infer spatial size from N={N}, provide spatial_size parameter"
        
        # Reshape inputs to spatial format [B, H, W, D]
        inputs_spatial = inputs.reshape(B, H, W, D)
        
        # Add learned spatial positional encodings
        # Extract the relevant portion of positional encodings for the actual spatial size
        pos_enc = self.spatial_pos_encoding[:, :H, :W, :]  # [1, H, W, feature_dim]
        inputs_spatial = inputs_spatial + pos_enc  # Broadcasting across batch dimension
        
        # Reshape back to [B, N, D]
        inputs = inputs_spatial.reshape(B, N, D)
        
        # Initialize slots from learned distribution
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        
        # Normalize inputs
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # [B, N, slot_dim]
        v = self.project_v(inputs)  # [B, N, slot_dim]
        
        # Iterative attention
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Attention
            q = self.project_q(slots)  # [B, num_slots, slot_dim]
            
            # Compute attention weights
            # [B, num_slots, slot_dim] @ [B, slot_dim, N] -> [B, num_slots, N]
            attn_logits = torch.bmm(q, k.transpose(-1, -2)) / (self.slot_dim ** 0.5)
            attn = F.softmax(attn_logits, dim=-1)  # Softmax over slots
            
            # Normalize attention weights across slots (competition)
            attn = attn + self.eps
            attn = attn / attn.sum(dim=1, keepdim=True)
            
            # Weighted mean
            # [B, num_slots, N] @ [B, N, slot_dim] -> [B, num_slots, slot_dim]
            updates = torch.bmm(attn, v)
            
            # Update slots with GRU
            slots = self.gru(
                updates.reshape(B * self.num_slots, self.slot_dim),
                slots_prev.reshape(B * self.num_slots, self.slot_dim)
            )
            slots = slots.reshape(B, self.num_slots, self.slot_dim)
            
            # Apply MLP
            slots = slots + self.mlp(slots)
        
        return slots


class SlotDecoder(nn.Module):
    """
    Decoder for slot attention that reconstructs the grid from slots.
    Each slot is decoded independently and combined via broadcasting.
    """
    def __init__(self, slot_dim, num_colors, hidden_dim, max_grid_size=30, num_slots=7):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_colors = num_colors
        self.hidden_dim = hidden_dim
        self.max_grid_size = max_grid_size
        self.num_slots = num_slots
        
        # Learned positional encodings for each slot
        # These give each slot position-specific information during decoding
        self.slot_pos_encoding = nn.Parameter(torch.randn(num_slots, slot_dim))
        
        # Decode each slot to a spatial feature map
        self.fc_decode = nn.Linear(slot_dim, hidden_dim * 4 * 4)
        
        self.deconv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                         kernel_size=4, stride=2, padding=1)
        self.bn_d1 = nn.BatchNorm2d(hidden_dim)
        
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                         kernel_size=4, stride=2, padding=1)
        self.bn_d2 = nn.BatchNorm2d(hidden_dim)
        
        self.deconv3 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                         kernel_size=4, stride=2, padding=1)
        self.bn_d3 = nn.BatchNorm2d(hidden_dim)
        
        self.conv_out1 = nn.Conv2d(hidden_dim, hidden_dim,
                                   kernel_size=3, padding=1)
        self.bn_out1 = nn.BatchNorm2d(hidden_dim)
        
        # Output: num_colors + 1 for alpha mask
        self.conv_out = nn.Conv2d(hidden_dim, num_colors + 1, kernel_size=1)
        
        self.relu = nn.ReLU()
    
    def decode_slot(self, slot):
        """Decode a single slot to spatial features."""
        x = self.relu(self.fc_decode(slot))
        x = x.reshape(-1, self.hidden_dim, 4, 4)
        
        x = self.relu(self.bn_d1(self.deconv1(x)))
        x = self.relu(self.bn_d2(self.deconv2(x)))
        x = self.relu(self.bn_d3(self.deconv3(x)))
        x = self.relu(self.bn_out1(self.conv_out1(x)))
        
        x = self.conv_out(x)  # [num_slots, num_colors+1, H, W]
        
        return x
    
    def forward(self, slots, target_size):
        """
        Args:
            slots: [B, num_slots, slot_dim]
            target_size: (H, W) target output size
            
        Returns:
            logits: [B, num_colors, H, W]
        """
        B, num_slots, slot_dim = slots.shape
        H, W = target_size
        
        # Add learned positional encodings to slots
        # Expand positional encodings for batch: [num_slots, slot_dim] -> [B, num_slots, slot_dim]
        pos_encodings = self.slot_pos_encoding.unsqueeze(0).expand(B, -1, -1)
        slots = slots + pos_encodings  # Add positional information to each slot
        
        # Decode all slots
        slots_flat = slots.reshape(B * num_slots, slot_dim)
        decoded = self.decode_slot(slots_flat)  # [B*num_slots, num_colors+1, 32, 32]
        
        # Reshape to [B, num_slots, num_colors+1, 32, 32]
        decoded = decoded.reshape(B, num_slots, self.num_colors + 1, 
                                 decoded.shape[-2], decoded.shape[-1])
        
        # Resize to target size
        if H != decoded.shape[-2] or W != decoded.shape[-1]:
            decoded = F.interpolate(decoded.reshape(B * num_slots, self.num_colors + 1,
                                                   decoded.shape[-2], decoded.shape[-1]),
                                   size=(H, W), mode='bilinear', align_corners=False)
            decoded = decoded.reshape(B, num_slots, self.num_colors + 1, H, W)
        
        # Split into reconstructions and masks
        recons = decoded[:, :, :self.num_colors, :, :]  # [B, num_slots, num_colors, H, W]
        masks = decoded[:, :, self.num_colors:, :, :]   # [B, num_slots, 1, H, W]
        
        # Normalize masks across slots
        masks = F.softmax(masks, dim=1)
        
        # Combine slots using masks (mixture of slots)
        recon_combined = (recons * masks).sum(dim=1)  # [B, num_colors, H, W]
        
        return recon_combined


class SenderAgent(nn.Module):
    """
    FIXED: Sender that returns both discrete and soft representations.
    Supports variable-length messages via stop tokens.
    """
    def __init__(self, encoder, vocab_size, max_length, use_stop_token=False, stop_token_id=None, lstm_hidden_dim=None):
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size  # Base vocabulary size (without stop token)
        self.max_length = max_length
        self.use_stop_token = use_stop_token
        
        # If using stop tokens, effective vocab is vocab_size + 1
        self.effective_vocab_size = vocab_size + 1 if use_stop_token else vocab_size
        self.stop_token_id = stop_token_id if use_stop_token else None
        
        # Allow custom LSTM hidden size; default to encoder latent dim
        self.lstm_hidden_dim = encoder.latent_dim if lstm_hidden_dim is None else lstm_hidden_dim
        self.lstm = nn.LSTM(self.effective_vocab_size, self.lstm_hidden_dim, batch_first=True)
        self.vocab_proj = nn.Linear(self.lstm_hidden_dim, self.effective_vocab_size)
        # Project encoder latent to LSTM hidden dim if needed
        self.latent_to_hidden = (nn.Identity() if self.lstm_hidden_dim == encoder.latent_dim
                                 else nn.Linear(encoder.latent_dim, self.lstm_hidden_dim))
        
    def forward(self, grids, sizes=None, temperature=1.0):
        """
        Returns:
            message: [batch, max_length] - discrete symbols (for logging)
            soft_message: [batch, max_length, effective_vocab_size] - soft for gradients
            message_lengths: [batch] - actual lengths of messages (position of stop token + 1, or max_length)
        """
        B = grids.shape[0]
        
        latent = self.encoder(grids, sizes=sizes)
        
        # Initialize LSTM hidden state from (projected) latent
        h = self.latent_to_hidden(latent).unsqueeze(0)
        c = torch.zeros_like(h)
        
        hard_messages = []
        soft_messages = []
        message_lengths = torch.full((B,), self.max_length, dtype=torch.long, device=grids.device)
        stopped = torch.zeros(B, dtype=torch.bool, device=grids.device)  # Track which sequences have stopped
        
        input_token = torch.zeros(B, self.effective_vocab_size, device=grids.device)
        
        for t in range(self.max_length):
            input_token_unsqueezed = input_token.unsqueeze(1)
            lstm_out, (h, c) = self.lstm(input_token_unsqueezed, (h, c))
            lstm_out = lstm_out.squeeze(1)
            
            logits = self.vocab_proj(lstm_out)
            
            # MODIFIED: Prevent stop token at first position
            if self.use_stop_token and t == 0:
                # Mask out stop token by setting its logit to negative infinity
                logits[:, self.stop_token_id] = -float('inf')
            
            if self.training:
                # Gumbel-softmax
                gumbel_noise = -torch.log(-torch.log(
                    torch.rand_like(logits) + 1e-20) + 1e-20)
                gumbel_logits = (logits + gumbel_noise) / temperature
                soft_token = F.softmax(gumbel_logits, dim=-1)
                
                # Straight-through
                hard_token = F.one_hot(soft_token.argmax(dim=-1), 
                                    num_classes=self.effective_vocab_size).float()
                
                # FIXED: Proper straight-through
                token_for_next_input = hard_token.detach() - soft_token.detach() + soft_token
                
                soft_messages.append(soft_token)
                hard_symbol = soft_token.argmax(dim=-1)
                hard_messages.append(hard_symbol)
                
                # Check for stop tokens (only if enabled and not already stopped)
                if self.use_stop_token:
                    is_stop = (hard_symbol == self.stop_token_id) & ~stopped
                    # Record the length for sequences that just stopped (t+1 because we include the stop token)
                    message_lengths[is_stop] = t + 1
                    stopped = stopped | is_stop
            else:
                # Eval mode
                soft_token = F.softmax(logits, dim=-1)
                hard_symbol = logits.argmax(dim=-1)
                hard_token = F.one_hot(hard_symbol, num_classes=self.effective_vocab_size).float()
                token_for_next_input = hard_token
                
                soft_messages.append(soft_token)
                hard_messages.append(hard_symbol)
                
                # Check for stop tokens (only if enabled and not already stopped)
                if self.use_stop_token:
                    is_stop = (hard_symbol == self.stop_token_id) & ~stopped
                    message_lengths[is_stop] = t + 1
                    stopped = stopped | is_stop
            
            input_token = token_for_next_input
        
        message = torch.stack(hard_messages, dim=1)
        soft_message = torch.stack(soft_messages, dim=1)
        
        return message, soft_message, message_lengths


class ReceiverAgent(nn.Module):
    """
    FIXED: Receiver that can accept soft message representations.
    Can optionally also receive the encoder's latent representation (shared common ground).
    Supports variable-length messages via masking.
    """
    def __init__(self, vocab_size, num_colors, hidden_dim, max_grid_size=30, 
                 receives_input_encoding=False, encoder_latent_dim=None,
                 use_stop_token=False, stop_token_id=None, lstm_hidden_dim=None):
        super().__init__()
        self.vocab_size = vocab_size  # Base vocabulary size (without stop token)
        self.num_colors = num_colors
        self.hidden_dim = hidden_dim
        self.max_grid_size = max_grid_size
        self.receives_input_encoding = receives_input_encoding
        self.use_stop_token = use_stop_token
        self.stop_token_id = stop_token_id if use_stop_token else None
        self.lstm_hidden_dim = hidden_dim if lstm_hidden_dim is None else lstm_hidden_dim
        
        # If using stop tokens, effective vocab is vocab_size + 1
        self.effective_vocab_size = vocab_size + 1 if use_stop_token else vocab_size
        
        self.symbol_embed = nn.Embedding(self.effective_vocab_size, hidden_dim)
        self.continuous_proj = nn.Linear(self.effective_vocab_size, hidden_dim)
        
        self.lstm = nn.LSTM(hidden_dim, self.lstm_hidden_dim, batch_first=True)
        
        # If receiver gets encoder's latent, combine it with message
        if receives_input_encoding:
            assert encoder_latent_dim is not None, "Must provide encoder_latent_dim when receives_input_encoding=True"
            self.combine_fc = nn.Linear(self.lstm_hidden_dim + encoder_latent_dim, hidden_dim)
        
        self.relu = nn.ReLU()
        
        # Map from message representation (from LSTM hidden) into decoder input
        self.fc_decode = nn.Linear(self.lstm_hidden_dim if not receives_input_encoding else hidden_dim, 
                                   hidden_dim * 4 * 4)
        
        self.deconv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                         kernel_size=4, stride=2, padding=1)
        self.bn_d1 = nn.BatchNorm2d(hidden_dim)
        
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                         kernel_size=4, stride=2, padding=1)
        self.bn_d2 = nn.BatchNorm2d(hidden_dim)
        
        self.deconv3 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                         kernel_size=4, stride=2, padding=1)
        self.bn_d3 = nn.BatchNorm2d(hidden_dim)
        
        self.conv_out1 = nn.Conv2d(hidden_dim, hidden_dim,
                                   kernel_size=3, padding=1)
        self.bn_out1 = nn.BatchNorm2d(hidden_dim)
        
        self.conv_out = nn.Conv2d(hidden_dim, num_colors, kernel_size=1)
    
    def forward(self, message, target_size, soft_message=None, input_encoding=None, message_lengths=None):
        """
        FIXED: Accepts soft_message for gradient flow during training.
        Can also accept input_encoding (encoder's latent) for shared common ground.
        Supports variable-length messages via masking.
        
        Args:
            message: Discrete message symbols [batch, max_length]
            target_size: (height, width) of target grid to reconstruct
            soft_message: Soft message representation [batch, max_length, effective_vocab_size]
            input_encoding: Optional encoder's latent [batch, encoder_latent_dim] (shared common ground)
            message_lengths: Optional [batch] tensor of actual message lengths (for masking)
        """
        if soft_message is not None and self.training:
            embedded = self.continuous_proj(soft_message)
        else:
            embedded = self.symbol_embed(message)
        
        # If we have message lengths, mask out padding (symbols after stop token)
        if message_lengths is not None and self.use_stop_token:
            # Create mask: True for valid positions, False for padding
            batch_size, seq_len = message.shape
            # Create position indices [batch, seq_len]
            positions = torch.arange(seq_len, device=message.device).unsqueeze(0).expand(batch_size, -1)
            # Mask is True where position < message_length
            mask = positions < message_lengths.unsqueeze(1)
            # Zero out embeddings after stop token
            embedded = embedded * mask.unsqueeze(-1).float()
        
        lstm_out, (h, c) = self.lstm(embedded)
        message_repr = h.squeeze(0)
        
        # If receiver gets encoder's latent, combine it with message representation
        if self.receives_input_encoding and input_encoding is not None:
            # Combine message and encoder's latent (shared common ground)
            combined = torch.cat([message_repr, input_encoding], dim=1)
            message_repr = self.relu(self.combine_fc(combined))
        
        H, W = target_size
        
        x_dec = self.relu(self.fc_decode(message_repr))
        x_dec = x_dec.reshape(1, self.hidden_dim, 4, 4)
        
        x_dec = self.relu(self.bn_d1(self.deconv1(x_dec)))
        x_dec = self.relu(self.bn_d2(self.deconv2(x_dec)))
        x_dec = self.relu(self.bn_d3(self.deconv3(x_dec)))
        
        x_dec = self.relu(self.bn_out1(self.conv_out1(x_dec)))
        
        logits = self.conv_out(x_dec)
        
        if H != logits.shape[2] or W != logits.shape[3]:
            logits = F.interpolate(logits, size=(H, W),
                                  mode='bilinear', align_corners=False)
        
        return logits

class ReceiverPuzzleClassifier(nn.Module):
    """Receiver for puzzle classification task. Supports variable-length messages."""
    def __init__(self, vocab_size, num_classes, hidden_dim, use_stop_token=False, stop_token_id=None, lstm_hidden_dim=None):
        super().__init__()
        self.vocab_size = vocab_size  # Base vocabulary size (without stop token)
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_stop_token = use_stop_token
        self.stop_token_id = stop_token_id if use_stop_token else None
        self.lstm_hidden_dim = hidden_dim if lstm_hidden_dim is None else lstm_hidden_dim
        
        # If using stop tokens, effective vocab is vocab_size + 1
        self.effective_vocab_size = vocab_size + 1 if use_stop_token else vocab_size
        
        self.symbol_embed = nn.Embedding(self.effective_vocab_size, hidden_dim)
        self.continuous_proj = nn.Linear(self.effective_vocab_size, hidden_dim)
        
        self.lstm = nn.LSTM(hidden_dim, self.lstm_hidden_dim, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, message, soft_message=None, message_lengths=None):
        """FIXED: Accepts soft_message and message_lengths for variable-length messages."""
        if soft_message is not None and self.training:
            msg_emb = self.continuous_proj(soft_message)
        else:
            msg_emb = self.symbol_embed(message)
        
        # If we have message lengths, mask out padding (symbols after stop token)
        if message_lengths is not None and self.use_stop_token:
            # Create mask: True for valid positions, False for padding
            batch_size, seq_len = message.shape
            # Create position indices [batch, seq_len]
            positions = torch.arange(seq_len, device=message.device).unsqueeze(0).expand(batch_size, -1)
            # Mask is True where position < message_length
            mask = positions < message_lengths.unsqueeze(1)
            # Zero out embeddings after stop token
            msg_emb = msg_emb * mask.unsqueeze(-1).float()
        
        _, (h, _) = self.lstm(msg_emb)
        msg_repr = h.squeeze(0)
        
        logits = self.classifier(msg_repr)
        return logits


class ReceiverSelector(nn.Module):
    """
    FIXED: Receiver selector that avoids gradient cancellation.
    
    Key change: Uses dot-product similarity instead of concatenation + linear layer.
    This prevents CrossEntropyLoss gradients from canceling out.
    Supports variable-length messages via masking.
    """
    def __init__(self, vocab_size, num_colors, embedding_dim, hidden_dim, num_conv_layers=2, use_stop_token=False, stop_token_id=None, lstm_hidden_dim=None):
        super().__init__()
        self.vocab_size = vocab_size  # Base vocabulary size (without stop token)
        self.num_colors = num_colors
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.use_stop_token = use_stop_token
        self.stop_token_id = stop_token_id if use_stop_token else None
        self.lstm_hidden_dim = hidden_dim if lstm_hidden_dim is None else lstm_hidden_dim
        
        assert embedding_dim == num_colors
        assert num_conv_layers >= 1
        
        # If using stop tokens, effective vocab is vocab_size + 1
        self.effective_vocab_size = vocab_size + 1 if use_stop_token else vocab_size
        
        self.symbol_embed = nn.Embedding(self.effective_vocab_size, hidden_dim)
        self.continuous_proj = nn.Linear(self.effective_vocab_size, hidden_dim)
        
        self.lstm = nn.LSTM(hidden_dim, self.lstm_hidden_dim, batch_first=True)
        
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
        # we use separate projections and compute similarity
        self.msg_proj = nn.Linear(self.lstm_hidden_dim, hidden_dim)
        self.cand_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Learnable temperature parameter
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
    
    def forward(self, message, candidates, candidate_sizes=None, soft_message=None, message_lengths=None):
        """
        FIXED: Computes similarity scores instead of concatenating features.
        
        This prevents gradient cancellation because the message representation
        is not directly shared across all candidates via expand().
        Supports variable-length messages via masking.
        """
        if soft_message is not None and self.training:
            msg_emb = self.continuous_proj(soft_message)
        else:
            msg_emb = self.symbol_embed(message)
        
        # If we have message lengths, mask out padding (symbols after stop token)
        if message_lengths is not None and self.use_stop_token:
            # Create mask: True for valid positions, False for padding
            batch_size, seq_len = message.shape
            # Create position indices [batch, seq_len]
            positions = torch.arange(seq_len, device=message.device).unsqueeze(0).expand(batch_size, -1)
            # Mask is True where position < message_length
            mask = positions < message_lengths.unsqueeze(1)
            # Zero out embeddings after stop token
            msg_emb = msg_emb * mask.unsqueeze(-1).float()
        
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


class DecoderSelector(nn.Module):
    """
    FIXED: Decoder selector for autoencoder mode.
    
    Uses similarity-based scoring to avoid gradient cancellation,
    same approach as ReceiverSelector.
    """
    def __init__(self, latent_dim, num_colors, embedding_dim, hidden_dim, num_conv_layers=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_colors = num_colors
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        
        assert embedding_dim == num_colors
        assert num_conv_layers >= 1
        
        # FIXED: Use separate projections instead of concatenation
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        
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
        self.cand_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Learnable temperature parameter
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
    
    def forward(self, latent, candidates, candidate_sizes=None):
        """
        FIXED: Computes similarity scores instead of concatenating features.
        
        This prevents gradient cancellation because the latent representation
        is not directly shared across all candidates via expand().
        """
        # Project latent to comparison space
        latent_proj = self.latent_proj(latent)  # [1, hidden_dim]
        
        # Encode and project candidates
        cand_repr = self.encode_candidates(candidates, candidate_sizes=candidate_sizes)  # [N, hidden_dim]
        cand_proj = self.cand_proj(cand_repr)  # [N, hidden_dim]
        
        # FIXED: Compute similarity using dot product
        # This prevents gradient cancellation because we're not using expand()
        logits = torch.matmul(cand_proj, latent_proj.squeeze(0)) / self.temperature
        
        return logits


class ARCAutoencoder(nn.Module):
    """
    FIXED: Main model with proper gradient flow through communication.
    Supports variable-length messages via stop tokens.
    Supports β-VAE with adjustable β hyperparameter.
    """
    def __init__(self, encoder, vocab_size, max_length, num_colors, embedding_dim, hidden_dim, 
                 max_grid_size=30, bottleneck_type='communication', task_type='reconstruction', 
                 num_conv_layers=2, num_classes=None, receiver_gets_input_puzzle=False,
                 use_stop_token=False, stop_token_id=None, lstm_hidden_dim=None,
                 use_beta_vae=False, beta=4.0,
                 num_slots=7, slot_dim=64, slot_iterations=3, slot_hidden_dim=128, slot_eps=1e-8):
        super().__init__()
        self.encoder = encoder
        self.bottleneck_type = bottleneck_type
        self.task_type = task_type
        self.receiver_gets_input_puzzle = receiver_gets_input_puzzle
        self.use_stop_token = use_stop_token
        self.stop_token_id = stop_token_id
        self.lstm_hidden_dim = lstm_hidden_dim
        self.use_beta_vae = use_beta_vae
        self.beta = beta
        
        if bottleneck_type == 'communication':
            self.sender = SenderAgent(encoder, vocab_size, max_length, use_stop_token=use_stop_token, stop_token_id=stop_token_id, lstm_hidden_dim=self.lstm_hidden_dim)
            if task_type == 'reconstruction':
                self.receiver = ReceiverAgent(vocab_size, num_colors, hidden_dim, max_grid_size, 
                                            receives_input_encoding=receiver_gets_input_puzzle,
                                            encoder_latent_dim=encoder.latent_dim,
                                            use_stop_token=use_stop_token, stop_token_id=stop_token_id,
                                            lstm_hidden_dim=self.lstm_hidden_dim)
            elif task_type == 'selection':
                # In selection mode, create BOTH receivers
                self.receiver = ReceiverSelector(vocab_size, num_colors, embedding_dim, hidden_dim, num_conv_layers,
                                                use_stop_token=use_stop_token, stop_token_id=stop_token_id,
                                                lstm_hidden_dim=self.lstm_hidden_dim)
                # Add reconstruction receiver for background training
                self.receiver_reconstructor = ReceiverAgent(vocab_size, num_colors, hidden_dim, max_grid_size,
                                                        receives_input_encoding=receiver_gets_input_puzzle,
                                                        encoder_latent_dim=encoder.latent_dim,
                                                        use_stop_token=use_stop_token, stop_token_id=stop_token_id,
                                                        lstm_hidden_dim=self.lstm_hidden_dim)
            elif task_type == 'puzzle_classification':
                assert num_classes is not None
                self.receiver = ReceiverPuzzleClassifier(vocab_size, num_classes, hidden_dim,
                                                        use_stop_token=use_stop_token, stop_token_id=stop_token_id,
                                                        lstm_hidden_dim=self.lstm_hidden_dim)
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
        elif bottleneck_type == 'autoencoder':
            if task_type == 'reconstruction':
                self.decoder = ARCDecoder(encoder.latent_dim, num_colors, hidden_dim, max_grid_size)
            elif task_type == 'selection':
                # In selection mode, create BOTH decoders
                self.decoder = DecoderSelector(encoder.latent_dim, num_colors, embedding_dim, hidden_dim, num_conv_layers)
                # Add reconstruction decoder for background training
                self.decoder_reconstructor = ARCDecoder(encoder.latent_dim, num_colors, hidden_dim, max_grid_size)
            elif task_type == 'puzzle_classification':
                raise NotImplementedError("Puzzle classification not yet supported in autoencoder mode")
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
        elif bottleneck_type == 'slot_attention':
            # Slot attention bottleneck
            # Get feature dimension from encoder's last conv layer
            feature_dim = encoder.conv_channels[-1]
            
            self.slot_attention = SlotAttention(
                num_slots=num_slots,
                slot_dim=slot_dim,
                feature_dim=feature_dim,
                num_iterations=slot_iterations,
                hidden_dim=slot_hidden_dim,
                eps=slot_eps,
                max_spatial_size=max_grid_size
            )
            
            if task_type == 'reconstruction':
                self.slot_decoder = SlotDecoder(slot_dim, num_colors, hidden_dim, max_grid_size, num_slots=num_slots)
            elif task_type == 'selection':
                raise NotImplementedError("Selection task not yet supported with slot_attention bottleneck")
            elif task_type == 'puzzle_classification':
                raise NotImplementedError("Puzzle classification not yet supported with slot_attention bottleneck")
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
        else:
            raise ValueError(f"Unknown bottleneck_type: {bottleneck_type}")
    
    def forward(self, x, sizes, temperature=1.0, candidates_list=None, candidates_sizes_list=None, 
                target_indices=None, labels=None, output_sizes=None):
        """
        Args:
            x: Input grids
            sizes: Input sizes
            output_sizes: Optional output sizes (for input->output transformation tasks)
                        If None, uses sizes (self-supervised reconstruction)
        """
        B = x.shape[0]
        
        # Use output_sizes if provided, otherwise use input sizes (self-supervised)
        target_sizes = output_sizes if output_sizes is not None else sizes
        
        if self.task_type == 'reconstruction':
            if self.bottleneck_type == 'communication':
                # Compute latent first (for shared common ground)
                latent = self.encoder(x, sizes=sizes)
                
                messages, soft_messages, message_lengths = self.sender(x, sizes=sizes, temperature=temperature)
                
                logits_list = []
                actual_sizes = []
                
                for i in range(B):
                    single_message = messages[i:i+1]
                    soft_single = soft_messages[i:i+1]
                    single_length = message_lengths[i:i+1]
                    target_h, target_w = target_sizes[i]
                    
                    # Pass encoder's latent to receiver if configured
                    if self.receiver_gets_input_puzzle:
                        # Pass encoder's latent (shared common ground)
                        input_encoding = latent[i:i+1]
                        logits = self.receiver(single_message, 
                                            target_size=(target_h, target_w),
                                            soft_message=soft_single,
                                            input_encoding=input_encoding,
                                            message_lengths=single_length)
                    else:
                        logits = self.receiver(single_message, 
                                            target_size=(target_h, target_w),
                                            soft_message=soft_single,
                                            message_lengths=single_length)
                    
                    logits_list.append(logits)
                    actual_sizes.append((target_h, target_w)) 
                
                return logits_list, actual_sizes, messages, message_lengths
                
            elif self.bottleneck_type == 'autoencoder':
                latent = self.encoder(x, sizes=sizes)
                
                logits_list = []
                actual_sizes = []
                
                for i in range(B):
                    single_latent = latent[i:i+1]
                    actual_h, actual_w = sizes[i]
                    
                    logits = self.decoder(single_latent, target_size=(actual_h, actual_w))
                    
                    logits_list.append(logits)
                    actual_sizes.append((actual_h, actual_w))
                
                return logits_list, actual_sizes, None
            
            elif self.bottleneck_type == 'slot_attention':
                # Extract spatial features before pooling
                spatial_features = self.encoder.get_spatial_features(x, sizes=sizes)
                # [B, C, H, W]
                
                B, C, H, W = spatial_features.shape
                
                # Reshape to [B, H*W, C] for slot attention
                features_flat = spatial_features.permute(0, 2, 3, 1).reshape(B, H * W, C)
                
                # Apply slot attention to get slots (pass spatial dimensions for positional encoding)
                slots = self.slot_attention(features_flat, spatial_size=(H, W))  # [B, num_slots, slot_dim]
                
                # Decode each sample separately
                logits_list = []
                actual_sizes = []
                
                for i in range(B):
                    single_slots = slots[i:i+1]
                    actual_h, actual_w = sizes[i]
                    
                    # Decode slots to reconstruction
                    logits = self.slot_decoder(single_slots, target_size=(actual_h, actual_w))
                    
                    logits_list.append(logits)
                    actual_sizes.append((actual_h, actual_w))
                
                return logits_list, actual_sizes, None
        
        
        elif self.task_type == 'selection':
            if self.bottleneck_type == 'communication':
                # Compute latent first (for shared common ground in reconstruction)
                latent = self.encoder(x, sizes=sizes)
                
                messages, soft_messages, message_lengths = self.sender(x, sizes=sizes, temperature=temperature)
                
                selection_logits_list = []
                reconstruction_logits_list = []
                actual_sizes = []
                
                for i in range(B):
                    single_message = messages[i:i+1]
                    soft_single = soft_messages[i:i+1]
                    single_length = message_lengths[i:i+1]
                    candidates = candidates_list[i]
                    candidate_sizes = candidates_sizes_list[i] if candidates_sizes_list is not None else None
                    actual_h, actual_w = sizes[i]
                    
                    # Selection task
                    sel_logits = self.receiver(single_message, candidates, 
                                            candidate_sizes=candidate_sizes,
                                            soft_message=soft_single,
                                            message_lengths=single_length)
                    
                    selection_logits_list.append(sel_logits)
                    
                    # Background reconstruction task
                    # Detach soft_message so gradients only flow to receiver_reconstructor
                    if self.receiver_gets_input_puzzle:
                        # Pass encoder's latent (shared common ground)
                        # Detach to prevent gradients flowing back through encoder via reconstruction path
                        input_encoding = latent[i:i+1].detach() if self.training else latent[i:i+1]
                        if self.training:
                            recon_logits = self.receiver_reconstructor(
                                single_message, 
                                target_size=(actual_h, actual_w),
                                soft_message=soft_single.detach(),
                                input_encoding=input_encoding,
                                message_lengths=single_length
                            )
                        else:
                            recon_logits = self.receiver_reconstructor(
                                single_message, 
                                target_size=(actual_h, actual_w),
                                soft_message=soft_single,
                                input_encoding=input_encoding,
                                message_lengths=single_length
                            )
                    else:
                        if self.training:
                            recon_logits = self.receiver_reconstructor(
                                single_message, 
                                target_size=(actual_h, actual_w),
                                soft_message=soft_single.detach(),
                                message_lengths=single_length
                            )
                        else:
                            recon_logits = self.receiver_reconstructor(
                                single_message, 
                                target_size=(actual_h, actual_w),
                                soft_message=soft_single,
                                message_lengths=single_length
                            )
                    reconstruction_logits_list.append(recon_logits)
                    
                    actual_sizes.append((actual_h, actual_w))
                
                return selection_logits_list, reconstruction_logits_list, actual_sizes, messages, message_lengths
                
            else:  # autoencoder
                latent = self.encoder(x, sizes=sizes)
                
                selection_logits_list = []
                reconstruction_logits_list = []
                actual_sizes = []
                
                for i in range(B):
                    single_latent = latent[i:i+1]
                    candidates = candidates_list[i]
                    candidate_sizes = candidates_sizes_list[i] if candidates_sizes_list is not None else None
                    actual_h, actual_w = sizes[i]
                    
                    # Selection task
                    sel_logits = self.decoder(single_latent, candidates, candidate_sizes=candidate_sizes)
                    selection_logits_list.append(sel_logits)
                    
                    # Background reconstruction task
                    # Detach latent so gradients only flow to decoder_reconstructor
                    if self.training:
                        recon_logits = self.decoder_reconstructor(single_latent.detach(), target_size=(actual_h, actual_w))
                    else:
                        recon_logits = self.decoder_reconstructor(single_latent, target_size=(actual_h, actual_w))
                    reconstruction_logits_list.append(recon_logits)
                    
                    actual_sizes.append((actual_h, actual_w))
                
                return selection_logits_list, reconstruction_logits_list, actual_sizes, None
        
        else:  # puzzle_classification
            if self.bottleneck_type == 'communication':
                messages, soft_messages, message_lengths = self.sender(x, sizes=sizes, temperature=temperature)
                
                classification_logits = self.receiver(messages, soft_message=soft_messages, message_lengths=message_lengths)
                
                return classification_logits, sizes, messages, message_lengths
            
            else:
                raise NotImplementedError("Puzzle classification not yet supported in autoencoder mode")
    
    def compute_beta_vae_loss(self, reconstruction_loss):
        """
        Compute β-VAE loss by adding the KL divergence term.
        
        According to the β-VAE paper:
        L(θ, φ; x, z, β) = E[log p(x|z)] - β * KL(q(z|x)||p(z))
        
        where p(z) = N(0, I) is the isotropic unit Gaussian prior.
        
        Args:
            reconstruction_loss: The reconstruction loss E[-log p(x|z)]
            
        Returns:
            total_loss: Combined reconstruction loss and KL divergence
            kl_div: The KL divergence value (for logging)
        """
        if not self.use_beta_vae or not hasattr(self.encoder, 'last_mu'):
            # β-VAE not enabled or encoder doesn't support it
            return reconstruction_loss, torch.tensor(0.0)
        
        mu = self.encoder.last_mu
        logvar = self.encoder.last_logvar
        
        if mu is None or logvar is None:
            # No latent variables stored (shouldn't happen during training)
            return reconstruction_loss, torch.tensor(0.0)
        
        # KL divergence between q(z|x) and p(z) = N(0, I)
        # KL(q(z|x)||p(z)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Average over batch
        kl_div = kl_div / mu.size(0)
        
        # Combine with reconstruction loss using β weighting
        total_loss = reconstruction_loss + self.beta * kl_div
        
        return total_loss, kl_div
    
    def visualize_slots(self, x, sizes=None):
        """
        Visualize slot decomposition (only works with slot_attention bottleneck).
        
        Args:
            x: Input grids [B, H, W]
            sizes: Optional actual sizes
            
        Returns:
            Dict with:
                - slots: [B, num_slots, slot_dim]
                - slot_reconstructions: [B, num_slots, num_colors, H, W]
                - masks: [B, num_slots, 1, H, W]
        """
        if self.bottleneck_type != 'slot_attention':
            raise ValueError("Slot visualization only available with slot_attention bottleneck")
        
        self.eval()
        with torch.no_grad():
            # Extract spatial features
            spatial_features = self.encoder.get_spatial_features(x, sizes=sizes)
            B, C, H, W = spatial_features.shape
            
            # Reshape for slot attention
            features_flat = spatial_features.permute(0, 2, 3, 1).reshape(B, H * W, C)
            
            # Get slots
            slots = self.slot_attention(features_flat)
            
            # Decode each slot separately to get individual reconstructions
            slots_flat = slots.reshape(B * self.slot_attention.num_slots, self.slot_attention.slot_dim)
            decoded = self.slot_decoder.decode_slot(slots_flat)
            
            # Reshape
            num_slots = self.slot_attention.num_slots
            decoded = decoded.reshape(B, num_slots, self.slot_decoder.num_colors + 1,
                                    decoded.shape[-2], decoded.shape[-1])
            
            # Resize if needed
            actual_h, actual_w = sizes[0] if sizes is not None else (H, W)
            if actual_h != decoded.shape[-2] or actual_w != decoded.shape[-1]:
                decoded = F.interpolate(
                    decoded.reshape(B * num_slots, self.slot_decoder.num_colors + 1,
                                  decoded.shape[-2], decoded.shape[-1]),
                    size=(actual_h, actual_w), mode='bilinear', align_corners=False
                )
                decoded = decoded.reshape(B, num_slots, self.slot_decoder.num_colors + 1, 
                                        actual_h, actual_w)
            
            # Split reconstructions and masks
            slot_recons = decoded[:, :, :self.slot_decoder.num_colors, :, :]
            masks = decoded[:, :, self.slot_decoder.num_colors:, :, :]
            masks = F.softmax(masks, dim=1)
            
            return {
                'slots': slots,
                'slot_reconstructions': slot_recons,
                'masks': masks
            }


class ARCPuzzleSolver(nn.Module):
    """
    Puzzle solver that learns rules from input→output example pairs.
    
    Architecture:
    1. Encode each (input, output) pair
    2. Aggregate example encodings into a rule representation
    3. Apply rule to test inputs to generate test outputs
    """
    def __init__(self, encoder, vocab_size=None, max_length=None, num_colors=10, 
                 embedding_dim=10, hidden_dim=128, max_grid_size=30,
                 bottleneck_type='communication', rule_dim=256, 
                 pair_combination='concat', num_conv_layers=2,
                 use_stop_token=False, stop_token_id=None, lstm_hidden_dim=None):
        super().__init__()
        self.encoder = encoder
        self.bottleneck_type = bottleneck_type
        self.rule_dim = rule_dim
        self.pair_combination = pair_combination
        self.use_stop_token = use_stop_token
        self.stop_token_id = stop_token_id
        self.lstm_hidden_dim = lstm_hidden_dim or hidden_dim
        
        # Pair encoder: encodes (input, output) pairs
        if pair_combination == 'concat':
            # Concatenate input and output encodings
            self.pair_encoder = nn.Sequential(
                nn.Linear(encoder.latent_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, rule_dim)
            )
        elif pair_combination == 'delta':
            # Encode the transformation (delta between input and output)
            self.pair_encoder = nn.Sequential(
                nn.Linear(encoder.latent_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, rule_dim)
            )
        else:
            raise ValueError(f"Unknown pair_combination: {pair_combination}")
        
        # Rule aggregator: combines multiple examples into a single rule
        self.rule_aggregator = nn.LSTM(rule_dim, rule_dim, batch_first=True)
        
        # Rule applier: applies rule to test input
        if bottleneck_type == 'communication':
            assert vocab_size is not None and max_length is not None
            # Create sender that takes (rule + input) and outputs message
            self.rule_sender = nn.Sequential(
                nn.Linear(encoder.latent_dim + rule_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            # Effective vocab size includes stop token if enabled
            self.effective_vocab_size = vocab_size + 1 if use_stop_token else vocab_size
            
            # Message generator
            self.message_lstm = nn.LSTM(self.effective_vocab_size, self.lstm_hidden_dim, batch_first=True)
            self.vocab_proj = nn.Linear(self.lstm_hidden_dim, self.effective_vocab_size)
            
            # Project from rule_sender output to LSTM hidden size
            self.sender_to_hidden = nn.Linear(hidden_dim, self.lstm_hidden_dim)
            
            # Receiver that takes message and reconstructs output
            self.rule_receiver = ReceiverAgent(
                vocab_size, num_colors, hidden_dim, max_grid_size,
                receives_input_encoding=False,
                use_stop_token=use_stop_token,
                stop_token_id=stop_token_id,
                lstm_hidden_dim=self.lstm_hidden_dim
            )
        else:  # autoencoder
            # Direct decoder from (rule + input encoding)
            self.rule_decoder = ARCDecoder(
                encoder.latent_dim + rule_dim,
                num_colors,
                hidden_dim,
                max_grid_size
            )
    
    def encode_pair(self, input_grid, input_size, output_grid, output_size):
        """Encode a single (input, output) pair."""
        input_enc = self.encoder(input_grid.unsqueeze(0), sizes=[input_size])
        output_enc = self.encoder(output_grid.unsqueeze(0), sizes=[output_size])
        
        if self.pair_combination == 'concat':
            pair_enc = torch.cat([input_enc, output_enc], dim=1)
        elif self.pair_combination == 'delta':
            pair_enc = torch.cat([output_enc - input_enc, input_enc], dim=1)
        
        return self.pair_encoder(pair_enc)
    
    def aggregate_rule(self, example_encodings):
        """
        Aggregate multiple example encodings into a rule.
        
        Args:
            example_encodings: [num_examples, rule_dim]
        Returns:
            rule: [1, rule_dim]
        """
        # Use LSTM to process sequence of examples
        example_seq = example_encodings.unsqueeze(0)  # [1, num_examples, rule_dim]
        _, (h, _) = self.rule_aggregator(example_seq)
        rule = h.squeeze(0)  # [1, rule_dim]
        return rule
    
    def apply_rule(self, rule, test_input, test_input_size, test_output_size, temperature=1.0):
        """
        Apply rule to test input to generate output.
        
        Args:
            rule: [1, rule_dim]
            test_input: [H, W] test input grid
            test_input_size: (h, w) actual size
            test_output_size: (h, w) target output size
            temperature: Temperature for Gumbel-softmax
            
        Returns:
            logits: Output logits
            message: Discrete message (if communication mode)
            soft_message: Soft message (if communication mode)
            message_lengths: Actual message lengths (if using stop tokens)
        """
        # Encode test input
        test_input_enc = self.encoder(test_input.unsqueeze(0), sizes=[test_input_size])
        
        if self.bottleneck_type == 'communication':
            # Combine rule and input encoding
            combined = torch.cat([rule, test_input_enc], dim=1)
            sender_out = self.rule_sender(combined)
            
            # Generate message
            h = self.sender_to_hidden(sender_out).unsqueeze(0)
            c = torch.zeros_like(h)
            
            hard_messages = []
            soft_messages = []
            message_lengths = torch.full((1,), self.effective_vocab_size, 
                                       dtype=torch.long, device=test_input.device)
            stopped = False
            
            input_token = torch.zeros(1, self.effective_vocab_size, device=test_input.device)
            
            for t in range(self.effective_vocab_size):  # max_length
                input_token_unsqueezed = input_token.unsqueeze(1)
                lstm_out, (h, c) = self.message_lstm(input_token_unsqueezed, (h, c))
                lstm_out = lstm_out.squeeze(1)
                
                logits = self.vocab_proj(lstm_out)
                
                # Prevent stop token at first position
                if self.use_stop_token and t == 0:
                    logits[:, self.stop_token_id] = -float('inf')
                
                if self.training:
                    # Gumbel-softmax
                    gumbel_noise = -torch.log(-torch.log(
                        torch.rand_like(logits) + 1e-20) + 1e-20)
                    gumbel_logits = (logits + gumbel_noise) / temperature
                    soft_token = F.softmax(gumbel_logits, dim=-1)
                    
                    hard_token = F.one_hot(soft_token.argmax(dim=-1), 
                                        num_classes=self.effective_vocab_size).float()
                    
                    token_for_next_input = hard_token.detach() - soft_token.detach() + soft_token
                    
                    soft_messages.append(soft_token)
                    hard_symbol = soft_token.argmax(dim=-1)
                    hard_messages.append(hard_symbol)
                    
                    # Check for stop token
                    if self.use_stop_token and not stopped:
                        is_stop = (hard_symbol == self.stop_token_id)
                        if is_stop:
                            message_lengths[0] = t + 1
                            stopped = True
                else:
                    soft_token = F.softmax(logits, dim=-1)
                    hard_symbol = logits.argmax(dim=-1)
                    hard_token = F.one_hot(hard_symbol, num_classes=self.effective_vocab_size).float()
                    token_for_next_input = hard_token
                    
                    soft_messages.append(soft_token)
                    hard_messages.append(hard_symbol)
                    
                    # Check for stop token
                    if self.use_stop_token and not stopped:
                        is_stop = (hard_symbol == self.stop_token_id)
                        if is_stop:
                            message_lengths[0] = t + 1
                            stopped = True
                
                input_token = token_for_next_input
            
            message = torch.stack(hard_messages, dim=1)
            soft_message = torch.stack(soft_messages, dim=1)
            
            # Decode message to output
            output_h, output_w = test_output_size
            logits = self.rule_receiver(message, (output_h, output_w), 
                                       soft_message=soft_message,
                                       message_lengths=message_lengths)
            
            return logits, message, soft_message, message_lengths
        else:
            # Autoencoder mode
            combined = torch.cat([rule, test_input_enc], dim=1)
            output_h, output_w = test_output_size
            logits = self.rule_decoder(combined, (output_h, output_w))
            
            return logits, None, None, None
    
    def forward(self, train_inputs, train_input_sizes, train_outputs, train_output_sizes,
                test_input, test_input_size, test_output_size, temperature=1.0):
        """
        Full forward pass for a single puzzle.
        
        Args:
            train_inputs: List of [H, W] training input grids
            train_input_sizes: List of (h, w) actual sizes
            train_outputs: List of [H, W] training output grids  
            train_output_sizes: List of (h, w) actual sizes
            test_input: [H, W] test input grid
            test_input_size: (h, w) actual size
            test_output_size: (h, w) target output size
            
        Returns:
            logits: Output logits for test
            message: Message (if communication mode)
            soft_message: Soft message (if communication mode)
            message_lengths: Message lengths (if using stop tokens)
        """
        # Encode all training pairs
        example_encodings = []
        for inp, inp_size, out, out_size in zip(train_inputs, train_input_sizes, 
                                                 train_outputs, train_output_sizes):
            pair_enc = self.encode_pair(inp, inp_size, out, out_size)
            example_encodings.append(pair_enc)
        
        # Stack and aggregate
        example_encodings = torch.cat(example_encodings, dim=0)  # [num_examples, rule_dim]
        rule = self.aggregate_rule(example_encodings)
        
        # Apply rule to test input
        logits, message, soft_message, message_lengths = self.apply_rule(
            rule, test_input, test_input_size, test_output_size, temperature
        )
        
        return logits, message, soft_message, message_lengths, rule