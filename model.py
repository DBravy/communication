"""Communication model for ARC grids - Sender and Receiver agents."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ARCEncoder(nn.Module):
    """
    Encodes 30x30 ARC grids into fixed-size representations.
    Simplified CNN architecture.
    """
    def __init__(self, 
                 num_colors=10,
                 embedding_dim=10,
                 hidden_dim=128,
                 latent_dim=512,
                 num_conv_layers=3):
        super().__init__()
        
        self.num_colors = num_colors
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_conv_layers = num_conv_layers
        
        # One-hot encoding (no learnable parameters)
        # embedding_dim should equal num_colors
        assert embedding_dim == num_colors, "embedding_dim must equal num_colors for one-hot encoding"
        assert num_conv_layers >= 1, "Must have at least 1 convolutional layer"
        
        # Simple convolutional feature extraction (dynamic number of layers)
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for i in range(num_conv_layers):
            in_channels = embedding_dim if i == 0 else hidden_dim
            self.conv_layers.append(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
            )
            self.bn_layers.append(nn.BatchNorm2d(hidden_dim))
        
        # Pooling to fixed spatial size
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Project to latent representation
        self.fc = nn.Linear(hidden_dim * 4 * 4, latent_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, sizes=None):
        """
        Args:
            x: [batch, 30, 30] - grid with integer color values (padded)
            sizes: list of (H, W) tuples - actual grid sizes (optional)
        Returns:
            latent: [batch, latent_dim] - fixed-size representation
        """
        B = x.shape[0]
        
        # One-hot encode colors: [B, 30, 30] -> [B, 30, 30, num_colors]
        x = F.one_hot(x.long(), num_classes=self.num_colors).float()
        x = x.permute(0, 3, 1, 2)  # [B, num_colors, 30, 30]
        
        # Convolutional layers (dynamic)
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.relu(bn(conv(x)))  # [B, hidden_dim, 30, 30]
        
        # Pool only the actual content region, not the padding!
        if sizes is not None:
            # Adaptive pool per sample based on actual size
            pooled_features = []
            for i in range(B):
                h, w = sizes[i]
                # Extract only the actual content region
                content = x[i:i+1, :, :h, :w]  # [1, hidden_dim, h, w]
                # Pool this to 4x4
                pooled = self.pool(content)  # [1, hidden_dim, 4, 4]
                pooled_features.append(pooled)
            x = torch.cat(pooled_features, dim=0)  # [B, hidden_dim, 4, 4]
        else:
            # Fallback to regular pooling
            x = self.pool(x)
        
        x = x.reshape(B, -1)  # [B, hidden_dim * 16]
        x = self.dropout(x)
        latent = self.relu(self.fc(x))  # [B, latent_dim]
        
        return latent
    
    def extract_feature_maps(self, x, sizes=None):
        """
        Returns a dict of intermediate CNN feature maps for visualization.
        Args:
            x: [batch, 30, 30] (int color ids)
            sizes: list of (H, W) tuples - actual grid sizes (optional)
        Returns:
            feats: dict[str, torch.Tensor] with shapes [B, C, 30, 30] (except 'pooled' -> [B, C, 4, 4])
        """
        self.eval()
        with torch.no_grad():
            B = x.shape[0]

            # One-hot encode colors
            emb = F.one_hot(x.long(), num_classes=self.num_colors).float()
            emb = emb.permute(0, 3, 1, 2)  # [B, num_colors, 30, 30]

            # Convolutional layers (dynamic)
            feats = {"embed": emb}
            x_current = emb
            for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
                x_current = self.relu(bn(conv(x_current)))  # [B, hidden_dim, 30, 30]
                feats[f"conv{i+1}"] = x_current

            # Pool only the actual content region, not the padding!
            if sizes is not None:
                # Adaptive pool per sample based on actual size
                pooled_features = []
                for i in range(B):
                    h, w = sizes[i]
                    # Extract only the actual content region
                    content = x_current[i:i+1, :, :h, :w]  # [1, hidden_dim, h, w]
                    # Pool this to 4x4
                    pooled_sample = self.pool(content)  # [1, hidden_dim, 4, 4]
                    pooled_features.append(pooled_sample)
                pooled = torch.cat(pooled_features, dim=0)  # [B, hidden_dim, 4, 4]
            else:
                # Fallback to regular pooling
                pooled = self.pool(x_current)  # [B, hidden_dim, 4, 4]

            feats["pooled"] = pooled
            return feats



class ARCDecoder(nn.Module):
    """
    Decoder for autoencoder mode: reconstructs ARC grid from continuous latent vector.
    Target grid size is provided as input.
    """
    def __init__(self, latent_dim, num_colors, hidden_dim, max_grid_size=30):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_colors = num_colors
        self.hidden_dim = hidden_dim
        self.max_grid_size = max_grid_size
        
        # Project latent to spatial representation
        self.fc_decode = nn.Linear(latent_dim, hidden_dim * 4 * 4)
        
        # Upsampling decoder
        self.deconv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                         kernel_size=4, stride=2, padding=1)
        self.bn_d1 = nn.BatchNorm2d(hidden_dim)
        
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                         kernel_size=4, stride=2, padding=1)
        self.bn_d2 = nn.BatchNorm2d(hidden_dim)
        
        self.deconv3 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                         kernel_size=4, stride=2, padding=1)
        self.bn_d3 = nn.BatchNorm2d(hidden_dim)
        
        # Refinement
        self.conv_out1 = nn.Conv2d(hidden_dim, hidden_dim,
                                   kernel_size=3, padding=1)
        self.bn_out1 = nn.BatchNorm2d(hidden_dim)
        
        # Output layer
        self.conv_out = nn.Conv2d(hidden_dim, num_colors, kernel_size=1)
        
        self.relu = nn.ReLU()
    
    def forward(self, latent, target_size):
        """
        Args:
            latent: [1, latent_dim] - continuous latent vector (single sample only)
            target_size: (H, W) - target grid size to reconstruct
        Returns:
            logits: [1, num_colors, H, W] - reconstruction logits
        """
        # Use provided target size
        H, W = target_size
        
        # Decode to spatial representation
        x_dec = self.relu(self.fc_decode(latent))
        x_dec = x_dec.reshape(1, self.hidden_dim, 4, 4)
        
        # Upsample
        x_dec = self.relu(self.bn_d1(self.deconv1(x_dec)))  # 8x8
        x_dec = self.relu(self.bn_d2(self.deconv2(x_dec)))  # 16x16
        x_dec = self.relu(self.bn_d3(self.deconv3(x_dec)))  # 32x32
        
        # Refine
        x_dec = self.relu(self.bn_out1(self.conv_out1(x_dec)))
        
        # Output
        logits = self.conv_out(x_dec)
        
        # Interpolate to target size
        if H != logits.shape[2] or W != logits.shape[3]:
            logits = F.interpolate(logits, size=(H, W),
                                  mode='bilinear', align_corners=False)
        
        return logits


class SenderAgent(nn.Module):
    """
    Sender agent: encodes ARC grid into discrete symbol sequence.
    """
    def __init__(self, encoder, vocab_size, max_length):
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # LSTM for generating message
        self.lstm = nn.LSTM(vocab_size, encoder.latent_dim, batch_first=True)
        
        # Project to vocabulary logits
        self.vocab_proj = nn.Linear(encoder.latent_dim, vocab_size)
        
    def forward(self, grids, sizes=None, temperature=1.0):
        """
        Args:
            grids: [batch, H, W] - input grids
            sizes: list of (H, W) tuples - actual grid sizes (optional)
            temperature: Gumbel-softmax temperature
        Returns:
            message: [batch, max_length] - discrete symbols
            message_probs: [batch, max_length, vocab_size] - probabilities
        """
        B = grids.shape[0]
        
        # Encode grid to initial hidden state
        latent = self.encoder(grids, sizes=sizes)  # [B, latent_dim]
        
        # Initialize LSTM hidden state
        h = latent.unsqueeze(0)  # [1, B, latent_dim]
        c = torch.zeros_like(h)
        
        # Generate message token by token
        messages = []
        message_probs = []
        
        # Start with zeros (could also use a learned start token)
        input_token = torch.zeros(B, self.vocab_size, device=grids.device)
        
        for t in range(self.max_length):
            # LSTM step
            input_token = input_token.unsqueeze(1)  # [B, 1, vocab_size]
            lstm_out, (h, c) = self.lstm(input_token, (h, c))
            lstm_out = lstm_out.squeeze(1)  # [B, latent_dim]
            
            # Get vocabulary logits
            logits = self.vocab_proj(lstm_out)  # [B, vocab_size]
            
            # Gumbel-softmax with straight-through
            probs = F.softmax(logits, dim=-1)
            
            if self.training:
                # Sample with Gumbel-softmax
                gumbel_noise = -torch.log(-torch.log(
                    torch.rand_like(logits) + 1e-20) + 1e-20)
                gumbel_logits = (logits + gumbel_noise) / temperature
                soft_token = F.softmax(gumbel_logits, dim=-1)
                
                # Straight-through: discrete forward, continuous backward
                hard_token = F.one_hot(soft_token.argmax(dim=-1), 
                                      num_classes=self.vocab_size).float()
                token = hard_token - soft_token.detach() + soft_token
            else:
                # At test time, use argmax
                token = F.one_hot(logits.argmax(dim=-1), 
                                 num_classes=self.vocab_size).float()
            
            messages.append(token.argmax(dim=-1))
            message_probs.append(probs)
            
            # Use token as next input
            input_token = token
        
        message = torch.stack(messages, dim=1)  # [B, max_length]
        message_probs = torch.stack(message_probs, dim=1)  # [B, max_length, vocab_size]
        
        return message, message_probs


class ReceiverAgent(nn.Module):
    """
    Receiver agent: reconstructs ARC grid from discrete symbol sequence.
    Target grid size is provided as input.
    """
    def __init__(self, vocab_size, num_colors, hidden_dim, max_grid_size=30):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_colors = num_colors
        self.hidden_dim = hidden_dim
        self.max_grid_size = max_grid_size
        
        # Embed message symbols
        self.symbol_embed = nn.Embedding(vocab_size, hidden_dim)
        
        # LSTM to process message
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Project to decoder initial state
        self.fc_decode = nn.Linear(hidden_dim, hidden_dim * 4 * 4)
        
        # Upsampling decoder
        self.deconv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                         kernel_size=4, stride=2, padding=1)
        self.bn_d1 = nn.BatchNorm2d(hidden_dim)
        
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                         kernel_size=4, stride=2, padding=1)
        self.bn_d2 = nn.BatchNorm2d(hidden_dim)
        
        self.deconv3 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                         kernel_size=4, stride=2, padding=1)
        self.bn_d3 = nn.BatchNorm2d(hidden_dim)
        
        # Refinement
        self.conv_out1 = nn.Conv2d(hidden_dim, hidden_dim,
                                   kernel_size=3, padding=1)
        self.bn_out1 = nn.BatchNorm2d(hidden_dim)
        
        # Output layer
        self.conv_out = nn.Conv2d(hidden_dim, num_colors, kernel_size=1)
        
        self.relu = nn.ReLU()
    
    def forward(self, message, target_size):
        """
        Args:
            message: [1, max_length] - discrete symbols (single sample only)
            target_size: (H, W) - target grid size to reconstruct
        Returns:
            logits: [1, num_colors, H, W] - reconstruction logits
        """
        # Embed message
        embedded = self.symbol_embed(message)  # [1, max_length, hidden_dim]
        
        # Process with LSTM
        lstm_out, (h, c) = self.lstm(embedded)
        
        # Use final hidden state
        message_repr = h.squeeze(0)  # [1, hidden_dim]
        
        # Use provided target size
        H, W = target_size
        
        # Decode to spatial representation
        x_dec = self.relu(self.fc_decode(message_repr))
        x_dec = x_dec.reshape(1, self.hidden_dim, 4, 4)
        
        # Upsample
        x_dec = self.relu(self.bn_d1(self.deconv1(x_dec)))  # 8x8
        x_dec = self.relu(self.bn_d2(self.deconv2(x_dec)))  # 16x16
        x_dec = self.relu(self.bn_d3(self.deconv3(x_dec)))  # 32x32
        
        # Refine
        x_dec = self.relu(self.bn_out1(self.conv_out1(x_dec)))
        
        # Output
        logits = self.conv_out(x_dec)
        
        # Interpolate to target size
        if H != logits.shape[2] or W != logits.shape[3]:
            logits = F.interpolate(logits, size=(H, W),
                                  mode='bilinear', align_corners=False)
        
        return logits


class ReceiverPuzzleClassifier(nn.Module):
    """
    Receiver agent for puzzle classification task: classifies puzzle from message.
    """
    def __init__(self, vocab_size, num_classes, hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Embed message symbols
        self.symbol_embed = nn.Embedding(vocab_size, hidden_dim)
        
        # LSTM to process message
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, message):
        """
        Args:
            message: [batch, max_length] - discrete symbols
        Returns:
            logits: [batch, num_classes] - classification logits
        """
        # Embed and process message
        msg_emb = self.symbol_embed(message)  # [batch, max_length, hidden_dim]
        _, (h, _) = self.lstm(msg_emb)  # h: [1, batch, hidden_dim]
        msg_repr = h.squeeze(0)  # [batch, hidden_dim]
        
        # Classify
        logits = self.classifier(msg_repr)  # [batch, num_classes]
        return logits


class ReceiverSelector(nn.Module):
    """
    Receiver agent for selection task: selects correct grid from candidates based on message.
    """
    def __init__(self, vocab_size, num_colors, embedding_dim, hidden_dim, num_conv_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_colors = num_colors
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        
        # One-hot encoding (no learnable parameters)
        # embedding_dim should equal num_colors
        assert embedding_dim == num_colors, "embedding_dim must equal num_colors for one-hot encoding"
        assert num_conv_layers >= 1, "Must have at least 1 convolutional layer"
        
        # Embed message symbols
        self.symbol_embed = nn.Embedding(vocab_size, hidden_dim)
        
        # LSTM to process message
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Encoder for candidate grids (dynamic number of conv layers)
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
        
        # Scoring: compare message representation with grid representations
        self.score_fc = nn.Linear(hidden_dim * 2, 1)
        
        self.relu = nn.ReLU()
        
    def encode_candidates(self, candidates, candidate_sizes=None):
        """
        Encode candidate grids into representations.
        Args:
            candidates: [num_candidates, H, W]
            candidate_sizes: list of (H, W) tuples - actual sizes for each candidate (optional)
        Returns:
            representations: [num_candidates, hidden_dim]
        """
        N, H, W = candidates.shape
        
        # One-hot encode colors
        x = F.one_hot(candidates.long(), num_classes=self.num_colors).float()  # [N, H, W, num_colors]
        x = x.permute(0, 3, 1, 2)  # [N, num_colors, H, W]
        
        # Convolutional encoding (dynamic)
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.relu(bn(conv(x)))
        
        # Pool only the actual content region, not the padding!
        if candidate_sizes is not None:
            # Adaptive pool per candidate based on actual size
            pooled_features = []
            for i in range(N):
                h, w = candidate_sizes[i]
                # Extract only the actual content region
                content = x[i:i+1, :, :h, :w]  # [1, hidden_dim, h, w]
                # Pool this to 4x4
                pooled = self.adaptive_pool(content)  # [1, hidden_dim, 4, 4]
                pooled_features.append(pooled)
            x = torch.cat(pooled_features, dim=0)  # [N, hidden_dim, 4, 4]
        else:
            # Fallback to regular pooling
            x = self.adaptive_pool(x)  # [N, hidden_dim, 4, 4]
        
        x = x.reshape(N, -1)  # [N, hidden_dim * 16]
        x = self.relu(self.grid_fc(x))  # [N, hidden_dim]
        
        return x
    
    def forward(self, message, candidates, candidate_sizes=None):
        """
        Args:
            message: [1, max_length] - discrete symbols
            candidates: [num_candidates, H, W] - candidate grids
            candidate_sizes: list of (H, W) tuples - actual sizes for each candidate (optional)
        Returns:
            logits: [num_candidates] - selection logits
        """
        # Embed and process message
        msg_emb = self.symbol_embed(message)  # [1, max_length, hidden_dim]
        _, (h, _) = self.lstm(msg_emb)  # h: [1, 1, hidden_dim]
        msg_repr = h.squeeze(0)  # [1, hidden_dim]
        
        # Encode all candidates
        cand_repr = self.encode_candidates(candidates, candidate_sizes=candidate_sizes)  # [num_candidates, hidden_dim]
        
        # Compute similarity scores
        num_candidates = candidates.shape[0]
        msg_repr_expanded = msg_repr.expand(num_candidates, -1)  # [num_candidates, hidden_dim]
        
        # Concatenate message and candidate representations
        combined = torch.cat([msg_repr_expanded, cand_repr], dim=-1)  # [num_candidates, hidden_dim*2]
        
        # Score each candidate
        logits = self.score_fc(combined).squeeze(-1)  # [num_candidates]
        
        return logits


class DecoderSelector(nn.Module):
    """
    Decoder for selection task in autoencoder mode: selects correct grid from candidates based on latent.
    """
    def __init__(self, latent_dim, num_colors, embedding_dim, hidden_dim, num_conv_layers=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_colors = num_colors
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        
        # One-hot encoding (no learnable parameters)
        # embedding_dim should equal num_colors
        assert embedding_dim == num_colors, "embedding_dim must equal num_colors for one-hot encoding"
        assert num_conv_layers >= 1, "Must have at least 1 convolutional layer"
        
        # Project latent to hidden
        self.latent_fc = nn.Linear(latent_dim, hidden_dim)
        
        # Encoder for candidate grids (dynamic number of conv layers)
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
        
        # Scoring
        self.score_fc = nn.Linear(hidden_dim * 2, 1)
        
        self.relu = nn.ReLU()
        
    def encode_candidates(self, candidates, candidate_sizes=None):
        """
        Encode candidate grids into representations.
        Args:
            candidates: [num_candidates, H, W]
            candidate_sizes: list of (H, W) tuples - actual sizes for each candidate (optional)
        Returns:
            representations: [num_candidates, hidden_dim]
        """
        N, H, W = candidates.shape
        
        # One-hot encode colors
        x = F.one_hot(candidates.long(), num_classes=self.num_colors).float()
        x = x.permute(0, 3, 1, 2)
        
        # Convolutional encoding (dynamic)
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.relu(bn(conv(x)))
        
        # Pool only the actual content region, not the padding!
        if candidate_sizes is not None:
            # Adaptive pool per candidate based on actual size
            pooled_features = []
            for i in range(N):
                h, w = candidate_sizes[i]
                # Extract only the actual content region
                content = x[i:i+1, :, :h, :w]  # [1, hidden_dim, h, w]
                # Pool this to 4x4
                pooled = self.adaptive_pool(content)  # [1, hidden_dim, 4, 4]
                pooled_features.append(pooled)
            x = torch.cat(pooled_features, dim=0)  # [N, hidden_dim, 4, 4]
        else:
            # Fallback to regular pooling
            x = self.adaptive_pool(x)
        
        x = x.reshape(N, -1)
        x = self.relu(self.grid_fc(x))
        
        return x
    
    def forward(self, latent, candidates, candidate_sizes=None):
        """
        Args:
            latent: [1, latent_dim] - latent representation
            candidates: [num_candidates, H, W] - candidate grids
            candidate_sizes: list of (H, W) tuples - actual sizes for each candidate (optional)
        Returns:
            logits: [num_candidates] - selection logits
        """
        # Process latent
        latent_repr = self.relu(self.latent_fc(latent))  # [1, hidden_dim]
        
        # Encode all candidates
        cand_repr = self.encode_candidates(candidates, candidate_sizes=candidate_sizes)  # [num_candidates, hidden_dim]
        
        # Compute similarity scores
        num_candidates = candidates.shape[0]
        latent_repr_expanded = latent_repr.expand(num_candidates, -1)
        
        combined = torch.cat([latent_repr_expanded, cand_repr], dim=-1)
        logits = self.score_fc(combined).squeeze(-1)
        
        return logits


class ARCAutoencoder(nn.Module):
    """
    Flexible bottleneck system: supports both communication and autoencoder modes.
    Also supports reconstruction, selection, and puzzle_classification tasks.
    
    - Communication mode: Sender encodes grid to discrete message, Receiver reconstructs/selects/classifies.
    - Autoencoder mode: Encoder → latent vector → Decoder reconstructs/selects.
    
    - Reconstruction task: Receiver/Decoder reconstructs the grid.
    - Selection task: Receiver/Decoder selects correct grid from candidates.
    - Puzzle classification task: Receiver classifies puzzle category from message.
    
    Receiver/Decoder is given the target size (reconstruction) or candidates (selection).
    Processes samples one at a time.
    """
    def __init__(self, encoder, vocab_size, max_length, num_colors, embedding_dim, hidden_dim, 
                 max_grid_size=30, bottleneck_type='communication', task_type='reconstruction', 
                 num_conv_layers=2, num_classes=None):
        super().__init__()
        self.encoder = encoder
        self.bottleneck_type = bottleneck_type
        self.task_type = task_type
        
        if bottleneck_type == 'communication':
            self.sender = SenderAgent(encoder, vocab_size, max_length)
            if task_type == 'reconstruction':
                self.receiver = ReceiverAgent(vocab_size, num_colors, hidden_dim, max_grid_size)
            elif task_type == 'selection':
                self.receiver = ReceiverSelector(vocab_size, num_colors, embedding_dim, hidden_dim, num_conv_layers)
            elif task_type == 'puzzle_classification':
                assert num_classes is not None, "num_classes must be provided for puzzle_classification task"
                self.receiver = ReceiverPuzzleClassifier(vocab_size, num_classes, hidden_dim)
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
        elif bottleneck_type == 'autoencoder':
            if task_type == 'reconstruction':
                self.decoder = ARCDecoder(encoder.latent_dim, num_colors, hidden_dim, max_grid_size)
            elif task_type == 'selection':
                self.decoder = DecoderSelector(encoder.latent_dim, num_colors, embedding_dim, hidden_dim, num_conv_layers)
            elif task_type == 'puzzle_classification':
                raise NotImplementedError("Puzzle classification not yet supported in autoencoder mode")
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
        else:
            raise ValueError(f"Unknown bottleneck_type: {bottleneck_type}. "
                           f"Must be 'communication' or 'autoencoder'")
    
    def forward(self, x, sizes, temperature=1.0, candidates_list=None, candidates_sizes_list=None, target_indices=None, labels=None):
        """
        Args:
            x: [batch, H, W] input grids (padded)
            sizes: list of (H, W) tuples with actual grid sizes
            temperature: Gumbel-softmax temperature (only used in communication mode)
            candidates_list: (selection task only) list of [num_candidates, H, W] tensors
            candidates_sizes_list: (selection task only) list of lists of (H, W) tuples for each candidate
            target_indices: (selection task only) [batch] indices of correct grid in candidates
            labels: (puzzle_classification task only) [batch] puzzle classification labels
        
        Returns:
            For reconstruction task:
                logits_list: list of [1, num_colors, H, W] reconstruction logits per sample
                actual_sizes: list of (H, W) actual input grid sizes
                messages: [batch, max_length] discrete messages (only in communication mode, else None)
            
            For selection task:
                selection_logits: [batch, num_candidates] selection logits
                actual_sizes: list of (H, W) actual input grid sizes
                messages: [batch, max_length] discrete messages (only in communication mode, else None)
            
            For puzzle_classification task:
                classification_logits: [batch, num_classes] classification logits
                actual_sizes: list of (H, W) actual input grid sizes
                messages: [batch, max_length] discrete messages (only in communication mode, else None)
        """
        B = x.shape[0]
        
        if self.task_type == 'reconstruction':
            # Original reconstruction task
            if self.bottleneck_type == 'communication':
                # Sender creates messages for all samples
                messages, message_probs = self.sender(x, sizes=sizes, temperature=temperature)  # [B, max_length]
                
                # Process each sample individually through receiver
                logits_list = []
                actual_sizes = []
                
                for i in range(B):
                    single_message = messages[i:i+1]  # [1, max_length]
                    actual_h, actual_w = sizes[i]
                    
                    # Receiver reconstructs with given target size
                    logits = self.receiver(single_message, target_size=(actual_h, actual_w))
                    
                    logits_list.append(logits)
                    actual_sizes.append((actual_h, actual_w))
                
                return logits_list, actual_sizes, messages
                
            else:  # autoencoder mode
                # Encoder creates latent vectors for all samples
                latent = self.encoder(x, sizes=sizes)  # [B, latent_dim]
                
                # Process each sample individually through decoder
                logits_list = []
                actual_sizes = []
                
                for i in range(B):
                    single_latent = latent[i:i+1]  # [1, latent_dim]
                    actual_h, actual_w = sizes[i]
                    
                    # Decoder reconstructs with given target size
                    logits = self.decoder(single_latent, target_size=(actual_h, actual_w))
                    
                    logits_list.append(logits)
                    actual_sizes.append((actual_h, actual_w))
                
                return logits_list, actual_sizes, None
        
        elif self.task_type == 'selection':
            if self.bottleneck_type == 'communication':
                # Sender creates messages for all samples
                messages, message_probs = self.sender(x, sizes=sizes, temperature=temperature)  # [B, max_length]
                
                # Process each sample individually through receiver selector
                selection_logits_list = []
                actual_sizes = []
                
                for i in range(B):
                    single_message = messages[i:i+1]  # [1, max_length]
                    candidates = candidates_list[i]  # [num_candidates, H, W]
                    candidate_sizes = candidates_sizes_list[i] if candidates_sizes_list is not None else None
                    actual_h, actual_w = sizes[i]
                    
                    # Receiver selects from candidates
                    sel_logits = self.receiver(single_message, candidates, candidate_sizes=candidate_sizes)  # [num_candidates]
                    
                    selection_logits_list.append(sel_logits)
                    actual_sizes.append((actual_h, actual_w))
                
                return selection_logits_list, actual_sizes, messages
                
            else:  # autoencoder mode
                # Encoder creates latent vectors for all samples
                latent = self.encoder(x, sizes=sizes)  # [B, latent_dim]
                
                # Process each sample individually through decoder selector
                selection_logits_list = []
                actual_sizes = []
                
                for i in range(B):
                    single_latent = latent[i:i+1]  # [1, latent_dim]
                    candidates = candidates_list[i]  # [num_candidates, H, W]
                    candidate_sizes = candidates_sizes_list[i] if candidates_sizes_list is not None else None
                    actual_h, actual_w = sizes[i]
                    
                    # Decoder selects from candidates
                    sel_logits = self.decoder(single_latent, candidates, candidate_sizes=candidate_sizes)  # [num_candidates]
                    
                    selection_logits_list.append(sel_logits)
                    actual_sizes.append((actual_h, actual_w))
                
                return selection_logits_list, actual_sizes, None
        
        else:  # puzzle_classification task
            if self.bottleneck_type == 'communication':
                # Sender creates messages for all samples
                messages, message_probs = self.sender(x, sizes=sizes, temperature=temperature)  # [B, max_length]
                
                # Receiver classifies from messages
                classification_logits = self.receiver(messages)  # [B, num_classes]
                
                return classification_logits, sizes, messages
            
            else:  # autoencoder mode
                raise NotImplementedError("Puzzle classification not yet supported in autoencoder mode")
    
    