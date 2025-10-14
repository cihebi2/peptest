"""
Performer Attention Mechanism
Based on "Rethinking Attention with Performers" (ICLR 2021)
Reduces attention complexity from O(n²) to O(n)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PerformerAttention(nn.Module):
    """
    Performer Attention using FAVOR+ kernel approximation

    Key Features:
    - Linear complexity O(n) instead of O(n²)
    - Random feature approximation of softmax kernel
    - Maintains comparable performance to standard attention

    Args:
        hidden_dim: Hidden dimension size
        num_heads: Number of attention heads
        nb_features: Number of random features for approximation (default: 256)
        dropout: Dropout probability
    """

    def __init__(self, hidden_dim, num_heads, nb_features=256, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.nb_features = nb_features
        self.hidden_dim = hidden_dim

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Random projection matrix (fixed during inference)
        self.register_buffer(
            'projection_matrix',
            self._create_projection_matrix()
        )

    def _create_projection_matrix(self):
        """Create random projection matrix for FAVOR+"""
        return torch.randn(self.nb_features, self.head_dim)

    def kernel_feature_creator(self, data, projection_matrix, is_query):
        """
        FAVOR+ kernel feature mapping

        Maps input data to random features using Gaussian kernel approximation
        φ(x) = exp(xW - ||x||²/2) / √m

        Args:
            data: Input tensor [B, H, N, D/H]
            projection_matrix: Random projection [M, D/H]
            is_query: Whether this is query (for numerical stability)

        Returns:
            Kernel features [B, H, N, M]
        """
        # Normalize data for numerical stability
        data_normalizer = 1.0 / math.sqrt(math.sqrt(self.head_dim))
        data = data_normalizer * data

        # Projection: data @ W
        # [B, H, N, D/H] @ [D/H, M] -> [B, H, N, M]
        data_dash = torch.einsum('...nd,md->...nm', data, projection_matrix)

        # Compute ||x||²/2 for each position
        diag_data = torch.sum(data ** 2, dim=-1, keepdim=True) / 2.0
        diag_data = diag_data.expand_as(data_dash)

        # Compute kernel features: exp(xW - ||x||²/2)
        if is_query:
            # For numerical stability, use softmax-like normalization
            data_dash = data_dash - torch.max(data_dash, dim=-1, keepdim=True)[0]

        ratio = 1.0 / math.sqrt(self.nb_features)
        kernel_features = ratio * torch.exp(data_dash - diag_data)

        return kernel_features

    def forward(self, q, k, v, attn_mask=None):
        """
        Forward pass with linear attention

        Args:
            q: Query tensor [B, N, D]
            k: Key tensor [B, N, D]
            v: Value tensor [B, N, D]
            attn_mask: Optional attention mask (not fully supported in Performer)

        Returns:
            Output tensor [B, N, D]
        """
        B, N, D = q.shape
        H = self.num_heads

        # Project and reshape to multi-head format
        q = self.q_proj(q).view(B, N, H, self.head_dim).transpose(1, 2)  # [B, H, N, D/H]
        k = self.k_proj(k).view(B, N, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, N, H, self.head_dim).transpose(1, 2)

        # Apply kernel feature mapping (FAVOR+)
        q_prime = self.kernel_feature_creator(
            q, self.projection_matrix, is_query=True
        )  # [B, H, N, M]
        k_prime = self.kernel_feature_creator(
            k, self.projection_matrix, is_query=False
        )  # [B, H, N, M]

        # Linear attention with O(n) complexity
        # Standard attention: softmax(QK^T)V
        # Performer: (φ(Q)(φ(K)^TV)) / (φ(Q)(φ(K)^T1))

        # Compute K^TV: [B, H, M, D/H]
        kv = torch.einsum('...nm,...nd->...md', k_prime, v)

        # Compute normalization: K^T1: [B, H, M]
        k_sum = k_prime.sum(dim=2)

        # Compute attention output: Q(K^TV): [B, H, N, D/H]
        out = torch.einsum('...nm,...md->...nd', q_prime, kv)

        # Normalize: divide by Q(K^T1): [B, H, N]
        normalizer = torch.einsum('...nm,...m->...n', q_prime, k_sum)
        normalizer = normalizer.unsqueeze(-1)  # [B, H, N, 1]

        # Avoid division by zero
        normalizer = torch.clamp(normalizer, min=1e-6)
        out = out / normalizer

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.o_proj(out)
        out = self.dropout(out)

        return out


class PerformerEncoderLayer(nn.Module):
    """
    Complete Transformer encoder layer with Performer attention

    Includes:
    - Performer attention
    - Feed-forward network
    - Layer normalization
    - Residual connections
    """

    def __init__(self, hidden_dim, num_heads, ff_dim=None, dropout=0.1, nb_features=256):
        super().__init__()

        if ff_dim is None:
            ff_dim = hidden_dim * 4

        # Performer attention
        self.attention = PerformerAttention(
            hidden_dim, num_heads, nb_features, dropout
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, attn_mask=None):
        """
        Forward pass

        Args:
            x: Input tensor [B, N, D]
            attn_mask: Optional attention mask

        Returns:
            Output tensor [B, N, D]
        """
        # Attention with residual connection (Pre-LN)
        x_norm = self.norm1(x)
        x_attn = self.attention(x_norm, x_norm, x_norm, attn_mask)
        x = x + x_attn

        # FFN with residual connection
        x_norm = self.norm2(x)
        x_ffn = self.ffn(x_norm)
        x = x + x_ffn

        return x


if __name__ == '__main__':
    # Test Performer attention
    print("Testing Performer Attention...")

    batch_size = 4
    seq_len = 100
    hidden_dim = 512
    num_heads = 8

    # Create random input
    x = torch.randn(batch_size, seq_len, hidden_dim)

    # Test PerformerAttention
    performer_attn = PerformerAttention(hidden_dim, num_heads)
    out = performer_attn(x, x, x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"

    # Test PerformerEncoderLayer
    encoder_layer = PerformerEncoderLayer(hidden_dim, num_heads)
    out = encoder_layer(x)
    print(f"Encoder layer output shape: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"

    print("✓ All tests passed!")
