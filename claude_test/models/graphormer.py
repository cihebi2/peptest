"""
Graphormer Architecture
Based on Microsoft's Graphormer (NeurIPS 2021)
"Do Transformers Really Perform Bad for Graph Representation?"

Key Features:
1. Centrality Encoding (degree information)
2. Spatial Encoding (shortest path distance)
3. Edge Encoding in attention bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl


class MultiHeadAttentionWithEdge(nn.Module):
    """
    Multi-head attention with edge features integrated as bias

    This is the core innovation of Graphormer - edge features and
    spatial information are incorporated into attention computation
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_bias=None, attn_mask=None):
        """
        Args:
            q, k, v: [B, N, D]
            attn_bias: [B, N, N, H] or [B, H, N, N] - attention bias from edge/spatial encoding
            attn_mask: [B, N, N] - attention mask (optional)

        Returns:
            out: [B, N, D]
        """
        B, N, D = q.shape
        H = self.num_heads

        # Project and reshape: [B, N, D] -> [B, H, N, D/H]
        q = self.q_proj(q).view(B, N, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, N, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, N, H, self.head_dim).transpose(1, 2)

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, N, N]

        # Add edge/spatial bias if provided
        if attn_bias is not None:
            if attn_bias.dim() == 4 and attn_bias.shape[1] != H:
                # Shape: [B, N, N, H] -> [B, H, N, N]
                attn_bias = attn_bias.permute(0, 3, 1, 2)
            attn = attn + attn_bias

        # Apply attention mask if provided
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.unsqueeze(1) == 0, float('-inf'))

        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Aggregate values: [B, H, N, N] @ [B, H, N, D/H] -> [B, H, N, D/H]
        out = torch.matmul(attn, v)

        # Reshape and project: [B, H, N, D/H] -> [B, N, D]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.o_proj(out)

        return out


class GraphormerLayer(nn.Module):
    """
    Complete Graphormer layer with:
    - Centrality encoding (node degree)
    - Spatial encoding (shortest path distance)
    - Edge feature encoding
    - Multi-head attention with bias
    - Feed-forward network
    """

    def __init__(
        self,
        hidden_dim=512,
        num_heads=8,
        dropout=0.1,
        max_degree=100,
        max_spatial_pos=512,
        use_performer=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_performer = use_performer

        # 1. Multi-head attention (with edge features)
        if use_performer:
            from .performer import PerformerAttention
            self.attention = PerformerAttention(hidden_dim, num_heads, dropout=dropout)
        else:
            self.attention = MultiHeadAttentionWithEdge(hidden_dim, num_heads, dropout)

        # 2. Centrality encoding (degree encoding)
        # Separate embeddings for in-degree and out-degree
        self.in_degree_encoder = nn.Embedding(max_degree, hidden_dim)
        self.out_degree_encoder = nn.Embedding(max_degree, hidden_dim)

        # 3. Spatial encoding (shortest path distance)
        # Used as attention bias
        self.spatial_encoder = nn.Embedding(max_spatial_pos, num_heads)

        # 4. Edge feature encoding
        # This will be provided externally, just define the projection
        # The edge encoding is integrated into attention as bias

        # 5. Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        # 6. Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_attr_bias=None, spatial_pos=None, in_degree=None, out_degree=None):
        """
        Args:
            x: Node features [B, N, D]
            edge_attr_bias: Edge attention bias [B, N, N, H]
            spatial_pos: Shortest path distances [B, N, N]
            in_degree: In-degrees [B, N]
            out_degree: Out-degrees [B, N]

        Returns:
            x: Updated node features [B, N, D]
        """
        # Add centrality encoding to node features
        if in_degree is not None and out_degree is not None:
            # Clamp degrees to max_degree - 1
            in_degree = torch.clamp(in_degree, max=self.in_degree_encoder.num_embeddings - 1)
            out_degree = torch.clamp(out_degree, max=self.out_degree_encoder.num_embeddings - 1)

            x = x + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)

        # Compute attention bias from spatial encoding and edge features
        attn_bias = None
        if spatial_pos is not None or edge_attr_bias is not None:
            if spatial_pos is not None:
                # Clamp spatial positions
                spatial_pos = torch.clamp(
                    spatial_pos,
                    max=self.spatial_encoder.num_embeddings - 1
                )
                spatial_bias = self.spatial_encoder(spatial_pos)  # [B, N, N, H]
                attn_bias = spatial_bias

            if edge_attr_bias is not None:
                if attn_bias is None:
                    attn_bias = edge_attr_bias
                else:
                    attn_bias = attn_bias + edge_attr_bias

        # Self-attention with residual connection (Pre-LN)
        x_norm = self.norm1(x)
        if self.use_performer:
            # Performer doesn't support attn_bias in the same way
            x_attn = self.attention(x_norm, x_norm, x_norm)
        else:
            x_attn = self.attention(x_norm, x_norm, x_norm, attn_bias=attn_bias)
        x = x + x_attn

        # FFN with residual connection
        x_norm = self.norm2(x)
        x_ffn = self.ffn(x_norm)
        x = x + x_ffn

        return x


def compute_shortest_path_distance(graph, max_dist=512):
    """
    Compute shortest path distances for a batched DGL graph

    Args:
        graph: DGL graph (possibly batched)
        max_dist: Maximum distance to consider (clip larger distances)

    Returns:
        spatial_pos: Shortest path distance matrix [B, N, N]
    """
    # This is a simplified version - in practice, you'd use Floyd-Warshall
    # or BFS for each node
    # For now, return a dummy tensor
    # TODO: Implement actual shortest path computation

    num_nodes = graph.num_nodes()
    batch_num_nodes = graph.batch_num_nodes()
    batch_size = len(batch_num_nodes)

    # Initialize with max_dist (disconnected nodes)
    spatial_pos = torch.full(
        (batch_size, max(batch_num_nodes), max(batch_num_nodes)),
        max_dist - 1,
        dtype=torch.long,
        device=graph.device
    )

    # Set diagonal to 0
    for i in range(batch_size):
        n = batch_num_nodes[i]
        spatial_pos[i, :n, :n].fill_diagonal_(0)

    # For directly connected nodes, set distance to 1
    # This is a simplification - full implementation would use BFS
    src, dst = graph.edges()
    batch_ids = graph.batch_num_nodes()

    # TODO: Properly compute shortest paths
    # For now, just return the initialized matrix
    return spatial_pos


class GraphormerEncoder(nn.Module):
    """
    Stack of Graphormer layers

    This creates a complete Graphormer encoder with multiple layers
    """

    def __init__(
        self,
        num_layers=12,
        hidden_dim=512,
        num_heads=8,
        dropout=0.1,
        use_performer=False
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            GraphormerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_performer=use_performer
            )
            for _ in range(num_layers)
        ])

        self.num_layers = num_layers

    def forward(self, x, edge_attr_bias=None, spatial_pos=None, in_degree=None, out_degree=None):
        """
        Args:
            x: Node features [B, N, D]
            edge_attr_bias: Edge attention bias [B, N, N, H]
            spatial_pos: Shortest path distances [B, N, N]
            in_degree: In-degrees [B, N]
            out_degree: Out-degrees [B, N]

        Returns:
            layer_outputs: List of outputs from each layer
        """
        layer_outputs = []

        for layer in self.layers:
            x = layer(x, edge_attr_bias, spatial_pos, in_degree, out_degree)
            layer_outputs.append(x)

        return layer_outputs


if __name__ == '__main__':
    # Test Graphormer layer
    print("Testing Graphormer Layer...")

    batch_size = 4
    num_nodes = 50
    hidden_dim = 512
    num_heads = 8

    # Create random input
    x = torch.randn(batch_size, num_nodes, hidden_dim)
    spatial_pos = torch.randint(0, 20, (batch_size, num_nodes, num_nodes))
    in_degree = torch.randint(0, 10, (batch_size, num_nodes))
    out_degree = torch.randint(0, 10, (batch_size, num_nodes))

    # Test GraphormerLayer
    layer = GraphormerLayer(hidden_dim, num_heads)
    out = layer(x, spatial_pos=spatial_pos, in_degree=in_degree, out_degree=out_degree)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"

    # Test GraphormerEncoder
    encoder = GraphormerEncoder(num_layers=3, hidden_dim=hidden_dim, num_heads=num_heads)
    layer_outputs = encoder(x, spatial_pos=spatial_pos, in_degree=in_degree, out_degree=out_degree)
    print(f"Number of layer outputs: {len(layer_outputs)}")
    print(f"Last layer output shape: {layer_outputs[-1].shape}")

    print("✓ All tests passed!")
