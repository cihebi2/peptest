"""
Hierarchical Pooling for Graph Representations

Implements multiple pooling strategies to create comprehensive graph-level representations:
1. Global Attention Pooling
2. Set Transformer Pooling
3. Multi-layer aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class GlobalAttentionPooling(nn.Module):
    """
    Global attention pooling

    Uses a learnable attention mechanism to weight node importance
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_feat, batch_graph):
        """
        Args:
            node_feat: Node features [Total_N, D]
            batch_graph: DGL batched graph

        Returns:
            graph_repr: Graph representation [B, D]
        """
        # Compute attention scores for each node
        attn_scores = self.attention_gate(node_feat)  # [Total_N, 1]
        attn_scores = F.softmax(attn_scores, dim=0)

        # Weighted sum of node features
        weighted_feat = node_feat * attn_scores  # [Total_N, D]

        # Aggregate per graph in batch
        with batch_graph.local_scope():
            batch_graph.ndata['h'] = weighted_feat
            graph_repr = dgl.sum_nodes(batch_graph, 'h')  # [B, D]

        return graph_repr


class SetTransformer(nn.Module):
    """
    Set Transformer for permutation-invariant pooling

    Based on "Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks" (ICML 2019)
    """

    def __init__(self, hidden_dim, num_seeds=4, num_heads=4):
        super().__init__()
        self.num_seeds = num_seeds
        self.hidden_dim = hidden_dim

        # Learnable seed vectors
        self.seeds = nn.Parameter(torch.randn(num_seeds, hidden_dim))

        # Multi-head attention for seed aggregation
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_seeds * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, node_feat, batch_graph):
        """
        Args:
            node_feat: Node features [Total_N, D]
            batch_graph: DGL batched graph

        Returns:
            graph_repr: Graph representation [B, D]
        """
        batch_num_nodes = batch_graph.batch_num_nodes()
        batch_size = len(batch_num_nodes)
        device = node_feat.device

        # Expand seeds for batch
        seeds = self.seeds.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, S, D]

        # Split node features by graph
        node_feat_list = torch.split(node_feat, batch_num_nodes.tolist())

        # Pad to same length for batched attention
        max_nodes = max(batch_num_nodes)
        padded_node_feat = []
        masks = []

        for i, feat in enumerate(node_feat_list):
            num_nodes = feat.shape[0]
            if num_nodes < max_nodes:
                # Pad with zeros
                padding = torch.zeros(
                    max_nodes - num_nodes,
                    self.hidden_dim,
                    device=device
                )
                feat = torch.cat([feat, padding], dim=0)

            padded_node_feat.append(feat)

            # Create mask
            mask = torch.zeros(max_nodes, dtype=torch.bool, device=device)
            mask[num_nodes:] = True
            masks.append(mask)

        node_feat_batched = torch.stack(padded_node_feat)  # [B, N, D]
        key_padding_mask = torch.stack(masks)  # [B, N]

        # Apply attention: seeds attend to node features
        attended_seeds, _ = self.attention(
            query=seeds,  # [B, S, D]
            key=node_feat_batched,  # [B, N, D]
            value=node_feat_batched,
            key_padding_mask=key_padding_mask
        )  # [B, S, D]

        # Flatten and project
        attended_seeds = attended_seeds.reshape(batch_size, -1)  # [B, S*D]
        graph_repr = self.output_proj(attended_seeds)  # [B, D]

        return graph_repr


class HierarchicalPooling(nn.Module):
    """
    Hierarchical pooling that combines multiple strategies:
    1. Weighted layer aggregation
    2. Global attention pooling
    3. Set transformer pooling
    4. Mean pooling (as baseline)
    """

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Learnable weights for layer aggregation
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        # Different pooling strategies
        self.global_pool = GlobalAttentionPooling(hidden_dim)
        self.set_pool = SetTransformer(hidden_dim)

        # Multi-scale fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, layer_outputs, batch_graph):
        """
        Args:
            layer_outputs: List of node features from each layer [Total_N, D] * num_layers
            batch_graph: DGL batched graph

        Returns:
            graph_repr: Fused graph representation [B, D]
        """
        # 1. Weighted aggregation of layer outputs
        weighted_output = sum(
            w * output
            for w, output in zip(
                F.softmax(self.layer_weights, dim=0),
                layer_outputs
            )
        )  # [Total_N, D]

        # 2. Apply different pooling strategies
        global_repr = self.global_pool(weighted_output, batch_graph)  # [B, D]
        set_repr = self.set_pool(weighted_output, batch_graph)  # [B, D]

        # 3. Mean pooling as baseline
        with batch_graph.local_scope():
            batch_graph.ndata['h'] = layer_outputs[-1]  # Use last layer
            mean_repr = dgl.mean_nodes(batch_graph, 'h')  # [B, D]

        # 4. Concatenate and fuse
        multi_scale_repr = torch.cat([global_repr, set_repr, mean_repr], dim=-1)  # [B, 3*D]
        fused_repr = self.fusion(multi_scale_repr)  # [B, D]

        return fused_repr


class DifferentiablePooling(nn.Module):
    """
    DiffPool: Differentiable graph pooling

    Based on "Hierarchical Graph Representation Learning with
    Differentiable Pooling" (NeurIPS 2018)

    Note: This is a more advanced pooling method that learns
    cluster assignments
    """

    def __init__(self, input_dim, output_dim, num_clusters):
        super().__init__()
        self.num_clusters = num_clusters

        # GNN for cluster assignment
        self.assign_conv = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, num_clusters)
        )

        # GNN for feature embedding
        self.embed_conv = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x, adj):
        """
        Args:
            x: Node features [B, N, D]
            adj: Adjacency matrix [B, N, N]

        Returns:
            x_pooled: Pooled features [B, K, D']
            adj_pooled: Pooled adjacency [B, K, K]
            link_loss: Link prediction auxiliary loss
            entropy_loss: Entropy regularization loss
        """
        # Compute cluster assignment matrix S
        s = self.assign_conv(x)  # [B, N, K]
        s = F.softmax(s, dim=-1)

        # Compute embedding
        z = self.embed_conv(x)  # [B, N, D']

        # Pool features: X' = S^T @ Z
        x_pooled = torch.matmul(s.transpose(1, 2), z)  # [B, K, D']

        # Pool adjacency: A' = S^T @ A @ S
        adj_pooled = torch.matmul(
            torch.matmul(s.transpose(1, 2), adj),
            s
        )  # [B, K, K]

        # Auxiliary losses
        # Link prediction loss: ||A - SS^T||_F
        link_loss = torch.norm(
            adj - torch.matmul(s, s.transpose(1, 2)),
            p='fro'
        )

        # Entropy regularization
        entropy = -torch.sum(s * torch.log(s + 1e-8), dim=-1)
        entropy_loss = -torch.mean(entropy)

        return x_pooled, adj_pooled, link_loss, entropy_loss


if __name__ == '__main__':
    # Test pooling modules
    print("Testing Hierarchical Pooling...")

    import dgl

    # Create a batched graph
    g1 = dgl.graph(([0, 1, 2], [1, 2, 0]))
    g2 = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 0]))
    batched_g = dgl.batch([g1, g2])

    hidden_dim = 512
    num_nodes_total = batched_g.num_nodes()

    # Create node features
    node_feat = torch.randn(num_nodes_total, hidden_dim)

    # Test GlobalAttentionPooling
    global_pool = GlobalAttentionPooling(hidden_dim)
    global_repr = global_pool(node_feat, batched_g)
    print(f"Global pooling output shape: {global_repr.shape}")
    assert global_repr.shape == (2, hidden_dim), "Shape mismatch!"

    # Test SetTransformer
    set_pool = SetTransformer(hidden_dim)
    set_repr = set_pool(node_feat, batched_g)
    print(f"Set pooling output shape: {set_repr.shape}")
    assert set_repr.shape == (2, hidden_dim), "Shape mismatch!"

    # Test HierarchicalPooling
    num_layers = 12
    layer_outputs = [torch.randn(num_nodes_total, hidden_dim) for _ in range(num_layers)]
    hierarchical_pool = HierarchicalPooling(hidden_dim, num_layers)
    hier_repr = hierarchical_pool(layer_outputs, batched_g)
    print(f"Hierarchical pooling output shape: {hier_repr.shape}")
    assert hier_repr.shape == (2, hidden_dim), "Shape mismatch!"

    print("✓ All tests passed!")
