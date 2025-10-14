"""
Improved PepLand Model

Integrates all improvements:
1. Graphormer architecture with centrality/spatial/edge encoding
2. Optional Performer attention for efficiency
3. Hierarchical pooling
4. Enhanced heterogeneous graph modeling
5. Virtual nodes
6. Deeper network (12 layers) with larger hidden dim (512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from .graphormer import GraphormerEncoder, compute_shortest_path_distance
from .hierarchical_pool import HierarchicalPooling


class EnhancedHeteroGraph(nn.Module):
    """
    Enhanced heterogeneous graph modeling with:
    1. Virtual super-node connecting all nodes
    2. Edge-to-node feature propagation
    3. Multi-view representations (atom + fragment)
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Virtual node embedding (learnable)
        self.virtual_node_emb = nn.Parameter(torch.randn(1, hidden_dim))

        # Virtual node update MLP
        self.virtual_node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Edge-to-node feature propagation
        self.edge_to_node = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, g, node_feat, edge_feat=None):
        """
        Args:
            g: DGL graph (batched)
            node_feat: Node features [Total_N, D]
            edge_feat: Edge features [Total_E, D] (optional)

        Returns:
            node_feat: Updated node features [Total_N, D]
            virtual_node: Virtual node features [B, D]
        """
        batch_size = g.batch_size if hasattr(g, 'batch_size') else 1
        device = node_feat.device

        # 1. Initialize virtual node for each graph
        virtual_node = self.virtual_node_emb.repeat(batch_size, 1)  # [B, D]

        # 2. Aggregate node features to virtual node (mean pooling)
        with g.local_scope():
            g.ndata['h'] = node_feat
            graph_mean = dgl.mean_nodes(g, 'h')  # [B, D]

        # 3. Update virtual node
        virtual_node_input = torch.cat([virtual_node, graph_mean], dim=-1)
        virtual_node = self.virtual_node_mlp(virtual_node_input)  # [B, D]

        # 4. Broadcast virtual node back to all nodes
        # Map each node to its graph index
        batch_num_nodes = g.batch_num_nodes() if hasattr(g, 'batch_num_nodes') else [g.num_nodes()]
        node_graph_indices = torch.cat([
            torch.full((n,), i, dtype=torch.long, device=device)
            for i, n in enumerate(batch_num_nodes)
        ])

        node_feat = node_feat + virtual_node[node_graph_indices]

        # 5. Propagate edge features to nodes (if provided)
        if edge_feat is not None and g.num_edges() > 0:
            with g.local_scope():
                g.edata['e'] = self.edge_to_node(edge_feat)
                g.update_all(
                    dgl.function.copy_e('e', 'm'),
                    dgl.function.mean('m', 'edge_agg')
                )
                if 'edge_agg' in g.ndata:
                    node_feat = node_feat + g.ndata['edge_agg']

        return node_feat, virtual_node


class ImprovedPepLand(nn.Module):
    """
    Improved PepLand model with state-of-the-art graph neural network techniques

    Key improvements over original PepLand:
    1. Graphormer architecture (vs HGT)
       - Centrality encoding (degree information)
       - Spatial encoding (shortest path)
       - Edge feature integration
    2. Deeper network (12 layers vs 5)
    3. Larger hidden dimension (512 vs 300)
    4. Optional Performer attention for efficiency
    5. Hierarchical pooling (vs simple GRU pooling)
    6. Virtual nodes for global information
    """

    def __init__(
        self,
        # Input dimensions
        atom_dim=42,
        bond_dim=14,
        fragment_dim=196,
        # Model architecture
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
        dropout=0.1,
        # Options
        use_performer=False,
        use_virtual_node=True,
        # Graphormer settings
        max_degree=100,
        max_spatial_pos=512,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_virtual_node = use_virtual_node

        # ============= Input Encoding =============
        # Encode different node types
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.fragment_encoder = nn.Sequential(
            nn.Linear(fragment_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Encode edge features
        self.bond_encoder = nn.Sequential(
            nn.Linear(bond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Edge features -> attention bias
        self.edge_bias_proj = nn.Linear(hidden_dim, num_heads)

        # ============= Enhanced Heterogeneous Graph =============
        if use_virtual_node:
            self.hetero_graph_layer = EnhancedHeteroGraph(hidden_dim)

        # ============= Graphormer Encoder =============
        self.encoder = GraphormerEncoder(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_performer=use_performer
        )

        # ============= Hierarchical Pooling =============
        self.pooling = HierarchicalPooling(hidden_dim, num_layers)

        # ============= Output Projection =============
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def encode_graph(self, batch_graph):
        """
        Encode node and edge features

        Args:
            batch_graph: DGL batched graph with:
                - ndata['atom_feat']: Atom features
                - ndata['fragment_feat']: Fragment features (optional)
                - ndata['node_type']: Node type (0=atom, 1=fragment)
                - edata['bond_feat']: Edge features

        Returns:
            node_feat: Encoded node features [Total_N, D]
            edge_feat: Encoded edge features [Total_E, D]
        """
        # Encode atoms and fragments separately
        node_types = batch_graph.ndata.get('node_type', None)

        if node_types is not None:
            # Heterogeneous graph with both atoms and fragments
            is_atom = (node_types == 0)
            is_fragment = (node_types == 1)

            node_feat = torch.zeros(
                batch_graph.num_nodes(),
                self.hidden_dim,
                device=batch_graph.device
            )

            # Check if tensor (could be single bool if only 1 node)
            if isinstance(is_atom, bool):
                is_atom = torch.tensor([is_atom], device=batch_graph.device)
            if isinstance(is_fragment, bool):
                is_fragment = torch.tensor([is_fragment], device=batch_graph.device)

            if is_atom.any():
                node_feat[is_atom] = self.atom_encoder(batch_graph.ndata['atom_feat'][is_atom])

            if is_fragment.any() and 'fragment_feat' in batch_graph.ndata:
                node_feat[is_fragment] = self.fragment_encoder(
                    batch_graph.ndata['fragment_feat'][is_fragment]
                )
        else:
            # Homogeneous graph (all atoms)
            node_feat = self.atom_encoder(batch_graph.ndata['atom_feat'])

        # Encode edges
        if batch_graph.num_edges() > 0 and 'bond_feat' in batch_graph.edata:
            edge_feat = self.bond_encoder(batch_graph.edata['bond_feat'])
        else:
            edge_feat = None

        return node_feat, edge_feat

    def compute_structural_encodings(self, batch_graph):
        """
        Compute structural encodings for Graphormer

        Args:
            batch_graph: DGL graph

        Returns:
            in_degree: In-degree for each node [Total_N]
            out_degree: Out-degree for each node [Total_N]
            spatial_pos: Shortest path distances [B, max_N, max_N]
            edge_attr_bias: Edge attention bias [B, max_N, max_N, H]
        """
        # Compute degrees
        in_degree = batch_graph.in_degrees()
        out_degree = batch_graph.out_degrees()

        # Compute shortest paths
        # Note: This is computationally expensive, consider caching
        spatial_pos = compute_shortest_path_distance(batch_graph)

        # Compute edge attention bias from edge features
        # This requires converting edge features to a dense matrix
        edge_attr_bias = None
        if batch_graph.num_edges() > 0 and 'edge_encoded' in batch_graph.edata:
            # Project edge features to attention bias
            edge_bias_values = self.edge_bias_proj(
                batch_graph.edata['edge_encoded']
            )  # [Total_E, H]

            # Convert to dense matrix
            # TODO: Implement efficient sparse-to-dense conversion
            # For now, return None
            edge_attr_bias = None

        return in_degree, out_degree, spatial_pos, edge_attr_bias

    def forward(self, batch_graph):
        """
        Forward pass

        Args:
            batch_graph: DGL batched graph

        Returns:
            graph_repr: Graph-level representation [B, D]
        """
        # 1. Encode node and edge features
        node_feat, edge_feat = self.encode_graph(batch_graph)

        # Store encoded edge features for later use
        if edge_feat is not None:
            batch_graph.edata['edge_encoded'] = edge_feat

        # 2. Apply enhanced heterogeneous graph layer (with virtual nodes)
        if self.use_virtual_node:
            node_feat, virtual_node = self.hetero_graph_layer(
                batch_graph, node_feat, edge_feat
            )

        # 3. Compute structural encodings for Graphormer
        in_degree, out_degree, spatial_pos, edge_attr_bias = \
            self.compute_structural_encodings(batch_graph)

        # 4. Reshape node features for batched processing
        # Graphormer expects [B, N, D] format
        batch_num_nodes = batch_graph.batch_num_nodes() if hasattr(batch_graph, 'batch_num_nodes') \
            else [batch_graph.num_nodes()]
        batch_size = len(batch_num_nodes)
        max_num_nodes = max(batch_num_nodes)

        # Pad node features to [B, max_N, D]
        padded_node_feat = torch.zeros(
            batch_size, max_num_nodes, self.hidden_dim,
            device=node_feat.device
        )

        node_splits = torch.split(node_feat, batch_num_nodes.tolist())
        for i, nodes in enumerate(node_splits):
            padded_node_feat[i, :nodes.shape[0]] = nodes

        # Similarly prepare in_degree and out_degree
        padded_in_degree = torch.zeros(batch_size, max_num_nodes, dtype=torch.long, device=node_feat.device)
        padded_out_degree = torch.zeros(batch_size, max_num_nodes, dtype=torch.long, device=node_feat.device)

        in_degree_splits = torch.split(in_degree, batch_num_nodes.tolist())
        out_degree_splits = torch.split(out_degree, batch_num_nodes.tolist())

        for i, (in_deg, out_deg) in enumerate(zip(in_degree_splits, out_degree_splits)):
            padded_in_degree[i, :in_deg.shape[0]] = in_deg
            padded_out_degree[i, :out_deg.shape[0]] = out_deg

        # 5. Apply Graphormer encoder
        layer_outputs = self.encoder(
            padded_node_feat,
            edge_attr_bias=edge_attr_bias,
            spatial_pos=spatial_pos,
            in_degree=padded_in_degree,
            out_degree=padded_out_degree
        )

        # 6. Flatten layer outputs back to [Total_N, D] for pooling
        flattened_outputs = []
        for layer_out in layer_outputs:
            flattened = []
            for i, num_nodes in enumerate(batch_num_nodes):
                flattened.append(layer_out[i, :num_nodes])
            flattened_outputs.append(torch.cat(flattened, dim=0))

        # 7. Hierarchical pooling to get graph representation
        graph_repr = self.pooling(flattened_outputs, batch_graph)  # [B, D]

        # 8. Output projection
        graph_repr = self.output_proj(graph_repr)

        return graph_repr


if __name__ == '__main__':
    # Test ImprovedPepLand
    print("Testing ImprovedPepLand...")

    import dgl

    # Create sample graphs
    g1 = dgl.graph(([0, 1, 2, 1], [1, 2, 0, 0]))
    g1.ndata['atom_feat'] = torch.randn(3, 42)
    g1.ndata['node_type'] = torch.tensor([0, 0, 0])
    g1.edata['bond_feat'] = torch.randn(4, 14)

    g2 = dgl.graph(([0, 1, 2, 3, 2], [1, 2, 3, 0, 0]))
    g2.ndata['atom_feat'] = torch.randn(4, 42)
    g2.ndata['node_type'] = torch.tensor([0, 0, 0, 0])
    g2.edata['bond_feat'] = torch.randn(5, 14)

    batch_g = dgl.batch([g1, g2])

    # Create model
    model = ImprovedPepLand(
        atom_dim=42,
        bond_dim=14,
        hidden_dim=128,  # Smaller for testing
        num_layers=3,    # Fewer layers for testing
        num_heads=4,
        use_performer=False,
        use_virtual_node=True
    )

    # Forward pass
    graph_repr = model(batch_g)
    print(f"Input: Batched graph with {batch_g.batch_size} graphs")
    print(f"Output shape: {graph_repr.shape}")
    assert graph_repr.shape == (2, 128), "Shape mismatch!"

    print("✓ All tests passed!")
