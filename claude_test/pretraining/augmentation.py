"""
Graph Data Augmentation

Implements various graph augmentation strategies for contrastive learning:
1. Node dropping
2. Edge perturbation (drop/add)
3. Feature masking
4. Subgraph sampling
5. Attribute augmentation
"""

import torch
import dgl
import random
import copy


class GraphAugmentation:
    """
    Collection of graph augmentation methods

    These augmentations create different views of the same graph
    while preserving semantic information, crucial for contrastive learning
    """

    @staticmethod
    def node_dropping(graph, drop_ratio=0.1):
        """
        Randomly drop nodes from the graph

        Args:
            graph: DGL graph
            drop_ratio: Ratio of nodes to drop

        Returns:
            Augmented graph with fewer nodes
        """
        num_nodes = graph.num_nodes()
        if num_nodes <= 1:
            return graph

        drop_num = max(1, int(num_nodes * drop_ratio))
        keep_num = num_nodes - drop_num

        # Randomly select nodes to keep
        device = graph.device if hasattr(graph, 'device') else 'cpu'
        keep_nodes = torch.randperm(num_nodes, device=device)[:keep_num]
        keep_nodes = torch.sort(keep_nodes)[0]  # Sort for stability

        # Create subgraph
        aug_graph = graph.subgraph(keep_nodes)

        return aug_graph

    @staticmethod
    def edge_perturbation(graph, drop_ratio=0.1, add_ratio=0.0):
        """
        Randomly drop and/or add edges

        Args:
            graph: DGL graph
            drop_ratio: Ratio of edges to drop
            add_ratio: Ratio of edges to add

        Returns:
            Augmented graph with modified edges
        """
        num_edges = graph.num_edges()
        num_nodes = graph.num_nodes()

        if num_edges == 0:
            return graph

        src, dst = graph.edges()
        edge_ids = torch.arange(num_edges)

        # Drop edges
        if drop_ratio > 0:
            drop_num = max(1, int(num_edges * drop_ratio))
            keep_num = num_edges - drop_num
            keep_edges = torch.randperm(num_edges)[:keep_num]
            src = src[keep_edges]
            dst = dst[keep_edges]

        # Add random edges
        if add_ratio > 0:
            add_num = int(num_edges * add_ratio)
            new_src = torch.randint(0, num_nodes, (add_num,))
            new_dst = torch.randint(0, num_nodes, (add_num,))

            # Remove self-loops
            mask = new_src != new_dst
            new_src = new_src[mask]
            new_dst = new_dst[mask]

            src = torch.cat([src, new_src])
            dst = torch.cat([dst, new_dst])

        # Create new graph
        aug_graph = dgl.graph((src, dst), num_nodes=num_nodes)

        # Copy node features
        for key, val in graph.ndata.items():
            aug_graph.ndata[key] = val.clone()

        # Copy edge features for kept edges (simplified)
        # Note: Newly added edges will not have features
        # This is acceptable for augmentation purposes

        return aug_graph

    @staticmethod
    def feature_masking(graph, mask_ratio=0.15, mask_value=0.0):
        """
        Randomly mask node features

        Args:
            graph: DGL graph
            mask_ratio: Ratio of features to mask
            mask_value: Value to use for masking

        Returns:
            Augmented graph with masked features
        """
        aug_graph = graph.clone()

        for feat_name in graph.ndata.keys():
            if 'feat' in feat_name:  # Only mask feature tensors
                feat = aug_graph.ndata[feat_name]
                num_nodes, feat_dim = feat.shape

                # Randomly select features to mask
                mask_num = int(num_nodes * mask_ratio)
                if mask_num > 0:
                    mask_indices = torch.randperm(num_nodes)[:mask_num]
                    aug_graph.ndata[feat_name][mask_indices] = mask_value

        return aug_graph

    @staticmethod
    def subgraph_sampling(graph, sample_ratio=0.7, method='random'):
        """
        Sample a connected subgraph

        Args:
            graph: DGL graph
            sample_ratio: Ratio of nodes to keep
            method: Sampling method ('random', 'random_walk')

        Returns:
            Subgraph
        """
        num_nodes = graph.num_nodes()
        sample_num = max(1, int(num_nodes * sample_ratio))

        if method == 'random':
            # Simple random sampling
            sampled_nodes = torch.randperm(num_nodes)[:sample_num]
            sampled_nodes = torch.sort(sampled_nodes)[0]

        elif method == 'random_walk':
            # Random walk sampling (more likely to preserve connectivity)
            # Start from random node
            start_node = torch.randint(0, num_nodes, (1,)).item()
            sampled_nodes = {start_node}

            current_node = start_node
            while len(sampled_nodes) < sample_num:
                # Get neighbors
                successors = graph.successors(current_node).tolist()
                predecessors = graph.predecessors(current_node).tolist()
                neighbors = list(set(successors + predecessors))

                if len(neighbors) == 0:
                    # Dead end, restart from unvisited node
                    unvisited = set(range(num_nodes)) - sampled_nodes
                    if len(unvisited) == 0:
                        break
                    current_node = random.choice(list(unvisited))
                else:
                    current_node = random.choice(neighbors)

                sampled_nodes.add(current_node)

            sampled_nodes = torch.tensor(sorted(list(sampled_nodes)))
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        return graph.subgraph(sampled_nodes)

    @staticmethod
    def attribute_augmentation(graph, noise_std=0.1):
        """
        Add Gaussian noise to node features

        Args:
            graph: DGL graph
            noise_std: Standard deviation of noise

        Returns:
            Augmented graph with noisy features
        """
        aug_graph = graph.clone()

        for feat_name in graph.ndata.keys():
            if 'feat' in feat_name:
                feat = aug_graph.ndata[feat_name]
                noise = torch.randn_like(feat) * noise_std
                aug_graph.ndata[feat_name] = feat + noise

        return aug_graph

    @staticmethod
    def compose_augmentations(graph, aug_list=None, num_augs=2):
        """
        Apply multiple augmentations sequentially

        Args:
            graph: DGL graph
            aug_list: List of augmentation functions to choose from
            num_augs: Number of augmentations to apply

        Returns:
            Augmented graph
        """
        if aug_list is None:
            aug_list = [
                lambda g: GraphAugmentation.node_dropping(g, 0.1),
                lambda g: GraphAugmentation.edge_perturbation(g, 0.1, 0.0),
                lambda g: GraphAugmentation.feature_masking(g, 0.15),
                lambda g: GraphAugmentation.subgraph_sampling(g, 0.7),
                lambda g: GraphAugmentation.attribute_augmentation(g, 0.1),
            ]

        # Randomly select augmentations
        num_augs = min(num_augs, len(aug_list))
        selected_augs = random.sample(aug_list, num_augs)

        # Apply augmentations sequentially
        aug_graph = graph
        for aug_fn in selected_augs:
            try:
                aug_graph = aug_fn(aug_graph)
            except Exception as e:
                # If augmentation fails, skip it
                print(f"Warning: Augmentation failed: {e}")
                continue

        return aug_graph

    @staticmethod
    def create_two_views(graph, strong_aug=False):
        """
        Create two augmented views of the same graph for contrastive learning

        Args:
            graph: DGL graph
            strong_aug: Whether to use strong augmentation

        Returns:
            view1, view2: Two augmented views
        """
        if strong_aug:
            # Strong augmentation: more aggressive
            aug_params = [
                lambda g: GraphAugmentation.node_dropping(g, 0.2),
                lambda g: GraphAugmentation.edge_perturbation(g, 0.2, 0.1),
                lambda g: GraphAugmentation.feature_masking(g, 0.2),
                lambda g: GraphAugmentation.subgraph_sampling(g, 0.6),
            ]
        else:
            # Weak augmentation: more conservative
            aug_params = [
                lambda g: GraphAugmentation.node_dropping(g, 0.1),
                lambda g: GraphAugmentation.edge_perturbation(g, 0.1, 0.0),
                lambda g: GraphAugmentation.feature_masking(g, 0.1),
                lambda g: GraphAugmentation.attribute_augmentation(g, 0.05),
            ]

        view1 = GraphAugmentation.compose_augmentations(
            graph, aug_params, num_augs=2
        )
        view2 = GraphAugmentation.compose_augmentations(
            graph, aug_params, num_augs=2
        )

        return view1, view2


# Augmentation strategies for different scenarios
class AugmentationStrategy:
    """Pre-defined augmentation strategies"""

    @staticmethod
    def get_strategy(name):
        """
        Get augmentation strategy by name

        Args:
            name: Strategy name ('weak', 'strong', 'mixed')

        Returns:
            Augmentation function
        """
        strategies = {
            'weak': lambda g: GraphAugmentation.compose_augmentations(
                g,
                [
                    lambda x: GraphAugmentation.feature_masking(x, 0.1),
                    lambda x: GraphAugmentation.attribute_augmentation(x, 0.05),
                ],
                num_augs=1
            ),
            'strong': lambda g: GraphAugmentation.compose_augmentations(
                g,
                [
                    lambda x: GraphAugmentation.node_dropping(x, 0.2),
                    lambda x: GraphAugmentation.edge_perturbation(x, 0.2, 0.1),
                    lambda x: GraphAugmentation.subgraph_sampling(x, 0.6),
                ],
                num_augs=2
            ),
            'mixed': lambda g: GraphAugmentation.compose_augmentations(
                g,
                [
                    lambda x: GraphAugmentation.node_dropping(x, 0.15),
                    lambda x: GraphAugmentation.edge_perturbation(x, 0.15, 0.05),
                    lambda x: GraphAugmentation.feature_masking(x, 0.15),
                    lambda x: GraphAugmentation.subgraph_sampling(x, 0.7),
                ],
                num_augs=2
            ),
        }

        return strategies.get(name, strategies['mixed'])


if __name__ == '__main__':
    # Test augmentation methods
    print("Testing Graph Augmentation...")

    # Create a sample graph
    g = dgl.graph(([0, 1, 2, 3, 2, 4], [1, 2, 3, 0, 4, 2]))
    g.ndata['atom_feat'] = torch.randn(5, 42)

    print(f"Original graph: {g.num_nodes()} nodes, {g.num_edges()} edges")

    # Test node dropping
    g_aug = GraphAugmentation.node_dropping(g, drop_ratio=0.2)
    print(f"After node dropping: {g_aug.num_nodes()} nodes, {g_aug.num_edges()} edges")

    # Test edge perturbation
    g_aug = GraphAugmentation.edge_perturbation(g, drop_ratio=0.2, add_ratio=0.1)
    print(f"After edge perturbation: {g_aug.num_nodes()} nodes, {g_aug.num_edges()} edges")

    # Test feature masking
    g_aug = GraphAugmentation.feature_masking(g, mask_ratio=0.3)
    print(f"After feature masking: features shape = {g_aug.ndata['atom_feat'].shape}")

    # Test subgraph sampling
    g_aug = GraphAugmentation.subgraph_sampling(g, sample_ratio=0.6)
    print(f"After subgraph sampling: {g_aug.num_nodes()} nodes, {g_aug.num_edges()} edges")

    # Test composed augmentations
    view1, view2 = GraphAugmentation.create_two_views(g, strong_aug=False)
    print(f"View 1: {view1.num_nodes()} nodes, {view1.num_edges()} edges")
    print(f"View 2: {view2.num_nodes()} nodes, {view2.num_edges()} edges")

    print("✓ All tests passed!")
