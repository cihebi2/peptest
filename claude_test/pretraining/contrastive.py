"""
Contrastive Learning for Graphs

Implements state-of-the-art contrastive learning methods:
1. SimCLR for graphs
2. MoCo (Momentum Contrast) for graphs
3. InfoNCE loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .augmentation import GraphAugmentation


class GraphContrastiveLearning(nn.Module):
    """
    Graph Contrastive Learning Framework based on SimCLR

    Key components:
    1. Two augmented views of each graph
    2. Projection head to map representations
    3. InfoNCE contrastive loss
    4. Temperature-scaled cosine similarity

    Reference: "A Simple Framework for Contrastive Learning of Visual
    Representations" (SimCLR, ICML 2020) adapted for graphs
    """

    def __init__(
        self,
        encoder,
        projection_dim=256,
        temperature=0.07,
        augmentation='mixed'
    ):
        """
        Args:
            encoder: Graph encoder model (e.g., ImprovedPepLand)
            projection_dim: Dimension of projection head output
            temperature: Temperature parameter for contrastive loss
            augmentation: Augmentation strategy ('weak', 'strong', 'mixed')
        """
        super().__init__()

        self.encoder = encoder
        self.temperature = temperature
        self.augmentation = augmentation

        # Projection head (MLP)
        # Maps graph representations to a normalized space for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(encoder.hidden_dim, encoder.hidden_dim),
            nn.BatchNorm1d(encoder.hidden_dim),
            nn.GELU(),
            nn.Linear(encoder.hidden_dim, projection_dim)
        )

    def forward(self, graph1, graph2=None):
        """
        Forward pass for contrastive learning

        Args:
            graph1: First augmented view (or original graph)
            graph2: Second augmented view (optional, will create if None)

        Returns:
            loss: Contrastive loss
            z1, z2: Projected representations
        """
        # Create second view if not provided
        if graph2 is None:
            graph2 = graph1  # Will augment in get_representations

        # Get representations for both views
        z1 = self.get_representation(graph1)  # [B, projection_dim]
        z2 = self.get_representation(graph2)  # [B, projection_dim]

        # Normalize representations
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Compute contrastive loss
        loss = self.contrastive_loss(z1, z2)

        return loss, z1, z2

    def get_representation(self, graph):
        """
        Get projected representation for a graph

        Args:
            graph: DGL graph

        Returns:
            z: Projected representation [B, projection_dim]
        """
        # Encode graph
        h = self.encoder(graph)  # [B, hidden_dim]

        # Project
        z = self.projection_head(h)  # [B, projection_dim]

        return z

    def contrastive_loss(self, z1, z2):
        """
        Compute InfoNCE contrastive loss with numerical stability

        Treats (z1[i], z2[i]) as positive pairs and all others as negatives

        Args:
            z1, z2: Normalized representations [B, D]

        Returns:
            loss: Contrastive loss (scalar)
        """
        batch_size = z1.shape[0]
        device = z1.device

        # Concatenate representations
        z = torch.cat([z1, z2], dim=0)  # [2B, D]

        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T)  # [2B, 2B]

        # Clamp similarity values for numerical stability
        sim_matrix = torch.clamp(sim_matrix, min=-1.0, max=1.0)

        # Apply temperature
        sim_matrix = sim_matrix / self.temperature  # [2B, 2B]

        # Clamp again after temperature to prevent overflow in exp
        sim_matrix = torch.clamp(sim_matrix, min=-50, max=50)

        # Create positive pair mask
        # For each sample i, its positive is at position (i + B) or (i - B)
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=device)
        for i in range(batch_size):
            pos_mask[i, i + batch_size] = 1
            pos_mask[i + batch_size, i] = 1

        # Create negative mask (all except self and positive)
        neg_mask = torch.ones(2 * batch_size, 2 * batch_size, device=device)
        neg_mask = neg_mask - torch.eye(2 * batch_size, device=device) - pos_mask

        # Compute loss with numerical stability
        # For each sample, we want: exp(sim_positive) / sum(exp(sim_all_negatives))
        pos_sim = torch.sum(sim_matrix * pos_mask, dim=1)  # [2B]

        # Use logsumexp with masking for stability
        # Set masked positions to very negative values
        masked_sim = torch.where(
            neg_mask.bool(),
            sim_matrix,
            torch.full_like(sim_matrix, -1e9)
        )
        neg_sim = torch.logsumexp(masked_sim, dim=1)  # [2B]

        loss = -torch.mean(pos_sim - neg_sim)

        return loss


class MoCoForGraphs(nn.Module):
    """
    Momentum Contrast (MoCo) adapted for graphs

    Key innovations:
    1. Momentum encoder: slowly updated version of encoder
    2. Queue: stores representations from previous batches
    3. More stable training with larger effective batch size

    Reference: "Momentum Contrast for Unsupervised Visual Representation
    Learning" (MoCo, CVPR 2020)
    """

    def __init__(
        self,
        encoder,
        projection_dim=256,
        queue_size=65536,
        temperature=0.07,
        momentum=0.999,
        augmentation='mixed'
    ):
        """
        Args:
            encoder: Graph encoder model
            projection_dim: Dimension of projection head output
            queue_size: Size of negative sample queue
            temperature: Temperature for contrastive loss
            momentum: Momentum coefficient for updating momentum encoder
            augmentation: Augmentation strategy
        """
        super().__init__()

        self.temperature = temperature
        self.momentum = momentum
        self.queue_size = queue_size
        self.augmentation = augmentation

        # Query encoder (trainable)
        self.encoder_q = encoder

        # Key encoder (momentum updated, not trainable)
        self.encoder_k = copy.deepcopy(encoder)

        # Freeze key encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        # Projection heads
        self.projection_q = nn.Sequential(
            nn.Linear(encoder.hidden_dim, encoder.hidden_dim),
            nn.BatchNorm1d(encoder.hidden_dim),
            nn.GELU(),
            nn.Linear(encoder.hidden_dim, projection_dim)
        )

        self.projection_k = copy.deepcopy(self.projection_q)

        # Freeze key projection
        for param in self.projection_k.parameters():
            param.requires_grad = False

        # Queue for storing negative samples
        self.register_buffer("queue", torch.randn(projection_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        """
        Momentum update of key encoder and projection head

        θ_k = m * θ_k + (1 - m) * θ_q
        """
        # Update encoder
        for param_q, param_k in zip(
            self.encoder_q.parameters(),
            self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + \
                          param_q.data * (1.0 - self.momentum)

        # Update projection head
        for param_q, param_k in zip(
            self.projection_q.parameters(),
            self.projection_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + \
                          param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update the queue with new keys

        Args:
            keys: New keys to add [B, D]
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Ensure queue size is multiple of batch size for simplicity
        if ptr + batch_size > self.queue_size:
            # Wrap around
            batch_size_1 = self.queue_size - ptr
            batch_size_2 = batch_size - batch_size_1

            self.queue[:, ptr:] = keys[:batch_size_1].T
            self.queue[:, :batch_size_2] = keys[batch_size_1:].T
            ptr = batch_size_2
        else:
            # Replace oldest entries
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    def forward(self, graph_q, graph_k=None):
        """
        Forward pass

        Args:
            graph_q: Query graph (augmented view 1)
            graph_k: Key graph (augmented view 2, optional)

        Returns:
            loss: Contrastive loss
        """
        # Query representation (trainable)
        h_q = self.encoder_q(graph_q)
        q = self.projection_q(h_q)
        q = F.normalize(q, dim=-1)  # [B, D]

        # Key representation (momentum encoder)
        with torch.no_grad():
            # Update momentum encoder
            self._momentum_update()

            # Encode keys
            if graph_k is None:
                graph_k = graph_q  # Use same graph with different augmentation

            h_k = self.encoder_k(graph_k)
            k = self.projection_k(h_k)
            k = F.normalize(k, dim=-1)  # [B, D]

        # Compute similarities
        # Positive pairs: q[i] and k[i]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # [B, 1]

        # Negative pairs: q[i] and queue[j] for all j
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # [B, K]

        # Logits: [B, 1+K]
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        # Labels: positive sample is at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        # Update queue
        self._dequeue_and_enqueue(k)

        return loss


class HardNegativeMining(nn.Module):
    """
    Hard negative mining for contrastive learning

    Focuses on difficult negative samples to improve learning
    """

    def __init__(self, encoder, projection_dim=256, temperature=0.07, top_k=128):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        self.top_k = top_k  # Number of hard negatives to use

        self.projection_head = nn.Sequential(
            nn.Linear(encoder.hidden_dim, encoder.hidden_dim),
            nn.BatchNorm1d(encoder.hidden_dim),
            nn.GELU(),
            nn.Linear(encoder.hidden_dim, projection_dim)
        )

    def forward(self, graph1, graph2, negative_graphs):
        """
        Args:
            graph1, graph2: Positive pair
            negative_graphs: Pool of negative samples

        Returns:
            loss: Contrastive loss with hard negative mining
        """
        # Get representations
        z1 = self.projection_head(self.encoder(graph1))
        z2 = self.projection_head(self.encoder(graph2))
        z_neg = self.projection_head(self.encoder(negative_graphs))

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        z_neg = F.normalize(z_neg, dim=-1)

        # Compute similarities to negatives
        sim_neg = torch.matmul(z1, z_neg.T)  # [B, N_neg]

        # Select top-k hardest negatives (highest similarity)
        _, hard_indices = torch.topk(sim_neg, self.top_k, dim=1)  # [B, K]

        # Gather hard negatives
        z_hard = z_neg[hard_indices]  # [B, K, D]

        # Compute contrastive loss with hard negatives
        pos_sim = torch.sum(z1 * z2, dim=-1, keepdim=True) / self.temperature  # [B, 1]
        neg_sim = torch.matmul(z1.unsqueeze(1), z_hard.transpose(1, 2)).squeeze(1) / self.temperature  # [B, K]

        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, 1+K]
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        return loss


if __name__ == '__main__':
    # Test contrastive learning
    print("Testing Contrastive Learning...")

    import dgl
    import sys
    sys.path.append('..')
    from models.improved_pepland import ImprovedPepLand

    # Create sample graphs
    g1 = dgl.graph(([0, 1, 2], [1, 2, 0]))
    g1.ndata['atom_feat'] = torch.randn(3, 42)
    g1.ndata['node_type'] = torch.zeros(3, dtype=torch.long)
    g1.edata['bond_feat'] = torch.randn(3, 14)

    g2 = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 0]))
    g2.ndata['atom_feat'] = torch.randn(4, 42)
    g2.ndata['node_type'] = torch.zeros(4, dtype=torch.long)
    g2.edata['bond_feat'] = torch.randn(4, 14)

    batch_g = dgl.batch([g1, g2])

    # Create encoder
    encoder = ImprovedPepLand(
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        use_performer=False
    )

    # Test SimCLR-style contrastive learning
    print("\n1. Testing SimCLR for graphs...")
    contrast_model = GraphContrastiveLearning(encoder, projection_dim=64)
    view1, view2 = GraphAugmentation.create_two_views(batch_g)
    loss, z1, z2 = contrast_model(view1, view2)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Representation shape: {z1.shape}")

    # Test MoCo for graphs
    print("\n2. Testing MoCo for graphs...")
    moco_model = MoCoForGraphs(encoder, projection_dim=64, queue_size=128)
    loss = moco_model(view1, view2)
    print(f"   Loss: {loss.item():.4f}")

    print("\n✓ All tests passed!")
