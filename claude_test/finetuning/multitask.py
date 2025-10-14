"""Multi-task Learning"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLearning(nn.Module):
    """Multi-task learning with shared encoder"""
    
    def __init__(self, encoder, tasks):
        super().__init__()
        self.encoder = encoder
        hidden_dim = encoder.hidden_dim
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            'binding': nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            ),
            'cpp': nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.GELU(),
                nn.Linear(256, 2)
            ),
            'solubility': nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.GELU(),
                nn.Linear(256, 1)
            ),
        })
        
        # Learnable task weights
        self.task_weights = nn.Parameter(torch.ones(len(tasks)))
    
    def forward(self, batch, task_name=None):
        # Shared encoding
        graph_repr = self.encoder(batch)
        
        if task_name:
            return self.task_heads[task_name](graph_repr)
        else:
            outputs = {}
            for task, head in self.task_heads.items():
                outputs[task] = head(graph_repr)
            return outputs
    
    def compute_loss(self, outputs, targets):
        """Weighted multi-task loss"""
        losses = {}
        for i, (task, pred) in enumerate(outputs.items()):
            if task in targets:
                target = targets[task]
                if task in ['binding', 'solubility']:
                    losses[task] = F.mse_loss(pred, target)
                else:
                    losses[task] = F.cross_entropy(pred, target)
        
        # Uncertainty-weighted loss
        weighted_loss = sum(
            torch.exp(-self.task_weights[i]) * loss + self.task_weights[i]
            for i, loss in enumerate(losses.values())
        )
        
        return weighted_loss, losses
