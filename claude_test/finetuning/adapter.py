"""Adapter Fine-tuning"""

import torch
import torch.nn as nn


class Adapter(nn.Module):
    """Adapter module for parameter-efficient fine-tuning"""
    
    def __init__(self, hidden_dim, adapter_dim=64):
        super().__init__()
        self.down_project = nn.Linear(hidden_dim, adapter_dim)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(adapter_dim, hidden_dim)
        
        # Initialize near zero for identity mapping
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual


class ModelWithAdapters(nn.Module):
    """Add adapters to a pre-trained model"""
    
    def __init__(self, pretrained_model, adapter_dim=64):
        super().__init__()
        self.model = pretrained_model
        
        # Freeze pre-trained model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Add adapters to each layer
        self.adapters = nn.ModuleList([
            Adapter(self.model.hidden_dim, adapter_dim)
            for _ in range(self.model.num_layers)
        ])
    
    def forward(self, x):
        # Apply model with adapters
        for i, layer in enumerate(self.model.encoder.layers):
            x = layer(x)
            x = self.adapters[i](x)
        return x
