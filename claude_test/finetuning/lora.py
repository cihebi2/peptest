"""LoRA (Low-Rank Adaptation)"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """LoRA layer for efficient fine-tuning"""
    
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Low-rank decomposition: ΔW = BA
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        self.scaling = alpha / rank
    
    def forward(self, x, original_weight):
        # Original: out = x @ W
        out = F.linear(x, original_weight)
        
        # LoRA delta: out += x @ A @ B * scaling
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        
        return out + lora_out


class ModelWithLoRA(nn.Module):
    """Apply LoRA to model attention layers"""
    
    def __init__(self, pretrained_model, rank=8, alpha=16):
        super().__init__()
        self.model = pretrained_model
        
        # Freeze pre-trained model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Add LoRA to Q and V projections
        self.lora_layers = nn.ModuleDict()
        for i, layer in enumerate(self.model.encoder.layers):
            hidden_dim = layer.hidden_dim
            self.lora_layers[f'layer_{i}_q'] = LoRALayer(hidden_dim, hidden_dim, rank, alpha)
            self.lora_layers[f'layer_{i}_v'] = LoRALayer(hidden_dim, hidden_dim, rank, alpha)
