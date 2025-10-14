"""Optimizer and Scheduler Configurations"""

import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, LinearLR, SequentialLR


def get_optimizer(model, config):
    """
    Get optimizer based on configuration
    
    Args:
        model: PyTorch model
        config: Dict with optimizer config
            - type: 'AdamW', 'Adam', 'SGD'
            - lr: learning rate
            - weight_decay: L2 regularization
            - betas: for Adam/AdamW
    
    Returns:
        optimizer: PyTorch optimizer
    """
    opt_type = config.get('type', 'AdamW')
    lr = config.get('lr', 1e-4)
    weight_decay = config.get('weight_decay', 0.01)
    
    if opt_type == 'AdamW':
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    elif opt_type == 'Adam':
        optimizer = Adam(
            model.parameters(),
            lr=lr,
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=weight_decay
        )
    elif opt_type == 'SGD':
        optimizer = SGD(
            model.parameters(),
            lr=lr,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")
    
    return optimizer


def get_scheduler(optimizer, config):
    """
    Get learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        config: Dict with scheduler config
            - type: 'cosine', 'onecycle', 'linear'
            - warmup_steps: number of warmup steps
            - total_steps: total training steps
    
    Returns:
        scheduler: PyTorch LR scheduler
    """
    sched_type = config.get('type', 'cosine')
    warmup_steps = config.get('warmup_steps', 1000)
    total_steps = config.get('total_steps', 100000)
    
    if sched_type == 'cosine':
        # Cosine annealing with warmup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get('T_0', 10000),
            T_mult=config.get('T_mult', 1),
            eta_min=config.get('min_lr', 1e-6)
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    elif sched_type == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.get('max_lr', 1e-3),
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps
        )
    elif sched_type == 'linear':
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.get('end_factor', 0.1),
            total_iters=total_steps
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")
    
    return scheduler
