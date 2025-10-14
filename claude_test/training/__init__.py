"""
Training Pipeline

Core training components:
1. Trainer with contrastive + generative pre-training
2. Optimizer configurations
3. Learning rate schedulers
4. Regularization techniques
"""

from .trainer import PreTrainer, DownstreamTrainer
from .optimizer import get_optimizer, get_scheduler
from .regularization import DropPath, LabelSmoothing

__all__ = [
    'PreTrainer',
    'DownstreamTrainer',
    'get_optimizer',
    'get_scheduler',
    'DropPath',
    'LabelSmoothing',
]
