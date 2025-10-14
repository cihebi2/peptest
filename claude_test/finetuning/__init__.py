"""
Fine-tuning Strategies

Parameter-efficient fine-tuning methods:
1. Adapter - insert trainable adapter modules
2. LoRA - low-rank adaptation
3. Multi-task learning
"""

from .adapter import Adapter, ModelWithAdapters
from .lora import LoRALayer, ModelWithLoRA
from .multitask import MultiTaskLearning

__all__ = [
    'Adapter',
    'ModelWithAdapters',
    'LoRALayer',
    'ModelWithLoRA',
    'MultiTaskLearning',
]
