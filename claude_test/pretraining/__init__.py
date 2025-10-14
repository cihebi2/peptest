"""
Pre-training Strategies

Advanced self-supervised learning techniques for graph representation:
1. Contrastive learning (SimCLR, MoCo)
2. Graph augmentation
3. Generative pre-training
4. Curriculum learning
"""

from .contrastive import GraphContrastiveLearning, MoCoForGraphs
from .augmentation import GraphAugmentation
from .generative import GenerativePretraining, MaskedGraphModeling
from .curriculum import CurriculumLearning, DifficultyMetrics

__all__ = [
    'GraphContrastiveLearning',
    'MoCoForGraphs',
    'GraphAugmentation',
    'GenerativePretraining',
    'MaskedGraphModeling',
    'CurriculumLearning',
    'DifficultyMetrics',
]
