"""
Feature Encoders for Enhanced Molecular Representations

Provides additional features beyond 2D topology:
1. 3D conformer features
2. Physicochemical properties
3. Sequence embeddings
"""

from .conformer_3d import Conformer3DEncoder, generate_3d_conformer
from .physicochemical import PhysicoChemicalEncoder
from .sequence import SequenceEncoder

__all__ = [
    'Conformer3DEncoder',
    'generate_3d_conformer',
    'PhysicoChemicalEncoder',
    'SequenceEncoder',
]
