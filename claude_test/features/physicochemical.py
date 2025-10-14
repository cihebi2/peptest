"""
Physicochemical Property Encoder

Encodes molecular descriptors and properties:
- RDKit descriptors (200+ properties)
- Lipinski's Rule of Five
- TPSA, LogP, etc.

Requires: rdkit
"""

import torch
import torch.nn as nn
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class PhysicoChemicalEncoder(nn.Module):
    """
    Encode physicochemical properties as features
    """

    def __init__(self, hidden_dim, descriptor_dim=200):
        super().__init__()
        self.descriptor_dim = descriptor_dim

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Learnable statistics for normalization
        self.register_buffer('mean', torch.zeros(descriptor_dim))
        self.register_buffer('std', torch.ones(descriptor_dim))
        self.register_buffer('initialized', torch.tensor(False))

    def update_statistics(self, descriptor_batch):
        """Update running mean and std from a batch of descriptors"""
        if descriptor_batch.shape[1] != self.descriptor_dim:
            return

        if not self.initialized:
            self.mean = descriptor_batch.mean(dim=0)
            self.std = descriptor_batch.std(dim=0) + 1e-8
            self.initialized = torch.tensor(True)
        else:
            # Exponential moving average
            momentum = 0.1
            self.mean = (1 - momentum) * self.mean + momentum * descriptor_batch.mean(dim=0)
            self.std = (1 - momentum) * self.std + momentum * (descriptor_batch.std(dim=0) + 1e-8)

    def forward(self, descriptors):
        """
        Args:
            descriptors: [B, descriptor_dim] molecular descriptors

        Returns:
            encoded: [B, hidden_dim] encoded features
        """
        # Normalize
        normalized = (descriptors - self.mean) / (self.std + 1e-8)

        # Encode
        encoded = self.encoder(normalized)

        return encoded


def compute_molecular_descriptors(smiles):
    """
    Compute RDKit molecular descriptors

    Args:
        smiles: SMILES string

    Returns:
        descriptors: Numpy array of descriptor values
    """
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required for descriptor computation")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Get all descriptor names
    descriptor_names = [x[0] for x in Descriptors._descList]

    # Compute descriptors
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
        descriptor_names
    )

    descriptors = np.array(calculator.CalcDescriptors(mol))

    # Handle NaN values
    descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=1e6, neginf=-1e6)

    return torch.FloatTensor(descriptors)


if __name__ == '__main__':
    print("Testing Physicochemical Encoder...")

    encoder = PhysicoChemicalEncoder(hidden_dim=512, descriptor_dim=200)

    # Dummy descriptors
    descriptors = torch.randn(4, 200)
    encoded = encoder(descriptors)

    print(f"Input shape: {descriptors.shape}")
    print(f"Output shape: {encoded.shape}")
    assert encoded.shape == (4, 512)

    print("✓ All tests passed!")
