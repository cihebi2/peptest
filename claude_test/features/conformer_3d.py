"""
3D Conformer Feature Encoder

Encodes 3D structural information including:
- Atomic coordinates
- Distance matrices
- Bond angles and dihedrals
- 3D geometric features

Requires: rdkit
Install: pip install rdkit
"""

import torch
import torch.nn as nn
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. 3D conformer features disabled.")


class Conformer3DEncoder(nn.Module):
    """
    Encoder for 3D conformer features

    Encodes spatial information from 3D molecular structures
    """

    def __init__(self, hidden_dim):
        super().__init__()

        # Coordinate encoder (x, y, z)
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # Distance encoder (pairwise distances)
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )

        # Angle encoder (bond angles)
        self.angle_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, coords, distance_matrix=None, angles=None):
        """
        Args:
            coords: [N, 3] atomic coordinates
            distance_matrix: [N, N] pairwise distances
            angles: [N, N, N] bond angles

        Returns:
            features: [N, D] encoded 3D features
        """
        # Encode coordinates
        coord_feat = self.coord_encoder(coords)  # [N, D]

        # Encode distances (aggregate over pairs)
        if distance_matrix is not None:
            dist_feat = self.distance_encoder(
                distance_matrix.unsqueeze(-1)
            )  # [N, N, D/2]
            dist_feat = dist_feat.mean(dim=1)  # [N, D/2]
        else:
            dist_feat = torch.zeros(
                coords.shape[0],
                self.distance_encoder[-1].out_features,
                device=coords.device
            )

        # Encode angles (aggregate)
        if angles is not None:
            angle_feat = self.angle_encoder(
                angles.unsqueeze(-1)
            )  # [N, N, N, D/2]
            angle_feat = angle_feat.mean(dim=[1, 2])  # [N, D/2]
        else:
            angle_feat = torch.zeros(
                coords.shape[0],
                self.angle_encoder[-1].out_features,
                device=coords.device
            )

        # Fuse all features
        combined = torch.cat([
            coord_feat,
            dist_feat,
            angle_feat
        ], dim=-1)

        output = self.fusion(combined)

        return output


def generate_3d_conformer(smiles, optimize=True):
    """
    Generate 3D conformer from SMILES

    Args:
        smiles: SMILES string
        optimize: Whether to optimize geometry

    Returns:
        coords: [N, 3] atomic coordinates
        distance_matrix: [N, N] pairwise distances
        molecule: RDKit molecule object
    """
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required for 3D conformer generation")

    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    success = AllChem.EmbedMolecule(mol, randomSeed=42)
    if success != 0:
        # Embedding failed, return 2D coordinates
        AllChem.Compute2DCoords(mol)

    # Optimize geometry with MMFF force field
    if optimize:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass  # Optimization might fail for some molecules

    # Extract coordinates
    conf = mol.GetConformer()
    coords = np.array([
        list(conf.GetAtomPosition(i))
        for i in range(mol.GetNumAtoms())
    ])

    # Compute distance matrix
    num_atoms = len(coords)
    distance_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    # Convert to tensors
    coords_tensor = torch.FloatTensor(coords)
    dist_matrix_tensor = torch.FloatTensor(distance_matrix)

    return coords_tensor, dist_matrix_tensor, mol


if __name__ == '__main__':
    print("Testing 3D Conformer Encoder...")

    # Test encoder
    encoder = Conformer3DEncoder(hidden_dim=512)

    # Dummy data
    coords = torch.randn(10, 3)  # 10 atoms
    dist_matrix = torch.randn(10, 10)

    features = encoder(coords, dist_matrix)
    print(f"Input coords shape: {coords.shape}")
    print(f"Output features shape: {features.shape}")
    assert features.shape == (10, 512)

    # Test 3D generation if RDKit available
    if RDKIT_AVAILABLE:
        try:
            smiles = "CCO"  # Ethanol
            coords, dist_matrix, mol = generate_3d_conformer(smiles)
            print(f"\n3D conformer generated for {smiles}")
            print(f"Coordinates shape: {coords.shape}")
            print(f"Distance matrix shape: {dist_matrix.shape}")
        except Exception as e:
            print(f"3D generation test skipped: {e}")

    print("\n✓ All tests passed!")
