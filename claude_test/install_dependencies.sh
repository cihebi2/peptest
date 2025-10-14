#!/bin/bash
# Installation script for Improved PepLand dependencies

set -e

echo "========================================================================"
echo "Installing Dependencies for Improved PepLand"
echo "========================================================================"
echo ""

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Check PyTorch
echo "Checking PyTorch installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || {
    echo "❌ PyTorch not found. Please install PyTorch first:"
    echo "   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
    exit 1
}
echo "✓ PyTorch is installed"
echo ""

# Install DGL
echo "========================================================================"
echo "Installing DGL (Deep Graph Library)"
echo "========================================================================"
echo ""
echo "DGL is required for graph neural network operations."
echo "Installing DGL for CUDA 12.1..."
echo ""

pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/cu121/repo.html

echo ""
echo "✓ DGL installation complete"
echo ""

# Verify DGL installation
echo "Verifying DGL installation..."
python -c "import dgl; print(f'DGL version: {dgl.__version__}')" && echo "✓ DGL verified" || {
    echo "❌ DGL verification failed"
    exit 1
}
echo ""

# Optional: Install other packages if missing
echo "========================================================================"
echo "Checking Optional Dependencies"
echo "========================================================================"
echo ""

# Check RDKit
python -c "from rdkit import Chem" 2>/dev/null && {
    echo "✓ RDKit is already installed"
} || {
    echo "⚠ RDKit not found (optional for 3D conformer features)"
    echo "  To install: conda install -c conda-forge rdkit"
    echo "           or: pip install rdkit"
}
echo ""

# Check ESM
python -c "import esm" 2>/dev/null && {
    echo "✓ fair-esm is already installed"
} || {
    echo "⚠ fair-esm not found (optional for sequence embeddings)"
    echo "  To install: pip install fair-esm"
}
echo ""

echo "========================================================================"
echo "Installation Summary"
echo "========================================================================"
echo ""

python check_environment.py

echo ""
echo "========================================================================"
echo "Installation complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Review the configuration: configs/pretrain_config.yaml"
echo "  2. Prepare your dataset"
echo "  3. Start pre-training: python scripts/pretrain.py --config configs/pretrain_config.yaml"
echo ""
