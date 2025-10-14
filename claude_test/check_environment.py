#!/usr/bin/env python
"""
Environment Check for Improved PepLand

Checks if all required dependencies are installed and provides
installation instructions for missing packages.
"""

import sys
import subprocess


def check_package(package_name, import_name=None, min_version=None):
    """Check if a package is installed and optionally verify version"""
    if import_name is None:
        import_name = package_name

    try:
        if import_name == 'yaml':
            import yaml
            version = getattr(yaml, '__version__', 'unknown')
        elif import_name == 'sklearn':
            import sklearn
            version = sklearn.__version__
        elif import_name == 'rdkit':
            from rdkit import Chem
            import rdkit
            version = getattr(rdkit, '__version__', 'installed')
        elif import_name == 'esm':
            import esm
            version = 'installed'
        else:
            mod = __import__(import_name)
            version = getattr(mod, '__version__', 'unknown')

        # Check version if specified
        if min_version and version != 'unknown' and version != 'installed':
            from packaging import version as pkg_version
            if pkg_version.parse(version) >= pkg_version.parse(min_version):
                return True, version
            else:
                return False, f"{version} (need >={min_version})"

        return True, version
    except ImportError:
        return False, None


def main():
    print("=" * 70)
    print("Environment Check for Improved PepLand")
    print("=" * 70)
    print()

    # Python version check
    print(f"Python Version: {sys.version}")
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    else:
        print("✓ Python version OK")
    print()

    # Core dependencies
    print("=" * 70)
    print("CORE DEPENDENCIES (Required)")
    print("=" * 70)

    core_deps = [
        ('torch', 'torch', '2.0.0'),
        ('dgl', 'dgl', '1.1.0'),
        ('pyyaml', 'yaml', None),
        ('tqdm', 'tqdm', None),
        ('numpy', 'numpy', None),
        ('scipy', 'scipy', None),
        ('scikit-learn', 'sklearn', None),
    ]

    missing_core = []

    for pkg_name, import_name, min_ver in core_deps:
        installed, version = check_package(pkg_name, import_name, min_ver)
        if installed:
            print(f"✓ {pkg_name:20s} {version}")
        else:
            version_str = f" (need >={min_ver})" if min_ver else ""
            print(f"✗ {pkg_name:20s} NOT INSTALLED{version_str}")
            missing_core.append(pkg_name)

    print()

    # Optional dependencies
    print("=" * 70)
    print("OPTIONAL DEPENDENCIES (For Enhanced Features)")
    print("=" * 70)

    optional_deps = [
        ('rdkit', 'rdkit', 'For 3D conformer generation and molecular descriptors'),
        ('fair-esm', 'esm', 'For protein sequence embeddings (ESM-2)'),
    ]

    missing_optional = []

    for pkg_name, import_name, description in optional_deps:
        installed, version = check_package(pkg_name, import_name)
        if installed:
            print(f"✓ {pkg_name:20s} {version}")
        else:
            print(f"✗ {pkg_name:20s} NOT INSTALLED")
            print(f"  → {description}")
            missing_optional.append(pkg_name)

    print()

    # GPU Check
    print("=" * 70)
    print("GPU AVAILABILITY")
    print("=" * 70)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA Available: {torch.version.cuda}")
            print(f"✓ GPU Count: {torch.cuda.device_count()}")
            print(f"✓ PyTorch Version: {torch.__version__}")

            # List GPUs
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / 1024**3
                print(f"  GPU {i}: {props.name} ({mem_gb:.1f} GB)")
        else:
            print("⚠ CUDA not available - will use CPU (much slower)")
    except ImportError:
        print("✗ PyTorch not installed - cannot check GPU")

    print()

    # Summary and instructions
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if missing_core:
        print("❌ MISSING CORE DEPENDENCIES:")
        print()
        print("To install missing core packages, run:")
        print()

        if 'dgl' in missing_core:
            print("# Install DGL (CUDA 12.1)")
            print("pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/cu121/repo.html")
            print()

        other_core = [p for p in missing_core if p != 'dgl']
        if other_core:
            print("# Install other core packages")
            print(f"pip install {' '.join(other_core)}")
            print()
    else:
        print("✓ All core dependencies installed!")

    if missing_optional:
        print()
        print("⚠ OPTIONAL DEPENDENCIES NOT INSTALLED:")
        print()
        print("For enhanced features (3D conformers, sequence embeddings):")
        print()
        if 'rdkit' in missing_optional:
            print("# Install RDKit")
            print("conda install -c conda-forge rdkit")
            print("# or")
            print("pip install rdkit")
            print()
        if 'fair-esm' in missing_optional:
            print("# Install ESM-2")
            print("pip install fair-esm")
            print()

    print()
    print("=" * 70)
    print("INSTALLATION COMMAND SUMMARY")
    print("=" * 70)
    print()

    if missing_core or missing_optional:
        print("# Complete installation command:")
        print()

        all_pip_packages = []

        # Core packages (except DGL)
        other_core = [p for p in missing_core if p != 'dgl']
        all_pip_packages.extend(other_core)

        # Optional packages
        if 'fair-esm' in missing_optional:
            all_pip_packages.append('fair-esm')

        if 'dgl' in missing_core:
            print("# 1. Install DGL first")
            print("pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/cu121/repo.html")
            print()

        if all_pip_packages:
            print("# 2. Install other packages")
            print(f"pip install {' '.join(all_pip_packages)}")
            print()

        if 'rdkit' in missing_optional:
            print("# 3. Install RDKit (optional, for 3D features)")
            print("conda install -c conda-forge rdkit")
            print("# or")
            print("pip install rdkit")
            print()
    else:
        print("✓ Environment is ready!")
        print()
        print("You can now run:")
        print("  python scripts/pretrain.py --config configs/pretrain_config.yaml")

    print()
    print("=" * 70)

    # Exit with appropriate code
    if missing_core:
        print("\n❌ Please install missing core dependencies before proceeding.")
        sys.exit(1)
    else:
        print("\n✓ Environment check passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
