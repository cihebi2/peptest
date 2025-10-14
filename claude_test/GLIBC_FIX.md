# GLIBC Issue and Solutions / GLIBC 问题和解决方案

## Problem / 问题

The current system has GLIBC 2.17, but DGL requires GLIBC 2.27 or higher.

当前系统使用 GLIBC 2.17，但 DGL 需要 GLIBC 2.27 或更高版本。

Error message:
```
OSError: /lib64/libm.so.6: version `GLIBC_2.27' not found (required by .../dgl/libdgl.so)
```

## Solutions / 解决方案

### Option 1: Install DGL from Source (Recommended) / 选项 1：从源码安装 DGL（推荐）

Build DGL from source in your current environment:

在当前环境从源码编译 DGL：

```bash
# Activate your environment
conda activate cuda12.1

# Install dependencies
pip install cmake ninja

# Clone DGL repository
git clone --recursive https://github.com/dmlc/dgl.git
cd dgl

# Build DGL with CUDA support
mkdir build
cd build
cmake -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda ..
make -j$(nproc)

# Install
cd ../python
python setup.py install
```

### Option 2: Use Pre-built DGL with Older GLIBC / 选项 2：使用支持旧版 GLIBC 的 DGL

Install an older version of DGL that supports GLIBC 2.17:

安装支持 GLIBC 2.17 的旧版本 DGL：

```bash
conda activate cuda12.1

# Uninstall current DGL
pip uninstall dgl dglgo -y

# Install older DGL version (0.9.x supports older GLIBC)
pip install dgl-cu117==0.9.1 -f https://data.dgl.ai/wheels/repo.html
```

Note: Replace `cu117` with your CUDA version (e.g., `cu116` for CUDA 11.6).

注意：将 `cu117` 替换为您的 CUDA 版本（例如 CUDA 11.6 使用 `cu116`）。

### Option 3: Create New Environment with Python 3.8 / 选项 3：创建 Python 3.8 新环境

Python 3.8 environments sometimes have better compatibility:

Python 3.8 环境有时有更好的兼容性：

```bash
# Create new environment
conda create -n pepland_py38 python=3.8 -y
conda activate pepland_py38

# Install PyTorch with CUDA 11.7
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install DGL
pip install dgl-cu117 -f https://data.dgl.ai/wheels/repo.html

# Install other dependencies
pip install rdkit-pypi transformers pyyaml tqdm tensorboard
```

### Option 4: Use Docker Container / 选项 4：使用 Docker 容器

Use a Docker container with compatible GLIBC:

使用具有兼容 GLIBC 的 Docker 容器：

```bash
# Pull DGL Docker image
docker pull dglai/dgl:latest-cu117

# Run container with GPU 3
docker run --gpus '"device=3"' -it -v /home/qlyu/AA_peptide:/workspace dglai/dgl:latest-cu117
```

### Option 5: Use the Original Pepland Environment / 选项 5：使用原始 Pepland 环境

Check what environment the original pepland code was using:

检查原始 pepland 代码使用的环境：

```bash
# Check if there's a requirements file
cat /home/qlyu/AA_peptide/pepland/requirements.txt

# Or check conda environment export
ls /home/qlyu/AA_peptide/pepland/*.yml
```

If the original code worked, use the same environment setup.

如果原始代码可以运行，使用相同的环境配置。

## Quick Test / 快速测试

### Test CUDA without DGL / 测试 CUDA（不使用 DGL）

```bash
cd /home/qlyu/AA_peptide/pepland/claude_test
CUDA_VISIBLE_DEVICES=3 python test_cuda_simple.py
```

This will verify that CUDA device 3 is accessible via PyTorch.

这将验证可以通过 PyTorch 访问 CUDA 设备 3。

### Test DGL after fixing / 修复后测试 DGL

```bash
cd /home/qlyu/AA_peptide/pepland/claude_test
CUDA_VISIBLE_DEVICES=3 python test_dgl_cuda.py
```

## Recommended Approach / 推荐方法

**For immediate testing**, use Option 3 (create new Python 3.8 environment) as it's the quickest and most reliable solution.

**立即测试**，建议使用选项 3（创建新的 Python 3.8 环境），这是最快且最可靠的解决方案。

**For production use**, consider Option 1 (build from source) for optimal performance.

**生产环境**，建议使用选项 1（从源码构建）以获得最佳性能。

## Alternative: Use PyTorch Geometric Instead / 替代方案：使用 PyTorch Geometric

If DGL compatibility remains an issue, consider using PyTorch Geometric instead:

如果 DGL 兼容性仍然有问题，可以考虑使用 PyTorch Geometric：

```bash
conda activate cuda12.1
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

The code would need minor modifications, but PyG often has better compatibility with older systems.

代码需要少量修改，但 PyG 通常对旧系统有更好的兼容性。
