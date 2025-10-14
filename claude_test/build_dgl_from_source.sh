#!/bin/bash
# DGL 源码编译脚本 / Build DGL from Source Script
#
# 此脚本将从源码编译 DGL，解决 GLIBC 兼容性问题
# This script builds DGL from source to resolve GLIBC compatibility issues

set -e  # 遇到错误立即退出 / Exit on error

echo "=========================================="
echo "DGL 源码编译安装 / Building DGL from Source"
echo "=========================================="

# 设置颜色输出 / Set color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查当前环境 / Check current environment
echo -e "\n${YELLOW}1. 检查当前环境 / Checking current environment${NC}"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 设置工作目录 / Set working directory
WORK_DIR="/home/qlyu/AA_peptide/pepland/claude_test"
BUILD_DIR="${WORK_DIR}/dgl_build"

echo -e "\n${YELLOW}2. 创建构建目录 / Creating build directory${NC}"
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# 安装编译依赖 / Install build dependencies
echo -e "\n${YELLOW}3. 安装编译依赖 / Installing build dependencies${NC}"
pip install cmake ninja pyyaml -q

# 克隆 DGL 仓库 / Clone DGL repository
echo -e "\n${YELLOW}4. 克隆 DGL 仓库 / Cloning DGL repository${NC}"
if [ -d "dgl" ]; then
    echo "DGL 目录已存在，删除旧版本... / DGL directory exists, removing..."
    rm -rf dgl
fi

echo "正在克隆 DGL (可能需要几分钟)... / Cloning DGL (may take a few minutes)..."
git clone --recursive https://github.com/dmlc/dgl.git
cd dgl

# 检出稳定版本 / Checkout stable version
echo -e "\n${YELLOW}5. 检出稳定版本 / Checking out stable version${NC}"
git checkout v1.1.3  # 使用稳定版本 / Use stable version
git submodule update --init --recursive

# 查找 CUDA 路径 / Find CUDA path
echo -e "\n${YELLOW}6. 查找 CUDA 安装路径 / Finding CUDA installation${NC}"
CUDA_PATH=$(python -c "import torch; print(torch.utils.cpp_extension.CUDA_HOME)" 2>/dev/null || echo "/usr/local/cuda")
if [ ! -d "$CUDA_PATH" ]; then
    # 尝试其他常见位置 / Try other common locations
    for path in /usr/local/cuda-12.4 /usr/local/cuda-12.1 /usr/local/cuda-11.7 /usr/local/cuda; do
        if [ -d "$path" ]; then
            CUDA_PATH=$path
            break
        fi
    done
fi

echo "CUDA path: $CUDA_PATH"

# 检查 nvcc / Check nvcc
if [ -f "${CUDA_PATH}/bin/nvcc" ]; then
    echo "nvcc found: ${CUDA_PATH}/bin/nvcc"
    ${CUDA_PATH}/bin/nvcc --version
else
    echo -e "${RED}警告: 未找到 nvcc，将尝试不使用 CUDA 编译${NC}"
    echo -e "${RED}Warning: nvcc not found, will try to build without CUDA${NC}"
    CUDA_PATH=""
fi

# 创建构建目录 / Create build directory
echo -e "\n${YELLOW}7. 配置 CMake / Configuring CMake${NC}"
mkdir -p build
cd build

# CMake 配置 / CMake configuration
if [ -n "$CUDA_PATH" ]; then
    echo "使用 CUDA 编译 / Building with CUDA support"
    cmake -DUSE_CUDA=ON \
          -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_PATH} \
          -DUSE_OPENMP=ON \
          -DBUILD_TORCH=ON \
          -DCMAKE_BUILD_TYPE=Release \
          ..
else
    echo "仅使用 CPU 编译 / Building with CPU only"
    cmake -DUSE_CUDA=OFF \
          -DUSE_OPENMP=ON \
          -DBUILD_TORCH=ON \
          -DCMAKE_BUILD_TYPE=Release \
          ..
fi

# 编译 / Build
echo -e "\n${YELLOW}8. 开始编译 (这可能需要 10-30 分钟)... / Building (may take 10-30 minutes)...${NC}"
echo "可以去喝杯咖啡休息一下 ☕ / Time for a coffee break ☕"

# 获取 CPU 核心数 / Get number of CPU cores
NPROC=$(nproc)
echo "使用 ${NPROC} 个核心并行编译 / Building with ${NPROC} cores"

make -j${NPROC}

# 安装 Python 包 / Install Python package
echo -e "\n${YELLOW}9. 安装 Python 包 / Installing Python package${NC}"
cd ../python
python setup.py install

# 清理旧的 DGL 安装 / Clean old DGL installation
echo -e "\n${YELLOW}10. 清理 / Cleanup${NC}"
pip uninstall dgl dglgo -y 2>/dev/null || true

# 重新安装 / Reinstall
python setup.py install

# 验证安装 / Verify installation
echo -e "\n${YELLOW}11. 验证安装 / Verifying installation${NC}"
python -c "import dgl; print(f'DGL version: {dgl.__version__}')"
python -c "import dgl; import torch; print(f'DGL CUDA available: {dgl.cuda.is_available()}')" 2>/dev/null || echo "CUDA check skipped"

echo -e "\n${GREEN}=========================================="
echo "✓ DGL 编译安装完成！ / DGL build completed!"
echo "==========================================${NC}"

echo -e "\n下一步: 运行测试 / Next step: Run tests"
echo "cd /home/qlyu/AA_peptide/pepland/claude_test"
echo "CUDA_VISIBLE_DEVICES=3 python test_dgl_cuda.py"
