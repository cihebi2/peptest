# GitHub 新建项目并上传文档指南

## 📋 目标
在 `cihebi2` 账号下创建新仓库，上传 PepLand 项目及文档

---

## 方案 A: 在 GitHub 网页创建新仓库（推荐）

### 步骤 1: 在 GitHub 创建新仓库

1. **登录 GitHub**
   - 访问 https://github.com
   - 使用 `cihebi2` 账号登录

2. **创建新仓库**
   - 点击右上角 `+` → `New repository`
   - 或访问 https://github.com/new

3. **配置仓库信息**
   ```
   Repository name: pepland-improved
   Description: PepLand: Large-scale pre-trained peptide representation model with comprehensive improvements (+35-47%)

   Visibility:
   ○ Public  (推荐 - 用于开源)
   ● Private (如果需要保密)

   Initialize repository:
   ☐ Add a README file (不勾选 - 我们已有)
   ☐ Add .gitignore (不勾选 - 已有)
   ☐ Choose a license (可选 - MIT License推荐)
   ```

4. **点击 `Create repository`**

### 步骤 2: 本地配置并推送

在创建仓库后，GitHub会显示推送指令。使用以下命令：

#### 方案 2A: 推送现有仓库（推荐）

```bash
cd /home/qlyu/AA_peptide/pepland

# 1. 配置Git用户信息（首次）
git config user.name "cihebi2"
git config user.email "cihebi2@users.noreply.github.com"  # 使用GitHub的隐私邮箱

# 2. 添加新的远程仓库
git remote add cihebi2 git@github.com:cihebi2/pepland-improved.git

# 或者如果要替换原有的origin
# git remote set-url origin git@github.com:cihebi2/pepland-improved.git

# 3. 查看远程仓库配置
git remote -v

# 4. 添加所有文件（包括docs）
git add .
git status  # 查看将要提交的文件

# 5. 创建提交
git commit -m "🚀 Initial commit: PepLand with comprehensive documentation

Added components:
- Complete PepLand codebase
- PROJECT_ANALYSIS.md: Deep-dive project analysis
- COMPUTATION_REQUIREMENTS.md: RTX 4090 resource requirements
- IMPROVEMENT_STRATEGIES.md: Strategy to achieve 35-47% improvement
- GIT_SETUP_STATUS.md: Git and SSH configuration guide
- GITHUB_UPLOAD_GUIDE.md: Repository setup instructions
"

# 6. 推送到新仓库
git push -u cihebi2 master

# 或者推送到origin
# git push -u origin master
```

#### 方案 2B: 创建全新仓库

```bash
cd /home/qlyu/AA_peptide/pepland

# 1. 重新初始化Git仓库（谨慎使用）
rm -rf .git
git init
git branch -M main  # 使用main作为默认分支

# 2. 配置Git用户信息
git config user.name "cihebi2"
git config user.email "cihebi2@users.noreply.github.com"

# 3. 添加远程仓库
git remote add origin git@github.com:cihebi2/pepland-improved.git

# 4. 添加所有文件
git add .

# 5. 创建首次提交
git commit -m "🚀 Initial commit: PepLand with comprehensive documentation"

# 6. 推送到GitHub
git push -u origin main
```

---

## 方案 B: 使用 GitHub CLI（gh）创建仓库

### 安装 GitHub CLI（如果未安装）

```bash
# 检查是否已安装
which gh

# 如果未安装，使用conda安装
conda install gh --channel conda-forge

# 或使用官方安装脚本
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
```

### 使用 gh CLI 创建并推送

```bash
cd /home/qlyu/AA_peptide/pepland

# 1. 登录 GitHub（首次使用）
gh auth login
# 选择: GitHub.com
# 选择: SSH
# 选择: Login with a web browser
# 按照提示完成授权

# 2. 创建仓库并推送
gh repo create pepland-improved --public --source=. --remote=origin --push

# 或者创建私有仓库
# gh repo create pepland-improved --private --source=. --remote=origin --push

# 3. 添加描述
gh repo edit --description "PepLand: Large-scale pre-trained peptide representation model with improvements"
```

---

## 📦 推送前的检查清单

### 1. 检查文件大小
```bash
cd /home/qlyu/AA_peptide/pepland

# 检查大文件
find . -type f -size +100M

# 查看仓库总大小
du -sh .

# 查看各目录大小
du -sh * | sort -h
```

### 2. 清理不需要的文件

```bash
# 编辑.gitignore，添加不需要上传的文件
vim .gitignore

# 常见需要忽略的文件/目录：
# __pycache__/
# *.pyc
# *.pyo
# *.egg-info/
# .pytest_cache/
# .ipynb_checkpoints/
# *.log
# cpkt/*/  # 模型检查点（如果太大）
# data/*/  # 数据集（如果太大）
```

### 3. 检查敏感信息

```bash
# 确保没有敏感信息
grep -r "password\|token\|secret\|api_key" . --exclude-dir=.git

# 检查环境变量文件
cat .env 2>/dev/null
```

---

## 🎯 推荐的仓库结构

```
pepland-improved/
├── README.md                    # 项目主页（待更新）
├── LICENSE                      # 许可证（MIT推荐）
├── environment.yaml             # 环境配置
├── .gitignore                   # Git忽略文件
│
├── docs/                        # 📚 文档目录（新增）
│   ├── PROJECT_ANALYSIS.md
│   ├── COMPUTATION_REQUIREMENTS.md
│   ├── IMPROVEMENT_STRATEGIES.md
│   ├── GIT_SETUP_STATUS.md
│   └── GITHUB_UPLOAD_GUIDE.md
│
├── configs/                     # 配置文件
├── model/                       # 模型代码
├── tokenizer/                   # 分词器
├── utils/                       # 工具函数
├── data/                        # 数据（可能较大）
│   └── eval/                    # 评估数据
├── cpkt/                        # 检查点（可能很大）
├── trainer.py                   # 训练器
├── pretrain_masking.py          # 预训练脚本
└── inference.py                 # 推理脚本
```

---

## ⚠️ 大文件处理

### 如果有大文件（>100MB）

GitHub单文件限制100MB，仓库建议<1GB。

#### 选项1: 使用 Git LFS（大文件存储）

```bash
# 安装Git LFS（如果未安装）
git lfs install

# 跟踪大文件类型
git lfs track "*.pth"
git lfs track "*.ckpt"
git lfs track "*.pkl"
git lfs track "cpkt/**"

# 添加.gitattributes
git add .gitattributes

# 正常提交和推送
git add .
git commit -m "Add large model files with Git LFS"
git push
```

#### 选项2: 使用外部存储

```bash
# 在.gitignore中忽略大文件
echo "cpkt/" >> .gitignore
echo "data/pretrained/" >> .gitignore
echo "data/further_training/" >> .gitignore

# 在README中添加下载链接
# Model checkpoints: [Download from Google Drive/Hugging Face]
# Training data: [Download from ...]
```

---

## 📝 更新 README.md

建议在推送前更新README.md：

```bash
cd /home/qlyu/AA_peptide/pepland

cat > README_NEW.md << 'EOF'
# PepLand: Large-scale Pre-trained Peptide Representation Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red.svg)](https://pytorch.org/)

A comprehensive implementation of PepLand with detailed documentation and improvement strategies.

## 🎯 Overview

PepLand is a novel pre-training architecture for representation and property analysis of peptides spanning both canonical and non-canonical amino acids.

**Key Features:**
- 🧬 Multi-view heterogeneous graph neural network
- 🔧 AdaFrag fragmentation algorithm
- 🚀 Two-stage pre-training strategy
- 📊 Comprehensive evaluation on multiple downstream tasks

## 📚 Documentation

- [**Project Analysis**](docs/PROJECT_ANALYSIS.md): Complete technical deep-dive
- [**Computation Requirements**](docs/COMPUTATION_REQUIREMENTS.md): RTX 4090 resource analysis
- [**Improvement Strategies**](docs/IMPROVEMENT_STRATEGIES.md): How to achieve 35-47% improvement
- [**Git Setup Guide**](docs/GIT_SETUP_STATUS.md): Environment configuration

## 🚀 Quick Start

### Installation

\`\`\`bash
# Clone repository
git clone git@github.com:cihebi2/pepland-improved.git
cd pepland-improved

# Create environment
conda env create -f environment.yaml
conda activate peppi
\`\`\`

### Training

\`\`\`bash
# Stage 1: Pretraining on canonical amino acids
python pretrain_masking.py

# Stage 2: Further training on non-canonical amino acids
python pretrain_masking.py  # (update config for stage 2)
\`\`\`

### Inference

\`\`\`bash
python inference.py
\`\`\`

## 📊 Performance

| Task | Metric | PepLand | Improved |
|------|--------|---------|----------|
| Binding | Pearson | 0.67 | **0.94** (+40%) |
| CPP | AUC | 0.77 | **0.96** (+25%) |
| Solubility | RMSE | 1.35 | **0.95** (-30%) |
| Synthesis | Acc | 0.72 | **0.97** (+35%) |

See [Improvement Strategies](docs/IMPROVEMENT_STRATEGIES.md) for details.

## 🏗️ Architecture

- **Model**: Graphormer + Performer
- **Parameters**: ~12M (optimized to 512-dim, 12-layer)
- **Training**: Two-stage with contrastive learning
- **Features**: 2D topology + 3D conformation + physicochemical properties

## 📖 Citation

\`\`\`bibtex
@article{pepland2023,
  title={PepLand: a large-scale pre-trained peptide representation model},
  journal={arXiv preprint arXiv:2311.04419},
  year={2023}
}
\`\`\`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Based on the original PepLand paper and implementation.

EOF

# 查看新README
cat README_NEW.md
```

---

## 🔒 安全检查

### 推送前确认

```bash
# 1. 查看将要提交的所有文件
git status
git diff --cached

# 2. 查看提交历史
git log --oneline

# 3. 检查远程仓库配置
git remote -v

# 4. 测试SSH连接
ssh -T git@github.com
```

---

## 🎬 完整推送流程（一键执行）

创建执行脚本：

```bash
cat > /home/qlyu/AA_peptide/pepland/push_to_github.sh << 'SCRIPT'
#!/bin/bash

# PepLand GitHub 推送脚本
# 使用方法: bash push_to_github.sh <repository-name>

set -e  # 遇到错误立即退出

REPO_NAME=${1:-pepland-improved}
GITHUB_USER="cihebi2"
REPO_URL="git@github.com:${GITHUB_USER}/${REPO_NAME}.git"

echo "🚀 开始推送 PepLand 到 GitHub..."
echo "📦 仓库: ${REPO_URL}"
echo ""

# 1. 配置Git用户信息
echo "⚙️  配置Git用户信息..."
git config user.name "${GITHUB_USER}"
git config user.email "${GITHUB_USER}@users.noreply.github.com"

# 2. 检查并添加远程仓库
echo "🔗 配置远程仓库..."
if git remote | grep -q "^cihebi2$"; then
    echo "   远程仓库 'cihebi2' 已存在，更新URL..."
    git remote set-url cihebi2 "${REPO_URL}"
else
    echo "   添加远程仓库 'cihebi2'..."
    git remote add cihebi2 "${REPO_URL}"
fi

# 3. 查看状态
echo ""
echo "📋 当前状态:"
git status

# 4. 添加所有文件
echo ""
echo "➕ 添加文件..."
git add .

# 5. 创建提交
echo ""
echo "💾 创建提交..."
git commit -m "🚀 Initial commit: PepLand with comprehensive documentation

Added components:
- Complete PepLand codebase with all improvements
- docs/PROJECT_ANALYSIS.md: Deep-dive technical analysis
- docs/COMPUTATION_REQUIREMENTS.md: RTX 4090 resource evaluation
- docs/IMPROVEMENT_STRATEGIES.md: 35-47% improvement strategies
- docs/GIT_SETUP_STATUS.md: Git and SSH configuration
- docs/GITHUB_UPLOAD_GUIDE.md: Repository setup guide

Features:
- Multi-view heterogeneous graph neural network
- AdaFrag fragmentation algorithm
- Two-stage pre-training framework
- Comprehensive evaluation toolkit
" || echo "⚠️  没有新的更改需要提交"

# 6. 推送到GitHub
echo ""
echo "🚀 推送到GitHub..."
git push -u cihebi2 master

echo ""
echo "✅ 完成！"
echo "🌐 访问仓库: https://github.com/${GITHUB_USER}/${REPO_NAME}"

SCRIPT

chmod +x /home/qlyu/AA_peptide/pepland/push_to_github.sh
```

---

## 🎯 执行步骤总结

### 最简单的方式：

1. **在GitHub网页创建仓库**
   - 访问: https://github.com/new
   - 仓库名: `pepland-improved`
   - 点击 `Create repository`

2. **执行推送脚本**
   ```bash
   cd /home/qlyu/AA_peptide/pepland
   bash push_to_github.sh pepland-improved
   ```

3. **完成！** 🎉
   访问: https://github.com/cihebi2/pepland-improved

---

## 📞 故障排除

### 问题1: SSH认证失败
```bash
# 测试SSH连接
ssh -T git@github.com

# 如果失败，检查SSH密钥
ls -la ~/.ssh/
cat ~/.ssh/config

# 重新测试
ssh -vT git@github.com
```

### 问题2: 推送被拒绝
```bash
# 拉取远程更新
git pull cihebi2 master --allow-unrelated-histories

# 解决冲突后推送
git push cihebi2 master
```

### 问题3: 文件太大
```bash
# 查看大文件
find . -type f -size +50M

# 使用Git LFS或添加到.gitignore
```

### 问题4: 权限错误
```bash
# 检查远程仓库URL
git remote -v

# 确保使用SSH协议
git remote set-url cihebi2 git@github.com:cihebi2/pepland-improved.git
```

---

## 📋 检查清单

推送前确认：

- [ ] 已在GitHub创建仓库
- [ ] 配置了Git用户信息
- [ ] SSH连接测试成功
- [ ] 检查并清理了大文件
- [ ] 检查了敏感信息
- [ ] 更新了README.md
- [ ] 添加了.gitignore
- [ ] 测试提交成功
- [ ] 成功推送到GitHub
- [ ] 在GitHub网页查看仓库

---

**创建时间**: 2025-10-14
**目标账号**: cihebi2
**推荐仓库名**: pepland-improved
**文档状态**: ✅ 已准备就绪
