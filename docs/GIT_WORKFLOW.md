# Git 工作流程指南

## 🎯 快速开始

### 已创建的工具脚本

```bash
pepland/
├── quick_push.sh      # 一键推送工具
├── git_history.sh     # 历史查看工具
└── git_rollback.sh    # 版本回滚工具
```

---

## 📤 一键上传代码

### 基本用法

```bash
# 1. 快速推送（使用默认提交信息）
./quick_push.sh

# 2. 推送并指定提交信息
./quick_push.sh "修复了模型训练的bug"

# 3. 推送并打标签
./quick_push.sh "完成模型优化" "v0.2.0"
```

### 详细示例

```bash
# 场景1: 日常代码更新
./quick_push.sh "Update training script and fix data loader"

# 场景2: 重要功能发布（打标签）
./quick_push.sh "Add Graphormer architecture" "v0.2.0"

# 场景3: 紧急bug修复
./quick_push.sh "🐛 Fix memory leak in data processing" "v0.1.1"

# 场景4: 文档更新
./quick_push.sh "📚 Update documentation"
```

---

## 🏷️ 版本标签系统

### 标签命名规范

使用 **语义化版本**（Semantic Versioning）：

```
v主版本.次版本.修订号

格式: vMAJOR.MINOR.PATCH
示例: v0.1.0, v1.2.3
```

**版本号含义**：
- **MAJOR（主版本）**: 重大架构变更，不向后兼容
  - 示例: v1.0.0 → v2.0.0
- **MINOR（次版本）**: 新功能添加，向后兼容
  - 示例: v1.0.0 → v1.1.0
- **PATCH（修订号）**: Bug修复，向后兼容
  - 示例: v1.0.0 → v1.0.1

### 当前版本规划

```
v0.1.0 - 文档阶段 ✅ (当前)
├── 完整的项目文档
├── 算力需求评估
└── 改进策略设计

v0.2.0 - 架构升级（计划中）
├── Graphormer替换HGT
├── Performer注意力机制
└── 深度提升到12层

v0.3.0 - 对比学习（计划中）
├── SimCLR实现
├── 数据增强策略
└── MoCo队列机制

v0.4.0 - 特征增强（计划中）
├── 3D构象集成
├── 物化性质特征
└── 序列编码

v1.0.0 - 正式发布（6周后）
├── 完整实现所有改进
├── 所有任务性能提升35-47%
└── 完整的论文和开源代码
```

### 标签操作命令

```bash
# 创建标签（本地）
git tag -a v0.2.0 -m "Add Graphormer architecture"

# 推送标签到GitHub
git push peptest v0.2.0

# 查看所有标签
git tag -l

# 查看标签详细信息
git show v0.1.0

# 删除本地标签
git tag -d v0.1.0

# 删除远程标签
git push peptest --delete v0.1.0

# 推送所有标签
git push peptest --tags
```

---

## 📜 查看历史

### 使用历史查看工具

```bash
# 查看所有信息（默认）
./git_history.sh

# 仅查看提交历史
./git_history.sh commits

# 仅查看标签
./git_history.sh tags

# 查看最近的更改
./git_history.sh diff

# 查看当前状态
./git_history.sh status

# 查看分支信息
./git_history.sh branches

# 查看统计信息
./git_history.sh stats
```

### 原生Git命令

```bash
# 查看提交历史（图形化）
git log --oneline --graph --decorate --all

# 查看最近5次提交
git log --oneline -5

# 查看某个文件的历史
git log --follow model/model.py

# 查看某人的提交
git log --author="cihebi2"

# 查看某个时间段的提交
git log --since="2 weeks ago"

# 查看提交统计
git shortlog -sn

# 查看某次提交的详细信息
git show <commit_hash>

# 查看两个版本之间的差异
git diff v0.1.0..v0.2.0
```

---

## ⏮️ 版本回滚

### 使用回滚工具

```bash
# 查看可用版本（无参数运行）
./git_rollback.sh

# 回滚到指定标签
./git_rollback.sh v0.1.0

# 回滚到指定提交
./git_rollback.sh d3eee13
```

**回滚工具特性**：
- ✅ 自动创建备份分支
- ✅ 显示详细的回滚信息
- ✅ 二次确认防止误操作
- ✅ 可选推送到GitHub

### 手动回滚方法

#### 方法1: 软回滚（保留更改）

```bash
# 回滚到上一次提交（保留更改在工作区）
git reset --soft HEAD~1

# 回滚到指定标签（保留更改）
git reset --soft v0.1.0
```

#### 方法2: 硬回滚（丢弃更改）⚠️

```bash
# 回滚到上一次提交（丢弃所有更改）
git reset --hard HEAD~1

# 回滚到指定标签（丢弃所有更改）
git reset --hard v0.1.0

# 强制推送到GitHub
git push -f peptest master
```

#### 方法3: 撤销提交（创建新提交）

```bash
# 撤销最近的提交（推荐，保留历史）
git revert HEAD

# 撤销指定提交
git revert <commit_hash>

# 推送到GitHub（无需强制）
git push peptest master
```

### 回滚后的恢复

```bash
# 查看所有操作历史（包括回滚）
git reflog

# 恢复到某个操作前的状态
git reset --hard HEAD@{2}

# 如果使用了回滚工具，可以恢复到备份分支
git reset --hard backup-20251014-123456
```

---

## 🌿 分支管理

### 创建和切换分支

```bash
# 创建新分支
git branch feature-graphormer

# 切换到分支
git checkout feature-graphormer

# 创建并切换（合并命令）
git checkout -b feature-graphormer

# 查看所有分支
git branch -a

# 删除分支
git branch -d feature-graphormer
```

### 分支工作流建议

```
master (主分支)
  ├── develop (开发分支)
  │   ├── feature-graphormer (功能分支)
  │   ├── feature-contrastive (功能分支)
  │   └── hotfix-bug (修复分支)
  └── release-v1.0 (发布分支)
```

**推荐工作流**：

```bash
# 1. 从master创建开发分支
git checkout -b develop master

# 2. 从develop创建功能分支
git checkout -b feature-graphormer develop

# 3. 在功能分支上开发
./quick_push.sh "Implement Graphormer layer"

# 4. 功能完成后合并回develop
git checkout develop
git merge feature-graphormer

# 5. develop测试通过后合并到master
git checkout master
git merge develop

# 6. 打标签发布
./quick_push.sh "Release v0.2.0" "v0.2.0"
```

---

## 🔍 差异对比

```bash
# 查看工作区更改
git diff

# 查看暂存区更改
git diff --staged

# 对比两个版本
git diff v0.1.0 v0.2.0

# 对比两个版本的统计
git diff --stat v0.1.0 v0.2.0

# 查看某个文件的更改
git diff HEAD model/model.py

# 对比两个分支
git diff master develop
```

---

## 📦 暂存和清理

### 暂存工作进度

```bash
# 暂存当前更改
git stash save "临时保存：正在开发的特征"

# 查看暂存列表
git stash list

# 恢复最近的暂存
git stash pop

# 恢复指定的暂存
git stash apply stash@{1}

# 删除暂存
git stash drop stash@{0}

# 清空所有暂存
git stash clear
```

### 清理工作区

```bash
# 撤销工作区的修改
git checkout -- <file>

# 撤销所有工作区修改
git checkout -- .

# 删除未跟踪的文件（预览）
git clean -n

# 删除未跟踪的文件（执行）
git clean -f

# 删除未跟踪的文件和目录
git clean -fd
```

---

## 🚨 常见问题处理

### 问题1: 推送冲突

```bash
# 拉取远程更新
git pull peptest master

# 如果有冲突，解决后提交
git add .
git commit -m "Resolve merge conflicts"
git push peptest master
```

### 问题2: 误提交敏感信息

```bash
# 从历史中删除文件
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch config/secrets.yaml" \
  --prune-empty --tag-name-filter cat -- --all

# 强制推送（谨慎使用）
git push peptest master --force
```

### 问题3: 提交信息写错

```bash
# 修改最近一次提交信息
git commit --amend -m "正确的提交信息"

# 强制推送（如果已推送）
git push -f peptest master
```

### 问题4: 推送了错误的文件

```bash
# 从暂存区移除文件
git reset HEAD <file>

# 删除文件并提交
git rm <file>
git commit -m "Remove wrong file"
git push peptest master
```

---

## 📊 工作流最佳实践

### 1. 提交信息规范

使用 **Conventional Commits** 格式：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**类型（type）**：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构（不是新功能也不是修复）
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建/工具配置

**示例**：

```bash
# 好的提交信息
git commit -m "feat(model): add Graphormer architecture

Implemented Graphormer layer with:
- Centrality encoding
- Spatial encoding
- Edge feature integration

Refs: #12"

# 不好的提交信息
git commit -m "update"
git commit -m "fix bug"
```

### 2. 常用Emoji

```
🎉 :tada: 初始提交
✨ :sparkles: 新功能
🐛 :bug: Bug修复
📚 :books: 文档
🎨 :art: 代码格式/结构
⚡ :zap: 性能优化
🔥 :fire: 删除代码/文件
✅ :white_check_mark: 测试
🔒 :lock: 安全
⬆️ :arrow_up: 升级依赖
⬇️ :arrow_down: 降级依赖
🚀 :rocket: 部署
```

### 3. 提交频率

- ✅ **频繁提交**: 每完成一个小功能就提交
- ✅ **有意义的提交**: 每次提交都是可工作的状态
- ❌ **避免大提交**: 不要积累太多更改再提交
- ❌ **避免无意义提交**: 不要提交"test"、"update"等

### 4. 标签策略

```
# 发布版本
v1.0.0, v1.1.0, v2.0.0

# 预发布版本
v1.0.0-alpha.1
v1.0.0-beta.1
v1.0.0-rc.1

# 实验版本
experiment-graphormer
experiment-contrastive-learning
```

---

## 🎯 完整工作流示例

### 场景1: 日常开发

```bash
# 1. 开始工作前，确保是最新代码
git pull peptest master

# 2. 修改代码
vim model/graphormer.py

# 3. 测试代码
python test_graphormer.py

# 4. 一键推送
./quick_push.sh "feat(model): implement Graphormer attention layer"

# 完成！
```

### 场景2: 重要版本发布

```bash
# 1. 确认所有功能已完成
./git_history.sh commits

# 2. 运行完整测试
pytest tests/

# 3. 更新文档
vim README.md
vim docs/CHANGELOG.md

# 4. 推送并打标签
./quick_push.sh "🚀 Release v0.2.0 - Graphormer Architecture" "v0.2.0"

# 5. 在GitHub上创建Release
# 访问: https://github.com/cihebi2/peptest/releases/new
```

### 场景3: 紧急回滚

```bash
# 1. 发现严重bug，需要回滚
./git_rollback.sh v0.1.0

# 2. 修复bug
vim model/bug.py

# 3. 重新发布
./quick_push.sh "🐛 hotfix: fix critical bug" "v0.1.1"
```

---

## 📞 快速参考

### 常用命令速查表

| 操作 | 命令 |
|------|------|
| **快速推送** | `./quick_push.sh "message"` |
| **推送+标签** | `./quick_push.sh "message" "v0.2.0"` |
| **查看历史** | `./git_history.sh` |
| **版本回滚** | `./git_rollback.sh v0.1.0` |
| **查看状态** | `git status` |
| **查看差异** | `git diff` |
| **查看提交** | `git log --oneline -10` |
| **查看标签** | `git tag -l` |
| **创建分支** | `git checkout -b branch-name` |
| **暂存更改** | `git stash` |

### GitHub仓库链接

- 🌐 **仓库首页**: https://github.com/cihebi2/peptest
- 📋 **提交历史**: https://github.com/cihebi2/peptest/commits/master
- 🏷️ **版本标签**: https://github.com/cihebi2/peptest/tags
- 📦 **发布页面**: https://github.com/cihebi2/peptest/releases

---

## 🎓 学习资源

- **Git官方文档**: https://git-scm.com/doc
- **GitHub Guides**: https://guides.github.com/
- **语义化版本规范**: https://semver.org/lang/zh-CN/
- **Conventional Commits**: https://www.conventionalcommits.org/

---

**文档版本**: 1.0
**最后更新**: 2025-10-14
**维护者**: PepLand Team
**仓库**: https://github.com/cihebi2/peptest
