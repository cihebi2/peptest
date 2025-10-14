# PepLand 项目深度分析文档

## 目录
- [项目概述](#项目概述)
- [项目架构](#项目架构)
- [数据收集与处理](#数据收集与处理)
- [模型架构设计](#模型架构设计)
- [训练策略](#训练策略)
- [模型评估](#模型评估)
- [代码完整性分析](#代码完整性分析)
- [使用指南](#使用指南)

---

## 项目概述

### 研究背景
PepLand 是一个针对肽段（peptide）表示学习的大规模预训练模型，专门设计用于处理包含规范氨基酸（canonical amino acids）和非规范氨基酸（non-canonical amino acids）的肽段分子。

### 核心创新
1. **多视图异构图神经网络（Multi-view Heterogeneous Graph Neural Network）**
   - 同时建模原子级（atom-level）和片段级（fragment-level）的分子表示
   - 融合多个粒度的结构信息

2. **自适应分片算法（AdaFrag）**
   - Amiibo算子：保留酰胺键的同时切割肽段
   - BRICS算法：进一步细化大型侧链结构

3. **两阶段预训练策略**
   - 第一阶段：在规范氨基酸数据上预训练
   - 第二阶段：在非规范氨基酸数据上继续训练

### 论文信息
- 标题：PepLand: a large-scale pre-trained peptide representation model for a comprehensive landscape of both canonical and non-canonical amino acids
- arXiv: https://arxiv.org/abs/2311.04419

---

## 项目架构

### 目录结构
```
pepland/
├── configs/              # 配置文件目录
│   ├── pretrain_masking.yaml  # 预训练配置
│   ├── inference.yaml         # 推理配置
│   └── *.json                 # 其他任务配置
├── model/                # 模型核心代码
│   ├── model.py         # PharmHGT模型定义
│   ├── hgt.py           # HGT层实现
│   ├── data.py          # 数据处理和加载
│   ├── core.py          # 特征提取器和预测器
│   └── util.py          # 工具函数
├── tokenizer/           # 分子分片工具
│   ├── pep2fragments.py # 分片算法实现
│   └── vocabs/          # 片段词汇表
├── utils/               # 通用工具
│   ├── metrics.py       # 评估指标
│   ├── distribution.py  # 分布式训练工具
│   ├── std_logger.py    # 日志记录
│   └── utils.py         # 通用函数
├── data/                # 数据目录
│   ├── pretrained/      # 预训练数据（规范氨基酸）
│   ├── further_training/# 进一步训练数据（非规范氨基酸）
│   └── eval/            # 评估数据集
├── cpkt/                # 模型检查点
├── test/                # 测试代码
├── trainer.py           # 训练器实现
├── pretrain_masking.py  # 预训练主脚本
├── inference.py         # 推理脚本
├── splitters.py         # 数据集划分工具
└── environment.yaml     # 环境依赖
```

### 核心模块关系
```
pretrain_masking.py (主入口)
    ↓
trainer.py (训练循环)
    ↓
model/model.py (PharmHGT模型)
    ↓
model/hgt.py (HGT层) + model/data.py (数据加载)
    ↓
tokenizer/pep2fragments.py (分子分片)
```

---

## 数据收集与处理

### 1. 数据来源

#### 预训练数据
- **位置**: `data/pretrained/`
- **内容**: 包含规范氨基酸的肽段SMILES数据
- **格式**: CSV文件（train.csv, valid.csv, test.csv）
- **字段**: `smiles` 列

#### 进一步训练数据
- **位置**: `data/further_training/`
- **内容**: 包含非规范氨基酸的肽段SMILES数据
- **格式**: CSV文件（train.csv, valid.csv, test.csv）

#### 评估数据集
- **位置**: `data/eval/`
- **数据集**:
  - `c-binding.csv`: 规范氨基酸结合数据
  - `nc-binding.csv`: 非规范氨基酸结合数据
  - `c-CPP.txt`: 规范氨基酸细胞穿透肽（Cell-Penetrating Peptides）
  - `nc-CPP.csv`: 非规范氨基酸CPP
  - `c-Sol.txt`: 规范氨基酸溶解度数据

### 2. 数据处理流程

#### 分子分片（Tokenization）
**核心算法位置**: `tokenizer/pep2fragments.py`

**AdaFrag算法流程**:
```python
1. 识别酰胺键结构: C(=O)N
2. Amiibo切割:
   - 保留酰胺键
   - 切割C-C单键和C-N单键（非酰胺键）
3. BRICS细化（可选）:
   - 对大型侧链应用BRICS算法进一步切割
```

**关键函数**:
- `get_cut_bond_idx(mol, side_chain_cut=True)`: 获取需要切割的键索引
  - `side_chain_cut=True`: 使用AdaFrag（Amiibo + BRICS）
  - `side_chain_cut=False`: 仅使用Amiibo
- `cut_peptide(mol, patt)`: 切割肽段并返回有序片段
- `brics_molecule(mol)`: 应用BRICS算法切割分子

#### 图构建（Graph Construction）
**代码位置**: `model/data.py:496-632`

**构建流程**:
```python
Mol2HeteroGraph(mol, frag='258'):
    1. 分子分片 → 获取片段（fragments）
    2. 构建异构图节点:
       - 'a' (atom): 原子节点
       - 'p' (pharm/fragment): 片段节点
       - 'junc' (junction): 连接节点（隐式）
    3. 构建异构图边:
       - ('a', 'b', 'a'): 原子间的键
       - ('p', 'r', 'p'): 片段间的反应边
       - ('a', 'j', 'p'): 原子到片段的连接
       - ('p', 'j', 'a'): 片段到原子的连接
    4. 提取特征:
       - 原子特征: 42维 (原子类型、度、电荷、手性、杂化等)
       - 键特征: 14维 (键类型、共轭性、环状等)
       - 片段特征: 196维 (MACCS指纹 + 药效团性质)
    5. 返回DGL异构图
```

**特征详情**:
```python
# 原子特征 (42维)
atom_features = [
    one_hot(atomic_num, 9种元素),      # 10维
    one_hot(degree, [0-5]),             # 6维
    one_hot(formal_charge, [-2,-1,0,1,2]), # 5维
    one_hot(chiral_tag, [0-3]),         # 4维
    one_hot(num_Hs, [0-4]),             # 5维
    one_hot(hybridization, 5种类型),    # 6维
    [is_aromatic],                      # 1维
    [mass * 0.01]                       # 1维
]

# 片段特征 (196维)
fragment_features = [
    maccs_keys,                         # 167维 (MACCS分子指纹)
    [padding],                          # 1维
    pharmacophore_properties,           # 27维 (药效团性质)
    [padding]                           # 1维
]

# 键特征 (14维)
bond_features = [
    [not_none],                         # 1维
    [is_single, is_double, is_triple, is_aromatic], # 4维
    [is_conjugated],                    # 1维
    [in_ring],                          # 1维
    one_hot(stereo, [0-5])              # 7维
]
```

#### 数据增强（Masking）
**代码位置**: `model/data.py:330-493` (`MaskAtom` 类)

**三种Masking策略**:
```python
1. Random Atom Masking (mask_rate=0.8):
   - 随机遮蔽80%的原子特征
   - 用特殊的mask特征向量替换

2. Amino Acid-based Masking (mask_amino=0.3):
   - 按氨基酸单元遮蔽
   - 随机选择30%的氨基酸，遮蔽其所有原子

3. Peptide Side Chain Masking (mask_pep=0.8):
   - 专门遮蔽侧链原子
   - 保留主链结构，遮蔽80%的侧链原子

4. Fragment Masking (mask_pharm=True):
   - 遮蔽片段节点特征
   - 使用mask片段特征替换

5. Edge Masking (mask_edge=False):
   - 可选：遮蔽连接到被mask原子的边
   - 默认关闭
```

#### 数据加载器
**代码位置**: `model/data.py:635-789`

**主要组件**:
```python
# 1. 可迭代数据集
class MolGraphSet(IterableDataset):
    - 支持多进程数据加载
    - 实时将SMILES转换为DGL图
    - 应用数据增强变换

# 2. DataLoader创建
make_loaders(cfg, ddp, dataset, ...):
    - 创建train/valid/test加载器
    - 支持分布式训练（DDP）
    - 使用DGL的GraphDataLoader

# 配置参数
batch_size: 512         # 批次大小
num_workers: 8          # 数据加载进程数
```

### 3. 数据集划分

**代码位置**: `splitters.py`

**支持的划分方式**:
1. **Random Split**: 随机划分（默认）
   - `random_split(dataset, frac_train=0.9, frac_valid=0.05, frac_test=0.05)`

2. **Scaffold Split**: 基于分子骨架的划分
   - `scaffold_split(dataset, smiles_list, ...)`
   - 使用Bemis-Murcko骨架
   - 确保结构相似的分子在同一集合

3. **Random Scaffold Split**: 随机骨架划分
   - `random_scaffold_split(dataset, smiles_list, ...)`

---

## 模型架构设计

### 1. 整体架构：PharmHGT

**代码位置**: `model/model.py:320-415`

```python
PharmHGT(
    hid_dim=300,        # 隐藏层维度
    act='ReLU',         # 激活函数
    depth=5,            # 消息传递层数
    atom_dim=42,        # 原子特征维度
    bond_dim=14,        # 键特征维度
    pharm_dim=196,      # 片段特征维度
    reac_dim=14         # 反应边特征维度
)
```

**架构组件**:
```
输入: DGL异构图
    ↓
特征初始化层
    ↓
多视图消息传递 (MVMP) × 5层
    ↓
图读出层 (Node_GRU)
    ↓
输出: 原子表示 + 片段表示
```

### 2. 核心模块详解

#### 2.1 多视图消息传递（MVMP）
**代码位置**: `model/model.py:174-318`

**设计思想**:
- 同时在原子视图和片段视图上进行消息传递
- 通过junction节点连接两个视图
- 实现跨视图信息交互

**实现细节**:
```python
class MVMP(nn.Module):
    def __init__(self, hid_dim=300, depth=3, view='apj'):
        # view='apj': atom + pharm + junction

        # 1. 同构边类型（homogeneous edges）
        homo_etypes = [
            ('a', 'b', 'a'),  # atom-bond-atom
            ('p', 'r', 'p')   # pharm-reaction-pharm
        ]

        # 2. 异构边类型（heterogeneous edges）
        hetero_etypes = [
            ('a', 'j', 'p'),  # atom-junction-pharm
            ('p', 'j', 'a')   # pharm-junction-atom
        ]

        # 3. 注意力机制
        self.attn = MultiHeadedAttention(n_heads=4, d_model=hid_dim)

        # 4. 消息传递层
        self.mp_list = nn.ModuleList([
            nn.Linear(hid_dim, hid_dim)
            for _ in range(depth-1)
        ])

    def forward(self, bg):
        # 迭代消息传递
        for i in range(depth-1):
            # (1) 同构边消息传递
            bg.multi_update_all(homo_update_funcs)

            # (2) 异构边消息传递
            apply_custom_copy_src(bg, hetero_etypes)

            # (3) 边特征更新
            update_edge_features(bg)

        # 最终节点特征更新
        final_node_update(bg)
```

**消息传递机制**:
```
时刻 t:
    节点特征: h_v^t
    边特征: e_uv^t

消息计算:
    m_uv = Attention(h_u^t, h_v^t) × e_uv^t

消息聚合:
    m_v = Σ_{u∈N(v)} m_uv

节点更新:
    h_v^{t+1} = MLP([h_v^t || m_v || f_v])
    其中 f_v 是初始特征
```

#### 2.2 异构图Transformer（HGT）
**代码位置**: `model/hgt.py:12-182`

**HGT层设计**:
```python
class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, node_dict, edge_dict, n_heads=4):
        # 1. 类型特定的线性变换
        self.k_linears = nn.ModuleList()  # 每种节点类型的K矩阵
        self.q_linears = nn.ModuleList()  # 每种节点类型的Q矩阵
        self.v_linears = nn.ModuleList()  # 每种节点类型的V矩阵

        # 2. 关系特定的注意力矩阵
        self.relation_att = nn.Parameter(
            torch.Tensor(num_relations, n_heads, d_k, d_k)
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(num_relations, n_heads, d_k, d_k)
        )

        # 3. 关系优先级
        self.relation_pri = nn.Parameter(
            torch.ones(num_relations, n_heads)
        )
```

**HGT注意力计算**:
```
对于边 (u, e, v):

1. 类型特定投影:
   K_u = W_k^{type(u)} h_u
   Q_v = W_q^{type(v)} h_v
   V_u = W_v^{type(u)} h_u

2. 关系特定变换:
   K'_u = K_u × R_att^e
   V'_u = V_u × R_msg^e

3. 注意力得分:
   α_{uv} = softmax((Q_v · K'_u) / √d_k × R_pri^e)

4. 消息聚合:
   m_v = Σ_{u∈N(v)} α_{uv} × V'_u

5. 节点更新:
   h_v^{new} = LayerNorm(α × MLP(m_v) + (1-α) × h_v^{old})
```

#### 2.3 图读出层（Graph Readout）
**代码位置**: `model/model.py:89-158`

**Node_GRU设计**:
```python
class Node_GRU(nn.Module):
    def __init__(self, hid_dim, bidirectional=True):
        # 1. 多头注意力混合
        self.att_mix = MultiHeadedAttention(6, hid_dim)

        # 2. 双向GRU
        self.gru = nn.GRU(
            hid_dim, hid_dim,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, bg, suffix='h'):
        # (1) 获取原子和片段表示
        p_pharmj = split_batch(bg, 'p', f'f_{suffix}')
        a_pharmj = split_batch(bg, 'a', f'f_{suffix}')

        # (2) 跨视图注意力
        h = self.att_mix(a_pharmj, p_pharmj, p_pharmj, mask)
        h = h + a_pharmj  # 残差连接

        # (3) 双向GRU处理序列
        h, hidden = self.gru(h)

        # (4) 聚合为图级表示
        graph_embed = mean_pooling(h)

        return graph_embed
```

### 3. 特征提取器和预测器

#### 3.1 PepLandFeatureExtractor
**代码位置**: `model/core.py:89-243`

**功能**:
```python
class PepLandFeatureExtractor(nn.Module):
    def __init__(self, model_path, pooling='avg'):
        # 加载预训练模型
        self.model = load_model(model_path)

        # 移除任务特定层
        remove_layers(['readout', 'out'])

        # 池化策略
        if pooling == 'avg':
            pooling_layer = AdaptiveAvgPool1d
        elif pooling == 'max':
            pooling_layer = AdaptiveMaxPool1d
        elif pooling == 'gru':
            pooling_layer = Node_GRU

    def forward(self, input_smiles):
        # (1) SMILES → 图
        graphs = self.tokenize(input_smiles)

        # (2) 提取原子和片段表示
        atom_rep, frag_rep = self.model(batch_graphs)

        # (3) 池化为肽段级表示
        pep_embeds = self.pooling_layer(
            concat([atom_rep, frag_rep])
        )

        return pep_embeds  # [batch_size, 300]
```

**使用缓存优化**:
```python
# LRU缓存机制
self._tokenize_cache = OrderedDict()
self.max_cache_size = 100000

# 避免重复计算SMILES → 图的转换
```

#### 3.2 PropertyPredictor
**代码位置**: `model/core.py:245-298`

**下游任务微调**:
```python
class PropertyPredictor(nn.Module):
    def __init__(self, model_path, hidden_dims=[256, 128]):
        # 特征提取器（冻结权重）
        self.feature_model = PepLandFeatureExtractor(
            model_path, pooling='avg'
        )

        # MLP预测头
        self.mlp = nn.Sequential(
            nn.Linear(300, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_smiles):
        graph_rep = self.feature_model(input_smiles)
        prediction = self.mlp(graph_rep)
        return prediction
```

### 4. 模型参数统计

```python
# PharmHGT模型参数
Total Parameters: ~5-10M (取决于具体配置)

# 各模块参数分布
- 特征初始化层: ~0.2M
  - w_atom: 42 × 300 = 12.6K
  - w_bond: 14 × 300 = 4.2K
  - w_pharm: 196 × 300 = 58.8K
  - w_junc: 238 × 300 = 71.4K

- MVMP层 (×5): ~4M
  - 注意力层: 4 × (300 × 300 × 4) = 1.44M per layer
  - MLP层: 2 × (300 × 300) = 0.18M per layer

- 读出层: ~0.5M
  - GRU: 2 × (300 × 300 × 6) = 1.08M

- 预测头: ~0.2M
  - Linear: 1200 × 300 + 300 × num_tasks
```

---

## 训练策略

### 1. 两阶段训练策略

#### 阶段1: 规范氨基酸预训练
**配置**: `configs/pretrain_masking.yaml`
```yaml
train:
  dataset: pretrained        # 使用规范氨基酸数据
  model: PharmHGT           # 从头训练
  batch_size: 512
  epochs: 50
  lr: 0.001
  num_layer: 5
  hid_dim: 300
  mask_rate: 0.8
  mask_pharm: True
  mask_pep: 0.8
```

**训练命令**:
```bash
python pretrain_masking.py
```

#### 阶段2: 非规范氨基酸继续训练
**配置修改**:
```yaml
train:
  dataset: further_training   # 使用非规范氨基酸数据
  model: fine-tune           # 加载阶段1模型
inference:
  model_path: ./inference/cpkt/  # 阶段1模型路径
```

**训练命令**:
```bash
python pretrain_masking.py  # 使用更新后的配置
```

### 2. 自监督学习任务

**代码位置**: `trainer.py:51-160` (`train_epoch` 方法)

#### 主要预训练任务:
```python
# 1. 原子掩码预测（Masked Atom Prediction）
loss_atom = CrossEntropy(
    pred_atom,           # 模型预测的原子类型
    true_atom_label      # 真实原子类型 (119类)
)

# 2. 片段掩码预测（Masked Fragment Prediction）
if cfg.train.mask_pharm:
    loss_pharm = CrossEntropy(
        pred_pharm,      # 模型预测的片段类型
        true_pharm_label # 真实片段类型 (264类片段词汇)
    )

# 3. 边掩码预测（Masked Edge Prediction）
if cfg.train.mask_edge:
    loss_edge = CrossEntropy(
        pred_edge,       # 模型预测的边类型
        true_edge_label  # 真实边类型 (4类)
    )

# 总损失
total_loss = loss_atom + loss_pharm + loss_edge
```

**预测头设计**:
```python
# 原子预测头
linear_pred_atoms = nn.Linear(300, 119)  # 119种原子类型

# 片段预测头
linear_pred_pharms = nn.Linear(300, 264)  # 264种片段

# 键预测头
linear_pred_bonds = nn.Linear(300, 4)  # 4种键类型
```

### 3. 优化策略

**优化器配置**:
```python
# Adam优化器
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,           # 学习率
    weight_decay=0      # L2正则化（默认关闭）
)

# 每个模块独立优化器
optimizer_model = optim.Adam(model.parameters(), lr=0.001)
optimizer_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=0.001)
optimizer_pred_pharms = optim.Adam(linear_pred_pharms.parameters(), lr=0.001)
optimizer_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=0.001)
```

**学习率策略**:
```python
# 注释代码中提到支持Plateau调度器（当前未激活）
# scheduler = ReduceLROnPlateau(
#     optimizer,
#     mode='max',
#     factor=0.5,
#     patience=5
# )
```

### 4. 训练监控

#### 日志记录
**代码位置**: `trainer.py:134-154`
```python
# MLflow日志
if cfg.logger.log:
    mlflow.log_metric("train/loss", loss, step=global_step)
    mlflow.log_metric("train/acc_atom", acc_atom, step=global_step)
    mlflow.log_metric("train/acc_pharm", acc_pharm, step=global_step)
    mlflow.log_metric("train/acc_edge", acc_edge, step=global_step)

# 控制台日志
Logger.info(f"train | epoch: {epoch} step: {step} | loss: {loss:.4f}")
```

#### 验证评估
**代码位置**: `trainer.py:161-271`
```python
# 每500步评估一次
if (global_train_step + 1) % 500 == 0:
    eval_epoch("valid")
    eval_epoch("test")

# 每个epoch结束评估
eval_epoch("valid")
eval_epoch("test")
```

#### 模型保存
**代码位置**: `trainer.py:210-270`
```python
# 保存最佳模型（基于验证集acc_atom）
if metrics['valid_acc_atom'] >= best_metric:
    best_metric = metrics['valid_acc_atom']

    # 保存4个模型组件
    mlflow.pytorch.save_model(model, path='model')
    mlflow.pytorch.save_model(linear_pred_atoms, path='linear_pred_atoms')
    mlflow.pytorch.save_model(linear_pred_pharms, path='linear_pred_pharms')
    mlflow.pytorch.save_model(linear_pred_bonds, path='linear_pred_bonds')
```

### 5. 分布式训练支持

**DDP配置**:
```python
# 启用DDP
if cfg.mode.ddp:
    # 初始化进程组
    setup_multinodes(local_rank, world_size)

    # 包装模型
    model = DDP(
        model,
        device_ids=[global_rank],
        output_device=global_rank
    )

    # 分布式采样器
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=global_rank
    )

    # 同步指标
    torch.distributed.all_reduce(loss_accum, op=ReduceOp.SUM)
    loss_accum /= world_size
```

**环境变量**:
```bash
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
```

### 6. 超参数总结

| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 512 | 批次大小 |
| epochs | 50 | 训练轮数 |
| lr | 0.001 | 学习率 |
| decay | 0 | 权重衰减 |
| num_layer | 5 | GNN层数 |
| hid_dim | 300 | 隐藏层维度 |
| atom_dim | 42 | 原子特征维度 |
| bond_dim | 14 | 键特征维度 |
| pharm_dim | 196 | 片段特征维度 |
| mask_rate | 0.8 | 掩码比例 |
| mask_pharm | True | 是否掩码片段 |
| mask_pep | 0.8 | 侧链掩码比例 |
| mask_edge | False | 是否掩码边 |
| num_workers | 8 | 数据加载进程数 |

---

## 模型评估

### 1. 预训练评估指标

**代码位置**: `trainer.py:272-367` (`evaluate` 方法)

#### 核心指标:
```python
metrics = {
    # 损失
    'loss': avg_loss,

    # 原子预测准确率
    'acc_atom': correct_atoms / total_atoms,

    # 片段预测准确率
    'acc_pharm': correct_pharms / total_pharms,

    # 边预测准确率
    'acc_edge': correct_edges / total_edges
}
```

**准确率计算**:
```python
def compute_accuracy(pred, target):
    pred_class = torch.argmax(pred, dim=-1)
    correct = (pred_class == target).sum().item()
    total = target.size(0)
    accuracy = correct / total
    return accuracy
```

### 2. 下游任务评估

**代码位置**: `utils/metrics.py`

#### 回归任务指标（亲和力预测）:
```python
class AffinityMetrics:
    def __call__(self, pred, true):
        return {
            # 均方误差
            'mse': mean_squared_error(pred, true),

            # 皮尔逊相关系数
            'pearson': pearsonr(pred, true)[0],

            # 斯皮尔曼相关系数
            'spearman': spearmanr(pred, true)[0],

            # Top-K召回率
            'recall@K': recall_at_k(pred, true, k)
        }
```

#### 分类任务指标:
```python
class MulticlassMetrics:
    def __call__(self, pred, true):
        return {
            # AUC-ROC
            'auc_score': roc_auc_score(true, pred, average='macro'),

            # 精确率
            'precision': precision_score(true, pred, average='macro'),

            # 召回率
            'recall': recall_score(true, pred, average='macro'),

            # 分类报告
            'classification_report': classification_report(true, pred)
        }
```

### 3. 评估数据集

根据`data/eval/`目录内容：

| 数据集 | 任务类型 | 描述 | 样本数 |
|--------|----------|------|--------|
| c-binding.csv | 回归 | 规范氨基酸蛋白质结合 | ~4M |
| nc-binding.csv | 回归 | 非规范氨基酸蛋白质结合 | ~550K |
| c-CPP.txt | 分类 | 规范氨基酸细胞穿透性 | ~50K |
| nc-CPP.csv | 分类 | 非规范氨基酸细胞穿透性 | ~16M |
| c-Sol.txt | 回归 | 规范氨基酸溶解度 | - |

### 4. 评估流程

#### 预训练评估:
```python
# 训练过程中评估
for epoch in range(epochs):
    train_epoch()

    # 每500步评估一次
    if step % 500 == 0:
        valid_metrics = evaluate("valid")
        test_metrics = evaluate("test")

    # 每轮结束评估
    valid_metrics = evaluate("valid")
    test_metrics = evaluate("test")

    # 保存最佳模型
    if valid_metrics['acc_atom'] > best_metric:
        save_model()
```

#### 下游任务评估:
```python
# 加载预训练模型
feature_extractor = PepLandFeatureExtractor(model_path)

# 构建预测器
predictor = PropertyPredictor(model_path)

# 微调和评估
for epoch in range(finetune_epochs):
    train_step()
    metrics = evaluate(test_loader)
```

### 5. 可视化

根据`data/eval/`中的PNG文件：

1. **t-SNE特征嵌入可视化**
   - 文件: `t-SNE_Feature_Embeddings_*.png`
   - 展示不同数据集的特征分布
   - 使用真实标签和聚类结果着色

2. **理化性质分析**
   - 文件: `*_physicochemical_properties.png`
   - 分析肽段的理化性质分布
   - 包括CPP和binding数据

3. **案例研究**
   - 文件: `case_study_predicted_scores_*.png`
   - 预测分数与真实值对比

---

## 代码完整性分析

### 1. 核心功能完整性

#### ✅ 完整实现的模块:

1. **数据处理模块** (`model/data.py`)
   - ✅ 分子分片算法（AdaFrag）
   - ✅ 异构图构建
   - ✅ 数据增强（多种masking策略）
   - ✅ 数据加载器（支持多进程和DDP）

2. **模型架构** (`model/model.py`, `model/hgt.py`)
   - ✅ PharmHGT完整实现
   - ✅ 多视图消息传递（MVMP）
   - ✅ HGT层实现
   - ✅ 图读出层（Node_GRU）

3. **训练框架** (`trainer.py`, `pretrain_masking.py`)
   - ✅ 两阶段预训练流程
   - ✅ 自监督学习任务
   - ✅ 分布式训练支持（DDP）
   - ✅ 模型保存和加载

4. **推理接口** (`inference.py`, `model/core.py`)
   - ✅ 特征提取器（PepLandFeatureExtractor）
   - ✅ 下游任务预测器（PropertyPredictor）
   - ✅ 批量推理支持

5. **评估工具** (`utils/metrics.py`)
   - ✅ 回归任务指标
   - ✅ 分类任务指标
   - ✅ Top-K召回率

6. **分片工具** (`tokenizer/pep2fragments.py`)
   - ✅ Amiibo算子
   - ✅ BRICS算法集成
   - ✅ 片段排序和标准化

### 2. 依赖完整性

**环境文件**: `environment.yaml`

**核心依赖**:
```yaml
# 深度学习框架
- pytorch=2.2.0 (CUDA 11.8)
- dgl=2.0.0 (CUDA 11.8)

# 化学信息学
- rdkit=2023.3.1

# 科学计算
- numpy=1.22.4
- scipy=1.7.1
- pandas=1.4.3
- scikit-learn=1.1.1

# 实验管理
- mlflow=1.28.0

# 配置管理
- hydra-core=1.2.0
- omegaconf=2.2.3

# 分布式训练
- nccl=2.12.12.1
```

**安装命令**:
```bash
conda env create -f environment.yaml
conda activate peppi  # 或 multiview
```

### 3. 配置文件完整性

**主配置**: `configs/pretrain_masking.yaml`
- ✅ 模型超参数
- ✅ 训练配置
- ✅ 数据配置
- ✅ 日志配置

**推理配置**: `configs/inference.yaml`
- ✅ 模型路径
- ✅ 推理参数
- ✅ 输入输出配置

### 4. 缺失或不完整的部分

#### ⚠️ 文档缺失:
- ❌ 详细的API文档
- ❌ 下游任务微调教程
- ❌ 数据格式说明
- ⚠️ 示例代码较少

#### ⚠️ 测试不足:
- ⚠️ `test/` 目录存在但内容未知
- ❌ 单元测试覆盖不足
- ❌ 集成测试缺失

#### ⚠️ 数据缺失:
- ⚠️ 预训练数据（`data/pretrained/`）需要用户提供
- ⚠️ 进一步训练数据（`data/further_training/`）需要用户提供
- ✅ 评估数据集（`data/eval/`）已提供

#### ⚠️ 预训练模型:
- ⚠️ `cpkt/` 目录有模型但路径需要配置
- ⚠️ README中提到的预训练权重下载链接可能需要更新

#### ⚠️ 可用性问题:
```python
# trainer.py中的注释代码
# 学习率调度器未激活
# scheduler = ...

# 氨基酸掩码预测未激活
# linear_pred_amino = ...
```

### 5. 代码质量

#### ✅ 优点:
- 代码结构清晰，模块化良好
- 使用Hydra进行配置管理
- 支持分布式训练
- 有MLflow日志支持

#### ⚠️ 改进空间:
- 注释主要用英文，但不够详细
- 部分功能被注释掉（如amino acid预测）
- 错误处理可以更完善
- 需要更多类型提示

### 6. 可运行性评估

#### 预训练:
```bash
# 前提条件:
1. ✅ 环境安装完成
2. ⚠️ 准备预训练数据（SMILES CSV文件）
3. ✅ 配置文件正确
4. ⚠️ 设置MLflow环境变量（如果启用日志）

# 运行命令:
python pretrain_masking.py

# 预期输出:
- 训练日志
- 模型检查点
- MLflow记录（如果启用）
```

#### 推理:
```bash
# 前提条件:
1. ✅ 环境安装完成
2. ⚠️ 预训练模型权重
3. ✅ 输入SMILES文件

# 运行命令:
python inference.py

# 或使用Python API:
from model.core import PepLandFeatureExtractor

model = PepLandFeatureExtractor(model_path, pooling='avg')
embeddings = model(['CCO', 'CCN'])
```

---

## 使用指南

### 1. 环境配置

```bash
# 1. 克隆仓库
git clone <repository_url>
cd pepland

# 2. 创建conda环境
conda env create -f environment.yaml

# 3. 激活环境
conda activate peppi

# 4. 验证安装
python -c "import torch; import dgl; import rdkit; print('OK')"
```

### 2. 数据准备

#### 预训练数据格式:
```csv
smiles
CC(C)C[C@H](NC(=O)...)C(=O)O
CC[C@H](C)[C@H](NC(=O)...)C(=O)O
...
```

#### 目录结构:
```
data/
├── pretrained/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
└── further_training/
    ├── train.csv
    ├── valid.csv
    └── test.csv
```

### 3. 训练模型

#### 阶段1: 规范氨基酸预训练
```bash
# 修改配置
vim configs/pretrain_masking.yaml

# 关键配置:
train:
  dataset: pretrained
  model: PharmHGT
  batch_size: 512
  epochs: 50

# 启动训练
python pretrain_masking.py

# DDP训练（多GPU）
torchrun --nproc_per_node=4 pretrain_masking.py
```

#### 阶段2: 非规范氨基酸继续训练
```bash
# 修改配置
vim configs/pretrain_masking.yaml

# 关键配置:
train:
  dataset: further_training
  model: fine-tune
inference:
  model_path: ./outputs/.../model_step_xxx/

# 启动训练
python pretrain_masking.py
```

### 4. 模型推理

#### 方式1: 使用脚本
```bash
# 准备输入文件（每行一个SMILES）
echo -e "CCO\nCCN\nCCC" > input_smiles.txt

# 配置推理参数
vim configs/inference.yaml

inference:
  model_path: ./cpkt/model/
  data: input_smiles.txt
  pool: avg  # avg, max, gru, or null
  device_ids: [0]

# 运行推理
python inference.py
```

#### 方式2: 使用Python API
```python
from model.core import PepLandFeatureExtractor

# 初始化模型
model = PepLandFeatureExtractor(
    model_path='./cpkt/model/',
    pooling='avg'  # 'avg', 'max', 'gru', or None
)
model.eval()

# 提取特征
smiles_list = ['CCO', 'CCN', 'CCC']
embeddings = model(smiles_list)
print(embeddings.shape)  # [3, 300]

# 提取原子和片段表示
atom_embeds, frag_embeds = model.extract_atom_fragment_embedding(smiles_list)
print(atom_embeds.shape)  # [3, num_atoms, 300]
print(frag_embeds.shape)  # [3, num_frags, 300]

# 提取特定原子的表示
atom_embed = model(smiles_list, atom_index=0)  # 第0个原子
print(atom_embed.shape)  # [3, 1, 300]
```

### 5. 下游任务微调

```python
from model.core import PropertyPredictor
import torch.nn as nn
import torch.optim as optim

# 1. 初始化预测器
predictor = PropertyPredictor(
    model_path='./cpkt/model/',
    pooling='avg',
    hidden_dims=[256, 128],
    mlp_dropout=0.1
)

# 2. 准备数据
train_smiles = [...]
train_labels = [...]

# 3. 训练
optimizer = optim.Adam(predictor.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(epochs):
    for batch_smiles, batch_labels in dataloader:
        pred = predictor(batch_smiles)
        loss = criterion(pred, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 4. 预测
with torch.no_grad():
    predictions = predictor(test_smiles)
```

### 6. 分子分片工具

```python
from tokenizer.pep2fragments import get_cut_bond_idx
from rdkit import Chem

# 加载分子
smiles = 'CC(C)C[C@H](NC(=O)...)C(=O)O'
mol = Chem.MolFromSmiles(smiles)

# AdaFrag分片（Amiibo + BRICS）
break_bonds, break_bonds_atoms = get_cut_bond_idx(
    mol,
    side_chain_cut=True
)

# 仅Amiibo分片
break_bonds, break_bonds_atoms = get_cut_bond_idx(
    mol,
    side_chain_cut=False
)

# 可视化切割键
from rdkit.Chem import Draw
highlight_bonds = break_bonds
img = Draw.MolToImage(
    mol,
    highlightBonds=highlight_bonds,
    size=(1000, 1000)
)
img.show()
```

### 7. 常见问题

#### Q1: 内存不足
```python
# 解决方案:
# 1. 减小batch_size
train.batch_size: 256  # 从512减小

# 2. 减少num_workers
train.num_workers: 4  # 从8减小

# 3. 使用梯度累积
accumulation_steps = 4
```

#### Q2: CUDA错误
```bash
# 检查CUDA版本
nvidia-smi

# 重新安装匹配的PyTorch
conda install pytorch=2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### Q3: RDKit错误
```python
# SMILES无法解析
try:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
except Exception as e:
    print(f"Error: {e}")
```

#### Q4: MLflow错误
```bash
# 设置环境变量
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=pepland_pretrain

# 或在配置中禁用
logger:
  log: False
```

### 8. 性能优化建议

#### 训练优化:
```python
# 1. 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 2. 使用更大的batch size
train.batch_size: 1024  # 如果内存允许

# 3. 使用分布式训练
torchrun --nproc_per_node=8 pretrain_masking.py

# 4. 优化数据加载
train.num_workers: 16  # 增加worker数量
pin_memory: True       # 使用固定内存
```

#### 推理优化:
```python
# 1. 批量推理
batch_size = 128
for i in range(0, len(smiles_list), batch_size):
    batch = smiles_list[i:i+batch_size]
    embeddings = model(batch)

# 2. 使用缓存
model._tokenize_cache  # 自动缓存SMILES→图转换

# 3. 半精度推理
model.half()  # 使用FP16
```

---

## 总结

### 项目优势:
1. ✅ **创新的模型架构**: 多视图异构图网络，同时建模原子和片段
2. ✅ **完整的训练框架**: 两阶段预训练，支持分布式训练
3. ✅ **专门的分片算法**: AdaFrag算法针对肽段结构优化
4. ✅ **良好的代码结构**: 模块化设计，易于扩展
5. ✅ **丰富的评估数据**: 提供多个下游任务数据集

### 需要改进:
1. ⚠️ **文档不足**: 缺少详细的API文档和教程
2. ⚠️ **测试覆盖**: 单元测试和集成测试需要补充
3. ⚠️ **数据可获得性**: 预训练数据需要用户自行准备
4. ⚠️ **模型权重**: 预训练权重的分发需要明确

### 建议:
1. 📝 补充详细的使用文档和示例
2. 🧪 增加单元测试和集成测试
3. 📦 提供预训练模型权重的下载链接
4. 📊 增加更多可视化和分析工具
5. 🔧 激活被注释的功能（如学习率调度器）

---

## 引用

如果使用PepLand，请引用：
```bibtex
@article{pepland2023,
  title={PepLand: a large-scale pre-trained peptide representation model for a comprehensive landscape of both canonical and non-canonical amino acids},
  author={...},
  journal={arXiv preprint arXiv:2311.04419},
  year={2023}
}
```

---

**文档版本**: 1.0
**最后更新**: 2025-10-14
**维护者**: PepLand Team
