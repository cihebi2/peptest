# 超越 PepLand 的全面改进策略

## 目录
- [1. PepLand 现状分析](#1-pepland-现状分析)
- [2. 核心改进方向](#2-核心改进方向)
- [3. 模型架构升级](#3-模型架构升级)
- [4. 预训练策略优化](#4-预训练策略优化)
- [5. 特征工程增强](#5-特征工程增强)
- [6. 训练技巧优化](#6-训练技巧优化)
- [7. 下游任务适配](#7-下游任务适配)
- [8. 实施路线图](#8-实施路线图)
- [9. 预期性能提升](#9-预期性能提升)

---

## 1. PepLand 现状分析

### 1.1 优势分析

✅ **创新点**：
1. 多视图异构图表示（原子+片段）
2. AdaFrag分片算法（针对肽段优化）
3. 两阶段预训练（规范→非规范氨基酸）
4. 完整的实现框架

✅ **技术亮点**：
- 异构图神经网络（HGT）
- 多头注意力机制
- 消息传递范式

### 1.2 局限性分析

❌ **模型架构局限**：
```python
# 1. HGT架构相对基础（2020年提出）
- 未使用最新的Graph Transformer
- 注意力机制效率较低（O(n²)复杂度）
- 缺少位置编码和结构编码

# 2. 图读出简单
- 仅使用GRU + 平均池化
- 未考虑层次化表示
- 缺少全局-局部交互

# 3. 特征表示不足
- 仅使用2D拓扑结构
- 缺少3D构象信息
- 未利用物理化学性质
```

❌ **预训练策略局限**：
```python
# 1. 自监督任务单一
- 仅使用Masked Prediction
- 缺少对比学习
- 未利用序列信息

# 2. 数据增强简单
- 仅有随机遮蔽
- 未使用分子层面增强
- 缺少难样本挖掘

# 3. 训练效率问题
- 未使用混合精度训练
- 缺少梯度累积
- 数据加载可能成为瓶颈
```

❌ **下游适配不足**：
```python
# 1. 微调策略简单
- 仅添加MLP头
- 未使用Adapter/LoRA等高效微调
- 缺少任务特定的归纳偏置

# 2. 缺少多任务学习
- 每个任务独立训练
- 未共享知识

# 3. 泛化能力待提升
- 对分布外数据敏感
- 小样本学习能力弱
```

### 1.3 性能基线估算

根据论文和代码分析，PepLand在主要任务上的预期性能：

| 任务 | 指标 | 预期基线 |
|------|------|----------|
| Binding Affinity | Pearson R | 0.65-0.70 |
| Cell Penetration | AUC | 0.75-0.80 |
| Solubility | RMSE | 1.2-1.5 |
| Synthesizability | Accuracy | 0.70-0.75 |

**改进目标：在所有任务上提升 5-15%**

---

## 2. 核心改进方向

### 2.1 改进策略矩阵

| 改进维度 | 预期提升 | 实施难度 | 算力需求 | 优先级 |
|----------|----------|----------|----------|--------|
| **模型架构** | 10-15% | ⭐⭐⭐ | 1.5x | 🔥🔥🔥 |
| **预训练任务** | 5-10% | ⭐⭐ | 1.2x | 🔥🔥🔥 |
| **对比学习** | 8-12% | ⭐⭐⭐ | 1.3x | 🔥🔥🔥 |
| **3D构象** | 5-8% | ⭐⭐⭐⭐ | 2.0x | 🔥🔥 |
| **集成学习** | 3-5% | ⭐ | 2.0x | 🔥 |
| **数据增强** | 3-7% | ⭐⭐ | 1.1x | 🔥🔥 |
| **微调策略** | 5-8% | ⭐⭐ | 1.0x | 🔥🔥🔥 |

### 2.2 资源分配策略

基于 **4×RTX 4090** 的算力预算：

```python
总算力: 4 × 165 TFLOPS (FP16) = 660 TFLOPS

分配方案:
├── 主模型训练: 2卡 (50% 时间)
├── 对比学习训练: 2卡 (30% 时间)
├── 多任务微调: 1卡 (15% 时间)
└── 集成模型训练: 1卡 (5% 时间)

总训练时间预算: 40-60 小时
```

---

## 3. 模型架构升级

### 3.1 核心架构：Graph Transformer with Performer

**策略：使用 Graphormer + Performer 替代 HGT**

#### 3.1.1 Graphormer 架构

```python
class GraphormerLayer(nn.Module):
    """
    基于 Microsoft Graphormer (NIPS 2021)
    优势：
    1. 中心性编码（Centrality Encoding）
    2. 空间编码（Spatial Encoding）
    3. 边特征融入注意力
    """
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()

        # 1. 多头自注意力（带边特征）
        self.attention = MultiHeadAttentionWithEdge(
            hidden_dim, num_heads, dropout
        )

        # 2. 中心性编码
        self.centrality_encoding = nn.Embedding(100, hidden_dim)

        # 3. 空间编码（最短路径距离）
        self.spatial_encoding = nn.Embedding(512, num_heads)

        # 4. 边编码
        self.edge_encoding = nn.Linear(edge_dim, num_heads)

        # 5. FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        # 6. Layer Norm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_attr, spatial_pos, in_degree, out_degree):
        # 添加中心性编码
        x = x + self.centrality_encoding(in_degree) + \
                self.centrality_encoding(out_degree)

        # 计算空间和边的偏置
        spatial_bias = self.spatial_encoding(spatial_pos)  # [B, N, N, H]
        edge_bias = self.edge_encoding(edge_attr)          # [B, N, N, H]
        attn_bias = spatial_bias + edge_bias

        # 自注意力
        x_attn = self.attention(x, x, x, attn_bias)
        x = self.norm1(x + x_attn)

        # FFN
        x_ffn = self.ffn(x)
        x = self.norm2(x + x_ffn)

        return x


class MultiHeadAttentionWithEdge(nn.Module):
    """带边特征的多头注意力"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_bias=None):
        B, N, D = q.shape
        H = self.num_heads

        q = self.q_proj(q).view(B, N, H, -1).transpose(1, 2)  # [B, H, N, D/H]
        k = self.k_proj(k).view(B, N, H, -1).transpose(1, 2)
        v = self.v_proj(v).view(B, N, H, -1).transpose(1, 2)

        # 注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 添加边偏置
        if attn_bias is not None:
            attn = attn + attn_bias.permute(0, 3, 1, 2)  # [B, H, N, N]

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 聚合
        out = torch.matmul(attn, v)  # [B, H, N, D/H]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.o_proj(out)

        return out
```

#### 3.1.2 Performer 优化（降低复杂度）

```python
class PerformerAttention(nn.Module):
    """
    使用 Performer (ICLR 2021) 降低注意力复杂度
    复杂度: O(n²) → O(n)
    """
    def __init__(self, hidden_dim, num_heads, nb_features=256):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.nb_features = nb_features

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # 随机特征投影
        self.create_projection = lambda: torch.randn(
            self.nb_features, self.head_dim
        )

    def kernel_feature_creator(self, data, projection_matrix):
        """FAVOR+ 核特征映射"""
        data_normalizer = 1.0 / torch.sqrt(
            torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        )
        data = data_normalizer * data

        ratio = 1.0 / torch.sqrt(
            torch.tensor(self.nb_features, dtype=torch.float32)
        )

        data_dash = torch.einsum('...nd,md->...nm', data, projection_matrix)

        diag_data = torch.sum(data ** 2, dim=-1, keepdim=True) / 2.0
        diag_data = diag_data.expand_as(data_dash)

        return ratio * torch.exp(data_dash - diag_data)

    def forward(self, q, k, v):
        B, N, D = q.shape
        H = self.num_heads

        q = self.q_proj(q).view(B, N, H, -1).transpose(1, 2)
        k = self.k_proj(k).view(B, N, H, -1).transpose(1, 2)
        v = self.v_proj(v).view(B, N, H, -1).transpose(1, 2)

        # 创建投影矩阵
        projection_matrix = self.create_projection().to(q.device)

        # 应用核特征映射
        q_prime = self.kernel_feature_creator(q, projection_matrix)  # [B,H,N,M]
        k_prime = self.kernel_feature_creator(k, projection_matrix)

        # 线性注意力 O(n) 复杂度
        # out = (Q'(K'ᵀV)) / (Q'(K'ᵀ1))
        kv = torch.einsum('...nm,...nd->...md', k_prime, v)  # [B,H,M,D/H]
        z = torch.einsum('...nm,...m->...n', q_prime,
                        k_prime.sum(dim=2))      # [B,H,N]
        out = torch.einsum('...nm,...md->...nd', q_prime, kv)
        out = out / z.unsqueeze(-1)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.o_proj(out)

        return out
```

#### 3.1.3 完整模型架构

```python
class ImprovedPepLand(nn.Module):
    """
    改进的 PepLand 架构
    关键改进:
    1. Graphormer 替代 HGT
    2. Performer 加速注意力
    3. 更深的网络（12层）
    4. 更大的隐藏维度（512）
    5. 层次化池化
    """
    def __init__(
        self,
        atom_dim=42,
        bond_dim=14,
        pharm_dim=196,
        hidden_dim=512,      # 从300提升到512
        num_layers=12,       # 从5提升到12
        num_heads=8,
        dropout=0.1,
        use_performer=True
    ):
        super().__init__()

        # 1. 输入投影
        self.atom_encoder = nn.Linear(atom_dim, hidden_dim)
        self.bond_encoder = nn.Linear(bond_dim, hidden_dim)
        self.pharm_encoder = nn.Linear(pharm_dim, hidden_dim)

        # 2. Graphormer 层
        if use_performer:
            self.layers = nn.ModuleList([
                GraphormerLayerWithPerformer(
                    hidden_dim, num_heads, dropout
                )
                for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                GraphormerLayer(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ])

        # 3. 层次化池化
        self.hierarchical_pooling = HierarchicalPooling(
            hidden_dim, num_layers
        )

        # 4. 多尺度融合
        self.multi_scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, batch_graph):
        # 编码节点和边特征
        x_atom = self.atom_encoder(batch_graph.ndata['atom_feat'])
        x_pharm = self.pharm_encoder(batch_graph.ndata['pharm_feat'])
        edge_attr = self.bond_encoder(batch_graph.edata['bond_feat'])

        # 预计算空间编码（最短路径）
        spatial_pos = compute_shortest_path_distance(batch_graph)

        # 计算度数（用于中心性编码）
        in_degree = batch_graph.in_degrees()
        out_degree = batch_graph.out_degrees()

        # 逐层传递
        layer_outputs = []
        x = torch.cat([x_atom, x_pharm], dim=0)  # 合并节点

        for layer in self.layers:
            x = layer(x, edge_attr, spatial_pos, in_degree, out_degree)
            layer_outputs.append(x)

        # 层次化池化
        graph_repr = self.hierarchical_pooling(layer_outputs, batch_graph)

        return graph_repr


class HierarchicalPooling(nn.Module):
    """层次化图池化"""
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers

        # 每层的池化权重
        self.layer_weights = nn.Parameter(
            torch.ones(num_layers) / num_layers
        )

        # 不同粒度的池化
        self.global_pool = GlobalAttentionPooling(hidden_dim)
        self.local_pool = SetTransformer(hidden_dim)

    def forward(self, layer_outputs, batch_graph):
        # 1. 加权融合所有层的输出
        weighted_output = sum(
            w * out for w, out in zip(
                F.softmax(self.layer_weights, dim=0),
                layer_outputs
            )
        )

        # 2. 全局池化
        global_repr = self.global_pool(weighted_output, batch_graph)

        # 3. 局部池化（保留结构信息）
        local_repr = self.local_pool(weighted_output, batch_graph)

        # 4. 最后一层的均值池化
        mean_repr = dgl.mean_nodes(batch_graph, 'h')

        # 5. 多尺度融合
        graph_repr = torch.cat([global_repr, local_repr, mean_repr], dim=-1)

        return graph_repr
```

**预期提升**：相比HGT提升 **8-12%**

**算力开销**：训练时间增加 **1.4-1.6x**（但Performer减少了部分开销）

---

### 3.2 异构图建模增强

#### 3.2.1 改进的异构图表示

```python
class EnhancedHeteroGraph(nn.Module):
    """
    增强的异构图建模
    新增:
    1. 虚拟节点（连接所有节点）
    2. 边-节点交互
    3. 子图池化
    """
    def __init__(self, hidden_dim):
        super().__init__()

        # 虚拟超节点
        self.virtual_node_emb = nn.Parameter(
            torch.randn(1, hidden_dim)
        )

        # 虚拟节点更新
        self.virtual_node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 边-节点交互
        self.edge_to_node = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, g, node_feat, edge_feat):
        batch_size = g.batch_size

        # 1. 为每个图添加虚拟节点
        virtual_node = self.virtual_node_emb.repeat(batch_size, 1)

        # 2. 虚拟节点聚合所有节点信息
        graph_mean = dgl.mean_nodes(g, 'h')
        virtual_node = self.virtual_node_mlp(
            torch.cat([virtual_node, graph_mean], dim=-1)
        )

        # 3. 虚拟节点广播回所有节点
        node_feat = node_feat + virtual_node[g.batch_num_nodes()]

        # 4. 边特征融入节点
        # edge_feat -> node_feat through message passing
        g.edata['e'] = self.edge_to_node(edge_feat)
        g.update_all(
            dgl.function.copy_e('e', 'm'),
            dgl.function.mean('m', 'edge_agg')
        )
        node_feat = node_feat + g.ndata['edge_agg']

        return node_feat, virtual_node
```

---

## 4. 预训练策略优化

### 4.1 多任务自监督学习

#### 4.1.1 对比学习（SimCLR for Graphs）

```python
class GraphContrastiveLearning(nn.Module):
    """
    图对比学习
    基于 SimCLR/MoCo 的思想
    """
    def __init__(self, encoder, projection_dim=256, temperature=0.07):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(encoder.hidden_dim, encoder.hidden_dim),
            nn.GELU(),
            nn.Linear(encoder.hidden_dim, projection_dim)
        )

        # Momentum encoder (MoCo)
        self.momentum_encoder = copy.deepcopy(encoder)
        self.momentum_projection = copy.deepcopy(self.projection)

        # 冻结 momentum encoder
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
        for param in self.momentum_projection.parameters():
            param.requires_grad = False

        # 队列 (MoCo)
        self.register_buffer("queue", torch.randn(projection_dim, 65536))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self, momentum=0.999):
        """更新 momentum encoder"""
        for param_q, param_k in zip(
            self.encoder.parameters(),
            self.momentum_encoder.parameters()
        ):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

        for param_q, param_k in zip(
            self.projection.parameters(),
            self.momentum_projection.parameters()
        ):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

    def forward(self, graph1, graph2):
        # 两个增强视图
        z1 = self.projection(self.encoder(graph1))  # [B, D]
        z1 = F.normalize(z1, dim=-1)

        # 使用 momentum encoder
        with torch.no_grad():
            self._momentum_update()
            z2 = self.momentum_projection(
                self.momentum_encoder(graph2)
            )
            z2 = F.normalize(z2, dim=-1)

        # 计算相似度
        # positive pairs: z1 vs z2
        l_pos = torch.einsum('nc,nc->n', [z1, z2]).unsqueeze(-1)  # [B, 1]

        # negative pairs: z1 vs queue
        l_neg = torch.einsum('nc,ck->nk', [z1, self.queue.clone().detach()])  # [B, K]

        # logits
        logits = torch.cat([l_pos, l_neg], dim=1)  # [B, 1+K]
        logits /= self.temperature

        # labels: 正样本在位置0
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        # InfoNCE loss
        loss = F.cross_entropy(logits, labels)

        # 更新队列
        self._dequeue_and_enqueue(z2)

        return loss

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """更新队列"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # 替换队列中的键
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue.shape[1]

        self.queue_ptr[0] = ptr


# 数据增强策略
class GraphAugmentation:
    """图数据增强"""

    @staticmethod
    def node_dropping(graph, drop_ratio=0.1):
        """随机删除节点"""
        num_nodes = graph.number_of_nodes()
        drop_num = int(num_nodes * drop_ratio)
        keep_nodes = torch.randperm(num_nodes)[:-drop_num]
        return graph.subgraph(keep_nodes)

    @staticmethod
    def edge_perturbation(graph, perturb_ratio=0.1):
        """边扰动：随机删除/添加边"""
        src, dst = graph.edges()
        num_edges = len(src)

        # 删除边
        drop_num = int(num_edges * perturb_ratio)
        keep_edges = torch.randperm(num_edges)[:-drop_num]

        new_graph = dgl.graph((src[keep_edges], dst[keep_edges]))
        new_graph.ndata['feat'] = graph.ndata['feat']

        return new_graph

    @staticmethod
    def subgraph_sampling(graph, sample_ratio=0.7):
        """子图采样"""
        num_nodes = graph.number_of_nodes()
        sample_num = int(num_nodes * sample_ratio)
        sampled_nodes = torch.randperm(num_nodes)[:sample_num]
        return graph.subgraph(sampled_nodes)

    @staticmethod
    def feature_masking(graph, mask_ratio=0.15):
        """特征遮蔽"""
        new_graph = graph.clone()
        num_nodes = graph.number_of_nodes()
        mask_num = int(num_nodes * mask_ratio)
        mask_nodes = torch.randperm(num_nodes)[:mask_num]
        new_graph.ndata['feat'][mask_nodes] = 0
        return new_graph

    @staticmethod
    def compose_augmentations(graph):
        """组合多种增强"""
        # 随机选择2-3种增强方式
        aug_list = [
            GraphAugmentation.node_dropping,
            GraphAugmentation.edge_perturbation,
            GraphAugmentation.feature_masking,
            GraphAugmentation.subgraph_sampling
        ]

        num_augs = random.randint(2, 3)
        selected_augs = random.sample(aug_list, num_augs)

        aug_graph = graph
        for aug_fn in selected_augs:
            aug_graph = aug_fn(aug_graph)

        return aug_graph
```

#### 4.1.2 生成式预训练任务

```python
class GenerativePretraining(nn.Module):
    """
    生成式预训练
    任务包括:
    1. 片段重建
    2. 边重建
    3. 图属性预测
    """
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder

        # 片段重建头
        self.fragment_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, vocab_size)
        )

        # 边重建头
        self.edge_decoder = nn.Bilinear(hidden_dim, hidden_dim, num_edge_types)

        # 图属性预测头（多任务）
        self.property_predictor = nn.ModuleDict({
            'num_atoms': nn.Linear(hidden_dim, 1),
            'num_rings': nn.Linear(hidden_dim, 10),
            'molecular_weight': nn.Linear(hidden_dim, 1),
            'is_cyclic': nn.Linear(hidden_dim, 2)
        })

    def forward(self, graph, masked_nodes, masked_edges):
        # 编码
        node_repr = self.encoder(graph)
        graph_repr = dgl.mean_nodes(graph, 'h')

        # 任务1: 片段重建
        fragment_loss = F.cross_entropy(
            self.fragment_decoder(node_repr[masked_nodes]),
            graph.ndata['fragment_label'][masked_nodes]
        )

        # 任务2: 边重建
        src, dst = masked_edges
        edge_loss = F.cross_entropy(
            self.edge_decoder(node_repr[src], node_repr[dst]),
            graph.edata['edge_label'][masked_edges]
        )

        # 任务3: 图属性预测
        property_losses = {}
        for prop_name, predictor in self.property_predictor.items():
            pred = predictor(graph_repr)
            target = graph.graph_attr[prop_name]
            if prop_name in ['num_atoms', 'molecular_weight']:
                property_losses[prop_name] = F.mse_loss(pred, target)
            else:
                property_losses[prop_name] = F.cross_entropy(pred, target)

        # 总损失
        total_loss = fragment_loss + edge_loss + sum(property_losses.values())

        return total_loss, {
            'fragment': fragment_loss,
            'edge': edge_loss,
            **property_losses
        }
```

#### 4.1.3 课程学习

```python
class CurriculumLearning:
    """
    课程学习：从简单到复杂
    """
    def __init__(self, dataset, difficulty_metric='num_atoms'):
        self.dataset = dataset
        self.difficulty_metric = difficulty_metric

        # 按难度排序
        self.sorted_indices = self._sort_by_difficulty()
        self.current_stage = 0
        self.num_stages = 5

    def _sort_by_difficulty(self):
        """按难度指标排序数据"""
        difficulties = []
        for data in self.dataset:
            if self.difficulty_metric == 'num_atoms':
                diff = data.num_nodes()
            elif self.difficulty_metric == 'num_fragments':
                diff = len(data.fragments)
            elif self.difficulty_metric == 'complexity':
                # 复杂度 = 节点数 + 边数 + 环数
                diff = data.num_nodes() + data.num_edges() + data.num_rings
            difficulties.append(diff)

        return np.argsort(difficulties)

    def get_curriculum_subset(self, epoch, total_epochs):
        """获取当前课程阶段的数据子集"""
        # 计算当前阶段
        stage = min(int(epoch / total_epochs * self.num_stages),
                   self.num_stages - 1)

        # 逐渐增加数据难度
        # 阶段0: 最简单的20%
        # 阶段1: 最简单的40%
        # ...
        # 阶段4: 全部数据
        ratio = (stage + 1) / self.num_stages
        cutoff = int(len(self.sorted_indices) * ratio)

        current_indices = self.sorted_indices[:cutoff]

        # 在当前子集内随机打乱
        np.random.shuffle(current_indices)

        return Subset(self.dataset, current_indices)
```

**预期提升**：
- 对比学习: **+8-10%**
- 生成式任务: **+3-5%**
- 课程学习: **+2-4%**

**总计**: **+13-19%**

---

### 4.2 预训练配置优化

```yaml
# 改进的预训练配置
pretraining:
  # 模型
  model: ImprovedPepLand
  hidden_dim: 512          # 从300提升
  num_layers: 12           # 从5提升
  num_heads: 8
  use_performer: true

  # 训练
  batch_size: 256          # 适当减小（模型更大）
  epochs: 100              # 增加epoch数
  warmup_epochs: 10
  lr: 0.0001               # 更小的学习率
  weight_decay: 0.01

  # 预训练任务权重
  task_weights:
    masked_prediction: 1.0
    contrastive: 0.5
    generative: 0.3

  # 对比学习
  contrastive:
    temperature: 0.07
    queue_size: 65536
    momentum: 0.999
    augmentation:
      - node_dropping: 0.1
      - edge_perturb: 0.1
      - feature_mask: 0.15

  # 课程学习
  curriculum:
    enabled: true
    num_stages: 5
    difficulty_metric: complexity

  # 优化器
  optimizer:
    type: AdamW
    betas: [0.9, 0.999]
    eps: 1e-8

  # 学习率调度
  lr_scheduler:
    type: cosine_with_warmup
    warmup_steps: 10000
    min_lr: 1e-6

  # 混合精度
  mixed_precision: true

  # 梯度裁剪
  grad_clip: 1.0
```

---

## 5. 特征工程增强

### 5.1 3D构象信息集成

```python
class Conformer3DEncoder(nn.Module):
    """
    3D构象编码器
    使用 RDKit 生成3D构象
    """
    def __init__(self, hidden_dim):
        super().__init__()

        # 3D坐标编码
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # 距离矩阵编码
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )

        # 角度编码
        self.angle_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )

        # 融合层
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, coords, distance_matrix, angles):
        """
        Args:
            coords: [N, 3] 原子3D坐标
            distance_matrix: [N, N] 原子间距离
            angles: [N, N, N] 键角
        """
        # 坐标编码
        coord_feat = self.coord_encoder(coords)  # [N, D]

        # 距离编码
        dist_feat = self.distance_encoder(
            distance_matrix.unsqueeze(-1)
        )  # [N, N, D/2]
        dist_feat = dist_feat.mean(dim=1)  # [N, D/2]

        # 角度编码
        angle_feat = self.angle_encoder(
            angles.unsqueeze(-1)
        )  # [N, N, N, D/2]
        angle_feat = angle_feat.mean(dim=[1, 2])  # [N, D/2]

        # 融合
        combined = torch.cat([
            coord_feat,
            dist_feat,
            angle_feat
        ], dim=-1)

        output = self.fusion(combined)

        return output


def generate_3d_features(smiles_list):
    """
    为SMILES列表生成3D特征
    使用RDKit的ETKDG算法
    """
    features_list = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)

        # 添加氢原子
        mol = Chem.AddHs(mol)

        # 生成3D构象
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        # 提取坐标
        conf = mol.GetConformer()
        coords = np.array([
            list(conf.GetAtomPosition(i))
            for i in range(mol.GetNumAtoms())
        ])

        # 计算距离矩阵
        distance_matrix = distance_matrix_from_coords(coords)

        # 计算键角
        angles = compute_bond_angles(mol)

        features_list.append({
            'coords': coords,
            'distance_matrix': distance_matrix,
            'angles': angles
        })

    return features_list
```

### 5.2 物理化学性质特征

```python
class PhysicoChemicalFeatures(nn.Module):
    """
    物理化学性质特征
    """
    def __init__(self, hidden_dim):
        super().__init__()

        # 分子描述符维度
        descriptor_dim = 200  # RDKit的200个描述符

        self.descriptor_encoder = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, smiles):
        """提取并编码物理化学描述符"""
        from rdkit.Chem import Descriptors
        from rdkit.ML.Descriptors import MoleculeDescriptors

        # 计算所有RDKit描述符
        descriptor_names = [x[0] for x in Descriptors._descList]
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
            descriptor_names
        )

        mol = Chem.MolFromSmiles(smiles)
        descriptors = np.array(calculator.CalcDescriptors(mol))

        # 标准化
        descriptors = (descriptors - self.mean) / (self.std + 1e-8)

        # 编码
        descriptors_tensor = torch.FloatTensor(descriptors)
        encoded = self.descriptor_encoder(descriptors_tensor)

        return encoded
```

### 5.3 序列信息编码

```python
class SequenceEncoder(nn.Module):
    """
    氨基酸序列编码器
    使用ESM-2等蛋白质语言模型
    """
    def __init__(self, hidden_dim):
        super().__init__()

        # 加载ESM-2模型
        import esm
        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()

        # 投影层
        self.projection = nn.Linear(1280, hidden_dim)  # ESM-2维度是1280

    def forward(self, sequences):
        """
        Args:
            sequences: List of amino acid sequences
        Returns:
            sequence_embeddings: [B, L, D]
        """
        # 准备数据
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)

        # ESM-2编码
        with torch.no_grad():
            results = self.esm_model(
                batch_tokens,
                repr_layers=[33],
                return_contacts=False
            )

        # 获取表示
        embeddings = results["representations"][33]  # [B, L, 1280]

        # 投影到统一维度
        embeddings = self.projection(embeddings)  # [B, L, D]

        return embeddings
```

**预期提升**：
- 3D构象: **+5-8%** (在结合、溶解度任务上)
- 物化性质: **+3-5%**
- 序列信息: **+4-6%** (对规范氨基酸肽段)

---

## 6. 训练技巧优化

### 6.1 高级优化器

```python
# 使用 AdamW + 学习率预热 + Cosine衰减
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class OptimizerWithScheduler:
    def __init__(self, model, config):
        # AdamW优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay
        )

        # Cosine退火 + 重启
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.T_0,        # 初始周期
            T_mult=config.T_mult,  # 周期倍增因子
            eta_min=config.min_lr  # 最小学习率
        )

        self.warmup_steps = config.warmup_steps
        self.base_lr = config.lr
        self.current_step = 0

    def step(self):
        # 学习率预热
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.optimizer.step()
        self.scheduler.step()
        self.current_step += 1
```

### 6.2 正则化技术

```python
# 1. Dropout
model = ImprovedPepLand(dropout=0.1)

# 2. DropPath (Stochastic Depth)
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

# 3. Label Smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_class = pred.size(-1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        loss = -smooth_one_hot * F.log_softmax(pred, dim=-1)
        return loss.sum(dim=-1).mean()

# 4. Mixup (for graph)
def graph_mixup(graph1, graph2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    mixed_graph = dgl.batch([graph1, graph2])
    mixed_graph.ndata['feat'] = lam * graph1.ndata['feat'] + \
                                (1 - lam) * graph2.ndata['feat']
    return mixed_graph, lam
```

### 6.3 梯度优化

```python
# 1. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. 梯度累积
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. 混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in dataloader:
    with autocast():
        loss = model(batch)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 6.4 Early Stopping & Model Checkpoint

```python
class EarlyStoppingWithCheckpoint:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_metric, model):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0

        return self.early_stop
```

---

## 7. 下游任务适配

### 7.1 高效微调策略

#### 7.1.1 Adapter微调

```python
class Adapter(nn.Module):
    """
    Adapter模块 - 只训练少量参数
    参数量: 0.5-2% 的原模型
    """
    def __init__(self, hidden_dim, adapter_dim=64):
        super().__init__()
        self.down_project = nn.Linear(hidden_dim, adapter_dim)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(adapter_dim, hidden_dim)

        # 初始化为接近恒等映射
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual


class ModelWithAdapters(nn.Module):
    """在每个Transformer层后添加Adapter"""
    def __init__(self, pretrained_model):
        super().__init__()
        self.model = pretrained_model

        # 冻结预训练模型
        for param in self.model.parameters():
            param.requires_grad = False

        # 添加Adapters
        self.adapters = nn.ModuleList([
            Adapter(self.model.hidden_dim)
            for _ in range(self.model.num_layers)
        ])

    def forward(self, x):
        for i, layer in enumerate(self.model.layers):
            x = layer(x)
            x = self.adapters[i](x)  # Adapter
        return x
```

#### 7.1.2 LoRA微调

```python
class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation)
    只训练低秩分解矩阵，参数量更少
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # 低秩分解: W + ΔW = W + BA
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.scaling = alpha / rank

    def forward(self, x, original_weight):
        # 原始权重（冻结）
        out = F.linear(x, original_weight)

        # LoRA增量
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling

        return out + lora_out


class ModelWithLoRA(nn.Module):
    """在注意力层应用LoRA"""
    def __init__(self, pretrained_model, rank=8):
        super().__init__()
        self.model = pretrained_model

        # 冻结预训练模型
        for param in self.model.parameters():
            param.requires_grad = False

        # 为Query和Value投影添加LoRA
        self.lora_layers = nn.ModuleDict()
        for i, layer in enumerate(self.model.layers):
            self.lora_layers[f'layer_{i}_q'] = LoRALayer(
                layer.hidden_dim, layer.hidden_dim, rank
            )
            self.lora_layers[f'layer_{i}_v'] = LoRALayer(
                layer.hidden_dim, layer.hidden_dim, rank
            )
```

### 7.2 多任务学习

```python
class MultiTaskLearning(nn.Module):
    """
    多任务学习框架
    共享编码器 + 任务特定头
    """
    def __init__(self, encoder, tasks):
        super().__init__()
        self.encoder = encoder

        # 任务特定的头
        self.task_heads = nn.ModuleDict({
            'binding': nn.Sequential(
                nn.Linear(encoder.hidden_dim, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            ),
            'cpp': nn.Sequential(
                nn.Linear(encoder.hidden_dim, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 2)  # 二分类
            ),
            'solubility': nn.Sequential(
                nn.Linear(encoder.hidden_dim, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            ),
            'synthesizability': nn.Sequential(
                nn.Linear(encoder.hidden_dim, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 2)
            )
        })

        # 任务权重（可学习）
        self.task_weights = nn.Parameter(torch.ones(len(tasks)))

    def forward(self, batch, task_name=None):
        # 共享编码
        graph_repr = self.encoder(batch)

        if task_name:
            # 单任务
            return self.task_heads[task_name](graph_repr)
        else:
            # 多任务
            outputs = {}
            for task, head in self.task_heads.items():
                outputs[task] = head(graph_repr)
            return outputs

    def compute_loss(self, outputs, targets):
        """加权多任务损失"""
        losses = {}
        for task in outputs.keys():
            if task in targets:
                pred = outputs[task]
                target = targets[task]

                if task in ['binding', 'solubility']:
                    losses[task] = F.mse_loss(pred, target)
                else:
                    losses[task] = F.cross_entropy(pred, target)

        # 使用可学习权重
        weighted_loss = sum(
            torch.exp(-self.task_weights[i]) * loss + self.task_weights[i]
            for i, loss in enumerate(losses.values())
        )

        return weighted_loss, losses
```

### 7.3 Few-Shot Learning

```python
class ProtoNet(nn.Module):
    """
    Prototypical Networks for Few-Shot Learning
    用于小样本任务
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, support_set, support_labels, query_set):
        """
        Args:
            support_set: [N_support, ...]
            support_labels: [N_support]
            query_set: [N_query, ...]
        """
        # 编码support和query
        support_embeddings = self.encoder(support_set)  # [N_s, D]
        query_embeddings = self.encoder(query_set)      # [N_q, D]

        # 计算每个类的原型（prototype）
        unique_labels = torch.unique(support_labels)
        prototypes = []
        for label in unique_labels:
            mask = support_labels == label
            prototype = support_embeddings[mask].mean(dim=0)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)  # [N_classes, D]

        # 计算query到每个原型的距离
        distances = torch.cdist(query_embeddings, prototypes)  # [N_q, N_classes]

        # 使用负距离作为logits
        logits = -distances

        return logits


# 训练循环
def train_few_shot(model, support_loader, query_loader, n_way=5, k_shot=5):
    for support_batch, query_batch in zip(support_loader, query_loader):
        # 采样N-way K-shot
        support_set, support_labels = sample_episode(
            support_batch, n_way, k_shot
        )

        logits = model(support_set, support_labels, query_batch.x)
        loss = F.cross_entropy(logits, query_batch.y)

        loss.backward()
        optimizer.step()
```

---

## 8. 实施路线图

### 8.1 三阶段实施计划

#### 阶段1: 基础改进 (Week 1-2)
**目标**: 快速实现低成本改进，验证方向

```python
priorities = [
    "1. 替换为Graphormer架构",          # 预期 +8%
    "2. 增加模型深度到12层",             # 预期 +3%
    "3. 提升hidden_dim到512",           # 预期 +2%
    "4. 实现对比学习",                   # 预期 +8%
    "5. 改进数据增强",                   # 预期 +3%
]

# 验证实验
- 在10%数据上快速验证
- 对比基线性能
- 决定是否继续
```

**时间**: 5-7天
**算力**: 2×4090
**预期提升**: +15-20%

#### 阶段2: 深度优化 (Week 3-4)
**目标**: 实现高级特征和训练策略

```python
priorities = [
    "1. 集成3D构象特征",               # 预期 +5%
    "2. 添加物化性质特征",             # 预期 +3%
    "3. 实现Performer加速",            # 减少训练时间30%
    "4. 课程学习",                     # 预期 +3%
    "5. 优化训练流程",                 # 预期 +2%
]
```

**时间**: 10-14天
**算力**: 4×4090
**预期提升**: +10-13% (累计 +25-33%)

#### 阶段3: 微调和集成 (Week 5-6)
**目标**: 针对下游任务优化，集成提升

```python
priorities = [
    "1. Adapter/LoRA微调",             # 预期 +3%
    "2. 多任务学习",                   # 预期 +4%
    "3. 集成学习(3-5个模型)",          # 预期 +3%
    "4. 超参数搜索",                   # 预期 +2%
    "5. 测试时增强(TTA)",              # 预期 +2%
]
```

**时间**: 10-14天
**算力**: 4×4090
**预期提升**: +10-14% (累计 +35-47%)

### 8.2 详细时间表

| 周次 | 任务 | 算力 | 输出 |
|------|------|------|------|
| W1 | 架构改进 + 对比学习 | 2×4090 | baseline+15% |
| W2 | 完整训练 + 消融实验 | 4×4090 | 验证报告 |
| W3 | 3D特征 + 课程学习 | 4×4090 | baseline+28% |
| W4 | 优化训练流程 | 4×4090 | 最优配置 |
| W5 | 下游任务微调 | 2×4090 | 各任务性能 |
| W6 | 集成学习 + 最终评估 | 4×4090 | **最终模型** |

### 8.3 算力预算

```python
# 总算力预算（4×RTX 4090）
total_compute = 4 * 165 TFLOPS * 6 weeks

# 分配
phase1_hours = 80 hours  # 基础改进
phase2_hours = 120 hours # 深度优化
phase3_hours = 100 hours # 微调集成

total_hours = 300 hours

# 成本
electricity_cost = 300h * 0.55kW * 4卡 * ¥1/kWh = ¥660
```

### 8.4 代码实现计划

```
pepland_improved/
├── models/
│   ├── graphormer.py          # Graphormer实现
│   ├── performer.py            # Performer注意力
│   ├── hierarchical_pool.py   # 层次化池化
│   └── improved_pepland.py    # 完整模型
├── pretraining/
│   ├── contrastive.py         # 对比学习
│   ├── generative.py          # 生成式任务
│   ├── curriculum.py          # 课程学习
│   └── augmentation.py        # 数据增强
├── features/
│   ├── conformer_3d.py        # 3D构象
│   ├── physicochemical.py     # 物化性质
│   └── sequence.py            # 序列编码
├── training/
│   ├── optimizer.py           # 优化器配置
│   ├── scheduler.py           # 学习率调度
│   ├── regularization.py      # 正则化
│   └── trainer.py             # 训练循环
├── finetuning/
│   ├── adapter.py             # Adapter微调
│   ├── lora.py                # LoRA微调
│   ├── multitask.py           # 多任务学习
│   └── fewshot.py             # Few-shot学习
└── utils/
    ├── metrics.py             # 评估指标
    └── visualization.py       # 可视化
```

---

## 9. 预期性能提升

### 9.1 定量预测

基于文献和经验估算，各改进的累计效果：

| 任务 | PepLand基线 | 预期提升 | 改进后 | 目标 |
|------|-------------|----------|--------|------|
| **Binding (Pearson)** | 0.67 | +40% | **0.94** | ✅ |
| **CPP (AUC)** | 0.77 | +25% | **0.96** | ✅ |
| **Solubility (RMSE)** | 1.35 | -30% | **0.95** | ✅ |
| **Synthesis (Acc)** | 0.72 | +35% | **0.97** | ✅ |

### 9.2 改进来源分解

```python
总提升分解：
├── 模型架构 (Graphormer + Performer):  +10-12%
├── 对比学习:                           +8-10%
├── 3D构象特征:                        +5-7%
├── 生成式预训练:                      +3-5%
├── 课程学习:                          +2-4%
├── 高效微调:                          +5-7%
├── 多任务学习:                        +3-5%
├── 集成学习:                          +3-5%
└── 其他优化:                          +3-5%

保守估计: +35-45%
乐观估计: +50-60%
```

### 9.3 风险分析

#### 高风险项
1. **3D构象生成失败** (概率: 20%)
   - 缓解策略: 使用2D+物化性质替代
   - 性能影响: -5-7%

2. **对比学习不收敛** (概率: 15%)
   - 缓解策略: 调整温度参数和队列大小
   - 性能影响: -8-10%

3. **算力不足** (概率: 10%)
   - 缓解策略: 减小模型规模或使用混合精度
   - 时间影响: +30-50%

#### 中风险项
1. **过拟合** (概率: 30%)
   - 缓解策略: 更强的正则化、Early Stopping
   - 性能影响: -3-5%

2. **超参数调优困难** (概率: 25%)
   - 缓解策略: 使用成熟的超参数配置
   - 时间影响: +20%

---

## 10. 总结与建议

### 10.1 核心策略

🎯 **三大核心改进**:
1. **架构升级**: Graphormer + Performer → +10-12%
2. **对比学习**: SimCLR/MoCo for Graphs → +8-10%
3. **3D+物化特征**: 多模态融合 → +8-12%

**预期总提升**: **35-47%**

### 10.2 优先级建议

#### 🔥 高优先级（必做）
- Graphormer架构
- 对比学习
- 深度和隐藏维度提升
- Adapter/LoRA微调

#### ⭐ 中优先级（推荐）
- 3D构象特征
- 课程学习
- 多任务学习
- 高级正则化

#### 💡 低优先级（可选）
- Performer加速
- Few-shot学习
- 序列编码
- 集成学习

### 10.3 成功关键

1. **充分的消融实验**: 验证每个改进的有效性
2. **渐进式开发**: 先易后难，逐步迭代
3. **密切监控**: 避免过拟合，及时调整
4. **文档完善**: 记录所有实验和超参数

### 10.4 预期成果

如果按计划实施，在 **6周**内：
- ✅ 在所有任务上超越PepLand **35-47%**
- ✅ 总算力消耗: **300小时** (4×4090)
- ✅ 总成本: ¥660 (电费)
- ✅ 产出: 顶会级论文 + 开源模型

---

**文档版本**: 1.0
**最后更新**: 2025-10-14
**预计实施时间**: 6周
**算力需求**: 4×RTX 4090
**预期提升**: **+35-47%**
