# PepLand 计算资源需求评估（基于RTX 4090）

## 硬件规格参考
**RTX 4090 规格**:
- VRAM: 24GB GDDR6X
- CUDA Cores: 16,384
- Tensor Cores: 512 (第4代)
- FP32性能: 82.6 TFLOPS
- FP16性能: 165.2 TFLOPS (with Tensor Cores)
- 功耗: 450W TDP

---

## 1. 模型内存占用分析

### 1.1 模型参数量估算

```python
# PharmHGT模型结构
PharmHGT(
    hid_dim=300,
    num_layer=5,
    atom_dim=42,
    bond_dim=14,
    pharm_dim=196,
    reac_dim=14
)
```

**详细参数计算**:

#### 特征初始化层
```
w_atom: 42 × 300 = 12,600 参数
w_bond: 14 × 300 = 4,200 参数
w_pharm: 196 × 300 = 58,800 参数
w_reac: 14 × 300 = 4,200 参数
w_junc: (42+196) × 300 = 71,400 参数
小计: ~151K 参数
```

#### MVMP层 (×5层)
每层包含:
```
# 多头注意力 (4 heads)
Attention Q,K,V: 3 × (300 × 300) × 2(atom+pharm) = 540,000
Attention output: 300 × 300 × 2 = 180,000

# 消息传递MLP
MP layers: 300 × 300 × 3(深度-1) × 2(atom+pharm边) = 540,000

# 节点更新层
Node update: 3×300 × 300 × 3(节点类型) = 810,000

每层小计: ~2.07M 参数
5层总计: ~10.35M 参数
```

#### 图读出层
```
MultiHeadedAttention: 4 × (300 × 300) = 360,000
GRU (bidirectional): 2 × (300 × 300 × 6) = 1,080,000
小计: ~1.44M 参数
```

#### 预测头
```
linear_pred_atoms: 300 × 119 = 35,700
linear_pred_pharms: 300 × 264 = 79,200
linear_pred_bonds: 300 × 4 = 1,200
小计: ~116K 参数
```

**模型总参数量**: **~12M 参数**

### 1.2 显存占用估算

#### 模型权重
```
FP32: 12M × 4 bytes = 48 MB
FP16: 12M × 2 bytes = 24 MB (混合精度训练)
```

#### 优化器状态 (Adam)
```
# Adam需要存储: 参数 + 一阶矩 + 二阶矩
FP32: 12M × 4 × 3 = 144 MB
FP16训练: 12M × (2 + 4 + 4) = 120 MB (参数FP16 + 优化器状态FP32)
```

#### 梯度
```
FP32: 12M × 4 bytes = 48 MB
FP16: 12M × 2 bytes = 24 MB
```

#### 训练中间激活值 (关键因素)
基于配置 `batch_size=512, num_layer=5`:

```python
# 单个分子平均统计（肽段）
avg_atoms = 100 原子
avg_fragments = 10 片段
avg_bonds = 110 键

# 批次大小
batch_size = 512

# 每层中间激活
atom_activations = 512 × 100 × 300 × 5层 = 76.8M 元素
fragment_activations = 512 × 10 × 300 × 5层 = 7.68M 元素
edge_activations = 512 × 110 × 300 × 5层 = 84.48M 元素
attention_scores = 512 × 100 × 100 × 4heads × 5层 = 102.4M 元素

总计约: 271M 元素
FP16: 271M × 2 bytes = 542 MB
FP32: 271M × 4 bytes = 1,084 MB
```

#### DGL图结构开销
```
# 图索引、边列表等
估计: 200-500 MB (取决于分子大小)
```

### 1.3 总显存需求

| 配置 | 模型 | 优化器 | 梯度 | 激活值 | 图结构 | **总计** |
|------|------|--------|------|--------|--------|----------|
| FP32 | 48MB | 144MB | 48MB | 1,084MB | 300MB | **1.6GB** |
| FP16混合 | 24MB | 120MB | 24MB | 542MB | 300MB | **1.0GB** |

**RTX 4090 (24GB) 可用配置**:

| Batch Size | 精度 | 估计显存 | 4090数量 | 备注 |
|------------|------|----------|----------|------|
| 512 | FP16 | ~1.0GB | **1张** | ✅ 推荐配置 |
| 1024 | FP16 | ~1.8GB | **1张** | ✅ 可行 |
| 2048 | FP16 | ~3.4GB | **1张** | ✅ 可行 |
| 4096 | FP16 | ~6.5GB | **1张** | ✅ 可行 |
| 8192 | FP16 | ~12.8GB | **1张** | ⚠️ 接近极限 |
| 512 | FP32 | ~1.6GB | **1张** | ✅ 可行但不推荐 |

---

## 2. 训练时间估算

### 2.1 前向传播时间

基于类似GNN模型的benchmark，RTX 4090的估算:

```python
# 单个batch前向传播
avg_forward_time = 50-80 ms (batch_size=512, FP16)
avg_forward_time = 80-120 ms (batch_size=512, FP32)

# 5层MVMP的计算量
FLOPs_per_sample ≈ 2 × 12M参数 × 100原子 × 5层
                  ≈ 12 GFLOPs/sample

batch_FLOPs = 12 GFLOPs × 512 = 6.144 TFLOPs
理论时间 = 6.144 TFLOPs / 165.2 TFLOPS (FP16) = 37 ms
实际时间 ≈ 60-80 ms (考虑内存带宽、图操作等)
```

### 2.2 反向传播时间

```python
# 反向传播通常是前向的2-3倍
avg_backward_time = 120-200 ms (batch_size=512, FP16)
```

### 2.3 单步训练时间

```python
# 前向 + 反向 + 优化器
single_step_time = 60ms + 150ms + 10ms = 220 ms (FP16)
single_step_time = 100ms + 250ms + 15ms = 365 ms (FP32)

throughput_FP16 = 512 / 0.22s = 2,327 样本/秒
throughput_FP32 = 512 / 0.365s = 1,403 样本/秒
```

### 2.4 完整训练时间估算

**假设数据集规模** (根据论文描述):
```
阶段1 (规范氨基酸): ~100万 肽段样本
阶段2 (非规范氨基酸): ~50万 肽段样本
```

#### 阶段1: 规范氨基酸预训练

```python
训练样本: 1,000,000
验证样本: 50,000
测试样本: 50,000

epochs: 50
batch_size: 512

# 每个epoch
steps_per_epoch = 1,000,000 / 512 = 1,954 steps
time_per_epoch = 1,954 × 0.22s = 430秒 ≈ 7.2分钟

# 验证评估 (每500步一次)
validation_per_epoch = (1,954 / 500) × (50,000/512) × 0.22s
                     ≈ 4 × 98 × 0.22s = 86秒 ≈ 1.4分钟

# 总时间
total_time_per_epoch = 7.2 + 1.4 = 8.6分钟
total_time_50_epochs = 8.6 × 50 = 430分钟 ≈ 7.2小时

**阶段1总计: 7-8小时 (单张4090, FP16)**
```

#### 阶段2: 非规范氨基酸继续训练

```python
训练样本: 500,000
epochs: 20-30 (通常比阶段1少)

time_per_epoch = (500,000 / 512) × 0.22s = 215秒 ≈ 3.6分钟
total_time_25_epochs = 3.6 × 25 = 90分钟 ≈ 1.5小时

**阶段2总计: 1.5-2小时 (单张4090, FP16)**
```

### 2.5 总训练时间汇总

| 训练阶段 | 数据规模 | Epochs | Batch Size | **单卡时间 (4090)** |
|----------|----------|--------|------------|---------------------|
| 阶段1 | 100万 | 50 | 512 | **7-8小时** |
| 阶段2 | 50万 | 25 | 512 | **1.5-2小时** |
| **总计** | - | - | - | **8.5-10小时** |

---

## 3. 多卡并行加速

### 3.1 数据并行 (DDP)

```python
# 支持的配置
cfg.mode.ddp: True

# 加速比
2卡 4090: ~1.8x 加速 (理论2x，实际有通信开销)
4卡 4090: ~3.4x 加速
8卡 4090: ~6.0x 加速 (通信开销增加)

# 训练时间
2卡: 10小时 / 1.8 = 5.6小时
4卡: 10小时 / 3.4 = 2.9小时
8卡: 10小时 / 6.0 = 1.7小时
```

### 3.2 多卡配置建议

| GPU数量 | 全局Batch Size | 每卡Batch | 预计时间 | 成本效益 |
|---------|----------------|-----------|----------|----------|
| 1× 4090 | 512 | 512 | 10小时 | ⭐⭐⭐⭐⭐ 最划算 |
| 2× 4090 | 1024 | 512 | 5.6小时 | ⭐⭐⭐⭐ 推荐 |
| 4× 4090 | 2048 | 512 | 2.9小时 | ⭐⭐⭐ 可行 |
| 8× 4090 | 4096 | 512 | 1.7小时 | ⭐⭐ 通信开销大 |

---

## 4. 推理性能

### 4.1 批量推理

```python
# 特征提取
batch_size = 256
inference_time = 30-40 ms/batch (FP16)
throughput = 256 / 0.035s = 7,314 样本/秒

# 100万样本推理
total_time = 1,000,000 / 7,314 = 137秒 ≈ 2.3分钟
```

### 4.2 单样本推理

```python
# 延迟 (包含SMILES→图转换)
cold_start: 50-80 ms
cached_graph: 5-10 ms (图已缓存)
```

### 4.3 推理显存占用

```python
batch_size = 256
显存占用: ~500MB (FP16)
batch_size = 1024
显存占用: ~1.5GB (FP16)

# RTX 4090可以轻松处理大批量推理
max_batch_size ≈ 8000-10000 (受图大小影响)
```

---

## 5. 实际使用建议

### 5.1 推荐配置

#### 预算有限 (个人研究)
```yaml
硬件: 1× RTX 4090 (24GB)
配置:
  batch_size: 512-1024
  precision: FP16
  num_workers: 8

预期:
  - 阶段1: 7-8小时
  - 阶段2: 1.5-2小时
  - 总计: 8.5-10小时

成本: 约 ¥15,000 (单卡)
```

#### 小型实验室 (快速迭代)
```yaml
硬件: 2× RTX 4090 (24GB)
配置:
  batch_size: 1024 (512/卡)
  precision: FP16
  ddp: True

预期:
  - 总训练时间: 5-6小时

成本: 约 ¥30,000 (双卡)
```

#### 工业/大规模研究
```yaml
硬件: 4× RTX 4090 或 8× A100 (40GB)
配置:
  batch_size: 2048-4096
  precision: FP16
  ddp: True

预期:
  - 总训练时间: 2-3小时

成本: ¥60,000 (4×4090) 或 更高 (A100)
```

### 5.2 优化建议

#### 内存优化
```python
# 1. 梯度累积 (减小显存)
accumulation_steps = 4
effective_batch_size = batch_size × accumulation_steps

# 2. 混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 3. 梯度检查点 (牺牲速度换内存)
torch.utils.checkpoint.checkpoint(layer, x)
```

#### 速度优化
```python
# 1. 增加num_workers
num_workers = 16  # CPU核心数的2倍

# 2. 使用pin_memory
pin_memory = True

# 3. 预处理数据集（提前转换为图）
# 避免训练时实时SMILES→图转换

# 4. 使用DataLoader的prefetch
persistent_workers = True
```

### 5.3 数据预处理建议

```python
# 方案1: 实时转换 (默认)
- 优点: 节省磁盘空间
- 缺点: CPU成为瓶颈，训练速度慢30-50%

# 方案2: 预处理 (推荐)
- 提前将SMILES转换为DGL图并序列化
- 训练时直接加载图
- 加速训练30-50%
- 需要额外磁盘空间: ~5-10GB (100万样本)

# 实现代码
import pickle
from tqdm import tqdm

# 预处理
graphs = []
for smiles in tqdm(smiles_list):
    g = Mol2HeteroGraph(smiles)
    graphs.append(g)

with open('preprocessed_graphs.pkl', 'wb') as f:
    pickle.dump(graphs, f)
```

---

## 6. 成本分析

### 6.1 硬件成本

| 配置 | 硬件成本 | 电费/天 | 10小时训练电费 |
|------|----------|---------|----------------|
| 1× 4090 | ¥15,000 | ¥11 | ¥4.5 |
| 2× 4090 | ¥30,000 | ¥22 | ¥5.5 |
| 4× 4090 | ¥60,000 | ¥44 | ¥7.3 |

```python
# 电费计算 (假设¥1/度)
单卡功耗 = 450W
实际功耗 = 450W × 0.8 (平均负载) = 360W
10小时电费 = 360W × 10h × ¥1/kWh = ¥3.6
系统总功耗 ≈ 450W (GPU) + 100W (CPU+其他) = 550W
10小时总电费 = 550W × 10h × ¥1/kWh = ¥5.5
```

### 6.2 云服务器成本

| 平台 | GPU类型 | 价格/小时 | 10小时成本 |
|------|---------|----------|-----------|
| AutoDL | RTX 4090 | ¥3-4 | ¥30-40 |
| 阿里云 | V100 | ¥15-20 | ¥150-200 |
| 腾讯云 | V100 | ¥15-18 | ¥150-180 |

**结论**: 如果训练次数 > 300次，购买自己的硬件更划算

---

## 7. 对比其他GPU

### 7.1 性能对比

| GPU型号 | 显存 | FP16性能 | 相对速度 | 训练时间 | 价格 |
|---------|------|----------|----------|----------|------|
| RTX 4090 | 24GB | 165 TFLOPS | 1.0x | 10小时 | ¥15,000 |
| RTX 3090 | 24GB | 71 TFLOPS | 0.43x | 23小时 | ¥10,000 |
| A100 (40GB) | 40GB | 312 TFLOPS | 1.9x | 5.3小时 | ¥50,000+ |
| A100 (80GB) | 80GB | 312 TFLOPS | 1.9x | 5.3小时 | ¥80,000+ |
| H100 | 80GB | 989 TFLOPS | 6.0x | 1.7小时 | ¥150,000+ |

### 7.2 性价比分析

```python
# 性价比 = 性能 / 价格
RTX 4090: 165 TFLOPS / ¥15,000 = 0.011 TFLOPS/¥
A100 (40GB): 312 TFLOPS / ¥50,000 = 0.006 TFLOPS/¥
RTX 3090: 71 TFLOPS / ¥10,000 = 0.007 TFLOPS/¥

结论: RTX 4090 性价比最高
```

---

## 8. 实际测试建议

### 8.1 小规模测试

```yaml
# 先用小数据集测试
train_samples: 10,000
valid_samples: 1,000
epochs: 5
batch_size: 512

预期时间: ~30分钟
目的: 验证代码、调试、确定最优batch_size
```

### 8.2 Profiling

```python
# 使用PyTorch Profiler
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # 训练一个epoch
    train_epoch()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 识别瓶颈
# - 如果GPU利用率 < 80%，增加num_workers
# - 如果显存不足，减小batch_size或使用梯度累积
# - 如果CPU瓶颈，考虑数据预处理
```

---

## 总结

### ✅ 核心结论

1. **单张RTX 4090完全够用**
   - 显存占用: 1-2GB (FP16)
   - 训练时间: 8.5-10小时
   - 成本: ¥15,000 + ¥5电费

2. **推荐配置**
   - GPU: 1-2张 RTX 4090
   - Batch Size: 512-1024
   - 精度: FP16混合精度
   - 数据预处理: 推荐

3. **可选加速**
   - 2卡DDP: 5-6小时
   - 4卡DDP: 2-3小时
   - 数据预处理: 额外提速30%

4. **性价比**
   - RTX 4090 > RTX 3090 > A100
   - 对于此项目，4090是最优选择

### 📊 快速参考表

| 需求 | 推荐配置 | 预期时间 | 成本 |
|------|----------|----------|------|
| 最低配置 | 1× 4090 | 10小时 | ¥15k |
| 推荐配置 | 2× 4090 | 5.6小时 | ¥30k |
| 快速训练 | 4× 4090 | 2.9小时 | ¥60k |

---

**评估日期**: 2025-10-14
**基准硬件**: NVIDIA RTX 4090 (24GB)
