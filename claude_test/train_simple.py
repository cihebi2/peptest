#!/usr/bin/env python3
"""简化训练脚本 - 使用合成数据测试完整流程"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import dgl
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/qlyu/AA_peptide/pepland/claude_test')

from models.improved_pepland import ImprovedPepLand

def create_synthetic_graph(num_nodes=30, device='cuda'):
    """创建合成图数据"""
    src = list(range(num_nodes - 1))
    dst = list(range(1, num_nodes))
    g = dgl.graph((src, dst), num_nodes=num_nodes).to(device)
    g.ndata['atom_feat'] = torch.randn(num_nodes, 42, device=device)
    g.ndata['node_type'] = torch.zeros(num_nodes, dtype=torch.long, device=device)
    g.edata['bond_feat'] = torch.randn(g.num_edges(), 14, device=device)
    return g

def create_dataset(num_samples=100, device='cuda'):
    """创建合成数据集"""
    return [create_synthetic_graph(torch.randint(20, 40, (1,)).item(), device)
            for _ in range(num_samples)]

print("="*60)
print("简化训练测试")
print("="*60)

device = torch.device('cuda:0')
print(f"\n设备: {device} (GPU 3)")

# 创建模型
print("\n1. 创建模型...")
model = ImprovedPepLand(
    atom_dim=42, bond_dim=14, fragment_dim=196,
    hidden_dim=256, num_layers=6, num_heads=8,
    dropout=0.1, use_performer=True, use_virtual_node=True
).to(device)

# 任务头
task_head = nn.Sequential(
    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)
).to(device)

print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

# 创建数据
print("\n2. 创建数据集...")
train_data = create_dataset(80, device)
val_data = create_dataset(20, device)
print(f"   训练集: {len(train_data)}, 验证集: {len(val_data)}")

# 优化器
optimizer = optim.AdamW(
    list(model.parameters()) + list(task_head.parameters()),
    lr=1e-4, weight_decay=1e-5
)
criterion = nn.MSELoss()
scaler = GradScaler()

# 训练
print("\n3. 开始训练...")
num_epochs = 3
batch_size = 8

for epoch in range(num_epochs):
    model.train()
    task_head.train()
    train_loss = 0

    # 分批训练
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        batch_g = dgl.batch(batch)

        optimizer.zero_grad()
        with autocast():
            graph_repr = model(batch_g)
            pred = task_head(graph_repr).squeeze(-1)
            target = torch.randn_like(pred)
            loss = criterion(pred, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    # 验证
    model.eval()
    task_head.eval()
    val_loss = 0

    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            batch = val_data[i:i+batch_size]
            batch_g = dgl.batch(batch)
            graph_repr = model(batch_g)
            pred = task_head(graph_repr).squeeze(-1)
            target = torch.randn_like(pred)
            loss = criterion(pred, target)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}: "
          f"Train Loss={train_loss/10:.4f}, Val Loss={val_loss/3:.4f}")

# 保存
print("\n4. 保存模型...")
torch.save({
    'model': model.state_dict(),
    'task_head': task_head.state_dict()
}, 'test_results/trained_model.pt')
print("   已保存到: test_results/trained_model.pt")

print("\n" + "="*60)
print("✓ 训练完成！")
print("="*60)
