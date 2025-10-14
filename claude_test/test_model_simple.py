#!/usr/bin/env python3
"""
简化的模型测试 - 只测试核心功能
"""

import torch
import dgl
import sys
sys.path.insert(0, '/home/qlyu/AA_peptide/pepland/claude_test')

from models.improved_pepland import ImprovedPepLand

def create_test_graph(device):
    """创建测试图"""
    num_nodes = 30

    # 创建链状图
    src = list(range(num_nodes - 1))
    dst = list(range(1, num_nodes))

    g = dgl.graph((src, dst), num_nodes=num_nodes).to(device)

    # 添加特征
    g.ndata['atom_feat'] = torch.randn(num_nodes, 42, device=device)
    g.ndata['node_type'] = torch.zeros(num_nodes, dtype=torch.long, device=device)
    g.edata['bond_feat'] = torch.randn(g.num_edges(), 14, device=device)

    return g

print("=" * 60)
print("简化模型测试")
print("=" * 60)

device = torch.device('cuda:0')
print(f"\n设备: {device} ({torch.cuda.get_device_name(0)})")

# 创建模型
print("\n1. 创建模型...")
model = ImprovedPepLand(
    atom_dim=42,
    bond_dim=14,
    fragment_dim=196,
    hidden_dim=256,  # 小一点
    num_layers=4,     # 少一点
    num_heads=8,
    dropout=0.1,
    use_performer=True,
    use_virtual_node=True
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"   参数量: {num_params:,}")

# 创建测试图
print("\n2. 创建测试图...")
g = create_test_graph(device)
print(f"   节点数: {g.num_nodes()}")
print(f"   边数: {g.num_edges()}")

# 前向传播
print("\n3. 测试前向传播...")
model.eval()
with torch.no_grad():
    try:
        output = model(g)
        print(f"   ✓ 输出形状: {output.shape}")
        print(f"   ✓ 输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# 测试批处理
print("\n4. 测试批处理...")
batch_g = dgl.batch([create_test_graph(device) for _ in range(4)])
with torch.no_grad():
    batch_output = model(batch_g)
    print(f"   ✓ 批输出形状: {batch_output.shape}")

# 测试反向传播
print("\n5. 测试反向传播...")
model.train()
output = model(g)
loss = output.mean()
loss.backward()
print(f"   ✓ 损失: {loss.item():.4f}")

print("\n" + "=" * 60)
print("✓ 所有测试通过！")
print("=" * 60)
