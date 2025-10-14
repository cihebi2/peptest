#!/usr/bin/env python3
"""对比学习预训练"""

import torch
import torch.optim as optim
import dgl
import sys

sys.path.insert(0, '/home/qlyu/AA_peptide/pepland/claude_test')

from models.improved_pepland import ImprovedPepLand
from pretraining.contrastive import GraphContrastiveLearning
from pretraining.augmentation import GraphAugmentation

def create_graph(num_nodes=30, device='cuda'):
    src = list(range(num_nodes - 1))
    dst = list(range(1, num_nodes))
    g = dgl.graph((src, dst), num_nodes=num_nodes).to(device)
    g.ndata['atom_feat'] = torch.randn(num_nodes, 42, device=device)
    g.ndata['node_type'] = torch.zeros(num_nodes, dtype=torch.long, device=device)
    g.edata['bond_feat'] = torch.randn(g.num_edges(), 14, device=device)
    return g

print("="*60)
print("对比学习预训练")
print("="*60)

device = torch.device('cuda:0')
print(f"\n设备: {device}")

# 创建编码器
encoder = ImprovedPepLand(
    atom_dim=42, bond_dim=14, fragment_dim=196,
    hidden_dim=256, num_layers=4, num_heads=8,
    use_performer=True
).to(device)

# 对比学习模型
contrastive_model = GraphContrastiveLearning(
    encoder=encoder, projection_dim=128, temperature=0.07
).to(device)

print(f"\n参数量: {sum(p.numel() for p in contrastive_model.parameters()):,}")

# 数据
data = [create_graph(torch.randint(20, 40, (1,)).item(), device) for _ in range(100)]
print(f"数据集: {len(data)}")

# 训练
optimizer = optim.AdamW(contrastive_model.parameters(), lr=1e-4)
aug = GraphAugmentation()

print("\n开始预训练...")
for epoch in range(3):
    total_loss = 0
    for i in range(0, len(data), 8):
        batch = data[i:i+8]
        batch_g = dgl.batch(batch)

        # 数据增强 - 使用简单的特征级增强避免设备问题
        view1 = aug.feature_masking(batch_g.clone(), 0.15)
        view2 = aug.attribute_augmentation(batch_g.clone(), 0.1)

        optimizer.zero_grad()
        loss, _, _ = contrastive_model(view1, view2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/3: Loss={total_loss/13:.4f}")

# 保存
torch.save(encoder.state_dict(), 'test_results/pretrained_encoder.pt')
print(f"\n✓ 预训练完成！已保存到: test_results/pretrained_encoder.pt")
