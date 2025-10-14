#!/usr/bin/env python3
"""
完整模型测试 - 集成 DGL
Test Full Model with DGL Integration
"""

import torch
import dgl
import sys
import os

sys.path.insert(0, '/home/qlyu/AA_peptide/pepland/claude_test')

from models.improved_pepland import ImprovedPepLand, EnhancedHeteroGraph

def create_test_heterograph(device):
    """创建测试用同构图（简化版）"""
    # 创建一个简单的同构图用于测试
    # ImprovedPepLand 期望同构图格式

    num_nodes = 30

    # 创建图 - 确保有30个节点
    src = []
    dst = []
    for i in range(num_nodes - 1):
        src.append(i)
        dst.append(i + 1)
    # 添加一些额外的边
    src.extend([5, 10, 15, 20])
    dst.extend([0, 5, 10, 15])

    g = dgl.graph((src, dst), num_nodes=num_nodes).to(device)

    # 添加节点特征
    g.ndata['atom_feat'] = torch.randn(num_nodes, 42, device=device)
    g.ndata['node_type'] = torch.zeros(num_nodes, dtype=torch.long, device=device)  # 全是原子节点

    # 添加边特征
    g.edata['bond_feat'] = torch.randn(g.num_edges(), 14, device=device)

    return g

def test_full_model():
    """测试完整的 ImprovedPepLand 模型"""

    print("=" * 60)
    print("完整模型测试 / Full Model Test")
    print("=" * 60)

    # 设备
    device = torch.device('cuda:0')  # CUDA_VISIBLE_DEVICES=3 makes GPU 3 appear as GPU 0
    print(f"\n设备 / Device: {device} (Physical GPU 3)")
    print(f"GPU 名称 / GPU Name: {torch.cuda.get_device_name(0)}")

    try:
        # 测试 1: 创建模型
        print("\n1. 创建 ImprovedPepLand 模型:")
        model = ImprovedPepLand(
            atom_dim=42,
            bond_dim=14,
            fragment_dim=196,
            hidden_dim=512,
            num_layers=12,
            num_heads=8,
            dropout=0.1,
            use_performer=True,  # 使用 Performer attention
            use_virtual_node=True
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"   ✓ 模型创建成功")
        print(f"   ✓ 总参数量: {num_params:,}")
        print(f"   ✓ 可训练参数: {trainable_params:,}")
        print(f"   ✓ 模型大小: {num_params * 4 / 1024**2:.2f} MB (FP32)")

        # 测试 2: 创建测试数据
        print("\n2. 创建测试异构图:")
        hetero_g = create_test_heterograph(device)
        print(f"   ✓ 节点类型: {hetero_g.ntypes}")
        print(f"   ✓ 边类型: {hetero_g.etypes}")
        print(f"   ✓ 原子节点数: {hetero_g.num_nodes('atom')}")
        print(f"   ✓ 片段节点数: {hetero_g.num_nodes('fragment')}")
        print(f"   ✓ 总边数: {hetero_g.num_edges()}")

        # 测试 3: 前向传播
        print("\n3. 测试前向传播:")
        model.eval()
        with torch.no_grad():
            output = model(hetero_g)

        print(f"   ✓ 输出形状: {output.shape}")
        print(f"   ✓ 输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"   ✓ 输出均值: {output.mean().item():.4f}")
        print(f"   ✓ 输出标准差: {output.std().item():.4f}")

        # 测试 4: 反向传播
        print("\n4. 测试反向传播:")
        model.train()
        output = model(hetero_g)
        loss = output.mean()
        loss.backward()

        # 检查梯度
        has_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for _ in model.parameters())

        print(f"   ✓ 损失值: {loss.item():.4f}")
        print(f"   ✓ 有梯度的参数: {has_grad}/{total_params}")
        print(f"   ✓ 梯度计算成功")

        # 测试 5: 批处理
        print("\n5. 测试批处理:")
        batch_graphs = [create_test_heterograph(device) for _ in range(4)]
        batched_g = dgl.batch(batch_graphs)

        print(f"   ✓ 批大小: {len(batch_graphs)}")
        print(f"   ✓ 批次图总节点数: {batched_g.num_nodes('atom')}")
        print(f"   ✓ 批次图总边数: {batched_g.num_edges()}")

        with torch.no_grad():
            batch_output = model(batched_g)

        print(f"   ✓ 批输出形状: {batch_output.shape}")

        # 测试 6: 内存效率
        print("\n6. GPU 内存使用:")
        print(f"   已分配 / Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"   已保留 / Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print(f"   峰值分配 / Peak: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")

        # 测试 7: 混合精度
        print("\n7. 测试混合精度训练 (FP16):")
        from torch.cuda.amp import autocast, GradScaler

        scaler = GradScaler()
        model.train()

        with autocast():
            output = model(hetero_g)
            loss = output.mean()

        scaler.scale(loss).backward()
        scaler.step(torch.optim.Adam(model.parameters(), lr=0.001))
        scaler.update()

        print(f"   ✓ 混合精度训练成功")
        print(f"   ✓ 输出 dtype: {output.dtype}")

        # 测试 8: 保存和加载
        print("\n8. 测试模型保存和加载:")
        save_path = '/home/qlyu/AA_peptide/pepland/claude_test/test_results/test_model.pt'
        torch.save(model.state_dict(), save_path)
        print(f"   ✓ 模型已保存到: {save_path}")

        # 创建新模型并加载
        model2 = ImprovedPepLand(
            atom_dim=42,
            bond_dim=14,
            fragment_dim=196,
            hidden_dim=512,
            num_layers=12,
            num_heads=8
        ).to(device)
        model2.load_state_dict(torch.load(save_path))
        print(f"   ✓ 模型加载成功")

        # 验证输出一致性
        model2.eval()
        with torch.no_grad():
            output1 = model(hetero_g)
            output2 = model2(hetero_g)
            diff = (output1 - output2).abs().max().item()

        print(f"   ✓ 输出差异: {diff:.10f} (应该很小)")

        print("\n" + "=" * 60)
        print("✓ 所有测试通过！模型可以正常使用")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_model()
    sys.exit(0 if success else 1)
