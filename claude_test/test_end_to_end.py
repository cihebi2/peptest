#!/usr/bin/env python3
"""
端到端测试：数据加载 -> 模型训练 -> 评估
End-to-End Test: Data Loading -> Training -> Evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import sys
import os
from tqdm import tqdm

sys.path.insert(0, '/home/qlyu/AA_peptide/pepland/claude_test')

from models.improved_pepland import ImprovedPepLand
from data_loader import PeptideDataModule, create_sample_dataset
from pretraining.contrastive import GraphContrastiveLearning, MoCoForGraphs
from pretraining.augmentation import GraphAugmentation

def test_simple_training():
    """测试简单的监督训练流程"""

    print("=" * 70)
    print("测试 1: 简单监督训练 / Test 1: Simple Supervised Training")
    print("=" * 70)

    device = torch.device('cuda:0')
    print(f"\n设备: {device} (Physical GPU 3)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. 准备数据
    print("\n[1/6] 准备数据...")
    sample_path = create_sample_dataset('test_results/train_data.csv', num_samples=100)

    # 创建简单的训练集
    import pandas as pd
    from data_loader import PeptideGraphDataset
    from dgl.dataloading import GraphDataLoader

    dataset = PeptideGraphDataset(sample_path)
    loader = GraphDataLoader(dataset, batch_size=8, num_workers=0)

    # 2. 创建模型
    print("\n[2/6] 创建模型...")
    model = ImprovedPepLand(
        atom_dim=42,
        bond_dim=14,
        fragment_dim=196,
        hidden_dim=256,  # 小一点加快测试
        num_layers=6,     # 少一点层数
        num_heads=8,
        dropout=0.1,
        use_performer=True
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数: {num_params:,}")

    # 3. 定义任务头（简单的回归任务）
    task_head = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 1)  # 输出单个值
    ).to(device)

    print(f"   任务头参数: {sum(p.numel() for p in task_head.parameters()):,}")

    # 4. 优化器和损失函数
    print("\n[3/6] 配置训练...")
    optimizer = optim.AdamW(
        list(model.parameters()) + list(task_head.parameters()),
        lr=1e-4,
        weight_decay=1e-5
    )
    criterion = nn.MSELoss()
    scaler = GradScaler()

    # 5. 训练循环
    print("\n[4/6] 开始训练...")
    model.train()
    task_head.train()

    num_epochs = 3
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch_g in enumerate(pbar):
            batch_g = batch_g.to(device)

            # 前向传播
            optimizer.zero_grad()

            with autocast():
                # 模型输出
                graph_repr = model(batch_g)  # [batch_size, hidden_dim]

                # 任务输出
                pred = task_head(graph_repr).squeeze(-1)  # [batch_size]

                # 生成伪标签（实际应用中应该是真实标签）
                target = torch.randn_like(pred)

                # 损失
                loss = criterion(pred, target)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # 只训练几个 batch 作为测试
            if batch_idx >= 10:
                break

        avg_loss = total_loss / num_batches
        print(f"   Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

    # 6. 评估
    print("\n[5/6] 评估模型...")
    model.eval()
    task_head.eval()

    eval_losses = []
    with torch.no_grad():
        for batch_idx, batch_g in enumerate(loader):
            batch_g = batch_g.to(device)

            graph_repr = model(batch_g)
            pred = task_head(graph_repr).squeeze(-1)
            target = torch.randn_like(pred)

            loss = criterion(pred, target)
            eval_losses.append(loss.item())

            if batch_idx >= 5:
                break

    print(f"   Eval Loss: {sum(eval_losses) / len(eval_losses):.4f}")

    # 7. 保存模型
    print("\n[6/6] 保存模型...")
    save_path = 'test_results/simple_model.pt'
    torch.save({
        'model': model.state_dict(),
        'task_head': task_head.state_dict(),
        'optimizer': optimizer.state_dict()
    }, save_path)
    print(f"   已保存到: {save_path}")

    print("\n" + "=" * 70)
    print("✓ 简单监督训练测试通过！")
    print("=" * 70)

    return True


def test_contrastive_pretraining():
    """测试对比学习预训练"""

    print("\n" + "=" * 70)
    print("测试 2: 对比学习预训练 / Test 2: Contrastive Pretraining")
    print("=" * 70)

    device = torch.device('cuda:0')

    # 1. 准备数据
    print("\n[1/5] 准备数据...")
    sample_path = create_sample_dataset('test_results/pretrain_data.csv', num_samples=50)

    from data_loader import PeptideGraphDataset
    from dgl.dataloading import GraphDataLoader

    dataset = PeptideGraphDataset(sample_path)
    loader = GraphDataLoader(dataset, batch_size=4, num_workers=0)

    # 2. 创建模型
    print("\n[2/5] 创建模型...")
    encoder = ImprovedPepLand(
        atom_dim=42,
        bond_dim=14,
        fragment_dim=196,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        use_performer=True
    ).to(device)

    # 3. 对比学习框架
    print("\n[3/5] 创建对比学习框架...")
    contrastive_model = GraphContrastiveLearning(
        encoder=encoder,
        projection_dim=128,
        temperature=0.07
    ).to(device)

    num_params = sum(p.numel() for p in contrastive_model.parameters())
    print(f"   总参数: {num_params:,}")

    # 4. 训练
    print("\n[4/5] 开始预训练...")
    optimizer = optim.AdamW(contrastive_model.parameters(), lr=1e-4)

    contrastive_model.train()
    num_epochs = 2

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch_g in enumerate(pbar):
            batch_g = batch_g.to(device)

            # 创建两个增强视图
            aug = GraphAugmentation()
            view1 = aug.create_two_views(batch_g, strong_aug=False)[0]
            view2 = aug.create_two_views(batch_g, strong_aug=False)[0]

            # 对比学习
            optimizer.zero_grad()
            loss, _, _ = contrastive_model(view1, view2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if batch_idx >= 8:
                break

        avg_loss = total_loss / num_batches
        print(f"   Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

    # 5. 保存预训练模型
    print("\n[5/5] 保存预训练模型...")
    save_path = 'test_results/pretrained_encoder.pt'
    torch.save(encoder.state_dict(), save_path)
    print(f"   已保存到: {save_path}")

    print("\n" + "=" * 70)
    print("✓ 对比学习预训练测试通过！")
    print("=" * 70)

    return True


def test_memory_efficiency():
    """测试内存效率"""

    print("\n" + "=" * 70)
    print("测试 3: 内存效率 / Test 3: Memory Efficiency")
    print("=" * 70)

    device = torch.device('cuda:0')

    print("\n初始 GPU 内存:")
    print(f"   已分配: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"   已保留: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    # 创建较大的模型
    print("\n创建模型...")
    model = ImprovedPepLand(
        atom_dim=42,
        bond_dim=14,
        fragment_dim=196,
        hidden_dim=512,
        num_layers=12,
        num_heads=8
    ).to(device)

    print(f"\n模型加载后:")
    print(f"   已分配: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"   已保留: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    # 测试推理
    sample_path = create_sample_dataset('test_results/memory_test.csv', num_samples=20)

    from data_loader import PeptideGraphDataset
    from dgl.dataloading import GraphDataLoader

    dataset = PeptideGraphDataset(sample_path)
    loader = GraphDataLoader(dataset, batch_size=16, num_workers=0)

    model.eval()
    with torch.no_grad():
        for batch_g in loader:
            batch_g = batch_g.to(device)
            output = model(batch_g)
            break

    print(f"\n推理后:")
    print(f"   已分配: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"   峰值: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")

    # 清理
    del model, batch_g, output
    torch.cuda.empty_cache()

    print(f"\n清理后:")
    print(f"   已分配: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    print("\n" + "=" * 70)
    print("✓ 内存效率测试通过！")
    print("=" * 70)

    return True


if __name__ == "__main__":
    """运行所有端到端测试"""

    print("\n" + "=" * 70)
    print("端到端测试套件 / End-to-End Test Suite")
    print("=" * 70)

    try:
        # 测试 1: 简单监督训练
        success1 = test_simple_training()

        # 测试 2: 对比学习预训练
        success2 = test_contrastive_pretraining()

        # 测试 3: 内存效率
        success3 = test_memory_efficiency()

        if success1 and success2 and success3:
            print("\n" + "=" * 70)
            print("✓✓✓ 所有端到端测试通过！ / All Tests Passed!")
            print("=" * 70)
            sys.exit(0)
        else:
            print("\n❌ 部分测试失败")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
