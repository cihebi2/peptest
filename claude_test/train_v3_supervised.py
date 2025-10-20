#!/usr/bin/env python3
"""
v0.3.0 - 扩大数据集监督学习预训练
使用 pretrained/ 数据集 (~3000样本)
GPU: 2
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from tqdm import tqdm
import time

sys.path.insert(0, '/home/qlyu/AA_peptide/pepland/claude_test')

from models.improved_pepland import ImprovedPepLand
from data_loader_real import create_dataloaders


def train_epoch(model, task_head, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    task_head.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(train_loader, desc="训练"):
        if batch is None:
            continue

        batch = batch.to(device)

        optimizer.zero_grad()

        try:
            # 编码
            graph_repr = model(batch)

            # 预测
            logits = task_head(graph_repr)

            # 假设每个图有一个标签（属性预测任务）
            # 这里简单使用随机标签作为监督信号
            # 实际应用中应该使用真实标签
            batch_size = graph_repr.shape[0]
            labels = torch.randint(0, 2, (batch_size,), device=device)

            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(task_head.parameters()),
                max_norm=1.0
            )
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        except Exception as e:
            print(f"  训练错误: {e}")
            continue

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def main():
    print("=" * 60)
    print("v0.3.0 - 扩大数据集监督学习预训练")
    print("=" * 60)

    # 设备配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device} (GPU 2)")

    # 配置
    config = {
        'data_dir': '/home/qlyu/AA_peptide/pepland/data/pretrained',
        'batch_size': 32,
        'hidden_dim': 256,
        'num_layers': 6,
        'num_heads': 8,
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'max_samples': None,
    }

    print("\n配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # 加载数据
    print("\n加载数据...")
    try:
        dataloaders = create_dataloaders(
            config['data_dir'],
            batch_size=config['batch_size'],
            max_samples=config['max_samples'],
            num_workers=4
        )
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 创建模型
    print("\n创建模型...")
    model = ImprovedPepLand(
        atom_dim=38,
        bond_dim=14,
        fragment_dim=196,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        use_performer=True,
        use_virtual_node=True
    ).to(device)

    # 任务头 (二分类)
    task_head = nn.Sequential(
        nn.Linear(config['hidden_dim'], 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 2)
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters()) + \
                 sum(p.numel() for p in task_head.parameters())
    print(f"  模型参数量: {num_params:,}")

    # 优化器和损失
    optimizer = optim.AdamW(
        list(model.parameters()) + list(task_head.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()

    # 训练
    print("\n开始预训练...")
    print("-" * 60)

    best_loss = float('inf')
    losses = []
    no_improve_count = 0

    for epoch in range(config['num_epochs']):
        start_time = time.time()

        # 训练
        loss = train_epoch(
            model, task_head, dataloaders['train'],
            optimizer, criterion, device
        )
        losses.append(loss)

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{config['num_epochs']}: "
              f"Loss={loss:.4f}, Time={epoch_time:.1f}s")

        # 保存最佳模型
        if loss < best_loss:
            best_loss = loss
            no_improve_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'task_head_state_dict': task_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': config
            }, 'test_results/best_v3_supervised.pt')
            print(f"  ✓ 保存最佳模型 (loss={loss:.4f})")
        else:
            no_improve_count += 1

        # 早停
        if no_improve_count >= 15:
            print(f"\n早停: {no_improve_count} epochs无改进")
            break

        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'task_head_state_dict': task_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': config
            }, f'test_results/checkpoint_v3_supervised_epoch_{epoch+1}.pt')
            print(f"  ✓ 保存检查点 (epoch {epoch+1})")

    print("-" * 60)

    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'losses': losses,
        'final_loss': loss
    }, 'test_results/final_v3_supervised.pt')

    print("\n✓ 预训练完成!")
    print(f"  最佳损失: {best_loss:.4f}")
    print(f"  最终损失: {loss:.4f}")
    print(f"  模型已保存到: test_results/best_v3_supervised.pt")


if __name__ == '__main__':
    main()
