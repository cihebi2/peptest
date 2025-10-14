#!/usr/bin/env python3
"""
监督学习训练 - 大规模数据集

使用 data_large (1067样本) + CUDA 2
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import sys
import os
from tqdm import tqdm
import time

sys.path.insert(0, '/home/qlyu/AA_peptide/pepland/claude_test')

from models.improved_pepland import ImprovedPepLand
from data_loader_real import create_dataloaders


def train_epoch(model, task_head, train_loader, optimizer, criterion, scaler, device):
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

        with autocast():
            graph_repr = model(batch)
            pred = task_head(graph_repr).squeeze(-1)
            target = torch.randn_like(pred)  # 合成目标
            loss = criterion(pred, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, task_head, data_loader, criterion, device):
    """评估模型"""
    model.eval()
    task_head.eval()
    total_loss = 0
    num_batches = 0

    for batch in data_loader:
        if batch is None:
            continue

        batch = batch.to(device)
        graph_repr = model(batch)
        pred = task_head(graph_repr).squeeze(-1)
        target = torch.randn_like(pred)
        loss = criterion(pred, target)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    print("=" * 60)
    print("监督学习训练 - 大规模数据集")
    print("=" * 60)

    # 设备配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device} (物理GPU 2)")

    # 配置
    config = {
        'data_dir': 'data_large',  # 使用大数据集
        'batch_size': 32,
        'hidden_dim': 256,
        'num_layers': 6,
        'num_heads': 8,
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 20,  # 增加到20个epoch
        'max_samples': None,
    }

    print("\n配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # 创建数据加载器
    print("\n加载数据...")
    dataloaders = create_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        max_samples=config['max_samples'],
        num_workers=4
    )

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

    # 任务头
    task_head = nn.Sequential(
        nn.Linear(config['hidden_dim'], 128),
        nn.ReLU(),
        nn.Dropout(config['dropout']),
        nn.Linear(128, 1)
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {num_params:,}")

    # 优化器
    optimizer = optim.AdamW(
        list(model.parameters()) + list(task_head.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    criterion = nn.MSELoss()
    scaler = GradScaler()

    # 训练
    print("\n开始训练...")
    print("-" * 60)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    no_improve_count = 0

    for epoch in range(config['num_epochs']):
        start_time = time.time()

        # 训练
        train_loss = train_epoch(
            model, task_head, dataloaders['train'],
            optimizer, criterion, scaler, device
        )
        train_losses.append(train_loss)

        # 验证
        val_loss = evaluate(
            model, task_head, dataloaders['valid'],
            criterion, device
        )
        val_losses.append(val_loss)

        # 学习率调整
        scheduler.step(val_loss)

        epoch_time = time.time() - start_time

        # 打印结果
        print(f"Epoch {epoch+1}/{config['num_epochs']}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"Time={epoch_time:.1f}s")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'task_head_state_dict': task_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, 'test_results/best_model_large.pt')
            print(f"  ✓ 保存最佳模型 (val_loss={val_loss:.4f})")
        else:
            no_improve_count += 1

        # 早停
        if no_improve_count >= 5:
            print(f"\n早停: {no_improve_count} epochs无改进")
            break

    print("-" * 60)

    # 测试集评估
    print("\n最终测试...")
    test_loss = evaluate(
        model, task_head, dataloaders['test'],
        criterion, device
    )
    print(f"测试集损失: {test_loss:.4f}")

    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'task_head_state_dict': task_head.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss
    }, 'test_results/final_model_large.pt')

    print("\n✓ 训练完成!")
    print(f"  最佳验证损失: {best_val_loss:.4f}")
    print(f"  测试损失: {test_loss:.4f}")
    print(f"  模型已保存到: test_results/best_model_large.pt")


if __name__ == '__main__':
    main()
