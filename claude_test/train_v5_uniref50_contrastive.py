#!/usr/bin/env python3
"""
v0.5.0 - UniRef50 大规模对比学习预训练
- 数据集: UniRef50 (292K 序列)
- 模型: 大型对比学习模型
- GPU: 1 和 2 并行训练
"""

import torch
import torch.optim as optim
import sys
import os
from tqdm import tqdm
import time
import numpy as np
import argparse

sys.path.insert(0, '/home/qlyu/AA_peptide/pepland/claude_test')

from models.improved_pepland import ImprovedPepLand
from pretraining.contrastive import GraphContrastiveLearning
from pretraining.augmentation import GraphAugmentation
from data_loader_real import create_dataloaders


def train_epoch(model, train_loader, optimizer, aug, device, epoch, accumulation_steps=2):
    """训练一个epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    num_batches = 0
    nan_count = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        if batch is None:
            continue

        batch = batch.to(device)

        # 数据增强
        try:
            view1 = aug.feature_masking(batch.clone(), 0.15)
            view2 = aug.attribute_augmentation(batch.clone(), 0.1)
        except Exception as e:
            print(f"数据增强失败: {e}")
            continue

        try:
            loss, logits, labels = model(view1, view2)

            # 数值检查
            if torch.isnan(loss).any():
                nan_count += 1
                continue

            if loss.item() > 100:
                print(f"  警告: 损失过大 {loss.item():.2f}，跳过")
                continue

            # 梯度累积
            loss = loss / accumulation_steps
            loss.backward()

            # 每accumulation_steps个batch更新一次
            if (batch_idx + 1) % accumulation_steps == 0:
                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                if torch.isnan(grad_norm) or grad_norm > 50.0:
                    print(f"  警告: 梯度异常 (norm={grad_norm:.2f})，跳过")
                    optimizer.zero_grad()
                    continue

                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            num_batches += 1

        except RuntimeError as e:
            print(f"  训练错误: {e}")
            continue

    # 处理剩余的梯度
    if (batch_idx + 1) % accumulation_steps != 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        if not (torch.isnan(grad_norm) or grad_norm > 50.0):
            optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(num_batches, 1)
    if nan_count > 0:
        print(f"  ⚠️ 本epoch有 {nan_count} 个批次出现NaN")

    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--name', type=str, default='v5_uniref50', help='实验名称')
    args = parser.parse_args()

    print("=" * 60)
    print(f"v0.5.0 - UniRef50 大规模对比学习预训练 (GPU {args.gpu})")
    print("=" * 60)

    # 设备配置
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")

    # 配置
    config = {
        'data_dir': '/home/qlyu/AA_peptide/pepland/data/uniref50_large',
        'batch_size': 32,
        'accumulation_steps': 2,  # 有效batch size = 64
        'hidden_dim': 512,
        'num_layers': 8,
        'num_heads': 8,
        'dropout': 0.1,
        'projection_dim': 256,
        'temperature': 0.5,
        'learning_rate': 5e-5,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'max_samples': None,  # 使用全部数据
        'warmup_epochs': 5,
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

    # 创建编码器
    print("\n创建编码器...")
    encoder = ImprovedPepLand(
        atom_dim=42,  # 20个氨基酸 + padding + special tokens
        bond_dim=14,
        fragment_dim=196,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        use_performer=True,
        use_virtual_node=True
    )

    # 对比学习模型
    contrastive_model = GraphContrastiveLearning(
        encoder=encoder,
        projection_dim=config['projection_dim'],
        temperature=config['temperature']
    ).to(device)

    num_params = sum(p.numel() for p in contrastive_model.parameters())
    print(f"  模型参数量: {num_params:,}")

    # 优化器
    optimizer = optim.AdamW(
        contrastive_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # 学习率调度器
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return (epoch + 1) / config['warmup_epochs']
        else:
            progress = (epoch - config['warmup_epochs']) / (config['num_epochs'] - config['warmup_epochs'])
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    aug = GraphAugmentation()

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
            contrastive_model, dataloaders['train'],
            optimizer, aug, device, epoch + 1,
            accumulation_steps=config['accumulation_steps']
        )
        losses.append(loss)

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - start_time

        # 检查损失
        if np.isnan(loss):
            print(f"Epoch {epoch+1}/{config['num_epochs']}: "
                  f"Loss=NaN ⚠️, LR={current_lr:.2e}, Time={epoch_time:.1f}s")
            print("  训练失败！")
            break
        else:
            print(f"Epoch {epoch+1}/{config['num_epochs']}: "
                  f"Loss={loss:.4f}, LR={current_lr:.2e}, Time={epoch_time:.1f}s")

        # 保存最佳模型
        if loss < best_loss and not np.isnan(loss):
            best_loss = loss
            no_improve_count = 0
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'model_state_dict': contrastive_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                'config': config
            }, f'test_results/best_{args.name}_gpu{args.gpu}.pt')
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
                'encoder_state_dict': encoder.state_dict(),
                'model_state_dict': contrastive_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                'config': config
            }, f'test_results/checkpoint_{args.name}_gpu{args.gpu}_epoch_{epoch+1}.pt')
            print(f"  ✓ 保存检查点 (epoch {epoch+1})")

    print("-" * 60)

    # 保存最终编码器
    if not np.isnan(loss):
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'config': config,
            'losses': losses,
            'final_loss': loss
        }, f'test_results/final_{args.name}_gpu{args.gpu}.pt')

        print("\n✓ 预训练完成!")
        print(f"  最佳损失: {best_loss:.4f}")
        print(f"  最终损失: {loss:.4f}")
        print(f"  编码器已保存到: test_results/best_{args.name}_gpu{args.gpu}.pt")
    else:
        print("\n✗ 训练失败（NaN损失）")


if __name__ == '__main__':
    main()
