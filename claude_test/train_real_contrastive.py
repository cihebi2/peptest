#!/usr/bin/env python3
"""
对比学习预训练 - 使用真实SMILES数据

使用 CUDA 2 (物理GPU 2)
"""

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import sys
import os
from tqdm import tqdm
import time

sys.path.insert(0, '/home/qlyu/AA_peptide/pepland/claude_test')

from models.improved_pepland import ImprovedPepLand
from pretraining.contrastive import GraphContrastiveLearning
from pretraining.augmentation import GraphAugmentation
from data_loader_real import create_dataloaders


def train_epoch(model, train_loader, optimizer, scaler, aug, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(train_loader, desc="训练"):
        if batch is None:
            continue

        batch = batch.to(device)

        # 数据增强 - 使用特征级增强避免设备问题
        view1 = aug.feature_masking(batch.clone(), 0.15)
        view2 = aug.attribute_augmentation(batch.clone(), 0.1)

        optimizer.zero_grad()

        with autocast():
            loss, logits, labels = model(view1, view2)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    print("=" * 60)
    print("对比学习预训练 - 真实数据")
    print("=" * 60)

    # 设备配置：使用 CUDA_VISIBLE_DEVICES=2，所以这里用 cuda:0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device} (物理GPU 2)")

    # 配置
    config = {
        'data_dir': '/home/qlyu/AA_peptide/pepland/data/pretrained',
        'batch_size': 32,
        'hidden_dim': 256,
        'num_layers': 6,
        'num_heads': 8,
        'dropout': 0.1,
        'projection_dim': 128,
        'temperature': 0.07,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 50,  # 对比学习需要更多epoch
        'max_samples': None,  # None表示使用全部数据
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

    # 创建编码器
    print("\n创建编码器...")
    encoder = ImprovedPepLand(
        atom_dim=38,  # 真实数据的原子特征维度（简化版）
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

    scaler = GradScaler()
    aug = GraphAugmentation()

    # 训练
    print("\n开始预训练...")
    print("-" * 60)

    best_loss = float('inf')
    losses = []

    for epoch in range(config['num_epochs']):
        start_time = time.time()

        # 训练
        loss = train_epoch(
            contrastive_model, dataloaders['train'],
            optimizer, scaler, aug, device
        )
        losses.append(loss)

        epoch_time = time.time() - start_time

        # 打印结果
        print(f"Epoch {epoch+1}/{config['num_epochs']}: "
              f"Loss={loss:.4f}, "
              f"Time={epoch_time:.1f}s")

        # 保存最佳模型
        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'model_state_dict': contrastive_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': config
            }, 'test_results/best_pretrained_encoder_real.pt')
            print(f"  ✓ 保存最佳模型 (loss={loss:.4f})")

        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'model_state_dict': contrastive_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': config
            }, f'test_results/checkpoint_epoch_{epoch+1}_real.pt')
            print(f"  ✓ 保存检查点 (epoch {epoch+1})")

    print("-" * 60)

    # 保存最终编码器
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'config': config,
        'losses': losses,
        'final_loss': loss
    }, 'test_results/final_pretrained_encoder_real.pt')

    print("\n✓ 预训练完成!")
    print(f"  最佳损失: {best_loss:.4f}")
    print(f"  最终损失: {loss:.4f}")
    print(f"  编码器已保存到: test_results/best_pretrained_encoder_real.pt")


if __name__ == '__main__':
    main()
