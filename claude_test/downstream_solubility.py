#!/usr/bin/env python3
"""
下游任务：溶解度预测
对比三种初始化方法：
1. Random - 随机初始化
2. Pretrained Contrastive - 对比学习预训练
3. Pretrained Supervised - 监督学习预训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, '/home/qlyu/AA_peptide/pepland/claude_test')

from models.improved_pepland import ImprovedPepLand
from data_loader_real import smiles_to_graph
from rdkit import Chem


# 溶解度数据集
class SolubilityDataset(Dataset):
    """
    溶解度数据集
    从氨基酸序列创建图
    """
    def __init__(self, txt_path, max_samples=None):
        self.data = []
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            if max_samples:
                lines = lines[:max_samples]
            
            for line in tqdm(lines, desc=f"加载 {os.path.basename(txt_path)}"):
                parts = line.strip().split(',')
                if len(parts) != 2:
                    continue
                
                sequence = parts[0]
                label = int(parts[1])
                
                # 将氨基酸序列转为SMILES（简化版：直接用序列）
                # 这里应该用真实的肽段到SMILES转换，暂时用占位符
                graph = self._sequence_to_graph(sequence)
                
                if graph is not None:
                    self.data.append((graph, label))
        
        print(f"  有效样本: {len(self.data)}")
    
    def _sequence_to_graph(self, sequence):
        """
        将氨基酸序列转为图
        简化版：创建线性图，每个氨基酸是一个节点
        """
        # 氨基酸单字母代码
        aa_dict = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
            'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
            'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
        }
        
        num_atoms = len(sequence)
        if num_atoms == 0 or num_atoms > 500:  # 过滤过长序列
            return None
        
        # 创建节点特征（42维，匹配原子特征维度）
        atom_features = []
        for aa in sequence:
            if aa not in aa_dict:
                return None
            
            # 创建one-hot编码（20个氨基酸）+ padding到42维
            feat = [0] * 42
            feat[aa_dict[aa]] = 1
            atom_features.append(feat)
        
        atom_features = torch.FloatTensor(atom_features)
        
        # 创建线性连接（肽键）
        edges_src = []
        edges_dst = []
        for i in range(num_atoms - 1):
            edges_src.extend([i, i+1])
            edges_dst.extend([i+1, i])
        
        # 如果只有一个氨基酸，添加自环
        if num_atoms == 1:
            edges_src = [0]
            edges_dst = [0]
        
        # 创建键特征（14维，全零表示肽键）
        bond_features = torch.zeros(len(edges_src), 14)
        bond_features[:, 0] = 1  # 标记为肽键
        
        import dgl
        g = dgl.graph((edges_src, edges_dst), num_nodes=num_atoms)
        g.ndata['atom_feat'] = atom_features
        g.ndata['node_type'] = torch.zeros(num_atoms, dtype=torch.long)
        g.edata['bond_feat'] = bond_features
        
        return g
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """批处理函数"""
    import dgl
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    labels = torch.LongTensor(labels)
    return batched_graph, labels


def create_dataloaders(data_file, train_ratio=0.8, valid_ratio=0.1, batch_size=32, max_samples=None):
    """
    创建数据加载器
    """
    dataset = SolubilityDataset(data_file, max_samples=max_samples)
    
    # 分割数据
    total = len(dataset)
    train_size = int(total * train_ratio)
    valid_size = int(total * valid_ratio)
    test_size = total - train_size - valid_size
    
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"\n数据分割:")
    print(f"  训练集: {train_size}")
    print(f"  验证集: {valid_size}")
    print(f"  测试集: {test_size}")
    
    return train_loader, valid_loader, test_loader


def train_epoch(model, task_head, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    task_head.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_graph, labels in tqdm(train_loader, desc="训练"):
        batch_graph = batch_graph.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        graph_repr = model(batch_graph)
        logits = task_head(graph_repr)
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(task_head.parameters()), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, task_head, data_loader, criterion, device):
    """评估模型"""
    model.eval()
    task_head.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch_graph, labels in data_loader:
        batch_graph = batch_graph.to(device)
        labels = labels.to(device)
        
        graph_repr = model(batch_graph)
        logits = task_head(graph_repr)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # 正类概率
    
    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    
    return avg_loss, acc, precision, recall, f1, auc


def train_model(init_method, encoder, device, train_loader, valid_loader, test_loader, 
                num_epochs=30, lr=1e-4, freeze_encoder=False):
    """
    训练模型
    
    Args:
        init_method: 初始化方法名称
        encoder: 编码器
        device: 设备
        train_loader, valid_loader, test_loader: 数据加载器
        num_epochs: 训练轮数
        lr: 学习率
        freeze_encoder: 是否冻结编码器
    """
    print(f"\n{'='*60}")
    print(f"训练配置: {init_method}")
    print(f"{'='*60}")
    print(f"冻结编码器: {freeze_encoder}")
    print(f"学习率: {lr}")
    print(f"训练轮数: {num_epochs}")
    
    # 冻结编码器（如果需要）
    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        print("✓ 编码器已冻结")
    
    # 创建任务头
    task_head = nn.Sequential(
        nn.Linear(encoder.hidden_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2)  # 二分类
    ).to(device)
    
    # 优化器
    if freeze_encoder:
        optimizer = optim.Adam(task_head.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(
            list(encoder.parameters()) + list(task_head.parameters()),
            lr=lr
        )
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # 训练
    best_val_acc = 0
    results = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(encoder, task_head, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = evaluate(
            encoder, task_head, valid_loader, criterion, device
        )
        
        scheduler.step(val_acc)
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val AUC={val_auc:.4f}, "
              f"Time={epoch_time:.1f}s")
        
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_auc': val_auc
        })
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'encoder': encoder.state_dict(),
                'task_head': task_head.state_dict(),
                'val_acc': val_acc,
                'val_auc': val_auc
            }, f'test_results/solubility_{init_method}_best.pt')
    
    # 测试集评估
    print("\n最终测试...")
    test_loss, test_acc, test_precision, test_recall, test_f1, test_auc = evaluate(
        encoder, task_head, test_loader, criterion, device
    )
    
    print(f"\n{'='*60}")
    print(f"最终结果 ({init_method}):")
    print(f"  最佳验证准确率: {best_val_acc:.4f}")
    print(f"  测试准确率: {test_acc:.4f}")
    print(f"  测试精确率: {test_precision:.4f}")
    print(f"  测试召回率: {test_recall:.4f}")
    print(f"  测试F1: {test_f1:.4f}")
    print(f"  测试AUC: {test_auc:.4f}")
    print(f"{'='*60}")
    
    return {
        'init_method': init_method,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'history': results
    }


def main():
    print("="*60)
    print("下游任务：溶解度预测")
    print("="*60)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 加载数据
    data_file = '/home/qlyu/AA_peptide/pepland/data/eval/c-Sol.txt'
    train_loader, valid_loader, test_loader = create_dataloaders(
        data_file,
        batch_size=32,
        max_samples=None
    )
    
    # 实验配置
    experiments = []
    
    # 1. Random initialization
    print("\n\n" + "="*60)
    print("实验 1: Random Initialization")
    print("="*60)
    encoder_random = ImprovedPepLand(
        atom_dim=42,
        bond_dim=14,
        fragment_dim=196,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        use_performer=True,
        use_virtual_node=True
    ).to(device)
    
    result_random = train_model(
        'random',
        encoder_random,
        device,
        train_loader,
        valid_loader,
        test_loader,
        num_epochs=30,
        lr=1e-4,
        freeze_encoder=False
    )
    experiments.append(result_random)
    
    # 2. Pretrained Contrastive
    print("\n\n" + "="*60)
    print("实验 2: Pretrained Contrastive")
    print("="*60)
    encoder_contrastive = ImprovedPepLand(
        atom_dim=42,
        bond_dim=14,
        fragment_dim=196,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        use_performer=True,
        use_virtual_node=True
    ).to(device)
    
    # 加载预训练权重（跳过不兼容的atom_encoder层）
    checkpoint = torch.load('test_results/best_pretrained_encoder_large_fixed.pt')
    pretrained_dict = checkpoint['encoder_state_dict']
    model_dict = encoder_contrastive.state_dict()

    # 过滤掉atom_encoder层（维度不匹配），只加载其他层
    filtered_dict = {k: v for k, v in pretrained_dict.items()
                     if k in model_dict and 'atom_encoder' not in k and v.shape == model_dict[k].shape}

    # 更新当前模型的参数
    model_dict.update(filtered_dict)
    encoder_contrastive.load_state_dict(model_dict)

    loaded_layers = len(filtered_dict)
    total_layers = len(pretrained_dict)
    print(f"✓ 加载对比学习预训练权重 ({loaded_layers}/{total_layers} 层)")
    print(f"  跳过atom_encoder（维度不匹配: 38→42）")
    
    result_contrastive = train_model(
        'contrastive',
        encoder_contrastive,
        device,
        train_loader,
        valid_loader,
        test_loader,
        num_epochs=30,
        lr=1e-5,  # 更低的学习率用于微调
        freeze_encoder=False
    )
    experiments.append(result_contrastive)
    
    # 3. Pretrained Supervised
    print("\n\n" + "="*60)
    print("实验 3: Pretrained Supervised")
    print("="*60)
    encoder_supervised = ImprovedPepLand(
        atom_dim=42,
        bond_dim=14,
        fragment_dim=196,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        use_performer=True,
        use_virtual_node=True
    ).to(device)
    
    # 加载预训练权重（跳过不兼容的atom_encoder层）
    checkpoint = torch.load('test_results/best_model_large.pt')
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = encoder_supervised.state_dict()

    # 过滤掉atom_encoder层（维度不匹配），只加载其他层
    filtered_dict = {k: v for k, v in pretrained_dict.items()
                     if k in model_dict and 'atom_encoder' not in k and v.shape == model_dict[k].shape}

    # 更新当前模型的参数
    model_dict.update(filtered_dict)
    encoder_supervised.load_state_dict(model_dict)

    loaded_layers = len(filtered_dict)
    total_layers = len(pretrained_dict)
    print(f"✓ 加载监督学习预训练权重 ({loaded_layers}/{total_layers} 层)")
    print(f"  跳过atom_encoder（维度不匹配: 38→42）")
    
    result_supervised = train_model(
        'supervised',
        encoder_supervised,
        device,
        train_loader,
        valid_loader,
        test_loader,
        num_epochs=30,
        lr=1e-5,
        freeze_encoder=False
    )
    experiments.append(result_supervised)
    
    # 汇总结果
    print("\n\n" + "="*60)
    print("实验结果汇总")
    print("="*60)
    print(f"{'方法':<20} {'验证准确率':<12} {'测试准确率':<12} {'测试F1':<10} {'测试AUC':<10}")
    print("-"*60)
    for exp in experiments:
        print(f"{exp['init_method']:<20} "
              f"{exp['best_val_acc']:<12.4f} "
              f"{exp['test_acc']:<12.4f} "
              f"{exp['test_f1']:<10.4f} "
              f"{exp['test_auc']:<10.4f}")
    
    # 保存结果
    import json
    with open('test_results/solubility_comparison.json', 'w') as f:
        json.dump(experiments, f, indent=2)
    
    print("\n✓ 结果已保存到 test_results/solubility_comparison.json")


if __name__ == '__main__':
    main()
