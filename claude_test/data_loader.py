#!/usr/bin/env python3
"""
数据加载器 - 兼容原始 pepland 数据格式
Data Loader - Compatible with Original PepLand Format
"""

import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
import dgl
from dgl.dataloading import GraphDataLoader

# 导入原始 pepland 的数据处理函数
sys.path.append('/home/qlyu/AA_peptide/pepland/cpkt/linear_pred_atoms/code/model')
from data import Mol2HeteroGraph, MaskAtom

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class PeptideGraphDataset(IterableDataset):
    """
    肽链图数据集 - 从 SMILES 生成异构图
    Compatible with original pepland format
    """

    def __init__(self, csv_path, transform=None, log=print):
        """
        Args:
            csv_path: CSV 文件路径，包含 'smiles' 列
            transform: 数据转换函数（如 MaskAtom）
            log: 日志函数
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.log = log

        # 统计
        self.total_count = len(self.df)
        self.success_count = 0
        self.failed_count = 0

    def __len__(self):
        return len(self.df)

    def get_data(self, data):
        """逐行处理数据"""
        for i, row in data.iterrows():
            smi = row['smiles']
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    self.log(f'Invalid SMILES: {smi}')
                    self.failed_count += 1
                    continue

                # 使用原始 pepland 的图构建函数
                g = Mol2HeteroGraph(mol)

                if g.num_nodes('a') == 0:
                    self.log(f'No atoms in graph: {smi}')
                    self.failed_count += 1
                    continue

                self.success_count += 1

                # 应用转换（如数据增强、mask）
                if self.transform:
                    g = self.transform(g)

                yield g

            except Exception as e:
                self.log(f'Error processing {smi}: {e}')
                self.failed_count += 1
                continue

    def __iter__(self):
        """迭代器实现 - 支持多进程"""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # 单进程
            return self.get_data(self.df)
        else:
            # 多进程：分配数据
            per_worker = len(self.df) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.df)
            return self.get_data(self.df.iloc[start:end])


class PeptideDataModule:
    """
    数据模块 - 管理训练/验证/测试数据集
    """

    def __init__(self, data_dir, batch_size=32, num_workers=4, transform=None):
        """
        Args:
            data_dir: 数据目录，应包含 train.csv, valid.csv, test.csv
            batch_size: 批大小
            num_workers: 数据加载的进程数
            transform: 数据转换函数
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

        self.train_path = os.path.join(data_dir, 'train.csv')
        self.valid_path = os.path.join(data_dir, 'valid.csv')
        self.test_path = os.path.join(data_dir, 'test.csv')

    def setup(self):
        """设置数据集"""
        print(f"Loading datasets from {self.data_dir}")

        self.train_dataset = PeptideGraphDataset(
            self.train_path,
            transform=self.transform
        )

        self.valid_dataset = PeptideGraphDataset(
            self.valid_path,
            transform=None  # 验证集不需要数据增强
        )

        self.test_dataset = PeptideGraphDataset(
            self.test_path,
            transform=None
        )

        print(f"Train size: {len(self.train_dataset)}")
        print(f"Valid size: {len(self.valid_dataset)}")
        print(f"Test size: {len(self.test_dataset)}")

    def train_dataloader(self):
        """训练数据加载器"""
        return GraphDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        """验证数据加载器"""
        return GraphDataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        """测试数据加载器"""
        return GraphDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )


def create_sample_dataset(output_path='test_results/sample_data.csv', num_samples=100):
    """
    创建测试用的样本数据集
    """
    # 一些示例肽链 SMILES
    sample_smiles = [
        'CC(C)C[C@H](NC(=O)[C@@H]1CCCN1)C(=O)N[C@@H](C)C(=O)O',  # 简单二肽
        'CC[C@H](C)[C@H](NC(=O)[C@H](C)N)C(=O)N[C@@H](C)C(=O)O',  # 三肽
        'N[C@@H](CC(=O)O)C(=O)N[C@@H](CC(=O)O)C(=O)O',  # 谷氨酸二肽
        'N[C@@H](CC(N)=O)C(=O)N[C@@H](Cc1ccccc1)C(=O)O',  # 天冬酰胺-苯丙氨酸
        'CC(C)C[C@H](NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](C)N)C(=O)O',  # 四肽
    ]

    # 复制生成足够数量的样本
    smiles_list = []
    for i in range(num_samples):
        smiles_list.append(sample_smiles[i % len(sample_smiles)])

    # 创建 DataFrame
    df = pd.DataFrame({'smiles': smiles_list})

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Created sample dataset: {output_path}")

    return output_path


if __name__ == "__main__":
    """测试数据加载器"""

    print("=" * 60)
    print("测试数据加载器 / Test Data Loader")
    print("=" * 60)

    # 创建样本数据
    print("\n1. 创建样本数据集...")
    sample_path = create_sample_dataset(num_samples=50)

    # 测试数据集
    print("\n2. 测试 PeptideGraphDataset...")
    dataset = PeptideGraphDataset(sample_path)
    print(f"   Dataset size: {len(dataset)}")

    # 测试迭代
    print("\n3. 测试数据迭代...")
    count = 0
    for g in dataset:
        count += 1
        if count == 1:
            print(f"   First graph: {g}")
            print(f"   Atom nodes: {g.num_nodes('a')}")
            print(f"   Fragment nodes: {g.num_nodes('p')}")
            print(f"   Atom features: {g.nodes['a'].data['f'].shape}")
            print(f"   Fragment features: {g.nodes['p'].data['f'].shape}")
        if count >= 5:
            break

    print(f"   ✓ Iterated {count} graphs")
    print(f"   Success: {dataset.success_count}, Failed: {dataset.failed_count}")

    # 测试带 transform 的数据集
    print("\n4. 测试数据转换 (MaskAtom)...")
    transform = MaskAtom(
        num_atom_type=119,
        num_edge_type=5,
        mask_rate=0.15,
        mask_edge=True,
        mask_fragment=True
    )

    dataset_with_mask = PeptideGraphDataset(sample_path, transform=transform)

    for g in dataset_with_mask:
        if 'mask' in g.nodes['a'].data:
            print(f"   ✓ Masked atoms: {g.nodes['a'].data['mask'].sum().item()}")
        if 'mask' in g.nodes['p'].data:
            print(f"   ✓ Masked fragments: {g.nodes['p'].data['mask'].sum().item()}")
        break

    # 测试 DataLoader
    print("\n5. 测试 GraphDataLoader...")
    loader = GraphDataLoader(
        dataset,
        batch_size=8,
        num_workers=0  # 单进程测试
    )

    for batch_g in loader:
        print(f"   Batch graph: {batch_g}")
        print(f"   Batch size: {batch_g.batch_size}")
        print(f"   Total atom nodes: {batch_g.num_nodes('a')}")
        print(f"   Total fragment nodes: {batch_g.num_nodes('p')}")
        break

    print("\n" + "=" * 60)
    print("✓ 数据加载器测试完成！")
    print("=" * 60)
