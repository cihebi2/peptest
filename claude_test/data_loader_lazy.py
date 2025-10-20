#!/usr/bin/env python3
"""
懒加载数据加载器 - 按需从SMILES创建图

优化版本，避免预加载所有数据到内存
"""

import torch
import dgl
import pandas as pd
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader


# 原子特征提取
def get_atom_features(atom):
    """提取原子特征 (42维)"""
    # 原子类型 one-hot (常见9种元素 + 其他)
    atom_types = [6, 7, 8, 16, 9, 15, 17, 35, 53]  # C, N, O, S, F, P, Cl, Br, I
    atom_type_encoding = [0] * 10
    atomic_num = atom.GetAtomicNum()
    if atomic_num in atom_types:
        atom_type_encoding[atom_types.index(atomic_num)] = 1
    else:
        atom_type_encoding[9] = 1

    # 度数 (0-5+)
    degree = min(atom.GetDegree(), 5)
    degree_encoding = [0] * 6
    degree_encoding[degree] = 1

    # 形式电荷 (-2, -1, 0, 1, 2)
    charge = atom.GetFormalCharge()
    charge_encoding = [0] * 5
    charge_idx = max(-2, min(2, charge)) + 2
    charge_encoding[charge_idx] = 1

    # 手性 (4种)
    chiral_tag = int(atom.GetChiralTag())
    chiral_encoding = [0] * 4
    if chiral_tag < 4:
        chiral_encoding[chiral_tag] = 1

    # 氢原子数 (0-4+)
    num_hs = min(atom.GetTotalNumHs(), 4)
    h_encoding = [0] * 5
    h_encoding[num_hs] = 1

    # 杂化方式 (5种)
    hybridization = int(atom.GetHybridization())
    hybrid_encoding = [0] * 6
    if hybridization <= 5:
        hybrid_encoding[hybridization] = 1

    # 芳香性
    aromatic = [1 if atom.GetIsAromatic() else 0]

    # 原子质量 (归一化)
    mass = [atom.GetMass() * 0.01]

    features = (atom_type_encoding + degree_encoding + charge_encoding +
                chiral_encoding + h_encoding + hybrid_encoding +
                aromatic + mass)

    return features


# 键特征提取
def get_bond_features(bond):
    """提取键特征 (14维)"""
    if bond is None:
        return [1] + [0] * 13

    # 键类型 (4种 + None)
    bond_type = bond.GetBondType()
    type_encoding = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
    ]

    # 共轭
    conjugated = [1 if bond.GetIsConjugated() else 0]

    # 在环中
    in_ring = [1 if bond.IsInRing() else 0]

    # 立体化学 (6种)
    stereo = int(bond.GetStereo())
    stereo_encoding = [0] * 7
    if stereo < 7:
        stereo_encoding[stereo] = 1

    features = [0] + type_encoding + conjugated + in_ring + stereo_encoding

    return features


def smiles_to_graph(smiles):
    """
    将SMILES字符串转换为DGL图

    Args:
        smiles: SMILES字符串

    Returns:
        DGL图对象，如果转换失败则返回None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 标准化分子
        mol = Chem.AddHs(mol)

        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return None

        # 提取原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            feat = get_atom_features(atom)
            atom_features.append(feat)

        atom_features = torch.FloatTensor(atom_features)

        # 提取键信息
        edges_src = []
        edges_dst = []
        bond_features = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # 添加双向边
            edges_src.extend([i, j])
            edges_dst.extend([j, i])

            # 键特征（两个方向相同）
            feat = get_bond_features(bond)
            bond_features.extend([feat, feat])

        # 如果没有键，创建自环
        if len(edges_src) == 0:
            edges_src = list(range(num_atoms))
            edges_dst = list(range(num_atoms))
            bond_features = [get_bond_features(None)] * num_atoms

        bond_features = torch.FloatTensor(bond_features)

        # 创建DGL图（在CPU上）
        g = dgl.graph((edges_src, edges_dst), num_nodes=num_atoms)
        g.ndata['atom_feat'] = atom_features
        g.ndata['node_type'] = torch.zeros(num_atoms, dtype=torch.long)
        g.edata['bond_feat'] = bond_features

        return g

    except Exception as e:
        return None


class PeptideDatasetLazy(Dataset):
    """
    肽段数据集 - 懒加载版本

    不预处理所有数据，而是在 __getitem__ 时动态创建图
    """

    def __init__(self, csv_path, max_samples=None):
        """
        Args:
            csv_path: CSV文件路径
            max_samples: 最大样本数（用于调试）
        """
        self.df = pd.read_csv(csv_path)

        if max_samples is not None:
            self.df = self.df.head(max_samples)

        print(f"加载数据: {csv_path}")
        print(f"  总样本数: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        smiles = self.df.iloc[idx]['smiles']
        graph = smiles_to_graph(smiles)
        return graph


def collate_fn(batch):
    """批处理函数"""
    # 过滤None值
    batch = [g for g in batch if g is not None]
    if len(batch) == 0:
        return None
    return dgl.batch(batch)


def create_dataloaders(data_dir, batch_size=32, max_samples=None, num_workers=4):
    """
    创建数据加载器 - 懒加载版本

    Args:
        data_dir: 数据目录（包含train.csv, valid.csv, test.csv）
        batch_size: 批大小
        max_samples: 最大样本数（用于调试）
        num_workers: 数据加载工作进程数

    Returns:
        {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
    """
    import os

    train_dataset = PeptideDatasetLazy(
        os.path.join(data_dir, 'train.csv'),
        max_samples=max_samples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )

    return {
        'train': train_loader,
    }


if __name__ == '__main__':
    # 测试数据加载
    print("测试懒加载数据加载器...")

    data_dir = '/home/qlyu/AA_peptide/pepland/data/uniref50_smiles'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataloaders = create_dataloaders(
        data_dir,
        batch_size=16,
        max_samples=100,
        num_workers=4
    )

    # 测试一个批次
    train_loader = dataloaders['train']
    batch = next(iter(train_loader))

    if batch is not None:
        print(f"\n✓ 批次创建成功!")
        print(f"  节点数: {batch.num_nodes()}")
        print(f"  边数: {batch.num_edges()}")
        print(f"  原子特征形状: {batch.ndata['atom_feat'].shape}")
        print(f"  键特征形状: {batch.edata['bond_feat'].shape}")

        # 测试移动到GPU
        batch = batch.to(device)
        print(f"\n✓ 成功移动到 {device}")
    else:
        print("✗ 批次创建失败")
