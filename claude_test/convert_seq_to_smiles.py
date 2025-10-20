#!/usr/bin/env python3
"""
将氨基酸序列转换为肽段 SMILES 表示

使用 RDKit 的 Chem.MolFromSequence() 将标准氨基酸序列转换为 SMILES
"""

import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import os

# 标准20个氨基酸
STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')


def sequence_to_smiles(sequence):
    """
    将氨基酸序列转换为 SMILES

    Args:
        sequence: 氨基酸序列字符串 (如 'MKLAL')

    Returns:
        SMILES 字符串，失败返回 None
    """
    try:
        # 检查是否只包含标准氨基酸
        if not all(aa in STANDARD_AA for aa in sequence):
            return None

        # 使用 RDKit 将序列转换为分子
        mol = Chem.MolFromSequence(sequence)

        if mol is None:
            return None

        # 转换为 SMILES
        smiles = Chem.MolToSmiles(mol)

        return smiles

    except Exception as e:
        return None


def convert_csv(input_file, output_file):
    """
    转换 CSV 文件中的序列为 SMILES

    Args:
        input_file: 输入 CSV（包含 'smiles' 列，实际是序列）
        output_file: 输出 CSV（转换后的真实 SMILES）
    """
    print(f"\n处理: {input_file}")

    # 读取数据
    df = pd.read_csv(input_file)
    original_count = len(df)
    print(f"  原始序列数: {original_count:,}")

    # 转换序列为 SMILES
    smiles_list = []
    failed_count = 0

    for seq in tqdm(df['smiles'], desc="  转换"):
        smiles = sequence_to_smiles(seq)

        if smiles is not None:
            smiles_list.append(smiles)
        else:
            failed_count += 1

    # 创建新的 DataFrame
    new_df = pd.DataFrame({'smiles': smiles_list})

    # 保存
    new_df.to_csv(output_file, index=False)

    success_count = len(smiles_list)
    print(f"  成功转换: {success_count:,} 条")
    print(f"  失败: {failed_count:,} 条")
    print(f"  成功率: {100*success_count/original_count:.2f}%")
    print(f"  保存到: {output_file}")


def main():
    print("=" * 60)
    print("氨基酸序列 → SMILES 转换")
    print("=" * 60)

    # 输入输出路径
    input_dir = '/home/qlyu/AA_peptide/pepland/data/uniref50_large'
    output_dir = '/home/qlyu/AA_peptide/pepland/data/uniref50_smiles'

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 转换三个文件
    for split in ['train', 'valid', 'test']:
        input_file = os.path.join(input_dir, f'{split}.csv')
        output_file = os.path.join(output_dir, f'{split}.csv')

        convert_csv(input_file, output_file)

    print("\n" + "=" * 60)
    print("✓ 转换完成！")
    print(f"输出目录: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
