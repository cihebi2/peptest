#!/usr/bin/env python3
"""
处理 UniRef50 数据集
将 FASTA 格式转换为 CSV 格式（用于对比学习预训练）
"""

import pandas as pd
from tqdm import tqdm
import random
import os

def parse_fasta(fasta_file):
    """解析 FASTA 文件"""
    sequences = []
    current_seq = ""

    print(f"读取 FASTA 文件: {fasta_file}")
    with open(fasta_file, 'r') as f:
        for line in tqdm(f, desc="解析序列"):
            line = line.strip()
            if line.startswith('>'):
                # 保存上一个序列
                if current_seq:
                    sequences.append(current_seq)
                current_seq = ""
            else:
                # 累积序列
                current_seq += line

        # 保存最后一个序列
        if current_seq:
            sequences.append(current_seq)

    print(f"  总共读取 {len(sequences)} 条序列")
    return sequences

def filter_sequences(sequences):
    """过滤序列（只保留标准20个氨基酸）"""
    standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
    filtered = []

    print("\n过滤非标准氨基酸序列...")
    for seq in tqdm(sequences, desc="过滤"):
        # 检查是否只包含标准氨基酸
        if all(aa in standard_aa for aa in seq):
            # 检查长度 5-30
            if 5 <= len(seq) <= 30:
                filtered.append(seq)

    print(f"  过滤后保留 {len(filtered)} 条序列")
    print(f"  过滤掉 {len(sequences) - len(filtered)} 条序列")
    return filtered

def split_dataset(sequences, train_ratio=0.8, valid_ratio=0.1):
    """划分训练/验证/测试集"""
    print("\n划分数据集...")

    # 打乱
    random.seed(42)
    random.shuffle(sequences)

    n = len(sequences)
    train_size = int(n * train_ratio)
    valid_size = int(n * valid_ratio)

    train_seqs = sequences[:train_size]
    valid_seqs = sequences[train_size:train_size + valid_size]
    test_seqs = sequences[train_size + valid_size:]

    print(f"  训练集: {len(train_seqs)} 条")
    print(f"  验证集: {len(valid_seqs)} 条")
    print(f"  测试集: {len(test_seqs)} 条")

    return train_seqs, valid_seqs, test_seqs

def save_to_csv(sequences, output_file):
    """保存为 CSV 格式（smiles 列）"""
    df = pd.DataFrame({'smiles': sequences})
    df.to_csv(output_file, index=False)
    print(f"  保存到: {output_file}")

def analyze_dataset(sequences):
    """分析数据集统计信息"""
    print("\n数据集统计:")
    lengths = [len(seq) for seq in sequences]

    print(f"  序列数量: {len(sequences):,}")
    print(f"  长度范围: {min(lengths)} - {max(lengths)}")
    print(f"  平均长度: {sum(lengths)/len(lengths):.1f}")

    # 长度分布
    length_dist = {}
    for l in lengths:
        length_dist[l] = length_dist.get(l, 0) + 1

    print("\n  长度分布 (top 10):")
    for length, count in sorted(length_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {length}aa: {count:,} 条 ({100*count/len(sequences):.2f}%)")

def main():
    print("=" * 60)
    print("UniRef50 数据处理")
    print("=" * 60)

    # 输入输出路径
    fasta_file = '/home/qlyu/AA_peptide/pepland/data/uniref50_5_30_20aa.fasta'
    output_dir = '/home/qlyu/AA_peptide/pepland/data/uniref50_large'

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. 解析 FASTA
    sequences = parse_fasta(fasta_file)

    # 2. 过滤序列
    sequences = filter_sequences(sequences)

    # 3. 分析数据集
    analyze_dataset(sequences)

    # 4. 划分数据集
    train_seqs, valid_seqs, test_seqs = split_dataset(sequences)

    # 5. 保存为 CSV
    print("\n保存 CSV 文件...")
    save_to_csv(train_seqs, os.path.join(output_dir, 'train.csv'))
    save_to_csv(valid_seqs, os.path.join(output_dir, 'valid.csv'))
    save_to_csv(test_seqs, os.path.join(output_dir, 'test.csv'))

    print("\n✓ 数据处理完成!")
    print(f"  输出目录: {output_dir}")

if __name__ == '__main__':
    main()
