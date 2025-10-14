#!/usr/bin/env python3
"""
合并所有可用数据创建大规模训练集
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

print("=" * 60)
print("数据集合并脚本")
print("=" * 60)

# 数据源
data_sources = [
    '/home/qlyu/AA_peptide/pepland/data/pretrained/train.csv',
    '/home/qlyu/AA_peptide/pepland/data/pretrained/valid.csv',
    '/home/qlyu/AA_peptide/pepland/data/pretrained/test.csv',
    '/home/qlyu/AA_peptide/pepland/data/eval/nc-CPP.csv',
    '/home/qlyu/AA_peptide/pepland/data/eval/c-binding.csv',
    '/home/qlyu/AA_peptide/pepland/data/eval/nc-binding.csv',
    '/home/qlyu/AA_peptide/pepland/data/further_training/train.csv',
    '/home/qlyu/AA_peptide/pepland/data/further_training/valid.csv',
    '/home/qlyu/AA_peptide/pepland/data/further_training/test.csv',
]

# 读取所有数据
all_data = []
total_samples = 0

for source in data_sources:
    if os.path.exists(source):
        try:
            df = pd.read_csv(source)
            # 确保有smiles列
            if 'smiles' in df.columns:
                print(f"✓ {os.path.basename(source)}: {len(df)} 样本")
                all_data.append(df[['smiles']])
                total_samples += len(df)
            else:
                print(f"⚠ {os.path.basename(source)}: 无smiles列")
        except Exception as e:
            print(f"✗ {os.path.basename(source)}: 错误 - {e}")
    else:
        print(f"✗ {os.path.basename(source)}: 文件不存在")

# 合并数据
print(f"\n总计: {total_samples} 样本")
combined_df = pd.concat(all_data, ignore_index=True)

# 去重
print(f"合并前: {len(combined_df)} 样本")
combined_df = combined_df.drop_duplicates(subset=['smiles'])
print(f"去重后: {len(combined_df)} 样本（去除 {total_samples - len(combined_df)} 重复）")

# 过滤无效SMILES
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

valid_smiles = []
invalid_count = 0

print("\n验证SMILES...")
for idx, row in combined_df.iterrows():
    smiles = row['smiles']
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        valid_smiles.append(smiles)
    else:
        invalid_count += 1

    if (idx + 1) % 1000 == 0:
        print(f"  已处理: {idx + 1}/{len(combined_df)}")

print(f"有效SMILES: {len(valid_smiles)}")
print(f"无效SMILES: {invalid_count}")

# 创建最终数据集
final_df = pd.DataFrame({'smiles': valid_smiles})

# 划分数据集 (80% train, 10% valid, 10% test)
train_data, temp_data = train_test_split(final_df, test_size=0.2, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"\n数据划分:")
print(f"  训练集: {len(train_data)} 样本")
print(f"  验证集: {len(valid_data)} 样本")
print(f"  测试集: {len(test_data)} 样本")

# 保存
output_dir = 'data_large'
os.makedirs(output_dir, exist_ok=True)

train_data.to_csv(f'{output_dir}/train.csv', index=False)
valid_data.to_csv(f'{output_dir}/valid.csv', index=False)
test_data.to_csv(f'{output_dir}/test.csv', index=False)

print(f"\n✓ 数据已保存到: {output_dir}/")
print(f"  {output_dir}/train.csv")
print(f"  {output_dir}/valid.csv")
print(f"  {output_dir}/test.csv")

print("\n" + "=" * 60)
print("数据准备完成！")
print("=" * 60)
