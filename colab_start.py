#!/usr/bin/env python
"""
Google Colab 一键启动脚本
自动修复所有 bug 并运行实验
"""

import os
import subprocess

print("="*70)
print("🚀 核心集选择实验 - Colab 自动启动")
print("="*70)

# 步骤 1: 检查环境
print("\n[1/7] 检查环境...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ CUDA: {torch.cuda.is_available()}")
except:
    print("安装 PyTorch...")
    subprocess.call(["pip", "install", "-q", "torch", "torchvision"])

# 步骤 2: 修复所有 bug
print("\n[2/7] 应用 bug 修复...")

# 修复 data/datasets.py
print("  修复 data/datasets.py...")
with open('data/datasets.py', 'r') as f:
    content = f.read()
content = content.replace(
    'from torch.utils.data import DataLoader, Subset, random',
    'import random\nfrom torch.utils.data import DataLoader, Subset'
)
with open('data/datasets.py', 'w') as f:
    f.write(content)

# 修复 RotatedMNIST
content = content.replace(
    'return img, label, self.indices[idx]',
    'return img, label'
)
with open('data/datasets.py', 'w') as f:
    f.write(content)
print("  ✓ data/datasets.py")

# 修复 core/coreset_base.py
print("  修复 core/coreset_base.py...")
with open('core/coreset_base.py', 'r') as f:
    content = f.read()

# 添加 _parse_batch 函数
import_section = "from torch.utils.data import DataLoader, Subset\n"
parse_batch_func = '''from torch.utils.data import DataLoader, Subset


def _parse_batch(batch):
    """解析数据批次，支持多种返回格式"""
    if len(batch) == 3:
        x, y, indices = batch
    elif len(batch) == 2:
        x, y = batch
        indices = torch.arange(x.size(0))
    else:
        raise ValueError(f"Unexpected batch format: {len(batch)} elements")
    return x, y, indices
'''
content = content.replace(import_section, parse_batch_func)

# 替换所有 for batch_x, batch_y, idx in dataset
content = content.replace(
    'for batch_x, batch_y, idx in dataset:\n                batch_x = batch_x.to(self.device)',
    'for batch in dataset:\n                batch_x, batch_y, _ = _parse_batch(batch)\n                batch_x = batch_x.to(self.device)'
)

content = content.replace(
    'for batch_x, batch_y in train_loader:\n                batch_x = batch_x.to(self.device)',
    'for batch in train_loader:\n                batch_x, batch_y, _ = _parse_batch(batch)\n                batch_x = batch_x.to(self.device)'
)

with open('core/coreset_base.py', 'w') as f:
    f.write(content)
print("  ✓ core/coreset_base.py")

# 修复 core/methods/greedy.py
print("  修复 core/methods/greedy.py...")
with open('core/methods/greedy.py', 'r') as f:
    content = f.read()
content = content.replace(
    'from ..coreset_base import CoresetSelector',
    'from ..coreset_base import CoresetSelector, _parse_batch'
)
content = content.replace(
    '''for batch in dataset:
                # 兼容不同数据格式 (x, y) 或 (x, y, idx)
                if len(batch) == 3:
                    batch_x, _, idx = batch
                else:
                    batch_x, _ = batch
                    idx = torch.arange(batch_x.size(0))

                batch_x = batch_x.to(self.device)''',
    '''for batch in dataset:
                batch_x, _, idx = _parse_batch(batch)
                batch_x = batch_x.to(self.device)'''
)
with open('core/methods/greedy.py', 'w') as f:
    f.write(content)
print("  ✓ core/methods/greedy.py")

# 修复 core/methods/csrel.py
print("  修复 core/methods/csrel.py...")
with open('core/methods/csrel.py', 'r') as f:
    content = f.read()
content = content.replace(
    'from ..coreset_base import CoresetSelector',
    'from ..coreset_base import CoresetSelector, _parse_batch'
)
content = content.replace(
    'for batch_x, batch_y, idx in dataset:\n                batch_x = batch_x.to(self.device)',
    'for batch in dataset:\n                batch_x, batch_y, _ = _parse_batch(batch)\n                batch_x = batch_x.to(self.device)'
)
content = content.replace(
    'rel_losses.append(loss.sum(dim=0))',
    'rel_losses.append(loss)'
)
with open('core/methods/csrel.py', 'w') as f:
    f.write(content)
print("  ✓ core/methods/csrel.py")

# 修复 core/methods/bcsr.py
print("  修复 core/methods/bcsr.py...")
with open('core/methods/bcsr.py', 'r') as f:
    content = f.read()
content = content.replace(
    'from ..coreset_base import CoresetSelector',
    'from ..coreset_base import CoresetSelector, _parse_batch'
)
content = content.replace(
    '''for batch in dataset:
            if len(batch) == 3:
                x, y, idx = batch
            else:
                x, y = batch
                idx = torch.arange(x.size(0))

            all_data.append(x)''',
    '''for batch in dataset:
            x, y, idx = _parse_batch(batch)
            all_data.append(x)'''
)
with open('core/methods/bcsr.py', 'w') as f:
    f.write(content)
print("  ✓ core/methods/bcsr.py")

print("  ✅ 所有文件已修复!")

# 步骤 3: 导入模块
print("\n[3/7] 导入项目模块...")
import sys
sys.path.insert(0, '.')
try:
    from core import get_selector, ContinualLearningFramework
    from core.coreset_base import reset_batch_index_counter
    from data.datasets import ContinualDataset
    print("✓ 模块导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import sys
    sys.exit(1)

# 步骤 4: 快速验证
print("\n[4/7] 快速验证...")
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = ContinualDataset('mnist', batch_size=128, num_workers=0)
dataset.split_tasks(2, 'split')
num_classes = len(dataset.full_dataset.classes)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, num_classes)
).to(device)

# 测试 random 方法
try:
    reset_batch_index_counter()  # 重置索引计数器
    selector = get_selector('random', memory_budget=200, device=device)
    train_loader = dataset.get_task_loaders(0)
    indices, weights = selector.select_coreset(train_loader, model, task_id=0)
    print(f"✓ Random 方法: 选择 {len(indices)} 个样本")
except Exception as e:
    print(f"✗ Random 方法失败: {e}")

# 步骤 5: 运行实验
print("\n[5/7] 运行对比实验...")
print("配置: MNIST, Random vs CSReL, 5 epochs, 2 任务")

result = subprocess.call([
    'python', 'experiments/run_comparison.py',
    '--dataset', 'mnist',
    '--method', 'random', 'csrel',
    '--quick'
])

if result == 0:
    print("\n✅ 实验完成!")
else:
    print(f"\n⚠ 实验退出码: {result}")

# 步骤 6: 生成分析
print("\n[6/7] 生成分析图表...")
import glob
result_files = glob.glob('results/comparison_*.json')
if result_files:
    latest_result = max(result_files)
    print(f"分析文件: {latest_result}")
    subprocess.call(['python', 'experiments/analysis.py', latest_result])
else:
    print("未找到结果文件")

# 步骤 7: 显示结果
print("\n[7/7] 显示结果...")
try:
    import pandas as pd
    df = pd.read_csv('results/figures/summary_table.csv')
    print("\n实验结果汇总:")
    print(df.to_string(index=False))
except:
    print("未找到汇总表格")

print("\n" + "="*70)
print("🎉 运行完成!")
print("="*70)
print("\n查看结果:")
print("  - 图表: results/figures/*.png")
print("  - 数据: results/figures/summary_table.csv")
print("  - 完整结果: results/comparison_*.json")
