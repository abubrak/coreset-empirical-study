"""
调试脚本 - 检查数据维度和模型匹配
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from data.datasets import ContinualDataset

print("="*60)
print("数据维度调试")
print("="*60)

# 创建数据集
dataset = ContinualDataset('mnist', batch_size=32, num_workers=0)
dataset.split_tasks(2, 'split')

# 获取第一个任务的数据加载器
train_loader = dataset.get_task_loaders(0)

# 检查第一个批次
for batch in train_loader:
    if len(batch) == 3:
        x, y, idx = batch
        print(f"\n批次格式: (x, y, idx)")
        print(f"  x shape: {x.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  idx: {idx[:5]}...")
    elif len(batch) == 2:
        x, y = batch
        print(f"\n批次格式: (x, y)")
        print(f"  x shape: {x.shape}")
        print(f"  y shape: {y.shape}")

    # 测试简单模型
    print(f"\n创建模型...")

    # 计算输入维度
    input_dim = x.shape[1] * x.shape[2] * x.shape[3] if len(x.shape) == 4 else x.shape[1]
    print(f"  计算的输入维度: {input_dim}")
    print(f"  实际数据形状: {x.shape}")

    # 测试展平
    x_flat = x.view(x.size(0), -1)
    print(f"  展平后形状: {x_flat.shape}")

    num_classes = len(dataset.full_dataset.classes)
    print(f"  类别数: {num_classes}")

    # 创建匹配的模型
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )

    # 测试前向传播
    try:
        output = model(x)
        print(f"\n✓ 模型前向传播成功!")
        print(f"  输入: {x.shape}")
        print(f"  输出: {output.shape}")
    except Exception as e:
        print(f"\n✗ 模型前向传播失败: {e}")

        # 尝试手动展平
        x_manual_flat = x.reshape(x.size(0), -1)
        print(f"  手动展平形状: {x_manual_flat.shape}")

    break

# 检查数据集信息
print(f"\n数据集信息:")
print(f"  总样本数: {len(dataset.full_dataset)}")
print(f"  类别数: {len(dataset.full_dataset.classes)}")
print(f"  任务数: {len(dataset.tasks)}")
print(f"  任务0样本数: {len(dataset.tasks[0])}")

# 测试模型创建
print(f"\n\n模型创建测试:")
try:
    from experiments.run_comparison import create_model
    model = create_model('mnist', 10)
    print(f"✓ create_model 成功")

    # 测试前向传播
    with torch.no_grad():
        output = model(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {output.shape}")

except Exception as e:
    print(f"✗ create_model 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
