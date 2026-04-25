"""
快速验证脚本
在 MNIST 上运行小规模实验，验证代码框架可用性
"""
import sys
import os
# 设置 UTF-8 编码输出（Windows 兼容）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from core import get_selector
from data.datasets import ContinualDataset


def quick_test():
    """快速测试所有方法"""
    print("=" * 50)
    print("快速验证测试")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载 MNIST
    dataset = ContinualDataset('mnist', batch_size=64, num_workers=0)
    dataset.split_tasks(2, 'split')
    num_classes = len(dataset.full_dataset.classes)
    print(f"数据集: MNIST, {num_classes} 类")

    # 简单模型
    import torch.nn as nn
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    ).to(device)

    methods = ['random', 'greedy', 'csrel', 'bcsr', 'ensemble']
    memory_budget = 500

    for method_name in methods:
        print(f"\n--- 测试 {method_name} ---")
        try:
            selector = get_selector(
                method_name,
                memory_budget=memory_budget,
                device=device
            )
            train_loader = dataset.get_task_loaders(0)
            indices, weights = selector.select_coreset(
                train_loader, model, task_id=0
            )
            print(f"  选择 {len(indices)} 样本, "
                  f"权重范围 [{weights.min():.4f}, {weights.max():.4f}]")
            print(f"  [OK] {method_name} 通过")
        except Exception as e:
            print(f"  [FAIL] {method_name} 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print("验证完成!")


if __name__ == '__main__':
    quick_test()
