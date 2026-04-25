"""
Colab 快速运行脚本
简化版的实验运行，适合在 Google Colab 中使用
"""
import os
import sys
import time
from pathlib import Path

# 确保在项目根目录
if os.path.exists('content/coreset-empirical-study'):
    os.chdir('content/coreset-empirical-study')

print("=" * 60)
print("核心集选择实证研究 - Colab 快速运行")
print("=" * 60)

# 1. 检查环境
print("\n[1/5] 检查环境...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("✗ PyTorch 未安装，请先运行：")
    print("  !pip install torch torchvision")
    sys.exit(1)

# 2. 导入模块
print("\n[2/5] 导入模块...")
try:
    from core import get_selector, ContinualLearningFramework
    from core.coreset_base import reset_batch_index_counter
    from data.datasets import ContinualDataset
    print("✓ 模块导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 3. 快速验证
print("\n[3/5] 快速验证...")
print("运行快速测试...")

import torch.nn as nn
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
dataset = ContinualDataset('mnist', batch_size=64, num_workers=0)
dataset.split_tasks(2, 'split')
num_classes = len(dataset.full_dataset.classes)

# 简单模型
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes)
).to(device)

# 测试方法
methods = ['random', 'greedy', 'csrel']
memory_budget = 500

for method_name in methods:
    try:
        reset_batch_index_counter()  # 重置索引计数器
        selector = get_selector(method_name, memory_budget=memory_budget, device=device)
        train_loader = dataset.get_task_loaders(0)
        indices, weights = selector.select_coreset(train_loader, model, task_id=0)
        print(f"  ✓ {method_name}: {len(indices)} 样本")
    except Exception as e:
        print(f"  ✗ {method_name}: {e}")

# 4. 运行实验
print("\n[4/5] 运行快速实验...")
print("数据集: MNIST, 方法: random, csrel, epochs: 10")

from experiments.run_comparison import ExperimentRunner

runner = ExperimentRunner()
runner.config['comparison'] = {
    'datasets': ['mnist'],
    'methods': ['random', 'csrel'],
    'task_type': 'split',
    'num_tasks_list': [2],
    'memory_ratios': [0.1],
    'num_runs': 1
}
runner.config['training']['num_epochs'] = 10
runner.config['training']['batch_size'] = 64

print("开始实验... (预计 5-10 分钟)")
start_time = time.time()

try:
    results, results_file = runner.run_full_comparison()
    elapsed = time.time() - start_time
    print(f"✓ 实验完成! 耗时: {elapsed/60:.1f} 分钟")
    print(f"✓ 结果保存至: {results_file}")
except Exception as e:
    print(f"✗ 实验失败: {e}")
    import traceback
    traceback.print_exc()

# 5. 生成分析
print("\n[5/5] 生成分析...")

try:
    if results_file and os.path.exists(results_file):
        from experiments.analysis import ResultAnalyzer
        analyzer = ResultAnalyzer(str(results_file))
        analyzer.generate_all()
        print("✓ 分析完成!")
        print(f"✓ 图表保存在: {analyzer.output_dir}")
    else:
        print("⚠ 跳过分析（未找到结果文件）")
except Exception as e:
    print(f"✗ 分析失败: {e}")

print("\n" + "=" * 60)
print("完成! 查看 results/ 目录获取结果")
print("=" * 60)
