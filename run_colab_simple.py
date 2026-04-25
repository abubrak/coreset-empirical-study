#!/usr/bin/env python
"""
Google Colab 简化运行脚本
适用于在 Google Colab 中快速运行核心集选择实验
使用方法：
1. 上传此文件到 Colab
2. 运行: !python run_colab_simple.py
"""

import os
import sys
import time
from pathlib import Path

print("=" * 70)
print("核心集选择实证研究 - Colab 快速运行脚本")
print("=" * 70)

# 1. 检查环境
print("\n[步骤 1/6] 检查环境...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
except ImportError:
    print("✗ PyTorch 未安装，正在安装...")
    os.system("!pip install -q torch torchvision")
    import torch

# 检查其他依赖
required_packages = ['numpy', 'matplotlib', 'pyyaml', 'tqdm', 'pandas', 'seaborn']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg}")
    except ImportError:
        print(f"✗ {pkg} 缺失")
        missing.append(pkg)

if missing:
    print(f"\n正在安装缺失的包: {', '.join(missing)}")
    os.system(f"!pip install -q {' '.join(missing)}")

# 2. 导入项目模块
print("\n[步骤 2/6] 导入项目模块...")
sys.path.insert(0, '.')
try:
    from core import get_selector, ContinualLearningFramework
    from core.coreset_base import reset_batch_index_counter
    from data.datasets import ContinualDataset
    print("✓ 模块导入成功")
except Exception as e:
    print(f"✗ 模块导入失败: {e}")
    sys.exit(1)

# 3. 快速验证
print("\n[步骤 3/6] 运行快速验证...")
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建简单的 MNIST 数据集
print("加载 MNIST 数据集...")
dataset = ContinualDataset('mnist', batch_size=128, num_workers=2)
dataset.split_tasks(2, 'split')
num_classes = len(dataset.full_dataset.classes)
print(f"✓ MNIST: {num_classes} 类")

# 简单模型
print("创建模型...")
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes)
).to(device)

# 测试所有方法
print("\n测试核心集选择方法...")
methods_to_test = ['random', 'greedy', 'csrel']
memory_budget = 200

results = {}
for method_name in methods_to_test:
    print(f"\n  测试 {method_name}...", end=" ")
    try:
        reset_batch_index_counter()  # 重置索引计数器
        selector = get_selector(method_name, memory_budget=memory_budget, device=device)
        train_loader = dataset.get_task_loaders(0)
        indices, weights = selector.select_coreset(train_loader, model, task_id=0)
        print(f"✓ 选择 {len(indices)} 个样本")
        results[method_name] = {'success': True, 'num_samples': len(indices)}
    except Exception as e:
        print(f"✗ 失败: {str(e)[:50]}")
        results[method_name] = {'success': False, 'error': str(e)}

# 4. 运行小型对比实验
print("\n[步骤 4/6] 运行小型对比实验...")
print("配置: MNIST, 2个任务, 5 epochs, 2种方法")

from experiments.run_comparison import ExperimentRunner

# 创建配置
config = {
    'experiment': {
        'name': 'colab_quick_test',
        'seed': 42,
        'device': 'auto'
    },
    'training': {
        'num_epochs': 5,
        'batch_size': 128,
        'learning_rate': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0001
    },
    'comparison': {
        'datasets': ['mnist'],
        'methods': ['random', 'csrel'],
        'task_type': 'split',
        'num_tasks_list': [2],
        'memory_ratios': [0.05],
        'num_runs': 1
    },
    'output': {
        'results_dir': './results'
    }
}

# 创建临时配置文件
import yaml
config_path = '/tmp/colab_config.yaml'
with open(config_path, 'w') as f:
    yaml.dump(config, f)

# 运行实验
runner = ExperimentRunner(config_path=config_path, output_dir='./results')

print("\n开始实验...")
print("预计耗时: 2-5 分钟 (CPU) / 1-2 分钟 (GPU)")

start_time = time.time()
try:
    all_results, results_file = runner.run_full_comparison()
    elapsed = time.time() - start_time
    print(f"\n✓ 实验完成！耗时: {elapsed/60:.1f} 分钟")
    print(f"✓ 结果保存至: {results_file}")
except Exception as e:
    print(f"\n✗ 实验失败: {e}")
    import traceback
    traceback.print_exc()
    all_results = None
    results_file = None

# 5. 生成分析图表
print("\n[步骤 5/6] 生成分析图表...")
if results_file and os.path.exists(results_file):
    try:
        from experiments.analysis import ResultAnalyzer
        analyzer = ResultAnalyzer(str(results_file))
        analyzer.generate_all()
        print(f"✓ 图表已保存至: {analyzer.output_dir}")

        # 显示结果
        import pandas as pd
        csv_file = analyzer.output_dir / 'summary_table.csv'
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            print("\n实验结果汇总:")
            print(df.to_string(index=False))
    except Exception as e:
        print(f"✗ 分析失败: {e}")
else:
    print("⚠ 跳过分析（未找到结果文件）")

# 6. 下载结果
print("\n[步骤 6/6] 准备下载结果...")
try:
    from google.colab import files
    if results_file:
        # 打包结果
        !zip -r results.zip results/ > /dev/null 2>&1
        files.download('results.zip')
        print("✓ 结果已打包为 results.zip 并开始下载")
    else:
        print("⚠ 没有可下载的结果文件")
except ImportError:
    print("⚠ 不在 Colab 环境中，跳过下载")
    print("⚠ 本地运行时，结果保存在 ./results/ 目录")

print("\n" + "=" * 70)
print("运行完成！")
print("=" * 70)

# 显示汇总
print("\n实验摘要:")
print(f"  - 数据集: MNIST")
print(f"  - 任务数: 2")
print(f"  - 方法: Random, CSReL")
print(f"  - Epochs: 5")
if all_results:
    for r in all_results:
        s = r['summary']
        print(f"\n  {r['method'].upper()}:")
        print(f"    - 平均准确率: {s['average_accuracy']:.4f}")
        print(f"    - 遗忘度量: {s['forgetting_measure']:.4f}")
        print(f"    - 测试准确率: {s['final_test_accuracy']:.4f}")
