#!/usr/bin/env python
"""
测试 Ensemble 方法是否正常工作
"""
import sys
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import get_selector
from data.datasets import ContinualDataset


def test_ensemble_selector():
    """测试 Ensemble 选择器"""
    print("\n" + "="*60)
    print("🧪 测试 Ensemble 方法")
    print("="*60 + "\n")

    # 检查 CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 设备: {device}")

    # 加载数据
    print("\n📊 加载 MNIST 数据...")
    dataset = ContinualDataset(
        dataset_name='mnist',
        batch_size=128,
        num_workers=0
    )
    dataset.split_tasks(num_tasks=5, task_type='split')

    # 获取第一个任务的训练数据
    print("🔑 获取任务 0 数据...")
    train_loader = dataset.get_task_loaders(0)

    # 创建模型
    print("🤖 创建模型...")
    from experiments.run_comparison import create_model
    model = create_model('mnist', num_classes=10).to(device)

    # 创建 Ensemble 选择器
    print("🎯 创建 Ensemble 选择器...")
    selector = get_selector(
        'ensemble',
        memory_budget=600,  # 小规模快速测试
        device=device,
        switch_threshold=0.5,
        early_ratio=0.3
    )

    # 运行选择
    print("\n🚀 运行核心集选择...")
    print(f"  • 数据集大小: {len(train_loader.dataset)}")
    print(f"  • 核心集预算: {selector.memory_budget}")
    print(f"  • Task ID: 0")
    print(f"  • 总任务数: 5\n")

    try:
        indices, weights = selector.select_coreset(
            train_loader,
            model,
            task_id=0,
            previous_coresets=None,
            total_tasks=5
        )

        print("\n✅ 选择成功！")
        print(f"  • 选择的样本数: {len(indices)}")
        print(f"  • 权重形状: {weights.shape}")
        print(f"  • 前 10 个索引: {indices[:10]}")
        print(f"  • 权重范围: [{weights.min():.4f}, {weights.max():.4f}]")

        # 验证索引有效性
        max_idx = max(indices) if indices else -1
        dataset_size = len(train_loader.dataset)

        if max_idx < dataset_size:
            print(f"\n✅ 索引验证通过")
            print(f"  • 最大索引: {max_idx}")
            print(f"  • 数据集大小: {dataset_size}")
            print(f"  • 所有索引都在有效范围内")
        else:
            print(f"\n❌ 索引验证失败")
            print(f"  • 最大索引: {max_idx}")
            print(f"  • 数据集大小: {dataset_size}")
            return False

        return True

    except Exception as e:
        print(f"\n❌ 选择失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ensemble_with_previous():
    """测试 Ensemble 方法处理历史核心集"""
    print("\n" + "="*60)
    print("🧪 测试 Ensemble 方法（有历史核心集）")
    print("="*60 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    dataset = ContinualDataset('mnist', batch_size=128, num_workers=0)
    dataset.split_tasks(num_tasks=5, task_type='split')

    # 创建模型
    from experiments.run_comparison import create_model
    model = create_model('mnist', num_classes=10).to(device)

    # 创建选择器
    selector = get_selector('ensemble', memory_budget=600, device=device)

    # 模拟有历史核心集的情况
    train_loader_0 = dataset.get_task_loaders(0)
    train_loader_1 = dataset.get_task_loaders(1)

    # 为任务 0 选择核心集
    print("📊 任务 0: 选择核心集...")
    indices_0, weights_0 = selector.select_coreset(
        train_loader_0, model, task_id=0, previous_coresets=None, total_tasks=5
    )
    print(f"  ✅ 选择了 {len(indices_0)} 个样本")

    # 为任务 1 选择核心集（有历史）
    print("\n📊 任务 1: 选择核心集（有历史）...")
    previous_coresets = [(indices_0, weights_0)]

    try:
        indices_1, weights_1 = selector.select_coreset(
            train_loader_1, model, task_id=1, previous_coresets=previous_coresets, total_tasks=5
        )

        print(f"  ✅ 选择了 {len(indices_1)} 个样本")
        print(f"  • 前 10 个索引: {indices_1[:10]}")

        # 验证索引
        max_idx = max(indices_1) if indices_1 else -1
        dataset_size = len(train_loader_1.dataset)

        if max_idx < dataset_size:
            print(f"\n✅ 索引验证通过（任务 1）")
            return True
        else:
            print(f"\n❌ 索引越界（任务 1）")
            print(f"  • 最大索引: {max_idx}")
            print(f"  • 数据集大小: {dataset_size}")
            return False

    except Exception as e:
        print(f"\n❌ 选择失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n🔬 Ensemble 方法测试套件\n")

    # 测试 1: 基本功能
    test1_pass = test_ensemble_selector()

    # 测试 2: 有历史核心集
    test2_pass = test_ensemble_with_previous()

    # 总结
    print("\n" + "="*60)
    print("📋 测试结果总结")
    print("="*60)
    print(f"  基本功能测试: {'✅ 通过' if test1_pass else '❌ 失败'}")
    print(f"  历史核心集测试: {'✅ 通过' if test2_pass else '❌ 失败'}")
    print("="*60 + "\n")

    if test1_pass and test2_pass:
        print("🎉 所有测试通过！Ensemble 方法可以正常使用。\n")
        return 0
    else:
        print("⚠️  部分测试失败，请检查错误信息。\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
