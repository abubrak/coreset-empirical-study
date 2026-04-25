"""
核心集选择方法对比实验运行器
统一框架，支持多方法、多数据集、多配置的批量对比实验
"""
import os
import time
import json
import yaml
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# 项目内部导入
import sys
# 使用绝对路径的父目录，确保跨平台兼容
_current_file = Path(__file__).absolute()
_project_root = _current_file.parent.parent
sys.path.insert(0, str(_project_root))

from core import get_selector, ContinualLearningFramework
from data.datasets import ContinualDataset


# ==================== 模型定义 ====================

class SimpleCNN(nn.Module):
    """轻量 CNN，适用于 MNIST / CIFAR-10"""
    def __init__(self, num_classes=10, input_channels=1, input_size=28):
        """
        Args:
            num_classes: 分类类别数
            input_channels: 输入通道数 (1=灰度, 3=RGB)
            input_size: 输入图像尺寸 (MNIST=28, CIFAR=32)
        """
        super().__init__()
        self.input_size = input_size

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # 计算特征图尺寸: input_size -> Pool1 -> Pool2
        # 每次MaxPool2d(2)使尺寸减半
        feature_size = input_size // 4  # 两次MaxPool
        self.feature_dim = 64 * feature_size * feature_size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


class ResNetLike(nn.Module):
    """小型 ResNet 风格网络，适用于 CIFAR-100"""
    def __init__(self, num_classes=100, input_channels=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_c, out_c, blocks, stride=1):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ))
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def get_features(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def create_model(dataset_name, num_classes):
    """根据数据集创建对应模型"""
    if dataset_name == 'mnist':
        return SimpleCNN(num_classes=num_classes, input_channels=1, input_size=28)
    elif dataset_name == 'cifar10':
        return SimpleCNN(num_classes=num_classes, input_channels=3, input_size=32)
    elif dataset_name == 'cifar100':
        return ResNetLike(num_classes=num_classes, input_channels=3)
    else:
        return SimpleCNN(num_classes=num_classes, input_channels=3, input_size=32)


# ==================== 实验运行器 ====================

class ExperimentRunner:
    """统一实验运行器"""

    def __init__(self, config_path: str = None, output_dir: str = None):
        """
        Args:
            config_path: 实验配置文件路径
            output_dir: 结果输出目录
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir or self.config.get('output', {}).get(
            'results_dir', './results'
        ))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)

        # 设备
        device_str = self.config.get('experiment', {}).get('device', 'auto')
        if device_str == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_str)

        self.seed = self.config.get('experiment', {}).get('seed', 42)

    def _load_config(self, config_path):
        """加载配置文件"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def set_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run_single_experiment(
        self,
        dataset_name: str,
        method_name: str,
        task_type: str = 'split',
        num_tasks: int = 5,
        memory_ratio: float = 0.1,
        num_epochs: int = 50,
        run_id: int = 0
    ) -> dict:
        """
        运行单组实验

        Args:
            dataset_name: 数据集名称
            method_name: 方法名称
            task_type: 任务类型
            num_tasks: 任务数量
            memory_ratio: 核心集占比
            num_epochs: 训练轮数
            run_id: 运行编号

        Returns:
            实验结果字典
        """
        seed = self.seed + run_id
        self.set_seed(seed)

        print(f"\n{'='*60}")
        print(f"实验: {dataset_name} | {method_name} | ratio={memory_ratio}")
        print(f"任务: {num_tasks} x {task_type} | epochs={num_epochs} | run={run_id}")
        print(f"设备: {self.device} | seed={seed}")
        print(f"{'='*60}\n")

        # 加载数据
        training_cfg = self.config.get('training', {})
        batch_size = training_cfg.get('batch_size', 128)

        continual_dataset = ContinualDataset(
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_workers=0  # Windows/Colab 兼容
        )
        continual_dataset.split_tasks(num_tasks, task_type)

        num_classes = len(continual_dataset.full_dataset.classes)
        train_set_size = len(continual_dataset.full_dataset)
        memory_budget = max(1, int(train_set_size * memory_ratio))

        print(f"数据集: {train_set_size} 样本, {num_classes} 类")
        print(f"核心集预算: {memory_budget} ({memory_ratio*100:.1f}%)\n")

        # 重置批次索引计数器
        from core.coreset_base import reset_batch_index_counter
        reset_batch_index_counter()

        # 创建模型
        model = create_model(dataset_name, num_classes).to(self.device)

        # 创建选择器
        methods_cfg = self.config.get('methods', {})
        method_cfg = methods_cfg.get(method_name, {})
        method_params = method_cfg.get('params', {})

        selector = get_selector(
            method_name,
            memory_budget=memory_budget,
            device=self.device,
            **method_params
        )

        # 持续学习框架
        framework = ContinualLearningFramework(
            model=model,
            device=self.device,
            optimizer_kwargs={
                'lr': training_cfg.get('learning_rate', 0.01),
                'momentum': training_cfg.get('momentum', 0.9),
                'weight_decay': training_cfg.get('weight_decay', 0.0001)
            }
        )

        # 实验记录
        results = {
            'dataset': dataset_name,
            'method': method_name,
            'task_type': task_type,
            'num_tasks': num_tasks,
            'memory_ratio': memory_ratio,
            'memory_budget': memory_budget,
            'num_epochs': num_epochs,
            'run_id': run_id,
            'seed': seed,
            'device': str(self.device),
            'per_task_results': [],
            'task_accuracies': {},   # task_id -> [acc_after_task_0, acc_after_task_1, ...]
        }

        previous_coresets = []
        test_loader = continual_dataset.get_test_loader()

        for task_id in range(num_tasks):
            print(f"\n--- 任务 {task_id+1}/{num_tasks} ---")

            # 获取当前任务训练数据
            train_loader = continual_dataset.get_task_loaders(task_id)

            # 选择核心集
            select_start = time.time()
            coreset_indices, coreset_weights = selector.select_coreset(
                train_loader, model, task_id, previous_coresets
            )
            select_time = time.time() - select_start

            print(f"核心集选择: {len(coreset_indices)} 样本, 耗时 {select_time:.2f}s")

            # 保存核心集
            previous_coresets.append(coreset_indices)

            # 创建核心集 DataLoader
            coreset_loader = selector.get_coreset_subset(
                train_loader, coreset_indices
            )

            # 在核心集 + 当前任务数据上训练
            train_start = time.time()
            train_metrics = framework.train_task(
                train_loader=coreset_loader,
                num_epochs=num_epochs
            )
            train_time = time.time() - train_start

            # 评估所有已见任务
            task_results = {
                'task_id': task_id,
                'coreset_size': len(coreset_indices),
                'select_time': select_time,
                'train_time': train_time,
                'final_train_loss': train_metrics['train_losses'][-1],
                'final_train_acc': train_metrics['train_accuracy'][-1],
                'per_seen_task_acc': {}
            }

            # 在所有已见任务上评估
            for prev_task in range(task_id + 1):
                prev_loader = continual_dataset.get_task_loaders(prev_task)
                acc, per_class = framework.evaluate(prev_loader)
                task_results['per_seen_task_acc'][prev_task] = acc

                if prev_task not in results['task_accuracies']:
                    results['task_accuracies'][prev_task] = []
                results['task_accuracies'][prev_task].append(acc)

                print(f"  任务 {prev_task} 准确率: {acc:.4f}")

            # 全局测试集评估
            test_acc, _ = framework.evaluate(test_loader)
            task_results['test_accuracy'] = test_acc
            print(f"  全局测试准确率: {test_acc:.4f}")

            results['per_task_results'].append(task_results)

        # 计算汇总指标
        results['summary'] = self._compute_summary(results)
        print(f"\n汇总: AA={results['summary']['average_accuracy']:.4f}, "
              f"Forgetting={results['summary']['forgetting_measure']:.4f}")

        return results

    def run_full_comparison(self):
        """运行完整对比实验"""
        comp = self.config.get('comparison', {})
        datasets = comp.get('datasets', ['mnist', 'cifar10'])
        methods = comp.get('methods', ['random', 'greedy', 'csrel', 'bcsr', 'ensemble'])
        task_type = comp.get('task_type', 'split')
        num_tasks_list = comp.get('num_tasks_list', [5])
        memory_ratios = comp.get('memory_ratios', [0.1])
        num_runs = comp.get('num_runs', 3)

        training_cfg = self.config.get('training', {})
        num_epochs = training_cfg.get('num_epochs', 50)

        all_results = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.output_dir / f'comparison_{timestamp}.json'

        total_exp = len(datasets) * len(methods) * len(num_tasks_list) * len(memory_ratios) * num_runs
        print(f"总计 {total_exp} 组实验\n")

        exp_count = 0
        for dataset_name in datasets:
            for method_name in methods:
                for num_tasks in num_tasks_list:
                    for ratio in memory_ratios:
                        for run_id in range(num_runs):
                            exp_count += 1
                            print(f"\n[{exp_count}/{total_exp}]", end="")

                            try:
                                result = self.run_single_experiment(
                                    dataset_name=dataset_name,
                                    method_name=method_name,
                                    task_type=task_type,
                                    num_tasks=num_tasks,
                                    memory_ratio=ratio,
                                    num_epochs=num_epochs,
                                    run_id=run_id
                                )
                                all_results.append(result)

                                # 增量保存
                                with open(results_file, 'w', encoding='utf-8') as f:
                                    json.dump(all_results, f, indent=2, ensure_ascii=False)

                            except Exception as e:
                                print(f"实验失败: {e}")
                                import traceback
                                traceback.print_exc()

        print(f"\n所有实验完成! 结果保存至: {results_file}")
        return all_results, results_file

    def _compute_summary(self, results: dict) -> dict:
        """计算汇总指标"""
        task_accs = results['task_accuracies']

        # Average Accuracy (AA): 最终模型在所有任务上的平均准确率
        final_accs = []
        for task_id, acc_list in task_accs.items():
            final_accs.append(acc_list[-1])
        avg_acc = np.mean(final_accs) if final_accs else 0.0

        # Forgetting Measure (FM)
        forgetting_scores = []
        for task_id, acc_list in task_accs.items():
            if len(acc_list) > 1:
                max_acc = max(acc_list[:-1])
                current_acc = acc_list[-1]
                forgetting_scores.append(max_acc - current_acc)
        avg_forgetting = np.mean(forgetting_scores) if forgetting_scores else 0.0

        # Forward Transfer (FWT)
        # 使用随机初始化模型的基准性能
        fwt = 0.0  # 简化计算

        # 训练和选择总耗时
        total_select_time = sum(
            t['select_time'] for t in results['per_task_results']
        )
        total_train_time = sum(
            t['train_time'] for t in results['per_task_results']
        )

        # 最终测试准确率
        final_test_acc = results['per_task_results'][-1]['test_accuracy'] if results['per_task_results'] else 0.0

        return {
            'average_accuracy': float(avg_acc),
            'forgetting_measure': float(avg_forgetting),
            'forward_transfer': float(fwt),
            'final_test_accuracy': float(final_test_acc),
            'total_select_time': float(total_select_time),
            'total_train_time': float(total_train_time),
        }


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description='核心集选择方法对比实验'
    )
    parser.add_argument(
        '--config', type=str,
        default='configs/experiments.yaml',
        help='实验配置文件路径'
    )
    parser.add_argument(
        '--output', type=str,
        default=None,
        help='结果输出目录'
    )
    parser.add_argument(
        '--dataset', type=str, nargs='+',
        default=None,
        help='指定数据集（覆盖配置文件）'
    )
    parser.add_argument(
        '--method', type=str, nargs='+',
        default=None,
        help='指定方法（覆盖配置文件）'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='快速模式：减少实验配置'
    )

    args = parser.parse_args()

    runner = ExperimentRunner(
        config_path=args.config,
        output_dir=args.output
    )

    # 覆盖配置
    if args.dataset:
        runner.config.setdefault('comparison', {})['datasets'] = args.dataset
    if args.method:
        runner.config.setdefault('comparison', {})['methods'] = args.method
    if args.quick:
        runner.config.setdefault('comparison', {})['memory_ratios'] = [0.1]
        runner.config.setdefault('comparison', {})['num_runs'] = 1
        runner.config.setdefault('training', {})['num_epochs'] = 10

    runner.run_full_comparison()


if __name__ == '__main__':
    main()
