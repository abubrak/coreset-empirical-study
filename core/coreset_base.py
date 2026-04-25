"""
核心集选择基础接口
定义所有核心集选择方法的统一接口
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# 全局批次索引计数器，用于生成跨批次的唯一索引
_batch_index_counter = 0


def _parse_batch(batch):
    """
    解析数据批次，支持多种返回格式

    Args:
        batch: 数据批次，可能是 (x, y) 或 (x, y, idx)

    Returns:
        (x, y, indices): 数据、标签和索引
    """
    global _batch_index_counter

    if len(batch) == 3:
        x, y, indices = batch
    elif len(batch) == 2:
        x, y = batch
        # 生成全局唯一索引
        batch_size = x.size(0)
        indices = torch.arange(_batch_index_counter, _batch_index_counter + batch_size)
        _batch_index_counter += batch_size
    else:
        raise ValueError(f"Unexpected batch format: {len(batch)} elements")

    return x, y, indices


def reset_batch_index_counter():
    """
    重置全局批次索引计数器

    ⚠️ 并发限制: 此实现使用全局变量，不支持多线程/多进程环境。
    如果需要并行实验，请使用不同的方案。

    应该在每个新实验开始时调用，确保索引从0开始。
    """
    global _batch_index_counter
    _batch_index_counter = 0


class CoresetSelector(ABC):
    """核心集选择器基类"""

    def __init__(self, memory_budget: int, device: torch.device = None):
        """
        Args:
            memory_budget: 核心集大小限制
            device: 计算设备
        """
        if memory_budget <= 0:
            raise ValueError(f"memory_budget must be positive, got {memory_budget}")

        self.memory_budget = memory_budget
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.selected_indices = None
        self.selection_weights = None

    @abstractmethod
    def select_coreset(
        self,
        dataset: DataLoader,
        model: nn.Module,
        task_id: int,
        previous_coresets: Optional[list] = None
    ) -> Tuple[list, torch.Tensor]:
        """
        选择核心集

        Args:
            dataset: 当前任务的数据加载器
            model: 当前模型
            task_id: 任务ID
            previous_coresets: 之前任务的核心集索引列表

        Returns:
            (selected_indices, selection_weights): 选择的索引和权重
        """
        pass

    def get_coreset_subset(self, dataset: DataLoader, indices: list) -> DataLoader:
        """根据索引获取数据子集"""
        if not indices:
            raise ValueError("Cannot create subset from empty indices list")

        # 验证索引在有效范围内
        dataset_size = len(dataset.dataset)
        valid_indices = [i for i in indices if 0 <= i < dataset_size]

        if len(valid_indices) != len(indices):
            invalid_count = len(indices) - len(valid_indices)
            print(f"警告: {invalid_count} 个索引超出范围，已过滤")

        if not valid_indices:
            raise ValueError("No valid indices remaining after filtering")

        subset = Subset(dataset.dataset, valid_indices)
        return DataLoader(
            subset,
            batch_size=min(dataset.batch_size, len(valid_indices)),
            shuffle=False,
            num_workers=dataset.num_workers
        )

    def compute_reducible_loss(
        self,
        dataset: DataLoader,
        model_full: nn.Module,
        model_subset: nn.Module
    ) -> torch.Tensor:
        """
        计算可约损失 (用于CSReL方法)

        Args:
            dataset: 数据集
            model_full: 在全量数据上训练的模型
            model_subset: 在子集上训练的模型

        Returns:
            每个样本的可约损失值
        """
        model_full.eval()
        model_subset.eval()

        rel_losses = []

        with torch.no_grad():
            for batch in dataset:
                batch_x, batch_y, _ = _parse_batch(batch)
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # 全量模型的损失（每个样本）
                logits_full = model_full(batch_x)
                loss_full = nn.functional.cross_entropy(logits_full, batch_y, reduction='none')

                # 子集模型的损失（每个样本）
                logits_sub = model_subset(batch_x)
                loss_sub = nn.functional.cross_entropy(logits_sub, batch_y, reduction='none')

                # 可约损失 = 子集损失 - 全量损失
                rel_loss = loss_sub - loss_full
                rel_losses.append(rel_loss)

        return torch.cat(rel_losses)


class ContinualLearningFramework:
    """持续学习框架"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        optimizer_class = torch.optim.SGD,
        optimizer_kwargs = None
    ):
        """
        Args:
            model: 神经网络模型
            device: 计算设备
            optimizer_class: 优化器类
            optimizer_kwargs: 优化器参数
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 0.01, 'momentum': 0.9}
        self.optimizer = None
        self.current_task = 0

    def train_task(
        self,
        train_loader: DataLoader,
        num_epochs: int = 50,
        coreset_indices: list = None,
        coreset_weights: torch.Tensor = None
    ) -> dict:
        """
        训练单个任务

        Args:
            train_loader: 训练数据加载器
            num_epochs: 训练轮数
            coreset_indices: 核心集索引（如果使用回放）
            coreset_weights: 核心集权重

        Returns:
            训练指标字典
        """
        self.model.train()

        if self.optimizer is None:
            self.optimizer = self.optimizer_class(
                self.model.parameters(),
                **self.optimizer_kwargs
            )

        metrics = {
            'train_losses': [],
            'train_accuracy': []
        }

        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch in train_loader:
                batch_x, batch_y, batch_idx = _parse_batch(batch)
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)

                # 计算每个样本的损失
                per_sample_loss = nn.functional.cross_entropy(
                    outputs, batch_y, reduction='none'
                )

                # 如果提供了核心集权重，应用加权损失
                if coreset_weights is not None and coreset_indices is not None:
                    # 创建索引到权重的映射
                    index_to_weight = {
                        idx.item(): weight.item()
                        for idx, weight in zip(coreset_indices, coreset_weights)
                    }

                    # 获取当前批次中每个样本的权重
                    batch_weights = torch.tensor(
                        [index_to_weight.get(idx.item(), 1.0)
                         for idx in batch_idx],
                        device=self.device, dtype=per_sample_loss.dtype
                    )

                    # 加权损失
                    loss = (per_sample_loss * batch_weights).mean()
                else:
                    loss = per_sample_loss.mean()

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_x.size(0)

            avg_loss = epoch_loss / total
            accuracy = correct / total

            metrics['train_losses'].append(avg_loss)
            metrics['train_accuracy'].append(accuracy)

        return metrics

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, dict]:
        """
        评估模型性能

        Returns:
            (accuracy, per_class_accuracy)
        """
        self.model.eval()

        correct = 0
        total = 0
        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                _, predicted = outputs.max(1)

                correct += predicted.eq(batch_y).sum().item()
                total += batch_x.size(0)

                # 每类准确率
                for cls in range(outputs.size(1)):
                    cls_mask = (batch_y == cls)
                    if cls_mask.any():
                        class_correct[cls] = class_correct.get(cls, 0) + predicted[cls_mask].eq(batch_y[cls_mask]).sum().item()
                        class_total[cls] = class_total.get(cls, 0) + cls_mask.sum().item()

        accuracy = correct / total

        per_class_acc = {
            cls: class_correct[cls] / class_total[cls]
            for cls in class_total.keys()
        }

        return accuracy, per_class_acc

    def compute_forgetting_measure(self, task_accuracies: dict) -> float:
        """
        计算遗忘度量

        Args:
            task_accuracies: 每个任务的历史准确率字典 {task_id: [acc1, acc2, ...]}

        Returns:
            平均遗忘率
        """
        if not task_accuracies:
            return 0.0

        forgetting = 0.0
        count = 0

        for task_id, acc_history in task_accuracies.items():
            if len(acc_history) > 1:
                # 遗忘率 = 历史最高 - 当前值
                max_acc = max(acc_history[:-1])  # 除去当前值
                current_acc = acc_history[-1]
                task_forgetting = max_acc - current_acc
                forgetting += task_forgetting
                count += 1

        return forgetting / count if count > 0 else 0.0
