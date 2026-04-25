"""
随机采样基准方法
"""
import torch
import random
import numpy as np
from ..coreset_base import CoresetSelector


class RandomSelector(CoresetSelector):
    """随机采样核心集选择器"""

    def select_coreset(
        self,
        dataset,
        model,
        task_id: int,
        previous_coresets=None
    ):
        """
        随机选择核心集

        Args:
            dataset: 数据加载器
            model: 模型（此方法不使用）
            task_id: 任务ID
            previous_coresets: 之前的核心集（不用于选择，由上层处理合并）

        Returns:
            (indices, weights): 选择的索引和均匀权重
        """
        # 获取当前任务数据集的样本数
        dataset_size = len(dataset.dataset)
        all_indices = list(range(dataset_size))

        if dataset_size == 0:
            raise ValueError("Dataset is empty")

        # 随机选择 memory_budget 个样本
        if len(all_indices) <= self.memory_budget:
            selected_indices = all_indices
        else:
            selected_indices = random.sample(all_indices, self.memory_budget)

        # 均匀权重
        weights = torch.ones(len(selected_indices), device=self.device)

        self.selected_indices = selected_indices
        self.selection_weights = weights

        return selected_indices, weights
