"""
随机采样基准方法
"""
import torch
import random
import numpy as np
from .coreset_base import CoresetSelector


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
            previous_coresets: 之前的核心集

        Returns:
            (indices, weights): 选择的索引和均匀权重
        """
        # 获取所有样本的索引
        all_indices = list(range(len(dataset.dataset)))

        # 随机选择
        if len(all_indices) <= self.memory_budget:
            selected_indices = all_indices
        else:
            selected_indices = random.sample(all_indices, self.memory_budget)

        # 均匀权重
        weights = torch.ones(len(selected_indices), device=self.device)

        # 如果有之前的核心集，合并
        if previous_coresets is not None and len(previous_coresets) > 0:
            all_previous = []
            for prev in previous_coresets:
                all_previous.extend(prev)

            # 保持总预算，随机选择保留多少历史
            total_history = len(all_previous)
            max_new = self.memory_budget

            if total_history > 0:
                keep_ratio = min(1.0, max_new / total_history)
                keep_indices = random.sample(all_previous, int(total_history * keep_ratio))
            else:
                keep_indices = []

            # 添加新样本
            remaining_budget = self.memory_budget - len(keep_indices)
            new_indices = random.sample(all_indices, remaining_budget)

            selected_indices = keep_indices + new_indices
            weights = torch.ones(self.memory_budget, device=self.device)

        self.selected_indices = selected_indices
        self.selection_weights = weights

        return selected_indices, weights
