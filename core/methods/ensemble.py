"""
集成方法：自适应选择策略
结合 CSReL (早期) 和 BCSR (后期) 的优点
"""
import torch
import numpy as np
from ..coreset_base import CoresetSelector
from .csrel import CSReLSelector


class EnsembleSelector(CoresetSelector):
    """
    自适应集成核心集选择器

    策略：
    - 早期阶段 (task_id < switch_threshold): 使用 CSReL 快速构建
    - 后期阶段: 使用更精细的选择策略
    """

    def __init__(
        self,
        memory_budget: int,
        device=None,
        switch_threshold: float = 0.5,
        early_ratio: float = 0.3
    ):
        """
        Args:
            memory_budget: 核心集大小
            device: 计算设备
            switch_threshold: 切换阈值 (任务ID / 总任务数)
            early_ratio: 早期阶段比例
        """
        super().__init__(memory_budget, device)
        self.switch_threshold = switch_threshold
        self.early_ratio = early_ratio

        # 子选择器
        self.csrel_selector = CSReLSelector(
            memory_budget=memory_budget,
            device=device,
            temperature=1.0
        )

        self.use_early_strategy = True
        self.task_count = 0

    def select_coreset(
        self,
        dataset,
        model,
        task_id: int,
        previous_coresets=None,
        total_tasks: int = 10
    ):
        """
        自适应选择核心集

        Args:
            dataset: 数据加载器
            model: 模型
            task_id: 当前任务ID
            previous_coresets: 历史核心集
            total_tasks: 总任务数

        Returns:
            (indices, weights): 选择的索引和权重
        """
        self.task_count += 1

        # 判断是否使用早期策略
        progress_ratio = task_id / total_tasks
        self.use_early_strategy = progress_ratio < self.early_ratio

        if self.use_early_strategy:
            # 早期：使用 CSReL 快速构建
            print(f"Task {task_id}: Using CSReL strategy (early stage)")
            indices, weights = self.csrel_selector.select_coreset(
                dataset, model, task_id, previous_coresets
            )
        else:
            # 后期：精细选择策略
            print(f"Task {task_id}: Using refined strategy (later stage)")
            indices, weights = self._refined_selection(
                dataset, model, task_id, previous_coresets
            )

        self.selected_indices = indices
        self.selection_weights = weights

        return indices, weights

    def _refined_selection(self, dataset, model, task_id, previous_coresets):
        """
        后期精细选择策略

        结合可约损失和历史重要性进行选择
        """
        # 计算可约损失
        rel_losses = self.csrel_selector._compute_reducible_losses(dataset, model)

        # 如果有历史核心集，考虑历史重要性
        if previous_coresets and len(previous_coresets) > 0:
            all_previous = []
            for prev in previous_coresets:
                all_previous.extend(prev)

            # 计算历史样本重要性
            hist_importance = self._compute_historical_importance(
                dataset, model, all_previous
            )

            # 合并新旧样本
            new_budget = self.memory_budget // 2
            hist_budget = self.memory_budget - new_budget

            # 从历史中选择最重要的
            if len(all_previous) > hist_budget:
                hist_selected = self._top_k_indices(hist_importance, hist_budget)
            else:
                hist_selected = all_previous

            # 从新样本中选择 ReL 最高的
            if len(rel_losses) > new_budget:
                new_selected = self._top_k_indices(rel_losses, new_budget)
            else:
                new_selected = list(range(len(rel_losses)))

            selected_indices = hist_selected + new_selected
            weights = self._compute_combined_weights(
                len(selected_indices), hist_budget
            )
        else:
            # 没有历史，直接选择 ReL 最高的
            if len(rel_losses) <= self.memory_budget:
                selected_indices = list(range(len(rel_losses)))
            else:
                selected_indices = self._top_k_indices(rel_losses, self.memory_budget)

            weights = torch.ones(len(selected_indices), device=self.device)

        return selected_indices, weights

    def _compute_historical_importance(self, dataset, model, indices):
        """计算历史样本的重要性"""
        importance = torch.zeros(len(indices), device=self.device)

        with torch.no_grad():
            for i, idx in enumerate(indices):
                x, y = dataset.dataset[idx]
                x = x.unsqueeze(0).to(self.device)
                y = torch.tensor([y]).to(self.device)

                logits = model(x)
                probs = torch.softmax(logits, dim=1)

                # 重要性 = 置信度 * (1 - 预测概率)
                confidence, pred_idx = probs.max(1)
                importance[i] = confidence * (1 - probs[0, y])

        return importance

    def _top_k_indices(self, scores, k):
        """选择得分最高的 k 个索引"""
        return torch.topk(scores, k).indices.tolist()

    def _compute_combined_weights(self, total_size, hist_size):
        """计算组合权重"""
        weights = torch.ones(total_size, device=self.device)
        # 历史样本权重略高
        if hist_size > 0:
            weights[:hist_size] = 1.2
            weights[hist_size:] = 0.8
            # 归一化
            weights = weights / weights.sum()
        return weights


def _top_k_indices(scores, k):
    """辅助函数：选择top-k索引"""
    return torch.topk(scores, k).indices.tolist()
