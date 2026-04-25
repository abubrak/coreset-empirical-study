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
            device=device
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

        关键修复：不合并历史核心集，只选择当前任务样本
        """
        # 计算可约损失
        rel_losses = self._compute_reducible_losses(dataset, model)

        # 直接从当前任务选择 ReL 最高的样本
        if len(rel_losses) <= self.memory_budget:
            selected_indices = list(range(len(rel_losses)))
        else:
            selected_indices = self._top_k_indices(rel_losses, self.memory_budget)

        weights = torch.ones(len(selected_indices), device=self.device)

        return selected_indices, weights

    def _compute_reducible_losses(self, dataset, model):
        """
        计算可约损失（简化版本）

        使用模型的不确定性作为样本重要性
        高损失 = 高不确定性 = 更需要学习的样本
        """
        from ..coreset_base import _parse_batch

        losses_list = []

        model.eval()
        with torch.no_grad():
            for batch in dataset:
                batch_x, batch_y, _ = _parse_batch(batch)
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = model(batch_x)
                loss = torch.nn.functional.cross_entropy(
                    logits, batch_y, reduction='none'
                )
                losses_list.append(loss)

        return torch.cat(losses_list)

    def _top_k_indices(self, scores, k):
        """选择得分最高的 k 个索引"""
        k = min(k, len(scores))
        return torch.topk(scores, k).indices.tolist()

