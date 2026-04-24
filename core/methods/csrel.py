"""
CSReL: 基于可约损失的核心集选择
"""
import torch
import random
import numpy as np
from ..coreset_base import CoresetSelector


class CSReLSelector(CoresetSelector):
    """
    CSReL (Coreset Selection via Reducible Loss)

    基于可约损失进行核心集选择
    ReL = Loss(子集模型) - Loss(全量模型)
    ReL 越高说明样本包含越多当前模型缺失的信息
    """

    def __init__(self, memory_budget: int, device=None, temperature: float = 1.0):
        """
        Args:
            memory_budget: 核心集大小
            device: 计算设备
            temperature: Softmax 温度参数
        """
        super().__init__(memory_budget, device)
        self.temperature = temperature
        self.reloss_history = []

    def select_coreset(
        self,
        dataset,
        model,
        task_id: int,
        previous_coresets=None
    ):
        """
        基于可约损失选择核心集

        Args:
            dataset: 数据加载器
            model: 当前模型
            task_id: 任务ID
            previous_coresets: 之前的核心集索引列表

        Returns:
            (indices, weights): 选择的索引和权重
        """
        model.eval()

        # 第一步：计算所有样本的可约损失
        rel_losses = self._compute_reducible_losses(dataset, model)

        # 第二步：基于可约损失进行采样
        if len(rel_losses) <= self.memory_budget:
            selected_indices = list(range(len(rel_losses)))
            weights = torch.ones(len(selected_indices), device=self.device)
        else:
            # 使用 Softmax 将可约损失转换为采样概率
            probs = torch.softmax(rel_losses / self.temperature, dim=0)

            # 重要性加权随机采样
            selected_indices = torch.multinomial(
                probs,
                num_samples=self.memory_budget,
                replacement=False
            ).tolist()

            # 根据选中频率设置权重
            weights = torch.zeros(self.memory_budget, device=self.device)
            for idx in selected_indices:
                weights[selected_indices.index(idx)] = probs[idx].item()

        # 第三步：与历史核心集合并
        if previous_coresets is not None and len(previous_coresets) > 0:
            all_previous = []
            for prev in previous_coresets:
                all_previous.extend(prev)

            if len(all_previous) > 0:
                # 计算历史样本的可约损失
                hist_rel_losses = self._compute_reducible_losses_for_indices(
                    dataset, model, all_previous
                )

                # 分配预算：历史 vs 新样本
                budget_new = self.memory_budget // 2  # 50%给新样本
                budget_hist = self.memory_budget - budget_new

                # 从历史中选择 ReL 最高的
                if len(all_previous) > budget_hist:
                    hist_probs = torch.softmax(hist_rel_losses / self.temperature, dim=0)
                    hist_selected = torch.multinomial(
                        hist_probs,
                        num_samples=budget_hist,
                        replacement=False
                    ).tolist()

                    selected_indices = hist_selected + selected_indices[:budget_new]
                else:
                    selected_indices = all_previous + selected_indices[:budget_new]

        self.selected_indices = selected_indices
        self.selection_weights = weights
        self.reloss_history.append(rel_losses.mean().item())

        return selected_indices, weights

    def _compute_reducible_losses(self, dataset, model):
        """计算所有样本的可约损失"""
        rel_losses = []

        with torch.no_grad():
            for batch_x, batch_y, idx in dataset:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # 使用当前模型在样本上的损失作为代理
                # 这里简化处理：使用损失值近似
                logits = model(batch_x)
                probs = torch.softmax(logits, dim=1)
                loss = -probs[range(len(batch_y)), batch_y].log()

                # 可约损失越高，说明模型对该样本的不确定性越高
                rel_losses.append(loss.sum(dim=0))

        return torch.cat(rel_losses)

    def _compute_reducible_losses_for_indices(self, dataset, model, indices):
        """为指定索引计算可约损失"""
        rel_losses = []

        with torch.no_grad():
            for idx in indices:
                # 获取单个样本
                x, y = dataset.dataset[idx]
                x = x.unsqueeze(0).to(self.device)
                y = torch.tensor([y]).to(self.device)

                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                loss = -probs[0, y].log()

                rel_losses.append(loss)

        return torch.stack(rel_losses)
