"""
BCSR: Bi-level Coreset Selection via Reduction
基于双层优化的核心集选择方法

双层优化框架：
  外层问题：min_{S} L_val(θ*(S))
  内层问题：θ*(S) = argmin_{θ} L_train(θ; S)

其中 S 为核心集索引集合，L_val 为验证集损失，L_train 为训练集损失。
通过隐函数定理将双层问题转化为单层优化。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .coreset_base import CoresetSelector


class BCSRSelector(CoresetSelector):
    """
    BCSR (Bi-level Coreset Selection via Reduction)

    通过双层优化同时学习：
    1. 内层：在核心集上训练模型参数 θ
    2. 外层：优化核心集选择权重，使验证集损失最小化

    使用一步前瞻 (one-step lookahead) 近似隐函数，
    避免完整的双层优化计算开销。
    """

    def __init__(
        self,
        memory_budget: int,
        device=None,
        inner_lr: float = 0.01,
        outer_lr: float = 0.1,
        inner_steps: int = 5,
        meta_steps: int = 3,
        temperature: float = 1.0,
        use_val_set: bool = True,
        val_ratio: float = 0.2
    ):
        """
        Args:
            memory_budget: 核心集大小
            device: 计算设备
            inner_lr: 内层学习率（模型训练）
            outer_lr: 外层学习率（权重优化）
            inner_steps: 内层优化步数
            meta_steps: 元优化（外层）迭代步数
            temperature: Softmax 温度
            use_val_set: 是否使用验证集指导选择
            val_ratio: 验证集比例
        """
        super().__init__(memory_budget, device)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.meta_steps = meta_steps
        self.temperature = temperature
        self.use_val_set = use_val_set
        self.val_ratio = val_ratio

        self.selection_logits = None

    def select_coreset(
        self,
        dataset,
        model,
        task_id: int,
        previous_coresets=None
    ):
        """
        双层优化选择核心集

        流程：
        1. 将训练数据划分为训练/验证
        2. 初始化选择权重（logits）
        3. 交替优化：
           a. 内层：固定权重，在加权子集上训练模型
           b. 外层：固定模型，优化权重使验证损失最小
        4. 选择权重最高的 top-k 样本

        Returns:
            (indices, weights): 选择的索引和权重
        """
        model.eval()

        # 第一步：提取数据和划分验证集
        all_data, all_targets, all_indices = self._extract_data(dataset)

        num_samples = len(all_targets)
        if num_samples <= self.memory_budget:
            selected_indices = list(range(num_samples))
            weights = torch.ones(num_samples, device=self.device)
            self.selected_indices = selected_indices
            self.selection_weights = weights
            return selected_indices, weights

        # 划分训练/验证
        if self.use_val_set:
            perm = torch.randperm(num_samples)
            val_size = int(num_samples * self.val_ratio)
            val_idx = perm[:val_size]
            train_idx = perm[val_size:]
        else:
            train_idx = torch.arange(num_samples)
            val_idx = torch.arange(num_samples)

        train_data = all_data[train_idx]
        train_targets = all_targets[train_idx]
        val_data = all_data[val_idx]
        val_targets = all_targets[val_idx]

        # 第二步：初始化选择 logits
        n_train = train_data.shape[0]
        self.selection_logits = torch.zeros(
            n_train, device=self.device, requires_grad=True
        )

        # 第三步：双层优化
        optimizer = torch.optim.Adam(
            [self.selection_logits], lr=self.outer_lr
        )

        for meta_step in range(self.meta_steps):
            # 获取当前选择权重
            selection_probs = torch.softmax(
                self.selection_logits / self.temperature, dim=0
            )

            # --- 内层优化：在加权子集上训练模型 ---
            model_copy = self._copy_model(model)
            inner_opt = torch.optim.SGD(
                model_copy.parameters(), lr=self.inner_lr
            )

            for _ in range(self.inner_steps):
                inner_opt.zero_grad()
                logits = model_copy(train_data)
                # 加权交叉熵
                loss = F.cross_entropy(logits, train_targets, reduction='none')
                weighted_loss = (loss * selection_probs).sum()
                weighted_loss.backward()
                inner_opt.step()

            # --- 外层优化：在验证集上评估并优化权重 ---
            optimizer.zero_grad()
            val_logits = model_copy(val_data)
            val_loss = F.cross_entropy(val_logits, val_targets)

            val_loss.backward()
            optimizer.step()

        # 第四步：根据权重选择 top-k
        final_probs = torch.softmax(
            self.selection_logits / self.temperature, dim=0
        )
        k = min(self.memory_budget, n_train)
        _, top_k_pos = torch.topk(final_probs, k)

        # 映射回原始索引
        selected_indices = [all_indices[train_idx[i].item()] for i in top_k_pos.tolist()]
        weights = final_probs[top_k_pos].detach()

        # 第五步：与历史核心集合并
        if previous_coresets is not None and len(previous_coresets) > 0:
            all_previous = []
            for prev in previous_coresets:
                all_previous.extend(prev)

            budget_hist = self.memory_budget // 2
            budget_new = self.memory_budget - budget_hist

            if len(all_previous) > budget_hist:
                # 从历史中选择（使用模型评估重要性）
                hist_importance = self._evaluate_importance(
                    model, all_data, all_targets, all_indices, all_previous
                )
                _, hist_top = torch.topk(hist_importance, budget_hist)
                hist_selected = [all_previous[i] for i in hist_top.tolist()]
            else:
                hist_selected = all_previous

            new_selected = selected_indices[:budget_new]
            selected_indices = hist_selected + new_selected
            weights = torch.cat([
                torch.ones(len(hist_selected), device=self.device),
                weights[:budget_new]
            ])

        self.selected_indices = selected_indices
        self.selection_weights = weights

        return selected_indices, weights

    def _extract_data(self, dataset):
        """从 DataLoader 提取所有数据和标签"""
        all_data = []
        all_targets = []
        all_indices = []

        for batch in dataset:
            if len(batch) == 3:
                x, y, idx = batch
            else:
                x, y = batch
                idx = torch.arange(x.size(0))

            all_data.append(x)
            all_targets.append(y)
            all_indices.extend(idx.tolist())

        all_data = torch.cat(all_data, dim=0).to(self.device)
        all_targets = torch.cat(all_targets, dim=0).to(self.device)

        return all_data, all_targets, all_indices

    def _copy_model(self, model):
        """深拷贝模型（用于内层优化）"""
        import copy
        model_copy = copy.deepcopy(model)
        model_copy.to(self.device)
        return model_copy

    def _evaluate_importance(self, model, all_data, all_targets, all_indices, hist_indices):
        """评估历史样本的重要性"""
        importance = torch.zeros(len(hist_indices), device=self.device)

        with torch.no_grad():
            for i, idx in enumerate(hist_indices):
                # 找到在 all_indices 中的位置
                if idx in all_indices:
                    pos = all_indices.index(idx)
                    x = all_data[pos:pos+1]
                    y = all_targets[pos:pos+1]

                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    # 重要性 = 损失值（不确定性越大越重要）
                    importance[i] = F.cross_entropy(logits, y)

        return importance
