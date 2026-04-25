"""
CSReL: 基于可约损失的核心集选择
原始实现: https://github.com/RuilinTong/CSReL-Coreset-CL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..coreset_base import CoresetSelector, _parse_batch


class CSReLSelector(CoresetSelector):
    """
    CSReL (Coreset Selection via Reducible Loss)

    基于可约损失进行核心集选择

    Reducible Loss (ReL) = Loss(当前模型) - Loss(参考模型)

    - 参考模型: 在全量数据上训练的"理想模型"
    - ReL 高: 说明样本包含当前模型缺失的重要信息
    - 选择策略: 选择 ReL 最高的 Top-K 样本(确定性)

    参数:
        memory_budget: 核心集大小
        device: 计算设备
        num_incremental_steps: 增量选择步数(每步选择 memory_budget/num_steps 个样本)
        init_ratio: 初始随机选择比例
        class_balanced: 是否类别平衡
    """

    def __init__(
        self,
        memory_budget: int,
        device=None,
        num_incremental_steps: int = 5,
        init_ratio: float = 0.1,
        class_balanced: bool = False
    ):
        super().__init__(memory_budget, device)
        self.num_incremental_steps = num_incremental_steps
        self.init_ratio = init_ratio
        self.class_balanced = class_balanced
        self.reference_model = None
        self.ref_losses = None
        self.ref_num_samples = None  # 记录 ref_losses 对应的样本数
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

        流程:
        1. 训练/加载参考模型(在全量数据上)
        2. 计算参考模型的损失
        3. 初始化: 随机选择少量样本
        4. 增量式选择:
           a. 在当前核心集上训练模型
           b. 计算可约损失 = 当前损失 - 参考损失
           c. 选择 ReL 最高的样本加入核心集
        """
        model.eval()

        # 第一步: 提取所有数据和标签
        all_data, all_targets, all_indices = self._extract_full_dataset(dataset)
        num_samples = len(all_targets)

        if num_samples <= self.memory_budget:
            selected_indices = list(range(num_samples))
            weights = torch.ones(num_samples, device=self.device)
            self.selected_indices = selected_indices
            self.selection_weights = weights
            return selected_indices, weights

        # 第二步: 训练参考模型(首次或模型/任务大小变化时)
        need_new_ref = (
            self.reference_model is None or
            not self._is_same_architecture(model, self.reference_model) or
            self.ref_num_samples != num_samples  # 关键修复：任务大小变化时重新训练
        )

        if need_new_ref:
            self.reference_model = self._train_reference_model(
                all_data, all_targets, model
            )
            # 计算参考损失
            self.ref_losses = self._compute_losses(
                self.reference_model, all_data, all_targets
            )
            self.ref_num_samples = num_samples  # 记录样本数

        # 第三步: 初始化核心集(随机选择少量样本)
        init_size = max(1, int(self.memory_budget * self.init_ratio))
        selected_indices = np.random.choice(
            num_samples, size=init_size, replace=False
        ).tolist()

        incremental_size = (self.memory_budget - init_size) // self.num_incremental_steps

        # 第四步: 增量式选择
        current_model = self._copy_model(model)

        for step in range(self.num_incremental_steps):
            # 在当前核心集上训练模型
            current_model = self._train_on_coreset(
                current_model, all_data[selected_indices], all_targets[selected_indices]
            )

            # 计算当前损失
            cur_losses = self._compute_losses(
                current_model, all_data, all_targets
            )

            # 计算可约损失（现在大小匹配）
            rel_losses = cur_losses - self.ref_losses

            # 从未选样本中选择 ReL 最高的
            mask = torch.ones(num_samples, dtype=torch.bool, device=self.device)
            mask[selected_indices] = False

            if self.class_balanced:
                # 类别平衡选择
                new_indices = self._select_class_balanced(
                    rel_losses, all_targets, mask,
                    incremental_size
                )
            else:
                # 直接选择 ReL 最高的
                masked_rel = rel_losses.clone()
                masked_rel[~mask] = -float('inf')
                _, new_pos = torch.topk(masked_rel, incremental_size)
                new_indices = new_pos.tolist()

            selected_indices.extend(new_indices)

        # 确保不超过预算
        selected_indices = selected_indices[:self.memory_budget]

        # 计算权重(与ReL成正比)
        rel_final = (self._compute_losses(current_model, all_data, all_targets)
                     - self.ref_losses)
        weights = rel_final[selected_indices]

        # 归一化权重
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones(len(selected_indices), device=self.device)

        self.selected_indices = selected_indices
        self.selection_weights = weights
        self.reloss_history.append(rel_losses.mean().item())

        return selected_indices, weights

    def _extract_full_dataset(self, dataset):
        """提取数据集中的所有数据、标签和索引"""
        all_data = []
        all_targets = []
        all_indices = []

        position = 0  # 局部位置计数器

        for batch in dataset:
            x, y, _ = _parse_batch(batch)  # 不使用 _parse_batch 的索引
            batch_size = x.size(0)
            all_data.append(x)
            all_targets.append(y)
            # 使用局部位置索引
            all_indices.extend(range(position, position + batch_size))
            position += batch_size

        all_data = torch.cat(all_data, dim=0).to(self.device)
        all_targets = torch.cat(all_targets, dim=0).to(self.device)

        return all_data, all_targets, all_indices

    def _train_reference_model(self, all_data, all_targets, base_model):
        """在全量数据上训练参考模型"""
        import copy

        ref_model = copy.deepcopy(base_model)
        ref_model.train()

        optimizer = torch.optim.SGD(
            ref_model.parameters(), lr=0.01, momentum=0.9
        )

        # 训练10个epoch
        for epoch in range(10):
            # 小批量训练
            batch_size = 128
            for i in range(0, len(all_data), batch_size):
                batch_x = all_data[i:i+batch_size]
                batch_y = all_targets[i:i+batch_size]

                optimizer.zero_grad()
                logits = ref_model(batch_x)
                loss = F.cross_entropy(logits, batch_y)
                loss.backward()
                optimizer.step()

        ref_model.eval()
        return ref_model.to(self.device)

    def _compute_losses(self, model, all_data, all_targets):
        """计算模型在所有样本上的损失"""
        model.eval()
        losses = []

        with torch.no_grad():
            batch_size = 128
            for i in range(0, len(all_data), batch_size):
                batch_x = all_data[i:i+batch_size]
                batch_y = all_targets[i:i+batch_size]

                logits = model(batch_x)
                loss = F.cross_entropy(logits, batch_y, reduction='none')
                losses.append(loss)

        return torch.cat(losses)

    def _train_on_coreset(self, model, coreset_data, coreset_targets):
        """在核心集上训练模型"""
        import copy

        model_copy = copy.deepcopy(model)
        model_copy.train()

        optimizer = torch.optim.SGD(
            model_copy.parameters(), lr=0.01, momentum=0.9
        )

        # 训练5个epoch
        for epoch in range(5):
            optimizer.zero_grad()
            logits = model_copy(coreset_data)
            loss = F.cross_entropy(logits, coreset_targets)
            loss.backward()
            optimizer.step()

        model_copy.eval()
        return model_copy.to(self.device)

    def _copy_model(self, model):
        """深拷贝模型"""
        import copy
        model_copy = copy.deepcopy(model)
        return model_copy.to(self.device)

    def _is_same_architecture(self, model1, model2):
        """检查两个模型是否架构相同"""
        if type(model1) != type(model2):
            return False

        # 检查参数形状
        params1 = {name: p.shape for name, p in model1.named_parameters()}
        params2 = {name: p.shape for name, p in model2.named_parameters()}

        return params1 == params2

    def _select_class_balanced(self, rel_losses, all_targets, mask, k):
        """类别平衡的选择策略"""
        num_classes = all_targets.max().item() + 1
        per_class = k // num_classes

        selected = []

        for cls in range(num_classes):
            class_mask = (all_targets == cls) & mask
            available = class_mask.sum().item()

            if available == 0:
                continue  # 跳过空类别

            select_count = min(per_class, available)
            class_rel = rel_losses.clone()
            class_rel[~class_mask] = -float('inf')
            _, top_pos = torch.topk(class_rel, select_count)
            selected.extend(top_pos.tolist())

        # 剩余随机选择
        remaining = k - len(selected)
        if remaining > 0:
            masked_rel = rel_losses.clone()
            masked_rel[~mask] = -float('inf')
            # 排除已选的
            if len(selected) > 0:
                masked_rel[selected] = -float('inf')
            _, top_pos = torch.topk(masked_rel, remaining)
            selected.extend(top_pos.tolist())

        return selected

    def _merge_with_previous(
        self, selected_indices, weights, previous_coresets,
        all_data, all_targets
    ):
        """与历史核心集合并"""
        all_previous = []
        for prev in previous_coresets:
            all_previous.extend(prev)

        budget_hist = self.memory_budget // 2
        budget_new = self.memory_budget - budget_hist

        if len(all_previous) > budget_hist:
            # 评估历史样本重要性
            hist_losses = self._compute_losses(
                self.reference_model, all_data[all_previous], all_targets[all_previous]
            )
            _, hist_top = torch.topk(hist_losses, budget_hist)
            hist_selected = [all_previous[i] for i in hist_top.tolist()]
        else:
            hist_selected = all_previous

        new_selected = selected_indices[:budget_new]
        final_indices = hist_selected + new_selected

        # 重新计算权重
        final_weights = torch.zeros(len(final_indices), device=self.device)
        final_weights[:len(hist_selected)] = 1.2  # 历史权重略高
        final_weights[len(hist_selected):] = 0.8   # 新样本权重略低
        final_weights = final_weights / final_weights.sum()

        return final_indices, final_weights
