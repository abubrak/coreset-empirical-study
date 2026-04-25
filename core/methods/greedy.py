"""
Greedy Coreset Selection
贪心核心集选择：基于特征空间的最大覆盖贪心选择
"""
import torch
import numpy as np
from ..coreset_base import CoresetSelector, _parse_batch


class GreedySelector(CoresetSelector):
    """
    Greedy Coreset Selector

    通过贪心策略迭代选择最大化特征空间覆盖的样本。
    每轮选择距离已选集合最远的样本（最远优先遍历），
    保证核心集的多样性和代表性。
    """

    def __init__(
        self,
        memory_budget: int,
        device=None,
        distance_metric: str = 'euclidean',
        use_features: bool = True
    ):
        """
        Args:
            memory_budget: 核心集大小
            device: 计算设备
            distance_metric: 距离度量 ('euclidean', 'cosine')
            use_features: 是否使用模型特征（否则用原始像素）
        """
        super().__init__(memory_budget, device)
        self.distance_metric = distance_metric
        self.use_features = use_features

    def select_coreset(
        self,
        dataset,
        model,
        task_id: int,
        previous_coresets=None
    ):
        """
        贪心选择核心集

        策略：
        1. 提取所有样本的特征表示
        2. 选择距离中心最近的样本作为种子
        3. 迭代选择距离已选集合最远的样本
        4. 与历史核心集合并

        Returns:
            (indices, weights): 选择的索引和权重
        """
        model.eval()

        # 第一步：提取特征
        features, indices_map = self._extract_features(dataset, model)

        # 第二步：贪心选择
        num_samples = features.shape[0]
        if num_samples <= self.memory_budget:
            selected_indices = list(range(num_samples))
            weights = torch.ones(num_samples, device=self.device)
        else:
            selected_positions = self._greedy_selection(features)
            selected_indices = [indices_map[i] for i in selected_positions]
            weights = torch.ones(len(selected_indices), device=self.device)

        # 第三步：与历史核心集合并
        if previous_coresets is not None and len(previous_coresets) > 0:
            all_previous = []
            for prev in previous_coresets:
                all_previous.extend(prev)

            # 历史样本占一半预算
            budget_hist = self.memory_budget // 2
            budget_new = self.memory_budget - budget_hist

            if len(all_previous) > budget_hist:
                # 从历史中随机采样（贪心方法的简化处理）
                import random
                hist_selected = random.sample(all_previous, budget_hist)
            else:
                hist_selected = all_previous

            new_selected = selected_indices[:budget_new]
            selected_indices = hist_selected + new_selected
            weights = torch.ones(len(selected_indices), device=self.device)

        self.selected_indices = selected_indices
        self.selection_weights = weights

        return selected_indices, weights

    def _extract_features(self, dataset, model):
        """提取样本特征"""
        features_list = []
        indices_map = []

        with torch.no_grad():
            for batch in dataset:
                batch_x, _, idx = _parse_batch(batch)
                batch_x = batch_x.to(self.device)

                if self.use_features:
                    # 使用模型中间层特征
                    feat = self._get_intermediate_features(model, batch_x)
                else:
                    # 使用原始像素展平
                    feat = batch_x.view(batch_x.size(0), -1)

                features_list.append(feat)
                indices_map.extend(idx.tolist())

        features = torch.cat(features_list, dim=0)

        # L2 归一化（对 cosine 距离效果更好）
        if self.distance_metric == 'cosine':
            features = torch.nn.functional.normalize(features, dim=1)

        return features, indices_map

    def _get_intermediate_features(self, model, x):
        """
        获取模型中间层特征（智能回退机制）

        策略：
        1. 优先使用 model.get_features()（如果模型实现了该方法）
        2. 对于 Sequential 模型，提取分类器前的特征
        3. 对于普通模块，逐层前传并在分类器前停止
        4. 回退到展平的输入特征
        """
        # 策略1：检查是否有自定义特征提取方法
        if hasattr(model, 'get_features'):
            try:
                features = model.get_features(x)
                # 验证返回的是特征而非logits
                if features.dim() >= 2 and features.size(0) == x.size(0):
                    return features
            except Exception:
                pass  # 回退到其他策略

        # 策略2：Sequential 模型处理
        if isinstance(model, torch.nn.Sequential):
            # 找到最后一个线性层（分类器）的索引
            classifier_idx = -1
            for i, module in enumerate(model):
                if isinstance(module, torch.nn.Linear):
                    classifier_idx = i
                    break

            if classifier_idx > 0:
                # 在分类器之前停止
                with torch.no_grad():
                    features = x
                    for module in list(model)[:classifier_idx]:
                        features = module(features)
                    return features.view(features.size(0), -1)

        # 策略3：逐层前传，在第一个线性层后停止
        features = x
        for name, module in model.named_children():
            features = module(features)
            # 在第一个线性层后停止（通常是特征提取的终点）
            if isinstance(module, torch.nn.Linear):
                break

        # 确保特征是2D张量
        if features.dim() > 2:
            features = features.view(features.size(0), -1)

        # 策略4：如果特征维度不合理（如过小），回退到展平输入
        if features.size(-1) < 10:  # 启发式：特征维度太小可能返回的是logits
            return x.view(x.size(0), -1)

        return features

    def _greedy_selection(self, features: torch.Tensor):
        """
        贪心最远优先遍历选择

        每轮选择距离已选集合最远的样本点。
        O(n*k) 复杂度，k 为核心集大小。
        """
        if features.shape[0] == 0:
            raise ValueError("Cannot select from empty features")

        n = features.shape[0]
        k = min(self.memory_budget, n)

        if k == 0:
            return []

        # 以全局中心最近点作为种子
        center = features.mean(dim=0, keepdim=True)
        dists = self._compute_distances(features, center)
        first_idx = dists.argmax().item()

        selected = [first_idx]
        # 维护每个样本到已选集合的最小距离
        min_dists = self._compute_distances(features, features[first_idx:first_idx+1]).squeeze()

        for _ in range(k - 1):
            # 选择距离已选集合最远的样本
            # 排除已选中的
            mask = torch.ones(n, dtype=torch.bool, device=self.device)
            for s in selected:
                mask[s] = False

            masked_dists = min_dists.clone()
            masked_dists[~mask] = -1

            next_idx = masked_dists.argmax().item()
            selected.append(next_idx)

            # 更新最小距离
            new_dists = self._compute_distances(
                features, features[next_idx:next_idx+1]
            ).squeeze()
            min_dists = torch.minimum(min_dists, new_dists)

        return selected

    def _compute_distances(self, features, centers):
        """计算样本到中心点的距离"""
        # features: (n, d), centers: (1, d) 或 (m, d)
        if self.distance_metric == 'cosine':
            # 余弦距离 = 1 - cosine_similarity
            sim = torch.nn.functional.cosine_similarity(
                features.unsqueeze(1), centers.unsqueeze(0), dim=2
            )
            return 1 - sim
        else:
            # 欧氏距离
            diff = features.unsqueeze(1) - centers.unsqueeze(0)
            return (diff ** 2).sum(dim=2)
