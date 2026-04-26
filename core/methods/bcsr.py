"""
BCSR: Bi-level Coreset Selection via Reduction
基于双层优化的核心集选择方法

双层优化框架：
  外层问题：min_{S} L_val(θ*(S))
  内层问题：θ*(S) = argmin_{θ} L_train(θ; S)

通过隐函数定理实现正确的双层优化：
- 隐式梯度计算（避免手动设置梯度）
- Top-K正则化（促进稀疏性）
- Simplex投影（约束权重和为1）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..coreset_base import CoresetSelector, _parse_batch


class BCSRSelector(CoresetSelector):
    """
    BCSR (Bi-level Coreset Selection via Reduction)

    通过双层优化同时学习：
    1. 内层：在核心集上训练模型参数 θ
    2. 外层：优化核心集选择权重，使验证集损失最小化

    使用隐函数定理计算正确的梯度，并添加Top-K正则化促进稀疏性。
    """

    def __init__(
        self,
        memory_budget: int,
        device=None,
        inner_lr: float = 0.01,
        outer_lr: float = 0.1,
        inner_steps: int = 10,    # 5 → 10，更强的内层优化
        meta_steps: int = 5,      # 3 → 5，更多的外层迭代
        temperature: float = 1.0,
        use_val_set: bool = True,
        val_ratio: float = 0.2,
        beta: float = 0.2         # 1.0 → 0.2，降低正则化强度
    ):
        """
        Args:
            memory_budget: 核心集大小
            device: 计算设备
            inner_lr: 内层学习率（模型训练）
            outer_lr: 外层学习率（权重优化）
            inner_steps: 内层优化步数（默认10，适合复杂任务）
            meta_steps: 元优化（外层）迭代步数（默认5，适合复杂任务）
            temperature: Softmax 温度
            use_val_set: 是否使用验证集指导选择
            val_ratio: 验证集比例
            beta: Top-K正则化系数（默认0.2，降低以避免过度稀疏）
        """
        super().__init__(memory_budget, device)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.meta_steps = meta_steps
        self.temperature = temperature
        self.use_val_set = use_val_set
        self.val_ratio = val_ratio
        self.beta = beta

        self.selection_logits = None

    def _implicit_gradient(self, model, train_data, train_targets, val_data, val_targets, weights):
        """
        计算双层优化的隐式梯度（有限差分近似，内存友好）

        完全避免 create_graph=True，通过有限差分近似:
        - Hessian-vector 乘积 (Pearlmutter, 1994)
        - Jacobian-vector 乘积
        - 3次 Neumann 迭代提升精度
        - eps=1e-4 提升数值稳定性

        内存峰值仅为单批次大小，适合 T4 等低显存 GPU
        """
        batch_size = 128  # ResNetLike 需较小 batch 控制显存
        n_train = train_data.shape[0]
        eps = 1e-4  # 有限差分步长（提升精度）
        detached_weights = weights.detach()

        model.eval()

        # 1. 计算 ∂L_val/∂θ（标准梯度，无需 create_graph）
        val_loss_parts = []
        for vi in range(0, val_data.shape[0], batch_size):
            end_vi = min(vi + batch_size, val_data.shape[0])
            val_pred = model(val_data[vi:end_vi])
            val_loss_parts.append(F.cross_entropy(val_pred, val_targets[vi:end_vi], reduction='sum'))
            del val_pred
        val_loss = sum(val_loss_parts) / val_data.shape[0]
        del val_loss_parts
        torch.cuda.empty_cache()

        d_theta = torch.autograd.grad(val_loss, model.parameters())
        d_theta = [g.detach() for g in d_theta]
        del val_loss
        torch.cuda.empty_cache()

        # 2. Neumann 迭代近似 v = H^(-1) * d_theta
        # 使用有限差分近似 H*v（3次迭代提升精度）
        v = list(d_theta)
        for _ in range(3):  # 2 → 3，更精确的 H^(-1) 近似
            hvp = self._finite_diff_hvp(
                model, train_data, train_targets,
                detached_weights, v, batch_size, eps
            )
            v = [(vi - hi).detach() for vi, hi in zip(v, hvp)]
            del hvp
            torch.cuda.empty_cache()

        # 3. 计算每个样本的梯度与 v 的内积: g_i · v
        # 使用有限差分: g_i · v ≈ (L_i(θ+εv) - L_i(θ-εv)) / (2ε)
        # θ -> θ + εv
        with torch.no_grad():
            for p, vi in zip(model.parameters(), v):
                p.data.add_(vi * eps)

        losses_plus = self._compute_per_sample_losses(
            model, train_data, train_targets, batch_size
        )

        # θ + εv -> θ - εv
        with torch.no_grad():
            for p, vi in zip(model.parameters(), v):
                p.data.add_(-2 * vi * eps)

        losses_minus = self._compute_per_sample_losses(
            model, train_data, train_targets, batch_size
        )

        # θ - εv -> θ（恢复原始参数）
        with torch.no_grad():
            for p, vi in zip(model.parameters(), v):
                p.data.add_(vi * eps)

        # g_i · v ≈ (L_i+ - L_i-) / (2ε)
        grad_dot_v = (losses_plus - losses_minus) / (2 * eps)

        # ∂L_val/∂w_i = -g_i · v （负号来自隐函数定理）
        jacobian = -grad_dot_v

        del losses_plus, losses_minus, grad_dot_v, v, d_theta
        torch.cuda.empty_cache()

        return jacobian

    def _compute_weighted_grad(self, model, data, targets, weights, batch_size):
        """计算加权训练损失的梯度（无 create_graph）"""
        n = data.shape[0]
        loss_parts = []
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            logits = model(data[i:end_i])
            losses = F.cross_entropy(logits, targets[i:end_i], reduction='none')
            loss_parts.append((losses * weights[i:end_i]).sum())
            del logits, losses

        total_loss = sum(loss_parts) / n
        del loss_parts

        grads = torch.autograd.grad(total_loss, model.parameters())
        return [g.detach() for g in grads]

    def _finite_diff_hvp(self, model, data, targets, weights, vector, batch_size, eps):
        """
        有限差分近似 Hessian-vector 乘积
        H*v ≈ (∇L(θ+εv) - ∇L(θ-εv)) / (2ε)
        """
        # θ + εv
        with torch.no_grad():
            for p, vi in zip(model.parameters(), vector):
                p.data.add_(vi * eps)

        grad_plus = self._compute_weighted_grad(model, data, targets, weights, batch_size)

        # θ + εv -> θ - εv
        with torch.no_grad():
            for p, vi in zip(model.parameters(), vector):
                p.data.add_(-2 * vi * eps)

        grad_minus = self._compute_weighted_grad(model, data, targets, weights, batch_size)

        # θ - εv -> θ
        with torch.no_grad():
            for p, vi in zip(model.parameters(), vector):
                p.data.add_(vi * eps)

        # H*v ≈ (grad_plus - grad_minus) / (2ε)
        hvp = [(gp - gm) / (2 * eps) for gp, gm in zip(grad_plus, grad_minus)]

        del grad_plus, grad_minus
        return hvp

    def _compute_per_sample_losses(self, model, data, targets, batch_size):
        """计算每个样本的损失（无梯度）"""
        losses = []
        with torch.no_grad():
            for i in range(0, data.shape[0], batch_size):
                end_i = min(i + batch_size, data.shape[0])
                logits = model(data[i:end_i])
                batch_losses = F.cross_entropy(logits, targets[i:end_i], reduction='none')
                losses.append(batch_losses)

        return torch.cat(losses)

    def _topk_regularization(self, weights, beta=None, topk=None):
        """
        Top-K 正则化: -β * sum(topk_weights)

        激励只有 K 个样本有大权重，促进稀疏性

        Args:
            weights: 当前权重向量
            beta: 正则化系数（默认使用self.beta）
            topk: Top-K数量（默认使用memory_budget）

        Returns:
            regularizer: 正则化损失值
        """
        if beta is None:
            beta = self.beta
        if topk is None:
            topk = self.memory_budget

        # 获取最大的k个权重
        topk_weights, _ = weights.topk(min(topk, len(weights)))

        # 添加平滑噪声（提高数值稳定性）
        epsilon = 1e-3
        noise = torch.normal(0, 1, size=topk_weights.shape, device=weights.device)

        regularizer = -beta * (topk_weights + epsilon * noise).sum()

        return regularizer

    def _projection_onto_simplex(self, v, b=1.0):
        """
        投影到单纯形: Σw_i = b, w_i ≥ 0

        使用 Duchi et al. (2008) 的算法

        Args:
            v: 待投影向量
            b: 单纯形约束和（默认1.0）

        Returns:
            w: 投影后的权重向量
        """
        v = v.detach().cpu().numpy()
        n_features = v.shape[0]

        # 处理全零或全负情况
        if np.all(v <= 0):
            w = np.zeros(n_features)
            w[0] = b  # 将所有权重分配给第一个元素
            w = torch.from_numpy(w).to(self.device)
            return w

        # 排序(降序)
        u = np.sort(v)[::-1]

        # 计算累积和
        cssv = np.cumsum(u) - b

        # 找到阈值
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0

        # 边界情况：如果 cond 全为 False，返回均匀分布
        if not np.any(cond):
            w = np.ones(n_features) / n_features * b
            w = torch.from_numpy(w).to(self.device)
            return w

        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)

        # 投影
        w = np.maximum(v - theta, 0)

        # 转回 tensor
        w = torch.from_numpy(w).to(self.device)

        return w

    def select_coreset(
        self,
        dataset,
        model,
        task_id: int,
        previous_coresets=None
    ):
        """
        双层优化选择核心集

        正确的双层优化流程：
        1. 划分训练/验证集
        2. 初始化选择权重
        3. 双层优化循环：
           a. 内层：固定权重，训练模型参数
           b. 外层：使用隐式梯度更新权重
           c. 添加Top-K正则化
           d. 投影到单纯形
        4. 选择权重最高的top-k样本

        Args:
            dataset: 数据加载器
            model: 模型
            task_id: 任务ID
            previous_coresets: 历史核心集（不用于选择，由上层处理合并）

        Returns:
            (indices, weights): 选择的索引和权重
        """
        model.eval()

        # 第一步：提取数据和划分验证集
        all_data, all_targets, all_indices = self._extract_data(dataset)

        num_samples = len(all_targets)
        if num_samples <= self.memory_budget:
            selected_indices = list(range(num_samples))
            weights = torch.ones(num_samples, device=self.device) / num_samples
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

        # 第二步：初始化选择权重（直接初始化为均匀分布）
        n_train = train_data.shape[0]
        weights = torch.ones(n_train, device=self.device, requires_grad=True) / n_train

        # 第三步：双层优化
        for meta_step in range(self.meta_steps):
            # --- 内层优化：在加权子集上训练模型 ---
            # 使用 detach 断开梯度图
            model_copy = self._copy_model(model)
            inner_opt = torch.optim.SGD(
                model_copy.parameters(), lr=self.inner_lr
            )
            detached_weights = weights.detach()

            for _ in range(self.inner_steps):
                inner_opt.zero_grad()
                # 分批前向+反向，ResNetLike 模型需较小 batch_size
                inner_batch_size = 128
                for j in range(0, n_train, inner_batch_size):
                    end_j = min(j + inner_batch_size, n_train)
                    logits = model_copy(train_data[j:end_j])
                    loss = F.cross_entropy(logits, train_targets[j:end_j], reduction='none')
                    weighted_loss = (loss * detached_weights[j:end_j]).sum()
                    weighted_loss.backward()
                inner_opt.step()
                torch.cuda.empty_cache()  # 每步后释放缓存

            # --- 外层优化：使用隐式梯度更新权重 ---
            # 有限差分方案无 create_graph，但 ResNetLike 模型仍需控制规模
            max_grad_samples = 6000
            max_val_samples = 3000

            if n_train > max_grad_samples:
                grad_idx = torch.randperm(n_train)[:max_grad_samples]
                grad_train_data = train_data[grad_idx]
                grad_train_targets = train_targets[grad_idx]
                grad_weights = weights[grad_idx]
            else:
                grad_idx = None
                grad_train_data = train_data
                grad_train_targets = train_targets
                grad_weights = weights

            n_val = val_data.shape[0]
            if n_val > max_val_samples:
                val_sub_idx = torch.randperm(n_val)[:max_val_samples]
                grad_val_data = val_data[val_sub_idx]
                grad_val_targets = val_targets[val_sub_idx]
            else:
                grad_val_data = val_data
                grad_val_targets = val_targets

            # 释放内层优化的缓存
            torch.cuda.empty_cache()

            implicit_grad_sub = self._implicit_gradient(
                model_copy, grad_train_data, grad_train_targets,
                grad_val_data, grad_val_targets, grad_weights
            )

            # 将子采样梯度映射回全量权重
            if grad_idx is not None:
                implicit_grad = torch.zeros(n_train, device=self.device)
                implicit_grad[grad_idx] = implicit_grad_sub
            else:
                implicit_grad = implicit_grad_sub

            del model_copy
            torch.cuda.empty_cache()

            # 计算Top-K正则化梯度
            topk_reg = self._topk_regularization(weights)
            topk_grad = torch.autograd.grad(topk_reg, weights)[0]

            # 总梯度 = 隐式梯度 + 正则化梯度
            total_grad = implicit_grad + topk_grad

            # 更新权重（使用简单的梯度下降）
            with torch.no_grad():
                weights.data = weights.data - self.outer_lr * total_grad

            # 投影到单纯形
            with torch.no_grad():
                weights.data = self._projection_onto_simplex(weights.data, b=1.0)

        # 第四步：根据权重选择 top-k
        final_weights = weights.detach()
        k = min(self.memory_budget, n_train)
        _, top_k_pos = torch.topk(final_weights, k)

        # 映射回原始索引（局部索引）
        selected_indices = [train_idx[i].item() for i in top_k_pos.tolist()]
        selected_weights = final_weights[top_k_pos]

        # 归一化权重
        if selected_weights.sum() > 0:
            selected_weights = selected_weights / selected_weights.sum()
        else:
            selected_weights = torch.ones(len(selected_indices), device=self.device)

        self.selected_indices = selected_indices
        self.selection_weights = selected_weights

        return selected_indices, selected_weights

    def _merge_with_previous(
        self,
        new_indices,
        new_weights,
        previous_coresets,
        all_data,
        all_targets,
        all_indices,
        model
    ):
        """
        合并新选择的核心集与历史核心集

        Args:
            new_indices: 新选择的样本索引
            new_weights: 新选择的样本权重
            previous_coresets: 历史核心集列表
            all_data: 所有数据
            all_targets: 所有标签
            all_indices: 所有索引
            model: 当前模型（用于评估重要性）

        Returns:
            (merged_indices, merged_weights): 合并后的索引和权重
        """
        # 收集所有历史样本
        all_previous = []
        for prev in previous_coresets:
            all_previous.extend(prev)

        # 预算分配
        budget_hist = self.memory_budget // 2
        budget_new = self.memory_budget - budget_hist

        # 如果历史样本超过预算，选择最重要的
        if len(all_previous) > budget_hist:
            hist_importance = self._evaluate_importance(
                model, all_data, all_targets, all_indices, all_previous
            )
            _, hist_top = torch.topk(hist_importance, budget_hist)
            hist_selected = [all_previous[i] for i in hist_top.tolist()]
            hist_weights = torch.ones(budget_hist, device=self.device) / budget_hist
        else:
            hist_selected = all_previous
            hist_weights = torch.ones(len(hist_selected), device=self.device) / len(hist_selected)

        # 新选择的样本（可能需要截断）
        new_selected = new_indices[:budget_new]
        new_selected_weights = new_weights[:budget_new]

        # 归一化新样本权重
        if new_selected_weights.sum() > 0:
            new_selected_weights = new_selected_weights / new_selected_weights.sum()
        else:
            new_selected_weights = torch.ones(len(new_selected), device=self.device) / len(new_selected)

        # 合并
        merged_indices = hist_selected + new_selected
        merged_weights = torch.cat([hist_weights, new_selected_weights])

        return merged_indices, merged_weights

    def _extract_data(self, dataset):
        """从 DataLoader 提取所有数据和标签"""
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
