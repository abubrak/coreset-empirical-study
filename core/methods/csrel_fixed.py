"""
CSReL: 基于可约损失的核心集选择（修正版）

基于原始论文实现的核心集选择方法
论文：Coreset Selection via Reducible Loss in Continual Learning (ICLR 2025)
GitHub: https://github.com/RuilinTong/CSReL-Coreset-CL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import Tuple, Optional, List, Dict
from ..coreset_base import CoresetSelector, _parse_batch


class CSReLSelectorFixed(CoresetSelector):
    """
    CSReL (Coreset Selection via Reducible Loss) - 修正版

    关键修正：
    1. 正确实现 Reducible Loss = Loss_current - Loss_reference
    2. 使用确定性选择（Top-K）而非随机采样
    3. 实现增量式选择流程
    4. 支持类别平衡
    5. 移除不必要的温度参数
    """

    def __init__(
        self,
        memory_budget: int,
        device=None,
        init_size: int = 0,
        selection_steps: int = 5,
        class_balance: bool = False,
        ref_epochs: int = 10
    ):
        """
        Args:
            memory_budget: 核心集大小限制
            device: 计算设备
            init_size: 初始随机选择的大小
            selection_steps: 增量选择的步数
            class_balance: 是否使用类别平衡
            ref_epochs: 参考模型训练轮数
        """
        super().__init__(memory_budget, device)
        self.init_size = init_size
        self.selection_steps = selection_steps
        self.class_balance = class_balance
        self.ref_epochs = ref_epochs

        # 存储参考模型和参考损失
        self.ref_model = None
        self.ref_losses = None
        self.reloss_history = []

    def select_coreset(
        self,
        dataset,
        model,
        task_id: int,
        previous_coresets=None
    ) -> Tuple[List[int], torch.Tensor]:
        """
        基于可约损失选择核心集

        Args:
            dataset: 数据加载器
            model: 当前模型
            task_id: 任务ID
            previous_coresets: 之前任务的核心集索引列表

        Returns:
            (selected_indices, weights): 选择的索引和权重
        """
        # 转换为列表格式以便处理
        all_data = self._collect_all_data(dataset)
        num_samples = len(all_data)

        if num_samples <= self.memory_budget:
            # 如果数据量小于预算，全部选择
            selected_indices = list(range(num_samples))
            weights = torch.ones(num_samples, device=self.device)
            return selected_indices, weights

        # 第一步：训练参考模型（如果还没有）
        if self.ref_model is None:
            print("Training reference model on full dataset...")
            self.ref_model = self._train_reference_model(dataset, model)

        # 第二步：计算参考损失
        if self.ref_losses is None:
            print("Computing reference losses...")
            self.ref_losses = self._compute_losses(
                dataset,
                self.ref_model,
                batch_size=32
            )

        # 第三步：初始化选择
        selected_indices = []
        current_model = model

        # 如果有初始大小，先随机选择
        if self.init_size > 0:
            if self.class_balance:
                init_indices = self._balanced_sample(
                    all_data,
                    self.init_size
                )
            else:
                init_indices = random.sample(
                    list(range(num_samples)),
                    self.init_size
                )
            selected_indices.extend(init_indices)

            # 在初始核心集上训练模型
            init_subset = self._create_subset(dataset, selected_indices)
            current_model = self._train_model(
                current_model,
                init_subset,
                epochs=5
            )

        # 第四步：增量式选择
        incremental_size = max(
            (self.memory_budget - len(selected_indices)) // self.selection_steps,
            1
        )

        print(f"Starting incremental selection...")
        print(f"Already selected: {len(selected_indices)}, Target: {self.memory_budget}")
        print(f"Incremental size per step: {incremental_size}, Steps: {self.selection_steps}")

        step = 0
        while len(selected_indices) < self.memory_budget and step < self.selection_steps:
            # 计算当前模型的损失
            current_losses = self._compute_losses(
                dataset,
                current_model,
                batch_size=32
            )

            # 计算 Reducible Loss
            reducible_loss = current_losses - self.ref_losses

            # 候选样本池（未选择的样本）
            candidate_pool = [
                i for i in range(num_samples)
                if i not in selected_indices
            ]

            # 计算 Reducible Loss
            candidate_rel = reducible_loss[candidate_pool]

            # 选择策略
            if self.class_balance:
                # 类别平衡选择
                new_indices = self._select_by_reducible_loss_balanced(
                    all_data,
                    candidate_pool,
                    candidate_rel,
                    min(incremental_size, self.memory_budget - len(selected_indices))
                )
            else:
                # 直接选择 Top-K
                top_k = min(
                    incremental_size,
                    self.memory_budget - len(selected_indices)
                )
                _, top_indices = torch.topk(candidate_rel, top_k)
                new_indices = [candidate_pool[i] for i in top_indices.tolist()]

            # 添加到核心集
            selected_indices.extend(new_indices)
            print(f"Step {step + 1}: Selected {len(new_indices)} samples, "
                  f"Total: {len(selected_indices)}/{self.memory_budget}")

            # 在新的核心集上重新训练模型
            new_subset = self._create_subset(dataset, selected_indices)
            current_model = self._train_model(
                current_model,
                new_subset,
                epochs=3
            )

            step += 1

        # 如果还没选满，用剩余步骤继续选择
        while len(selected_indices) < self.memory_budget:
            # 计算当前模型的损失
            current_losses = self._compute_losses(
                dataset,
                current_model,
                batch_size=32
            )

            # 计算 Reducible Loss
            reducible_loss = current_losses - self.ref_losses

            # 候选样本池
            candidate_pool = [
                i for i in range(num_samples)
                if i not in selected_indices
            ]
            candidate_rel = reducible_loss[candidate_pool]

            # 选择剩余样本
            remaining = self.memory_budget - len(selected_indices)
            _, top_indices = torch.topk(candidate_rel, remaining)
            new_indices = [candidate_pool[i] for i in top_indices.tolist()]
            selected_indices.extend(new_indices)

            print(f"Final selection: Added {len(new_indices)} samples, "
                  f"Total: {len(selected_indices)}/{self.memory_budget}")
            break

        # 计算权重（简单起见，使用均等权重）
        weights = torch.ones(len(selected_indices), device=self.device)

        # 记录历史
        self.selected_indices = selected_indices
        self.selection_weights = weights
        self.reloss_history.append(reducible_loss[selected_indices].mean().item())

        print(f"Selection completed. Total samples: {len(selected_indices)}")
        print(f"Average reducible loss: {self.reloss_history[-1]:.4f}")

        return selected_indices, weights

    def _train_reference_model(self, dataset, base_model):
        """
        训练参考模型（在全量数据上）

        Args:
            dataset: 数据加载器
            base_model: 基础模型架构

        Returns:
            训练好的参考模型
        """
        # 创建模型副本
        ref_model = type(base_model)(
            *base_model.__init__.__code__.co_varnames[:1]
        ).to(self.device)

        # 复制参数
        ref_model.load_state_dict(base_model.state_dict())

        # 简单训练
        optimizer = torch.optim.SGD(
            ref_model.parameters(),
            lr=0.01,
            momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()

        ref_model.train()
        for epoch in range(self.ref_epochs):
            total_loss = 0.0
            for batch in dataset:
                batch_x, batch_y, _ = _parse_batch(batch)
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = ref_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"Reference model training epoch {epoch + 1}/{self.ref_epochs}, "
                      f"Loss: {total_loss / len(dataset):.4f}")

        return ref_model

    def _compute_losses(self, dataset, model, batch_size=32):
        """
        计算模型在数据集上每个样本的损失

        Args:
            dataset: 数据加载器
            model: 模型
            batch_size: 批大小

        Returns:
            每个样本的损失（Tensor）
        """
        model.eval()
        all_losses = []

        with torch.no_grad():
            batch_x_list = []
            batch_y_list = []
            sample_indices = []

            for batch_idx, batch in enumerate(dataset):
                batch_x, batch_y, _ = _parse_batch(batch)
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                batch_x_list.append(batch_x)
                batch_y_list.append(batch_y)

                # 累积到 batch_size 或最后一个批次
                current_batch_size = sum(x.size(0) for x in batch_x_list)

                if current_batch_size >= batch_size or batch_idx == len(dataset) - 1:
                    # 拼接批次
                    combined_x = torch.cat(batch_x_list, dim=0)
                    combined_y = torch.cat(batch_y_list, dim=0)

                    # 计算损失
                    outputs = model(combined_x)
                    loss = F.cross_entropy(
                        outputs,
                        combined_y,
                        reduction='none'
                    )

                    all_losses.append(loss.cpu())

                    # 清空
                    batch_x_list.clear()
                    batch_y_list.clear()

        # 合并所有损失
        all_losses = torch.cat(all_losses, dim=0)
        return all_losses

    def _train_model(self, model, dataset, epochs=5):
        """
        在数据集上训练模型

        Args:
            model: 模型
            dataset: 数据加载器
            epochs: 训练轮数

        Returns:
            训练好的模型
        """
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataset:
                batch_x, batch_y, _ = _parse_batch(batch)
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        return model

    def _collect_all_data(self, dataset):
        """
        收集所有数据到列表

        Args:
            dataset: 数据加载器

        Returns:
            数据列表 [(x, y), ...]
        """
        all_data = []
        for batch in dataset:
            batch_x, batch_y, _ = _parse_batch(batch)
            for i in range(batch_x.size(0)):
                all_data.append((batch_x[i], batch_y[i].item()))
        return all_data

    def _create_subset(self, original_dataset, indices):
        """
        根据索引创建数据子集

        Args:
            original_dataset: 原始数据加载器
            indices: 索引列表

        Returns:
            子集数据加载器
        """
        from torch.utils.data import Subset, DataLoader

        # 收集所有数据
        all_data = []
        for batch in original_dataset:
            batch_x, batch_y, _ = _parse_batch(batch)
            for i in range(batch_x.size(0)):
                all_data.append((batch_x[i], batch_y[i].item()))

        # 创建子集
        subset_data = [all_data[i] for i in indices]
        subset_dataset = SimpleDataset(subset_data)

        return DataLoader(
            subset_dataset,
            batch_size=original_dataset.batch_size,
            shuffle=False
        )

    def _balanced_sample(self, all_data, sample_size):
        """
        类别平衡采样

        Args:
            all_data: 所有数据
            sample_size: 采样大小

        Returns:
            采样索引列表
        """
        # 按类别分组
        class_to_indices = {}
        for idx, (_, label) in enumerate(all_data):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)

        num_classes = len(class_to_indices)
        samples_per_class = max(sample_size // num_classes, 1)

        selected_indices = []
        for class_indices in class_to_indices.values():
            if len(class_indices) >= samples_per_class:
                selected_indices.extend(random.sample(class_indices, samples_per_class))
            else:
                selected_indices.extend(class_indices)

        # 如果还不够，随机补充
        if len(selected_indices) < sample_size:
            remaining = sample_size - len(selected_indices)
            available = [i for i in range(len(all_data)) if i not in selected_indices]
            selected_indices.extend(random.sample(available, min(remaining, len(available))))

        return selected_indices[:sample_size]

    def _select_by_reducible_loss_balanced(
        self,
        all_data,
        candidate_pool,
        candidate_rel,
        select_size
    ):
        """
        类别平衡的可约损失选择

        Args:
            all_data: 所有数据
            candidate_pool: 候选样本池
            candidate_rel: 候选样本的可约损失
            select_size: 选择大小

        Returns:
            选择的索引列表
        """
        # 按类别分组候选样本
        class_to_candidates = {}
        for idx, rel_value in zip(candidate_pool, candidate_rel):
            label = all_data[idx][1]
            if label not in class_to_candidates:
                class_to_candidates[label] = []
            class_to_candidates[label].append((idx, rel_value.item()))

        num_classes = len(class_to_candidates)
        select_per_class = max(select_size // num_classes, 1)

        selected_indices = []

        # 从每个类别中选择 ReL 最高的
        for class_candidates in class_to_candidates.values():
            # 按 ReL 降序排序
            sorted_candidates = sorted(
                class_candidates,
                key=lambda x: x[1],
                reverse=True
            )

            # 选择前 k 个
            class_selected = [x[0] for x in sorted_candidates[:select_per_class]]
            selected_indices.extend(class_selected)

        # 如果还不够，从剩余候选中选择 ReL 最高的
        if len(selected_indices) < select_size:
            remaining = select_size - len(selected_indices)
            # 排除已选择的
            remaining_candidates = [
                (idx, rel.item())
                for idx, rel in zip(candidate_pool, candidate_rel)
                if idx not in selected_indices
            ]
            # 按 ReL 降序排序
            sorted_remaining = sorted(
                remaining_candidates,
                key=lambda x: x[1],
                reverse=True
            )
            # 选择前 remaining 个
            selected_indices.extend([x[0] for x in sorted_remaining[:remaining]])

        return selected_indices[:select_size]


class SimpleDataset(torch.utils.data.Dataset):
    """简单的数据集包装器"""

    def __init__(self, data):
        """
        Args:
            data: 数据列表 [(x, y), ...]
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y, idx
