#!/usr/bin/env python
"""
完整修复脚本 - 修复所有已知的 bug
"""

import os
import re

print("="*70)
print("核心集选择项目 - 完整 Bug 修复")
print("="*70)

# ==================== 修复 1: data/datasets.py ====================
print("\n[修复 1/8] data/datasets.py - import 错误")
with open('data/datasets.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修复 import
old_import = "from torch.utils.data import DataLoader, Subset, random"
new_import = """import random
from torch.utils.data import DataLoader, Subset"""
content = content.replace(old_import, new_import)

with open('data/datasets.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("✓ 已修复 import 错误")

# ==================== 修复 2: core/coreset_base.py ====================
print("\n[修复 2/8] core/coreset_base.py - 添加 _parse_batch")
with open('core/coreset_base.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 在 import 后添加 _parse_batch 函数
import_marker = "from torch.utils.data import DataLoader, Subset\n"
parse_batch_func = """from torch.utils.data import DataLoader, Subset


def _parse_batch(batch):
    \"\"\"
    解析数据批次，支持多种返回格式

    Args:
        batch: 数据批次，可能是 (x, y) 或 (x, y, idx)

    Returns:
        (x, y, indices): 数据、标签和索引
    \"\"\"
    if len(batch) == 3:
        x, y, indices = batch
    elif len(batch) == 2:
        x, y = batch
        indices = torch.arange(x.size(0))
    else:
        raise ValueError(f"Unexpected batch format: {len(batch)} elements")

    return x, y, indices
"""

content = content.replace(import_marker, parse_batch_func)

# 修复 compute_reducible_loss 方法
old_compute = """    def compute_reducible_loss(
        self,
        dataset: DataLoader,
        model_full: nn.Module,
        model_subset: nn.Module
    ) -> torch.Tensor:
        \"\"\"
        计算可约损失 (用于CSReL方法)

        Args:
            dataset: 数据集
            model_full: 在全量数据上训练的模型
            model_subset: 在子集上训练的模型

        Returns:
            每个样本的可约损失值
        \"\"\"
        model_full.eval()
        model_subset.eval()

        rel_losses = []

        with torch.no_grad():
            for batch_x, batch_y, idx in dataset:
                batch_x = batch_x.to(self.device)

                # 全量模型的损失
                logits_full = model_full(batch_x)
                loss_full = nn.functional.cross_entropy(logits_full, batch_y.to(self.device))

                # 子集模型的损失
                logits_sub = model_subset(batch_x)
                loss_sub = nn.functional.cross_entropy(logits_sub, batch_y.to(self.device))

                # 可约损失 = 子集损失 - 全量损失
                rel_loss = loss_sub - loss_full
                rel_losses.append(rel_loss)

        return torch.cat(rel_losses)"""

new_compute = """    def compute_reducible_loss(
        self,
        dataset: DataLoader,
        model_full: nn.Module,
        model_subset: nn.Module
    ) -> torch.Tensor:
        \"\"\"
        计算可约损失 (用于CSReL方法)

        Args:
            dataset: 数据集
            model_full: 在全量数据上训练的模型
            model_subset: 在子集上训练的模型

        Returns:
            每个样本的可约损失值
        \"\"\"
        model_full.eval()
        model_subset.eval()

        rel_losses = []

        with torch.no_grad():
            for batch in dataset:
                batch_x, batch_y, _ = _parse_batch(batch)
                batch_x = batch_x.to(self.device)

                # 全量模型的损失
                logits_full = model_full(batch_x)
                loss_full = nn.functional.cross_entropy(logits_full, batch_y.to(self.device))

                # 子集模型的损失
                logits_sub = model_subset(batch_x)
                loss_sub = nn.functional.cross_entropy(logits_sub, batch_y.to(self.device))

                # 可约损失 = 子集损失 - 全量损失
                rel_loss = loss_sub - loss_full
                rel_losses.append(rel_loss)

        return torch.cat(rel_losses)"""

content = content.replace(old_compute, new_compute)

# 修复 train_task 方法
old_train = """        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)

                # 如果提供了核心集权重，应用加权损失
                if coreset_weights is not None:
                    loss = (criterion(outputs, batch_y) * coreset_weights).mean()
                else:
                    loss = criterion(outputs, batch_y)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_x.size(0)

            avg_loss = epoch_loss / total
            accuracy = correct / total

            metrics['train_losses'].append(avg_loss)
            metrics['train_accuracy'].append(accuracy)"""

new_train = """        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch in train_loader:
                batch_x, batch_y, _ = _parse_batch(batch)
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)

                # 如果提供了核心集权重，应用加权损失
                if coreset_weights is not None:
                    loss = (criterion(outputs, batch_y) * coreset_weights).mean()
                else:
                    loss = criterion(outputs, batch_y)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_x.size(0)

            avg_loss = epoch_loss / total
            accuracy = correct / total

            metrics['train_losses'].append(avg_loss)
            metrics['train_accuracy'].append(accuracy)"""

content = content.replace(old_train, new_train)

# 添加 memory_budget 验证
old_init = """    def __init__(self, memory_budget: int, device: torch.device = None):
        \"\"\"
        Args:
            memory_budget: 核心集大小限制
            device: 计算设备
        \"\"\"
        self.memory_budget = memory_budget
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.selected_indices = None
        self.selection_weights = None"""

new_init = """    def __init__(self, memory_budget: int, device: torch.device = None):
        \"\"\"
        Args:
            memory_budget: 核心集大小限制
            device: 计算设备
        \"\"\"
        if memory_budget <= 0:
            raise ValueError(f"memory_budget must be positive, got {memory_budget}")

        self.memory_budget = memory_budget
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.selected_indices = None
        self.selection_weights = None"""

content = content.replace(old_init, new_init)

with open('core/coreset_base.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("✓ 已修复 core/coreset_base.py")

# ==================== 修复 3: core/methods/greedy.py ====================
print("\n[修复 3/8] core/methods/greedy.py")
with open('core/methods/greedy.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    "from ..coreset_base import CoresetSelector",
    "from ..coreset_base import CoresetSelector, _parse_batch"
)

old_extract = """    def _extract_features(self, dataset, model):
        \"\"\"提取样本特征\"\"\"
        features_list = []
        indices_map = []

        with torch.no_grad():
            for batch in dataset:
                # 兼容不同数据格式 (x, y) 或 (x, y, idx)
                if len(batch) == 3:
                    batch_x, _, idx = batch
                else:
                    batch_x, _ = batch
                    idx = torch.arange(batch_x.size(0))

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

        return features, indices_map"""

new_extract = """    def _extract_features(self, dataset, model):
        \"\"\"提取样本特征\"\"\"
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

        return features, indices_map"""

content = content.replace(old_extract, new_extract)

with open('core/methods/greedy.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("✓ 已修复 core/methods/greedy.py")

# ==================== 修复 4: core/methods/csrel.py ====================
print("\n[修复 4/8] core/methods/csrel.py")
with open('core/methods/csrel.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    "from ..coreset_base import CoresetSelector",
    "from ..coreset_base import CoresetSelector, _parse_batch"
)

old_compute_rel = """    def _compute_reducible_losses(self, dataset, model):
        \"\"\"计算所有样本的可约损失\"\"\"
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

        return torch.cat(rel_losses)"""

new_compute_rel = """    def _compute_reducible_losses(self, dataset, model):
        \"\"\"计算所有样本的可约损失\"\"\"
        rel_losses = []

        with torch.no_grad():
            for batch in dataset:
                batch_x, batch_y, _ = _parse_batch(batch)
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # 使用当前模型在样本上的损失作为代理
                # 这里简化处理：使用损失值近似
                logits = model(batch_x)
                probs = torch.softmax(logits, dim=1)
                loss = -probs[range(len(batch_y)), batch_y].log()

                # 可约损失越高，说明模型对该样本的不确定性越高
                rel_losses.append(loss)

        return torch.cat(rel_losses)"""

content = content.replace(old_compute_rel, new_compute_rel)

with open('core/methods/csrel.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("✓ 已修复 core/methods/csrel.py")

# ==================== 修复 5: core/methods/bcsr.py ====================
print("\n[修复 5/8] core/methods/bcsr.py")
with open('core/methods/bcsr.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    "from ..coreset_base import CoresetSelector",
    "from ..coreset_base import CoresetSelector, _parse_batch"
)

old_extract_data = """    def _extract_data(self, dataset):
        \"\"\"从 DataLoader 提取所有数据和标签\"\"\"
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

        return all_data, all_targets, all_indices"""

new_extract_data = """    def _extract_data(self, dataset):
        \"\"\"从 DataLoader 提取所有数据和标签\"\"\"
        all_data = []
        all_targets = []
        all_indices = []

        for batch in dataset:
            x, y, idx = _parse_batch(batch)
            all_data.append(x)
            all_targets.append(y)
            all_indices.extend(idx.tolist())

        all_data = torch.cat(all_data, dim=0).to(self.device)
        all_targets = torch.cat(all_targets, dim=0).to(self.device)

        return all_data, all_targets, all_indices"""

content = content.replace(old_extract_data, new_extract_data)

with open('core/methods/bcsr.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("✓ 已修复 core/methods/bcsr.py")

# ==================== 修复 6: data/datasets.py - RotatedMNIST 返回值 ====================
print("\n[修复 6/8] data/datasets.py - RotatedMNIST 返回格式")
with open('data/datasets.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修复 RotatedMNIST 的 __getitem__ 方法
old_getitem = """    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        img = self.transforms(img)
        return img, label, self.indices[idx]"""

new_getitem = """    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        img = self.transforms(img)
        return img, label"""

content = content.replace(old_getitem, new_getitem)

with open('data/datasets.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("✓ 已修复 RotatedMNIST 返回格式")

# ==================== 修复 7: experiments/run_comparison.py ====================
print("\n[修复 7/8] experiments/run_comparison.py - 路径兼容性")
with open('experiments/run_comparison.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_import = """# 项目内部导入
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import get_selector, ContinualLearningFramework
from data.datasets import ContinualDataset"""

new_import = """# 项目内部导入
import sys
# 使用绝对路径的父目录，确保跨平台兼容
_current_file = Path(__file__).absolute()
_project_root = _current_file.parent.parent
sys.path.insert(0, str(_project_root))

from core import get_selector, ContinualLearningFramework
from data.datasets import ContinualDataset"""

content = content.replace(old_import, new_import)

with open('experiments/run_comparison.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("✓ 已修复 experiments/run_comparison.py")

# ==================== 修复 8: run_quick.py ====================
print("\n[修复 8/8] run_quick.py - 编码问题")
with open('run_quick.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 移除 Windows 编码设置，简化为跨平台版本
quick_header = '''"""
快速验证脚本
在 MNIST 上运行小规模实验，验证代码框架可用性
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from core import get_selector
from data.datasets import ContinualDataset


def quick_test():
    """快速测试所有方法"""
    print("=" * 50)
    print("快速验证测试")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载 MNIST
    dataset = ContinualDataset('mnist', batch_size=64, num_workers=0)
    dataset.split_tasks(2, 'split')
    num_classes = len(dataset.full_dataset.classes)
    print(f"数据集: MNIST, {num_classes} 类")

    # 简单模型
    import torch.nn as nn
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    ).to(device)

    methods = ['random', 'greedy', 'csrel', 'bcsr', 'ensemble']
    memory_budget = 500

    for method_name in methods:
        print(f"\\n--- 测试 {method_name} ---")
        try:
            selector = get_selector(
                method_name,
                memory_budget=memory_budget,
                device=device
            )
            train_loader = dataset.get_task_loaders(0)
            indices, weights = selector.select_coreset(
                train_loader, model, task_id=0
            )
            print(f"  选择 {len(indices)} 样本, "
                  f"权重范围 [{weights.min():.4f}, {weights.max():.4f}]")
            print(f"  [OK] {method_name} 通过")
        except Exception as e:
            print(f"  [FAIL] {method_name} 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\\n" + "=" * 50)
    print("验证完成!")


if __name__ == '__main__':
    quick_test()'''

# 找到 """快速验证脚本""" 到 if __name__ == '__main__': 之间的内容并替换
pattern = r'""".*?""".*?(?=\nif __name__)'
content_new = re.sub(pattern, quick_header.split('\n\n\nif __name__')[0], content, flags=re.DOTALL, count=1)

# 确保有正确的 if __name__ 部分
if 'if __name__' not in content_new or 'quick_test()' not in content_new:
    # 如果替换失败，直接写入完整内容
    content_new = quick_header

with open('run_quick.py', 'w', encoding='utf-8') as f:
    f.write(content_new)
print("✓ 已修复 run_quick.py")

print("\n" + "="*70)
print("✅ 所有 Bug 已修复!")
print("="*70)
print("\n现在可以运行实验了:")
print("  !python run_quick.py          # 快速验证")
print("  !python experiments/run_comparison.py --quick  # 运行实验")
