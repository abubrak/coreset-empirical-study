#!/usr/bin/env python
"""
快速修复脚本 - 在 Colab 中运行此脚本以修复所有已知 bug
"""

print("正在修复项目中的 bug...")

# 修复 1: data/datasets.py 的 import 错误
print("\n[1/6] 修复 data/datasets.py...")
with open('data/datasets.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修复 import 错误
old_import = "from torch.utils.data import DataLoader, Subset, random"
new_import = """import random
from torch.utils.data import DataLoader, Subset"""

content = content.replace(old_import, new_import)

with open('data/datasets.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("✓ 已修复 data/datasets.py")

# 修复 2: core/coreset_base.py - 添加 _parse_batch 函数
print("\n[2/6] 修复 core/coreset_base.py...")
with open('core/coreset_base.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 在文件开头添加 _parse_batch 函数
import_section = """from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset"""

new_import_section = import_section + """


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

    return x, y, indices"""

content = content.replace(import_section, new_import_section)

# 更新所有使用 _parse_batch 的地方
content = content.replace(
    "for batch_x, batch_y, idx in dataset:",
    "for batch in dataset:\n                batch_x, batch_y, _ = _parse_batch(batch)"
)

content = content.replace(
    "for batch_x, batch_y in train_loader:",
    "for batch in train_loader:\n                batch_x, batch_y, _ = _parse_batch(batch)"
)

with open('core/coreset_base.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("✓ 已修复 core/coreset_base.py")

# 修复 3: core/methods/greedy.py
print("\n[3/6] 修复 core/methods/greedy.py...")
with open('core/methods/greedy.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    "from ..coreset_base import CoresetSelector",
    "from ..coreset_base import CoresetSelector, _parse_batch"
)

# 修复 _extract_features 方法
old_extract = """        with torch.no_grad():
            for batch in dataset:
                # 兼容不同数据格式 (x, y) 或 (x, y, idx)
                if len(batch) == 3:
                    batch_x, _, idx = batch
                else:
                    batch_x, _ = batch
                    idx = torch.arange(batch_x.size(0))

                batch_x = batch_x.to(self.device)"""

new_extract = """        with torch.no_grad():
            for batch in dataset:
                batch_x, _, idx = _parse_batch(batch)
                batch_x = batch_x.to(self.device)"""

content = content.replace(old_extract, new_extract)

with open('core/methods/greedy.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("✓ 已修复 core/methods/greedy.py")

# 修复 4: core/methods/csrel.py
print("\n[4/6] 修复 core/methods/csrel.py...")
with open('core/methods/csrel.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    "from ..coreset_base import CoresetSelector",
    "from ..coreset_base import CoresetSelector, _parse_batch"
)

content = content.replace(
    "for batch_x, batch_y, idx in dataset:",
    "for batch in dataset:\n                batch_x, batch_y, _ = _parse_batch(batch)"
)

with open('core/methods/csrel.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("✓ 已修复 core/methods/csrel.py")

# 修复 5: core/methods/bcsr.py
print("\n[5/6] 修复 core/methods/bcsr.py...")
with open('core/methods/bcsr.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    "from ..coreset_base import CoresetSelector",
    "from ..coreset_base import CoresetSelector, _parse_batch"
)

old_extract = """        for batch in dataset:
            if len(batch) == 3:
                x, y, idx = batch
            else:
                x, y = batch
                idx = torch.arange(x.size(0))

            all_data.append(x)"""

new_extract = """        for batch in dataset:
            x, y, idx = _parse_batch(batch)
            all_data.append(x)"""

content = content.replace(old_extract, new_extract)

with open('core/methods/bcsr.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("✓ 已修复 core/methods/bcsr.py")

# 修复 6: run_quick.py 的编码问题
print("\n[6/6] 修复 run_quick.py...")
with open('run_quick.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 移除 Windows 特定的编码设置（在 Colab 中不需要）
if 'io.TextIOWrapper' in content:
    old_header = '''"""
快速验证脚本
在 MNIST 上运行小规模实验，验证代码框架可用性
"""
import sys
import os
# 设置 UTF-8 编码输出（Windows 兼容）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))'''

    new_header = '''"""
快速验证脚本
在 MNIST 上运行小规模实验，验证代码框架可用性
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))'''

    content = content.replace(old_header, new_header)

with open('run_quick.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("✓ 已修复 run_quick.py")

print("\n" + "="*60)
print("✅ 所有 bug 已修复！")
print("="*60)
print("\n现在可以运行以下命令：")
print("  !python run_quick.py")
print("\n或者在 notebook 中运行：")
print("  from core import get_selector")
print("  from data.datasets import ContinualDataset")
