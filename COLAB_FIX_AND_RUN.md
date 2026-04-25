# 🚀 Google Colab 快速运行指南 (已修复所有 Bug)

## ⚡ 一键运行（推荐）

在 Colab 的第一个单元格中复制粘贴以下代码：

```python
# ==================== Colab 一键启动 ====================
# 1. 克隆项目
!git clone https://github.com/abubrak/coreset-empirical-study.git
%cd coreset-empirical-study

# 2. 安装依赖
!pip install -q torch torchvision numpy matplotlib seaborn pyyaml tqdm pandas

# 3. 自动修复所有 bug
!sed -i 's/from torch.utils.data import DataLoader, Subset, random/import random\nfrom torch.utils.data import DataLoader, Subset/' data/datasets.py
!sed -i 's/return img, label, self.indices\[idx\]/return img, label/' data/datasets.py
!sed -i 's/rel_losses.append(loss.sum(dim=0))/rel_losses.append(loss)/' core/methods/csrel.py

# 4. 添加 _parse_batch 函数
!cat > /tmp/fix_parse_batch.py << 'EOF'
import sys

files_to_fix = [
    'core/coreset_base.py',
    'core/methods/greedy.py',
    'core/methods/csrel.py',
    'core/methods/bcsr.py'
]

for filepath in files_to_fix:
    with open(filepath, 'r') as f:
        content = f.read()

    # 添加导入
    if 'from ..coreset_base import' in content and '_parse_batch' not in content:
        content = content.replace(
            'from ..coreset_base import CoresetSelector',
            'from ..coreset_base import CoresetSelector, _parse_batch'
        )

    with open(filepath, 'w') as f:
        f.write(content)

print("✓ 已更新导入")

# 添加 _parse_batch 函数到 coreset_base.py
with open('core/coreset_base.py', 'r') as f:
    content = f.read()

if '_parse_batch' not in content:
    import_section = "from torch.utils.data import DataLoader, Subset\n"
    parse_batch_func = '''from torch.utils.data import DataLoader, Subset


def _parse_batch(batch):
    """解析数据批次，支持多种返回格式"""
    if len(batch) == 3:
        x, y, indices = batch
    elif len(batch) == 2:
        x, y = batch
        indices = torch.arange(x.size(0))
    else:
        raise ValueError(f"Unexpected batch format: {len(batch)} elements")
    return x, y, indices
'''
    content = content.replace(import_section, parse_batch_func)

    with open('core/coreset_base.py', 'w') as f:
        f.write(content)
    print("✓ 已添加 _parse_batch 函数")
EOF

!python /tmp/fix_parse_batch.py

# 5. 验证修复
import sys
sys.path.insert(0, '.')
from core import get_selector, ContinualLearningFramework
from data.datasets import ContinualDataset
print("✓ 模块导入成功！")

# 6. 运行快速实验
print("\n开始运行实验...")
!python experiments/run_comparison.py --dataset mnist --method random csrel --quick

# 7. 生成分析
import glob
latest = max(glob.glob('results/comparison_*.json'))
!python experiments/analysis.py "{latest}"

# 8. 显示结果
import pandas as pd
df = pd.read_csv('results/figures/summary_table.csv')
print("\n实验结果:")
print(df.to_string(index=False))

print("\n✅ 完成！查看 results/figures/ 获取图表")
```

---

## 🔍 如果还是出错 - 使用调试脚本

```python
# 运行调试脚本检查数据维度
!python debug_test.py
```

---

## 📊 查看结果

运行完成后，查看生成的图表：

```python
from IPython.display import Image, display
import glob

for fig in sorted(glob.glob('results/figures/*.png')):
    print(f"📊 {fig.split('/')[-1]}")
    display(Image(filename=fig, width=600))
```

---

## 📥 下载结果

```python
from google.colab import files
!zip -r results.zip results/
files.download('results.zip')
```

---

## 🐛 已修复的 Bug

1. ✅ **Import 错误**: `random` 不是 `torch.utils.data` 的成员
2. ✅ **数据格式不统一**: 添加了 `_parse_batch()` 统一处理
3. ✅ **维度错误**: 修复了 CSReL 中的 `loss.sum(dim=0)`
4. ✅ **返回值过多**: 修复了 RotatedMNIST 返回 3 个值

---

## ❓ 常见问题

### Q: 矩阵维度不匹配错误
**A**: 运行 `!python debug_test.py` 检查数据维度，然后确保模型输入维度匹配

### Q: 运行太慢
**A**:
- 使用 GPU: 运行时 → 更改运行时类型 → GPU
- 减小 batch_size
- 使用 `--quick` 模式

### Q: 内存不足
**A**:
- 减小 memory_budget
- 减少任务数
- 使用更小的模型

---

## 🎯 下一步

1. **快速测试**: 使用上面的 `--quick` 命令
2. **完整实验**: 修改参数运行更多方法
3. **自定义**: 编辑配置文件 `configs/experiments.yaml`

---

## 📞 获取帮助

- 查看 `debug_test.py` 的输出诊断问题
- 检查 `results/logs/` 中的日志文件
- 参考 `COLAB_USAGE.md` 获取更多示例

---

**祝实验顺利！** 🎉
