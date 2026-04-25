# 📚 Google Colab 运行指南

## 🚀 快速开始

### 方法 1: 使用简化脚本（推荐）

1. **打开 Google Colab**
   - 访问: https://colab.research.google.com/
   - 点击 "文件" → "新建笔记本"

2. **上传项目文件**
   ```python
   # 在第一个单元格中运行
   !git clone https://github.com/abubrak/coreset-empirical-study.git
   %cd coreset-empirical-study
   ```

3. **一键运行**
   ```python
   # 运行简化脚本
   !python run_colab_simple.py
   ```

### 方法 2: 使用 Notebook

1. **打开 Colab Notebook**
   ```python
   # 在 Colab 中运行
   !git clone https://github.com/abubrak/coreset-empirical-study.git
   %cd coreset-empirical-study
   ```

2. **逐个运行以下单元格**:

```python
# 单元格 1: 安装依赖
!pip install -q torch torchvision numpy matplotlib seaborn pyyaml tqdm pandas

# 单元格 2: 导入模块
import sys
sys.path.insert(0, '.')
from core import get_selector, ContinualLearningFramework
from data.datasets import ContinualDataset
import torch
import torch.nn as nn

# 单元格 3: 检查 GPU
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 单元格 4: 快速验证
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = ContinualDataset('mnist', batch_size=128, num_workers=2)
dataset.split_tasks(2, 'split')

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).to(device)

# 测试方法
selector = get_selector('random', memory_budget=200, device=device)
train_loader = dataset.get_task_loaders(0)
indices, weights = selector.select_coreset(train_loader, model, task_id=0)
print(f"✓ Random 方法: 选择 {len(indices)} 个样本")

# 单元格 5: 运行对比实验
!python experiments/run_comparison.py --dataset mnist --method random csrel --quick

# 单元格 6: 生成分析图表
import glob
latest = max(glob.glob('results/comparison_*.json'))
!python experiments/analysis.py "{latest}"

# 单元格 7: 查看结果
from IPython.display import Image, display
import glob
for fig in glob.glob('results/figures/*.png'):
    display(Image(filename=fig, width=600))

# 单元格 8: 下载结果
from google.colab import files
!zip -r results.zip results/
files.download('results.zip')
```

## 🎯 快速实验模板

### 模板 1: 单方法测试
```python
# 测试单个方法
!python experiments/run_comparison.py \
    --dataset mnist \
    --method random \
    --quick
```

### 模板 2: 多方法对比
```python
# 对比多个方法
!python experiments/run_comparison.py \
    --dataset mnist \
    --method random greedy csrel bcsr \
    --quick
```

### 模板 3: 自定义配置
```python
# 使用配置文件
!python experiments/run_comparison.py \
    --config configs/experiments.yaml
```

## 📊 实验参数说明

### 快速模式 (--quick) 默认配置:
- 数据集: MNIST
- 任务数: 2
- 方法: random, csrel
- Epochs: 10
- 内存比例: 10%
- 运行次数: 1

### 完整实验参数:
```bash
--dataset      # 数据集: mnist, cifar10, cifar100
--method       # 方法: random, greedy, csrel, bcsr, ensemble
--config       # 配置文件路径
--output       # 结果输出目录
```

## 🛠️ 常见问题

### Q1: 如何切换 GPU？
```
菜单 → 运行时 → 更改运行时类型 → 硬件加速器 → GPU
```

### Q2: 如何上传本地文件？
```python
from google.colab import files
files.upload()  # 选择文件上传
```

### Q3: 如何下载结果？
```python
from google.colab import files
files.download('results.zip')
```

### Q4: 内存不足怎么办？
- 减小 `batch_size`
- 减小 `memory_budget`
- 使用更小的数据集 (MNIST vs CIFAR)
- 减少运行的任务数

### Q5: 运行太慢怎么办？
- 确保使用 GPU 运行时
- 使用 `--quick` 模式
- 减少训练 epochs
- 减少方法数量

## 📈 预期运行时间

| 配置 | CPU | GPU |
|------|-----|-----|
| 快速验证 | 2-5 分钟 | 1-2 分钟 |
| 单方法实验 | 5-10 分钟 | 2-3 分钟 |
| 完整对比 (5方法) | 30-60 分钟 | 10-20 分钟 |

## 💡 提示和技巧

1. **保存进度**: 定期下载结果，防止会话断开
2. **使用 GPU**: 深度学习实验强烈建议使用 GPU
3. **分步运行**: 出错时可以逐步调试
4. **查看日志**: 检查 `results/logs/` 目录
5. **可视化**: 使用 matplotlib 在 notebook 中直接绘图

## 📞 获取帮助

- GitHub Issues: https://github.com/abubrak/coreset-empirical-study/issues
- 项目文档: `README.md`, `COLAB_GUIDE.md`

## 🎓 学习资源

- 核心集选择基础: `IMPLEMENTATION_ANALYSIS.md`
- 方法对比: 实验结果中的 `summary_table.csv`
- 可视化图表: `results/figures/` 目录
