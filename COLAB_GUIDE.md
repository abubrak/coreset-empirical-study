# Google Colab 使用指南

## 🚀 快速开始

### 方式一：使用 Jupyter Notebook（推荐）

1. **打开 Colab**
   - 访问：https://colab.research.google.com/
   - 点击 `文件` → `打开笔记本` → `GitHub`
   - 输入仓库地址：`abubrak/coreset-empirical-study`
   - 选择 `colab_experiment.ipynb`

2. **运行步骤**
   ```
   单元格 1: 安装依赖          (运行)
   单元格 2: 克隆仓库          (运行)
   单元格 3: 快速验证          (运行，测试所有方法)
   单元格 4: 快速实验          (运行，约 5-10 分钟)
   单元格 5: 结果分析          (运行，生成图表)
   单元格 6: 查看结果          (运行，显示图表)
   ```

3. **下载结果**
   - 实验完成后，运行"下载结果"单元格
   - 会自动下载 `results.zip`

### 方式二：使用 Python 脚本

在 Colab 中新建代码笔记本，依次运行：

```python
# 1. 克隆仓库
!git clone https://github.com/abubrak/coreset-empirical-study.git
%cd coreset-empirical-study

# 2. 安装依赖
!pip install -q torch torchvision numpy scipy matplotlib seaborn pyyaml tqdm pandas

# 3. 快速验证
!python run_quick.py

# 4. 运行快速实验
!python experiments/run_comparison.py --dataset mnist --method random csrel bcsr --quick

# 5. 生成分析（假设最新结果是 comparison_XXX.json）
!python experiments/analysis.py results/comparison_XXX.json
```

## ⚙️ 实验配置

### 快速模式（推荐首次使用）
```bash
# MNIST，10 epochs，单次运行（约 5-10 分钟）
!python experiments/run_comparison.py --dataset mnist --method random csrel --quick
```

### 自定义实验
```bash
# 多数据集，多方法
!python experiments/run_comparison.py \
    --dataset mnist cifar10 \
    --method random greedy csrel bcsr ensemble \
    --config configs/experiments.yaml
```

### 完整实验（需要较长时间）
```bash
# 运行完整实验矩阵（1-2 小时）
!python experiments/run_comparison.py --config configs/experiments.yaml
```

## 📊 查看结果

### 在 Colab 中查看

```python
# 显示汇总表格
import pandas as pd
df = pd.read_csv('results/figures/summary_table.csv')
print(df.to_string(index=False))

# 显示图表
from IPython.display import Image, display
import glob
for fig in glob.glob('results/figures/*.png'):
    display(Image(filename=fig))
```

### 下载到本地

```python
# 打包并下载
!zip -r results.zip results/
from google.colab import files
files.download('results.zip')
```

## 💡 使用技巧

### 1. 启用 GPU 加速
- 点击 `运行时` → `更改运行时类型` → `硬件加速器` 选择 `GPU`
- GPU 可显著加速实验（约 3-5 倍）

### 2. 避免超时
- Colab 免费版有 90 分钟运行时间限制
- 使用 `--quick` 模式可在时间内完成
- 复杂实验可分段运行

### 3. 保存中间结果
- 实验结果会实时保存到 `results/comparison_XXX.json`
- 即使中断运行，已完成的实验不会丢失

### 4. 内存管理
- 如果遇到内存不足，减小 batch_size：
  ```python
  # 修改 configs/experiments.yaml
  training:
    batch_size: 64  # 从 128 改为 64
  ```

## 🐛 常见问题

### Q1: 导入错误 `ModuleNotFoundError`
```bash
# 解决方案：确保在项目根目录
%cd coreset-empirical-study
import sys
sys.path.insert(0, '.')
```

### Q2: 数据下载慢
```bash
# 手动下载数据到 Google Drive 后挂载
from google.colab import drive
drive.mount('/content/drive')
# 复制数据到项目目录
!cp /content/drive/MyDrive/data ./data -r
```

### Q3: CUDA 内存不足
```python
# 清理缓存
import torch
torch.cuda.empty_cache()

# 或减小批量大小
# 修改 configs/experiments.yaml 中的 batch_size
```

### Q4: 实验中断
```python
# 从上次中断处继续（results 中已有部分结果）
!python experiments/run_comparison.py --config configs/experiments.yaml
```

## 📈 预期结果

### 快速模式（5-10 分钟）
- MNIST × 3 方法 × 2 任务 × 1 次运行
- 约 6 组实验
- 生成 6-8 张图表

### 完整模式（1-2 小时）
- 多数据集 × 5 方法 × 多任务 × 3 次运行
- 约 90-150 组实验
- 生成完整分析报告

## 📚 参考资料

- **项目 README**：[README.md](README.md)
- **实现分析**：[IMPLEMENTATION_ANALYSIS.md](IMPLEMENTATION_ANALYSIS.md)
- **原始论文**：
  - BCSR: NeurIPS 2023
  - CSReL: ICLR 2025

## 🆘 需要帮助？

如果遇到问题：
1. 查看 [GitHub Issues](https://github.com/abubrak/coreset-empirical-study/issues)
2. 检查实现分析文档中的常见问题
3. 确保使用的是最新代码：`!git pull`

---

**祝实验顺利！** 🎉
