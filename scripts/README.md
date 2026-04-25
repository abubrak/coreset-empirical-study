# 实验运行脚本使用说明

本目录包含用于运行核心集选择实验的便捷脚本。

## 📁 文件列表

### Python 脚本（推荐，跨平台）

- **`run_experiments.py`** - 通用实验运行脚本
- **`colab_helper.py`** - Colab 专用脚本（带进度显示）
- **`analyze_results.py`** - 结果分析和可视化

### Shell 脚本（Linux/Mac）

- **`quick_test.sh`** - 快速验证测试
- **`run_single_dataset.sh`** - 单数据集实验
- **`run_full_experiments.sh`** - 完整论文实验

### 批处理脚本（Windows）

- **`quick_test.bat`** - 快速验证测试

---

## 🚀 快速开始

### 1. 快速验证（5-10 分钟）

验证所有 bug 已修复：

```bash
# 方法 1: Python 脚本（推荐）
python scripts/run_experiments.py --quick

# 方法 2: Shell 脚本（Linux/Mac）
bash scripts/quick_test.sh

# 方法 3: 批处理（Windows）
scripts\quick_test.bat

# 方法 4: 直接运行
python run_quick.py
```

**预期结果：**
- ✅ random: 无 IndexError
- ✅ greedy: 无 IndexError
- ✅ csrel: 无 tensor size mismatch
- ✅ bcsr: 无 CUDA OOM

---

### 2. 单数据集实验（20-30 分钟）

在单个数据集上运行所有方法：

```bash
# MNIST
python scripts/run_experiments.py --dataset mnist

# CIFAR-10
python scripts/run_experiments.py --dataset cifar10

# 指定方法
python scripts/run_experiments.py --dataset mnist --methods random,csrel
```

**结果位置：** `results/mnist_YYYYMMDD_HHMMSS/`

---

### 3. 完整论文实验（2-4 小时）

运行所有数据集和方法：

```bash
# Python 脚本
python scripts/run_experiments.py --full

# Shell 脚本
bash scripts/run_full_experiments.sh
```

**结果位置：** `results/full_experiments_YYYYMMDD_HHMMSS/`

---

### 4. Colab 运行（推荐）

在 Google Colab 中运行：

```python
# 在 Colab 单元格中
!python scripts/colab_helper.py --dataset mnist

# 或运行所有数据集
!python scripts/colab_helper.py --all
```

**优点：**
- 免费GPU访问
- 进度可视化
- 错误提示

---

## 📊 结果分析

### 查看结果汇总

```bash
# 查看所有结果的汇总表格
python scripts/analyze_results.py --input results/

# 生成 LaTeX 表格
python scripts/analyze_results.py --input results/ --latex results/table.tex

# 导出 CSV
python scripts/analyze_results.py --input results/ --csv results/results.csv
```

### 手动查看单个结果

```bash
# 查看结果 JSON
cat results/mist_*/comparison_*.json | python -m json.tool

# 或使用 jq（如果已安装）
cat results/mist_*/comparison_*.json | jq '.summary'
```

---

## 📋 实验配置说明

### 快速模式（--quick）

- **Epochs**: 10（而非 50）
- **运行次数**: 1（而非 3）
- **Memory ratios**: 仅 0.1
- **预计时间**: 单数据集 20-30 分钟

### 完整模式（无 --quick）

- **Epochs**: 50
- **运行次数**: 3
- **Memory ratios**: 0.05, 0.1, 0.2
- **预计时间**: 单数据集 2-4 小时

---

## 🛠️ 故障排除

### 问题 1: CUDA Out of Memory

**错误信息：**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解决方案：**
1. 减小 BCSR 的 batch_size：
   ```python
   # 在 core/methods/bcsr.py:79 修改
   batch_size = 128  # 从 256 改为 128
   ```

2. 或使用 CPU 模式：
   ```bash
   CUDA_VISIBLE_DEVICES="" python scripts/run_experiments.py --quick
   ```

### 问题 2: 实验速度慢

**解决方案：**
1. 使用 `--quick` 参数
2. 减少运行次数（修改配置文件）
3. 使用更少的方法：
   ```bash
   python scripts/run_experiments.py --dataset mnist --methods random,csrel
   ```

### 问题 3: 结果差异大

**解决方案：**
1. 增加运行次数到 5 或更多
2. 检查随机种子设置
3. 使用完整模式（更多 epochs）

---

## 📈 论文写作

### 所需文件

1. **结果数据**：`results/full_experiments_*/` 目录
2. **LaTeX 表格**：使用 `--latex` 生成
3. **CSV 数据**：使用 `--csv` 导出
4. **图表**：使用分析脚本生成

### 推荐流程

1. ✅ 运行快速验证（确保修复有效）
2. 📊 运行完整实验（收集论文数据）
3. 📈 分析结果（生成表格和图表）
4. 📝 撰写论文（使用生成的数据）

---

## 📞 获取帮助

- 📖 查看详细指南：`EXPERIMENT_GUIDE.md`
- 🐛 报告问题：GitHub Issues
- 💬 讨论：项目 Wiki

---

## 🎯 快速参考

| 任务 | 命令 | 时间 |
|------|------|------|
| 验证修复 | `python scripts/run_experiments.py --quick` | 5-10 分钟 |
| 单数据集 | `python scripts/run_experiments.py --dataset mnist` | 20-30 分钟 |
| 完整实验 | `python scripts/run_experiments.py --full` | 2-4 小时 |
| 查看结果 | `python scripts/analyze_results.py --input results/` | < 1 分钟 |
| 生成表格 | `python scripts/analyze_results.py --input results/ --latex table.tex` | < 1 分钟 |

祝实验顺利！🚀
