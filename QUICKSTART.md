# 🚀 快速入门指南

本指南帮助你在 5 分钟内开始运行实验。

## 前提条件

- Python 3.8+
- PyTorch 2.0+
- (可选) CUDA-capable GPU

---

## 步骤 1: 安装依赖（1 分钟）

```bash
cd f:/paper/coreset-empirical-study
pip install -r requirements.txt
```

---

## 步骤 2: 快速验证（5-10 分钟）

验证所有 bug 已修复：

```bash
python scripts/run_experiments.py --quick
```

**预期输出：**
```
========================================
🔍 快速验证实验
目的: 验证所有 4 个 bug 已修复
预计时间: 5-10 分钟
========================================

🚀 运行快速测试...
[1/4] random: ✅ 完成
[2/4] greedy: ✅ 完成
[3/4] csrel: ✅ 完成
[4/4] bcsr: ✅ 完成

✅ 所有测试通过！
```

---

## 步骤 3: 查看结果（1 分钟）

```bash
python scripts/analyze_results.py --input results/
```

**输出示例：**
```
================================================================================
📊 实验结果汇总
================================================================================

数据集: MNIST
--------------------------------------------------------------------------------
方法          平均准确率      遗忘度量       选择时间(s)     训练时间(s)
--------------------------------------------------------------------------------
random        0.8234±0.0123  0.1345±0.0234  0.5            45.2
greedy        0.8567±0.0098  0.1123±0.0187  123.4          48.7
csrel         0.8789±0.0076  0.0987±0.0156  89.3           46.8
bcsr          0.8901±0.0065  0.0891±0.0132  234.5          47.1
```

---

## 常用命令

### 单数据集实验
```bash
python scripts/run_experiments.py --dataset mnist
```

### 完整实验（所有数据集）
```bash
python scripts/run_experiments.py --full
```

### 指定方法
```bash
python scripts/run_experiments.py --dataset mnist --methods random,bcsr
```

### Colab 运行
```python
!python scripts/colab_helper.py --all
```

### 生成论文表格
```bash
python scripts/analyze_results.py --input results/ --latex table.tex
```

---

## 结果文件位置

```
results/
├── quick_test_20260425_120000/        # 快速测试结果
│   └── comparison_20260425_120000.json
├── mnist_20260425_130000/             # MNIST 完整结果
│   └── comparison_20260425_130000.json
└── full_experiments_20260425_140000/  # 所有数据集
    ├── mnist/
    ├── cifar10/
    └── cifar100/
```

---

## 故障排除

### ❌ CUDA Out of Memory

**解决方案：** 使用 CPU 模式或减小 batch_size

```bash
# 使用 CPU
CUDA_VISIBLE_DEVICES="" python scripts/run_experiments.py --quick
```

### ❌ ModuleNotFoundError

**解决方案：** 确保在项目根目录运行

```bash
cd f:/paper/coreset-empirical-study
python scripts/run_experiments.py --quick
```

### ❌ 实验太慢

**解决方案：** 使用 `--quick` 参数（已默认启用）

---

## 下一步

1. ✅ 验证修复有效（已完成）
2. 📊 运行单数据集实验（20-30 分钟）
3. 📈 运行完整实验（2-4 小时）
4. 📝 分析结果，生成论文数据

---

## 需要帮助？

- 📘 **[EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)** - 详细实验流程
- 🔧 **[scripts/README.md](scripts/README.md)** - 脚本使用说明
- 🐛 **GitHub Issues** - 报告问题

祝实验顺利！🚀
