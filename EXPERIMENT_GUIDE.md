# 核心集选择实验运行指南

本指南提供了完整的实验流程，用于生成论文所需的实验数据。

## 📋 实验概览

### 数据集
- **MNIST**: 60,000 训练样本，10 类
- **CIFAR-10**: 50,000 训练样本，10 类
- **CIFAR-100**: 50,000 训练样本，100 类

### 方法
- **Random**: 随机采样基准
- **Greedy**: 贪心特征覆盖选择
- **CSReL**: 基于可约损失的选择
- **BCSR**: 双层优化选择

### 指标
- **Average Accuracy (AA)**: 所有任务的平均准确率
- **Forgetting Measure (FM)**: 平均遗忘程度
- **Selection Time**: 选择时间（秒）
- **Training Time**: 训练时间（秒）

---

## 🚀 快速开始

### 1. 验证修复（5 分钟）

首先验证 bug 已修复：

```bash
# 快速测试：MNIST + random + 2 任务 + 1 轮
python run_quick.py
```

预期输出：
- ✅ 所有 4 个方法完成实验
- ✅ 无 IndexError
- ✅ 无 RuntimeError（tensor size mismatch）
- ✅ 无 CUDA OOM

---

### 2. 完整实验（2-4 小时）

#### 方案 A：本地运行（推荐用于开发）

```bash
# 完整实验：所有数据集 + 所有方法 + 多次运行
python experiments/run_comparison.py --quick
```

**--quick 参数说明：**
- 使用较少的 epoch (10 而非 50)
- 单次运行 (而非 3 次)
- 只测试 memory_ratio=0.1

#### 方案 B：Colab 运行（推荐用于论文）

```bash
# 在 Colab 中运行
python run_colab.py
```

---

## 📊 实验配置

### 快速验证实验（用于测试）

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据集 | mnist | 单个数据集 |
| 方法 | random, greedy, csrel, bcsr | 所有方法 |
| 任务数 | 5 | 分成 5 个任务 |
| Epochs | 10 | 快速训练 |
| 运行次数 | 1 | 单次运行 |
| Memory ratio | 0.1 | 10% 核心集 |

**预计时间**: 20-30 分钟

---

### 论文级实验（用于最终结果）

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据集 | mnist, cifar10, cifar100 | 多数据集 |
| 方法 | random, greedy, csrel, bcsr | 所有方法 |
| 任务数 | 5, 10 | 不同任务数 |
| Epochs | 50 | 充分训练 |
| 运行次数 | 3 | 报告平均值±标准差 |
| Memory ratio | 0.05, 0.1, 0.2 | 不同预算 |

**预计时间**: 6-12 小时（取决于硬件）

---

## 📁 结果文件

实验结果保存在 `results/` 目录：

```
results/
├── comparison_20260425_115624.json          # 完整实验结果
├── quick_test_20260425_120000.json         # 快速测试结果
└── figures/                                 # 生成的图表
    ├── accuracy_comparison.png
    ├── forgetting_measure.png
    └── timing_comparison.png
```

### 结果格式

```json
{
  "dataset": "mnist",
  "method": "bcsr",
  "memory_ratio": 0.1,
  "per_task_results": [
    {
      "task_id": 0,
      "coreset_size": 1266,
      "select_time": 45.2,
      "train_time": 123.4,
      "per_seen_task_acc": {0: 0.995},
      "test_accuracy": 0.923
    }
  ],
  "summary": {
    "average_accuracy": 0.856,
    "forgetting_measure": 0.123,
    "total_select_time": 234.5,
    "total_train_time": 567.8
  }
}
```

---

## 🛠️ 便捷运行脚本

### 脚本 1: 快速验证脚本

**文件**: `scripts/quick_test.sh`

```bash
#!/bin/bash
# 快速验证所有 bug 已修复

echo "========================================="
echo "快速验证实验"
echo "========================================="

python run_quick.py

if [ $? -eq 0 ]; then
    echo "✅ 所有测试通过！"
else
    echo "❌ 测试失败，请检查错误信息"
    exit 1
fi
```

### 脚本 2: 单数据集完整实验

**文件**: `scripts/run_single_dataset.sh`

```bash
#!/bin/bash
# 运行单个数据集的完整实验

DATASET=${1:-mnist}
METHODS=${2:-random,greedy,csrel,bcsr}

echo "========================================="
echo "数据集: $DATASET"
echo "方法: $METHODS"
echo "========================================="

python experiments/run_comparison.py \
    --dataset $DATASET \
    --method $METHODS \
    --quick \
    --output results/$DATASET
```

使用方法：
```bash
bash scripts/run_single_dataset.sh mnist
bash scripts/run_single_dataset.sh cifar10
```

### 脚本 3: 完整论文实验

**文件**: `scripts/run_full_experiments.sh`

```bash
#!/bin/bash
# 运行所有论文级实验

echo "========================================="
echo "完整论文实验"
echo "预计时间: 6-12 小时"
echo "========================================="

# 创建结果目录
mkdir -p results/full_experiments

# MNIST 完整实验
echo "📊 MNIST 完整实验..."
python experiments/run_comparison.py \
    --dataset mnist \
    --method random,greedy,csrel,bcsr \
    --output results/full_experiments/mnist

# CIFAR-10 完整实验
echo "📊 CIFAR-10 完整实验..."
python experiments/run_comparison.py \
    --dataset cifar10 \
    --method random,greedy,csrel,bcsr \
    --output results/full_experiments/cifar10

# CIFAR-100 完整实验
echo "📊 CIFAR-100 完整实验..."
python experiments/run_comparison.py \
    --dataset cifar100 \
    --method random,greedy,csrel,bcsr \
    --output results/full_experiments/cifar100

echo "✅ 所有实验完成！"
echo "结果保存在: results/full_experiments/"
```

---

## 📈 结果分析

### 生成论文表格

运行分析脚本：

```bash
python experiments/analysis.py --input results/full_experiments/
```

输出：
- LaTeX 表格（用于论文）
- CSV 文件（用于进一步分析）
- 可视化图表

---

## 🔧 故障排除

### 问题 1: CUDA OOM（BCSR）

**解决方案**：
- 减小 batch_size（在 BCSRSelector 中修改 `batch_size=256` 为 `128`）
- 使用更小的 meta_steps（从 3 降到 2）

### 问题 2: 实验太慢

**解决方案**：
- 使用 `--quick` 参数
- 减少运行次数（修改 `configs/experiments.yaml` 中的 `num_runs`）
- 使用更少的数据集

### 问题 3: 结果差异大

**解决方案**：
- 增加运行次数到 5 或更多
- 检查随机种子设置
- 确保 deterministic 行为

---

## 📝 论文写作建议

### 表格 1: 主要结果对比

| Method | MNIST AA | MNIST FM | CIFAR-10 AA | CIFAR-10 FM |
|--------|----------|----------|-------------|-------------|
| Random | XX.X±X.X | X.XXX±X.XX | XX.X±X.X | X.XXX±X.XX |
| Greedy | XX.X±X.X | X.XXX±X.XX | XX.X±X.X | X.XXX±X.XX |
| CSReL  | XX.X±X.X | X.XXX±X.XX | XX.X±X.X | X.XXX±X.XX |
| BCSR   | XX.X±X.X | X.XXX±X.XX | XX.X±X.X | X.XXX±X.XX |

### 图表 1: 准确率曲线

```python
# 绘制每个任务的准确率
import matplotlib.pyplot as plt
import json

with open('results/comparison_xxx.json') as f:
    results = json.load(f)

for method in ['random', 'greedy', 'csrel', 'bcsr']:
    # 提取该方法的准确率曲线
    # 绘图...
    plt.plot(task_ids, accuracies, label=method)

plt.legend()
plt.savefig('results/figures/accuracy_curves.png')
```

---

## 🎯 下一步

1. ✅ 运行快速验证，确保修复有效
2. 📊 运行完整实验，收集论文数据
3. 📈 分析结果，生成图表和表格
4. 📝 撰写论文
5. 🔄 根据审稿意见补充实验

---

## 📞 获取帮助

遇到问题？检查：
1. `results/` 目录中的日志文件
2. 实验输出中的错误信息
3. GitHub Issues

祝实验顺利！🚀
