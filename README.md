# 基于双层优化的 Coreset 选择的实证研究

## 项目概述

本项目用于支持毕业论文《基于双层优化的 Coreset 选择的实证研究》的实验部分。实现了**传统方法**与**双层优化方法**的统一对比框架，在持续学习场景下系统比较核心集选择策略的差异。

## 方法列表

| 方法 | 类型 | 核心思想 |
|------|------|---------|
| Random Sampling | Baseline | 均匀随机采样 |
| Greedy Coreset | Traditional | 最远优先遍历贪心选择 |
| CSReL | Traditional | 基于可约损失 (Reducible Loss) 采样 |
| **BCSR** | **Bi-level Optim.** | **双层优化：内层训练模型，外层优化选择权重** |
| Ensemble | Adaptive | 自适应集成 CSReL + 精细选择 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 快速验证（5-10 分钟）

验证所有 bug 已修复：

```bash
# 方法 1: 使用便捷脚本（推荐）
python scripts/run_experiments.py --quick

# 方法 2: 直接运行
python run_quick.py

# 方法 3: Shell 脚本（Linux/Mac）
bash scripts/quick_test.sh
```

### 3. 运行实验

#### 单数据集实验（20-30 分钟）

```bash
python scripts/run_experiments.py --dataset mnist
```

#### 完整论文实验（2-4 小时）

```bash
python scripts/run_experiments.py --full
```

#### Colab 运行（推荐，免费 GPU）

```python
# 在 Colab 单元格中
!python scripts/colab_helper.py --all
```

### 4. 分析结果

```bash
# 查看结果汇总
python scripts/analyze_results.py --input results/

# 生成论文表格
python scripts/analyze_results.py --input results/ --latex results/table.tex

# 导出 CSV
python scripts/analyze_results.py --input results/ --csv results/results.csv
```

### 📖 详细指南

查看完整的实验运行指南：
- 📘 **[EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)** - 详细实验流程
- 🔧 **[scripts/README.md](scripts/README.md)** - 脚本使用说明

## 项目结构

```
coreset-empirical-study/
├── configs/                    # 实验配置
│   ├── datasets.yaml           #   数据集配置
│   ├── methods.yaml            #   方法配置
│   └── experiments.yaml        #   实验参数
├── core/                       # 核心框架
│   ├── __init__.py             #   方法注册表
│   ├── coreset_base.py         #   基类 + 持续学习框架
│   └── methods/
│       ├── random.py           #   随机采样
│       ├── greedy.py           #   贪心核心集
│       ├── csrel.py            #   CSReL (可约损失)
│       ├── bcsr.py             #   BCSR (双层优化)
│       └── ensemble.py         #   自适应集成
├── data/
│   └── datasets.py             # 持续学习数据加载器
├── experiments/
│   ├── run_comparison.py       # 主实验运行器
│   └── analysis.py             # 结果分析与可视化
├── run_quick.py                # 快速验证脚本
├── run_quick.sh                # 一键运行脚本
└── requirements.txt
```

## 实验设计

### 对比维度

1. **准确率**: Average Accuracy (AA)、测试准确率
2. **抗遗忘**: Forgetting Measure (FM)
3. **核心集质量**: 不同记忆比例下的性能
4. **效率**: 选择耗时、训练耗时

### 数据集与任务

| 数据集 | 任务类型 | 任务数 | 每任务类别数 |
|--------|---------|--------|------------|
| MNIST | Split | 5 | 2 |
| CIFAR-10 | Split | 5 | 2 |
| CIFAR-100 | Split | 10 | 10 |
| MNIST | Permuted | 20 | - |
| MNIST | Rotated | 10 | - |

## 引用

```bibtex
@misc{coreset_empirical_2026,
  title={基于双层优化的Coreset选择的实证研究},
  author={[您的姓名]},
  year={2026}
}
```
