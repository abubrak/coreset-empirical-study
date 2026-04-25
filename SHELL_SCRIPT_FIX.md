# 🔧 Shell 脚本修复说明

## ✅ 已修复的问题

**问题：** Shell 脚本传递方法参数的方式不正确

```bash
# ❌ 错误（旧代码）
METHODS="random,greedy,csrel,bcsr"
--method $METHODS  # 传递一个字符串 "random,greedy,csrel,bcsr"

# ✅ 正确（新代码）
METHODS=("random" "greedy" "csrel" "bcsr")
--method "${METHODS[@]}"  # 展开为多个独立参数
```

---

## 📝 已修复的文件

| 文件 | 状态 |
|------|------|
| `scripts/run_full_experiments.sh` | ✅ 已修复 |
| `scripts/run_single_dataset.sh` | ✅ 已修复 |
| `scripts/run_experiments.py` | ✅ 原本正确（使用 `.split(",")`）|

---

## 🚀 现在可以正常运行

### 在本地运行

```bash
# 重新运行完整实验
bash scripts/run_full_experiments.sh

# 或使用 Python 脚本（推荐）
python scripts/run_experiments.py --full
```

### 在 Colab 中运行

```python
# 方法 1: 使用 run_colab.py（已修复）
!python run_colab.py

# 方法 2: 使用 colab_helper.py（推荐）
!python scripts/colab_helper.py --dataset mnist

# 方法 3: 直接运行实验（使用正确的参数）
!python experiments/run_comparison.py --dataset mnist --method random --method greedy --method csrel --method bcsr --quick
```

---

## 📋 完整的 Colab 运行步骤

### 1. 重新加载代码

在 Colab 中运行：

```python
# 重新挂载 Google Drive（如果使用）
from google.colab import drive
drive.mount('/content/drive')

# 进入项目目录
%cd /content/coreset-empirical-study

# 拉取最新代码（如果从 GitHub）
# !git pull
```

### 2. 运行快速验证

```python
# 验证修复
!python run_quick.py
```

### 3. 运行完整实验

```python
# 方法 1: 使用修复后的脚本
!bash scripts/run_full_experiments.sh

# 方法 2: 使用 Python 脚本（推荐）
!python scripts/run_experiments.py --full

# 方法 3: 单独测试每个数据集
!python scripts/run_experiments.py --dataset mnist
```

---

## 🔍 问题根源

### Shell 参数传递机制

当你在 Shell 中定义一个变量并传递给 Python 时：

```bash
# 示例 1: 字符串（错误）
METHODS="random,greedy,csrel,bcsr"
python script.py --method $METHODS
# Python 收到: sys.argv = ['script.py', '--method', 'random,greedy,csrel,bcsr']
#                 1 个参数  ❌

# 示例 2: 数组展开（正确）
METHODS=("random" "greedy" "csrel" "bcsr")
python script.py --method "${METHODS[@]}"
# Python 收到: sys.argv = ['script.py', '--method', 'random', '--method', 'greedy', ...]
#                 多个参数  ✅
```

### Python argparse 行为

```python
parser.add_argument('--method', nargs='+')
```

`nargs='+'` 表示：
- 接受**多个**独立的命令行参数
- 不是接受一个逗号分隔的字符串

---

## ✅ 验证修复

运行以下命令验证修复：

```bash
# 应该看到 4 个独立的方法参数
bash -x scripts/run_full_experiments.sh 2>&1 | grep "method"

# 或直接测试
python experiments/run_comparison.py --dataset mnist --method random --method greedy --help
```

---

## 📌 快速参考

### 正确的命令格式

```bash
# ✅ 正确：每个方法一个 --method 参数
python experiments/run_comparison.py \
    --dataset mnist \
    --method random \
    --method greedy \
    --method csrel \
    --method bcsr \
    --quick

# ❌ 错误：逗号分隔
python experiments/run_comparison.py \
    --dataset mnist \
    --method random,greedy,csrel,bcsr \
    --quick
```

---

## 🎯 推荐使用方式

**在 Colab 中，推荐使用 Python 脚本而不是 Shell 脚本：**

```python
# ✅ 推荐：Python 脚本
!python scripts/run_experiments.py --dataset mnist

# ✅ 也可：colab_helper.py
!python scripts/colab_helper.py --dataset mnist

# ⚠️  避免：Shell 脚本（参数传递复杂）
!bash scripts/run_single_dataset.sh mnist
```

---

## 🔄 如果仍然遇到问题

1. **确认脚本已更新**
   ```bash
   # 检查脚本是否包含修复
   grep 'METHODS=\(' scripts/run_full_experiments.sh
   # 应该看到: METHODS=("random" "greedy" "csrel" "bcsr")
   ```

2. **重新克隆/下载代码**
   ```bash
   # 如果从 GitHub 克隆
   !git pull origin main
   ```

3. **使用 Python 脚本作为备选**
   ```python
   # Python 脚本没有这个问题
   !python scripts/run_experiments.py --full
   ```

---

**问题已解决！现在可以正常运行实验了。** 🎉
