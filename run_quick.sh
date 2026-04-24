#!/bin/bash
# 一键运行完整对比实验

echo "=========================================="
echo "核心集选择方法对比实验"
echo "=========================================="

# 快速验证
echo "[1/3] 快速验证..."
python run_quick.py
if [ $? -ne 0 ]; then
    echo "快速验证失败，请检查环境"
    exit 1
fi

# 运行完整实验
echo "[2/3] 运行对比实验..."
python experiments/run_comparison.py --config configs/experiments.yaml

# 生成分析图表
echo "[3/3] 生成分析图表..."
LATEST=$(ls -t results/comparison_*.json 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    python experiments/analysis.py "$LATEST"
    echo "结果图表在 results/figures/"
else
    echo "未找到结果文件"
fi

echo "完成!"
