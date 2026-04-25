#!/bin/bash
# 单数据集完整实验脚本
# 用于在单个数据集上运行所有方法

set -e

# 默认参数
DATASET=${1:-mnist}
# 关键修复：使用数组而不是逗号分隔的字符串
METHODS=("random" "greedy" "csrel" "bcsr")

echo "========================================="
echo "📊 单数据集实验"
echo "数据集: $DATASET"
echo "方法: ${METHODS[@]}"
echo "========================================="
echo ""

# 创建结果目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/${DATASET}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "📁 结果将保存到: $OUTPUT_DIR"
echo ""

# 显示配置
echo "⚙️  实验配置:"
echo "  • 任务数: 5"
echo "  • Epochs: 10 (快速模式)"
echo "  • 运行次数: 1"
echo "  • Memory ratio: 0.1"
echo ""

# 运行实验 - 关键修复：正确传递多个方法参数
echo "🚀 开始实验..."
python experiments/run_comparison.py \
    --dataset $DATASET \
    --method "${METHODS[@]}" \
    --quick \
    --output "$OUTPUT_DIR"

echo ""
echo "========================================="
echo "✅ 实验完成！"
echo ""
echo "结果文件:"
echo "  • $OUTPUT_DIR/comparison_*.json"
echo ""
echo "查看结果:"
echo "  cat $OUTPUT_DIR/comparison_*.json | python -m json.tool"
echo "========================================="
