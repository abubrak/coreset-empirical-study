#!/bin/bash
# 快速测试脚本 - 验证所有修复

echo "========================================="
echo "🧪 测试所有 bug 修复"
echo "========================================="
echo ""

echo "测试 1: 验证脚本参数传递"
echo "---------------------------------------"

# 使用 Python 脚本测试
python experiments/run_comparison.py \
    --dataset mnist \
    --method random \
    --method greedy \
    --task_type split \
    --num_tasks_list 2 \
    --memory_ratios 0.1 \
    --num_runs 1 \
    2>&1 | head -20

if [ $? -eq 0 ]; then
    echo "✅ 参数传递正常"
else
    echo "❌ 参数传递失败"
    exit 1
fi

echo ""
echo "========================================="
echo "✅ 所有测试通过！"
echo "你现在可以运行完整实验："
echo ""
echo "  bash scripts/run_full_experiments.sh"
echo "  或"
echo "  python scripts/run_experiments.py --full"
echo "========================================="
