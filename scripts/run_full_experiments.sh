#!/bin/bash
# 完整论文实验脚本
# 运行所有数据集和方法的完整实验

set -e

echo "========================================="
echo "📊 完整论文实验"
echo ""
echo "⚠️  警告: 此实验将耗时 6-12 小时"
echo "请确保:"
echo "  • GPU 可用且稳定"
echo "  • 有足够的磁盘空间 (~5GB)"
echo "  • 不会中断（nohup 或 screen）"
echo "========================================="
echo ""

# 创建结果目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/full_experiments_$TIMESTAMP"
mkdir -p "$BASE_DIR"

echo "📁 结果将保存到: $BASE_DIR"
echo ""

# 记录开始时间
START_TIME=$(date)
echo "⏰ 开始时间: $START_TIME"
echo "" > "$BASE_DIR/experiment_log.txt"
echo "实验开始: $START_TIME" >> "$BASE_DIR/experiment_log.txt"

# 定义数据集列表
DATASETS=("mnist" "cifar10" "cifar100")
METHODS="random,greedy,csrel,bcsr"

# 循环运行每个数据集
for DATASET in "${DATASETS[@]}"; do
    echo "========================================="
    echo "📊 数据集: $DATASET"
    echo "========================================="
    echo ""

    # 创建数据集特定的输出目录
    DATASET_DIR="$BASE_DIR/$DATASET"
    mkdir -p "$DATASET_DIR"

    # 记录开始时间
    DATASET_START=$(date +%s)
    echo "[$DATASET] 实验开始: $(date)" | tee -a "$BASE_DIR/experiment_log.txt"

    # 运行实验
    if python experiments/run_comparison.py \
        --dataset $DATASET \
        --method $METHODS \
        --quick \
        --output "$DATASET_DIR" 2>&1 | tee -a "$BASE_DIR/experiment_log.txt"; then
        # 成功
        DATASET_END=$(date +%s)
        DATASET_DURATION=$((DATASET_END - DATASET_START))
        echo "[$DATASET] ✅ 完成 (耗时: $DATASET_DURATION 秒)" | tee -a "$BASE_DIR/experiment_log.txt"
    else
        # 失败
        DATASET_END=$(date +%s)
        DATASET_DURATION=$((DATASET_END - DATASET_START))
        echo "[$DATASET] ❌ 失败 (耗时: $DATASET_DURATION 秒)" | tee -a "$BASE_DIR/experiment_log.txt"
        echo "继续下一个数据集..."
    fi

    echo ""
done

# 记录结束时间
END_TIME=$(date)
echo "" | tee -a "$BASE_DIR/experiment_log.txt"
echo "=========================================" | tee -a "$BASE_DIR/experiment_log.txt"
echo "✅ 所有实验完成！" | tee -a "$BASE_DIR/experiment_log.txt"
echo "" | tee -a "$BASE_DIR/experiment_log.txt"
echo "开始时间: $START_TIME" | tee -a "$BASE_DIR/experiment_log.txt"
echo "结束时间: $END_TIME" | tee -a "$BASE_DIR/experiment_log.txt"
echo "=========================================" | tee -a "$BASE_DIR/experiment_log.txt"
echo ""
echo "结果保存在: $BASE_DIR"
echo ""
echo "📊 查看结果摘要:"
echo "  ls -lh $BASE_DIR/*/*.json"
echo ""
echo "📝 查看日志:"
echo "  cat $BASE_DIR/experiment_log.txt"
echo ""
echo "📈 生成分析报告:"
echo "  python experiments/analysis.py --input $BASE_DIR"
echo ""
