#!/usr/bin/env python
"""
便捷实验运行脚本
支持快速测试、单数据集实验、完整论文实验
"""
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\n❌ {description} 失败 (退出码: {result.returncode})")
        return False

    print(f"\n✅ {description} 完成")
    return True


def quick_test():
    """运行快速验证测试"""
    print("\n🔍 快速验证实验")
    print("目的: 验证所有 4 个 bug 已修复")
    print("预计时间: 5-10 分钟")

    cmd = [sys.executable, "run_quick.py"]
    return run_command(cmd, "快速验证测试")


def single_dataset(dataset, methods):
    """运行单数据集实验"""
    print(f"\n📊 单数据集实验: {dataset}")
    print(f"方法: {methods}")

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/{dataset}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 结果将保存到: {output_dir}")

    cmd = [
        sys.executable,
        "experiments/run_comparison.py",
        "--dataset", dataset,
        "--method", *methods.split(","),
        "--quick",
        "--output", str(output_dir)
    ]

    success = run_command(cmd, f"{dataset} 数据集实验")

    if success:
        print(f"\n✅ 实验完成！")
        print(f"结果文件: {output_dir}/comparison_*.json")

    return success


def full_experiments():
    """运行完整论文实验"""
    print("\n📊 完整论文实验")
    print("⚠️  警告: 此实验将耗时 6-12 小时")

    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(f"results/full_experiments_{timestamp}")
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 结果将保存到: {base_dir}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 创建日志文件
    log_file = base_dir / "experiment_log.txt"
    log_file.write_text(f"实验开始: {datetime.now()}\n")

    datasets = ["mnist", "cifar10", "cifar100"]
    methods = "random,greedy,csrel,bcsr"

    all_success = True

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"📊 数据集: {dataset}")
        print(f"{'='*60}\n")

        dataset_dir = base_dir / dataset
        dataset_dir.mkdir(exist_ok=True)

        dataset_start = datetime.now()
        log_file.write_text(f"\n[{dataset}] 实验开始: {dataset_start}\n", append_ok=True)

        cmd = [
            sys.executable,
            "experiments/run_comparison.py",
            "--dataset", dataset,
            "--method", *methods.split(","),
            "--quick",
            "--output", str(dataset_dir)
        ]

        success = run_command(cmd, f"{dataset} 数据集实验")

        dataset_end = datetime.now()
        duration = (dataset_end - dataset_start).total_seconds()

        if success:
            log_file.write_text(f"[{dataset}] ✅ 完成 (耗时: {duration:.0f} 秒)\n", append_ok=True)
        else:
            log_file.write_text(f"[{dataset}] ❌ 失败 (耗时: {duration:.0f} 秒)\n", append_ok=True)
            all_success = False

    # 记录结束时间
    log_file.write_text(f"\n实验结束: {datetime.now()}\n", append_ok=True)

    print(f"\n{'='*60}")
    if all_success:
        print("✅ 所有实验完成！")
    else:
        print("⚠️  部分实验失败，请检查日志")

    print(f"\n结果保存在: {base_dir}")
    print(f"日志文件: {log_file}")
    print(f"{'='*60}\n")

    return all_success


def main():
    parser = argparse.ArgumentParser(
        description="核心集选择实验运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速验证（5-10 分钟）
  python scripts/run_experiments.py --quick

  # 单数据集实验
  python scripts/run_experiments.py --dataset mnist

  # 完整论文实验（6-12 小时）
  python scripts/run_experiments.py --full
        """
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="运行快速验证测试（5-10 分钟）"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "cifar10", "cifar100"],
        help="运行单数据集实验"
    )

    parser.add_argument(
        "--methods",
        type=str,
        default="random,greedy,csrel,bcsr",
        help="要测试的方法（逗号分隔）"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="运行完整论文实验（所有数据集）"
    )

    args = parser.parse_args()

    # 如果没有指定任何选项，显示帮助
    if not any([args.quick, args.dataset, args.full]):
        parser.print_help()
        return

    # 执行相应的实验
    success = True

    if args.quick:
        success = quick_test()
    elif args.dataset:
        success = single_dataset(args.dataset, args.methods)
    elif args.full:
        success = full_experiments()

    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
