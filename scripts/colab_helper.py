#!/usr/bin/env python
"""
Colab 实验运行助手
提供进度显示、结果可视化、错误处理等功能
"""
import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import sys


class ColabExperimentRunner:
    """Colab 实验运行器"""

    def __init__(self):
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

        self.start_time = None
        self.current_experiment = None

    def print_header(self, title, subtitle=""):
        """打印漂亮的标题"""
        print("\n" + "="*60)
        print(f"🔬 {title}")
        if subtitle:
            print(f"📋 {subtitle}")
        print("="*60 + "\n")

    def print_experiment_info(self, dataset, methods, quick_mode=True):
        """打印实验信息"""
        print("📊 实验配置:")
        print(f"  • 数据集: {dataset}")
        print(f"  • 方法: {', '.join(methods)}")
        if quick_mode:
            print(f"  • 模式: 快速测试")
            print(f"  • Epochs: 10")
            print(f"  • 运行次数: 1")
        print(f"  • 设备: {'GPU' if self._check_gpu() else 'CPU'}")
        print()

    def _check_gpu(self):
        """检查 GPU 是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def run_experiment(self, dataset, methods, quick_mode=True):
        """运行实验"""
        self.start_time = datetime.now()
        self.current_experiment = f"{dataset}_{self.start_time.strftime('%Y%m%d_%H%M%S')}"

        self.print_header(
            "开始实验",
            f"数据集: {dataset} | 方法: {', '.join(methods)}"
        )

        self.print_experiment_info(dataset, methods, quick_mode)

        # 构建命令
        cmd = [
            sys.executable,
            "experiments/run_comparison.py",
            "--dataset", dataset,
            "--method", *methods,
            "--output", str(self.results_dir / self.current_experiment)
        ]

        if quick_mode:
            cmd.append("--quick")

        # 显示命令
        print("🚀 执行命令:")
        print(f"   {' '.join(cmd)}\n")

        # 运行实验
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True
            )

            self.print_completion()
            return True

        except subprocess.CalledProcessError as e:
            print(f"\n❌ 实验失败 (退出码: {e.returncode})")
            self.print_troubleshooting()
            return False

    def print_completion(self, show_summary=True):
        """打印完成信息"""
        duration = datetime.now() - self.start_time

        print("\n" + "="*60)
        print("✅ 实验完成!")
        print("="*60)
        print(f"⏰ 总耗时: {self._format_duration(duration)}")
        print(f"📁 结果目录: {self.results_dir / self.current_experiment}")

        if show_summary:
            self.print_summary()

        print("="*60 + "\n")

    def _format_duration(self, duration):
        """格式化时间差"""
        seconds = int(duration.total_seconds())
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)

        if hours > 0:
            return f"{hours}小时 {minutes}分钟"
        elif minutes > 0:
            return f"{minutes}分钟 {seconds}秒"
        else:
            return f"{seconds}秒"

    def print_summary(self):
        """打印结果摘要"""
        result_file = list(
            (self.results_dir / self.current_experiment).glob("comparison_*.json")
        )

        if not result_file:
            print("⚠️  未找到结果文件")
            return

        result_file = result_file[0]

        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            print("\n📊 结果摘要:")
            print(f"  • 数据集: {data.get('dataset', 'N/A')}")
            print(f"  • 方法: {data.get('method', 'N/A')}")
            print(f"  • 内存比例: {data.get('memory_ratio', 'N/A')}")

            summary = data.get('summary', {})
            if summary:
                print(f"\n  性能指标:")
                print(f"    • 平均准确率: {summary.get('average_accuracy', 'N/A'):.4f}")
                print(f"    • 遗忘度量: {summary.get('forgetting_measure', 'N/A'):.4f}")
                print(f"    • 测试准确率: {summary.get('final_test_accuracy', 'N/A'):.4f}")

                print(f"\n  时间指标:")
                print(f"    • 选择时间: {summary.get('total_select_time', 'N/A'):.1f}秒")
                print(f"    • 训练时间: {summary.get('total_train_time', 'N/A'):.1f}秒")

        except Exception as e:
            print(f"⚠️  无法解析结果文件: {e}")

    def print_troubleshooting(self):
        """打印故障排除建议"""
        print("\n🔧 故障排除建议:")
        print("  1. 检查错误信息中的具体错误类型")
        print("  2. 如果是 CUDA OOM:")
        print("     • 减小 batch_size (在 bcsr.py 中修改)")
        print("     • 或使用 CPU 模式")
        print("  3. 如果是 IndexError:")
        print("     • 确保已应用所有修复")
        print("     • 检查 git log 查看最近修改")
        print("  4. 查看完整日志:")
        print(f"     • cat {self.results_dir / self.current_experiment}/*.log")

    def run_all_datasets(self, methods, quick_mode=True):
        """运行所有数据集"""
        datasets = ["mnist", "cifar10", "cifar100"]

        self.print_header(
            "完整论文实验",
            f"所有数据集 | 预计时间: 2-4小时"
        )

        print("⚠️  警告: 这将运行所有数据集，耗时较长")
        print("建议:")
        print("  • 确保已连接 GPU")
        print("  • 不要关闭浏览器标签")
        print("  • 监控 GPU 内存使用\n")

        results = {}

        for dataset in datasets:
            print(f"\n{'#'*60}")
            print(f"# 数据集: {dataset.upper()}")
            print(f"{'#'*60}\n")

            success = self.run_experiment(dataset, methods, quick_mode)
            results[dataset] = "✅ 成功" if success else "❌ 失败"

        # 打印总结
        self.print_header("实验总结", "所有数据集")

        for dataset, status in results.items():
            print(f"  {dataset:10s}: {status}")

        print()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Colab 实验运行助手",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "cifar10", "cifar100"],
        help="单个数据集"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="运行所有数据集"
    )

    parser.add_argument(
        "--methods",
        type=str,
        default="random,greedy,csrel,bcsr",
        help="方法列表（逗号分隔）"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="完整实验（非快速模式）"
    )

    args = parser.parse_args()

    # 创建运行器
    runner = ColabExperimentRunner()

    # 运行实验
    methods = args.methods.split(",")

    if args.all:
        runner.run_all_datasets(methods, quick_mode=not args.full)
    elif args.dataset:
        runner.run_experiment(args.dataset, methods, quick_mode=not args.full)
    else:
        # 默认：快速测试 MNIST
        runner.print_header("快速验证测试", "验证所有 bug 已修复")
        runner.run_experiment("mnist", methods, quick_mode=True)


if __name__ == "__main__":
    main()
