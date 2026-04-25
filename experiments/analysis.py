"""
实验结果分析与可视化
生成对比表格、准确率曲线、遗忘度量分析等图表
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ResultAnalyzer:
    """实验结果分析器"""

    def __init__(self, results_path: str, output_dir: str = None):
        """
        Args:
            results_path: 实验结果 JSON 文件路径
            output_dir: 图表输出目录
        """
        if not Path(results_path).exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        with open(results_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)

        if not isinstance(self.results, list):
            raise ValueError("Results must be a list of experiment results")

        self.output_dir = Path(output_dir or Path(results_path).parent / 'figures')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 方法类型分类
        self.method_types = {
            'random': 'Baseline',
            'greedy': 'Traditional',
            'csrel': 'Traditional',
            'bcsr': 'Bi-level Optim.',
            'ensemble': 'Adaptive',
        }

        # 方法显示名称
        self.method_names = {
            'random': 'Random',
            'greedy': 'Greedy',
            'csrel': 'CSReL',
            'bcsr': 'BCSR',
            'ensemble': 'Ensemble',
        }

    def generate_all(self):
        """生成所有分析图表"""
        print("生成分析图表...")

        if not self.results:
            print("警告: 没有实验结果可供分析")
            return

        self._print_summary_table()
        self.plot_accuracy_comparison()
        self.plot_forgetting_comparison()
        self.plot_per_task_accuracy()
        self.plot_memory_ratio_sensitivity()
        self.plot_time_comparison()
        self.plot_accuracy_heatmap()
        self.export_latex_table()

        print(f"\n所有图表已保存至: {self.output_dir}")

    def _print_summary_table(self):
        """打印汇总表格"""
        rows = []
        for r in self.results:
            s = r['summary']
            rows.append({
                'Dataset': r['dataset'].upper(),
                'Method': self.method_names.get(r['method'], r['method']),
                'Type': self.method_types.get(r['method'], 'Unknown'),
                'Ratio': f"{r['memory_ratio']*100:.0f}%",
                'AA': f"{s['average_accuracy']:.4f}",
                'FM': f"{s['forgetting_measure']:.4f}",
                'Test Acc': f"{s['final_test_accuracy']:.4f}",
                'Select(s)': f"{s['total_select_time']:.1f}",
                'Train(s)': f"{s['total_train_time']:.1f}",
            })

        df = pd.DataFrame(rows)
        print("\n" + "="*90)
        print("实验结果汇总")
        print("="*90)
        print(df.to_string(index=False))
        print("="*90 + "\n")

        # 保存 CSV
        df.to_csv(self.output_dir / 'summary_table.csv', index=False)
        print(f"CSV 保存至: {self.output_dir / 'summary_table.csv'}")

    def plot_accuracy_comparison(self):
        """绘制平均准确率对比柱状图"""
        if not self.results:
            print("警告: 没有可用于准确率对比的结果")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # 按数据集分组
        datasets = sorted(set(r['dataset'] for r in self.results))

        all_accs = []
        for dataset in datasets:
            dataset_results = [
                r for r in self.results
                if r['dataset'] == dataset and r.get('run_id', 0) == 0
            ]

            if not dataset_results:
                continue

            methods = [r['method'] for r in dataset_results]
            accs = [r['summary']['average_accuracy'] for r in dataset_results]
            all_accs.extend(accs)
            labels = [self.method_names.get(m, m) for m in methods]

            x = np.arange(len(labels))
            width = 0.6 / len(datasets)
            offset = (datasets.index(dataset) - len(datasets)/2 + 0.5) * width

            bars = ax.bar(
                x + offset, accs, width,
                label=dataset.upper(),
                alpha=0.85
            )

            # 数值标注
            for bar, acc in zip(bars, accs):
                ax.text(
                    bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8
                )

        ax.set_xlabel('Method')
        ax.set_ylabel('Average Accuracy')
        ax.set_title('Average Accuracy Comparison')
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels([self.method_names.get(m, m) for m in
                           ['random', 'greedy', 'csrel', 'bcsr', 'ensemble']])
        ax.legend()
        if all_accs:
            ax.set_ylim(0, max(all_accs) * 1.15)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_comparison.png', dpi=150)
        plt.close()

    def plot_forgetting_comparison(self):
        """绘制遗忘度量对比"""
        fig, ax = plt.subplots(figsize=(10, 6))

        datasets = sorted(set(r['dataset'] for r in self.results))
        methods_order = ['random', 'greedy', 'csrel', 'bcsr', 'ensemble']

        for dataset in datasets:
            dataset_results = {
                r['method']: r['summary']['forgetting_measure']
                for r in self.results
                if r['dataset'] == dataset and r.get('run_id', 0) == 0
            }

            methods = [m for m in methods_order if m in dataset_results]
            fm_values = [dataset_results[m] for m in methods]
            labels = [self.method_names.get(m, m) for m in methods]

            x = np.arange(len(labels))
            width = 0.6 / len(datasets)
            offset = (datasets.index(dataset) - len(datasets)/2 + 0.5) * width

            ax.bar(x + offset, fm_values, width, label=dataset.upper(), alpha=0.85)

        ax.set_xlabel('Method')
        ax.set_ylabel('Forgetting Measure')
        ax.set_title('Catastrophic Forgetting Comparison (Lower is Better)')
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels([self.method_names.get(m, m) for m in methods_order])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'forgetting_comparison.png', dpi=150)
        plt.close()

    def plot_per_task_accuracy(self):
        """绘制每个任务随训练进程的准确率变化"""
        datasets = sorted(set(r['dataset'] for r in self.results))
        methods_order = ['random', 'greedy', 'csrel', 'bcsr', 'ensemble']

        for dataset in datasets:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            for method in methods_order:
                results = [
                    r for r in self.results
                    if r['dataset'] == dataset
                    and r['method'] == method
                    and r.get('run_id', 0) == 0
                ]
                if not results:
                    continue
                r = results[0]

                num_tasks = r['num_tasks']
                task_accs = r['task_accuracies']

                # 任务准确率矩阵: task_id x evaluation_point
                acc_matrix = np.zeros((num_tasks, num_tasks))
                for tid in range(num_tasks):
                    if str(tid) in task_accs:
                        accs = task_accs[str(tid)]
                        # 修复：安全计算列索引，确保不越界
                        start_col = max(0, num_tasks - len(accs))
                        for i, a in enumerate(accs):
                            col = start_col + i
                            if col < num_tasks and tid < num_tasks:
                                acc_matrix[tid, col] = a

                # 平均准确率曲线
                avg_accs = []
                for eval_point in range(num_tasks):
                    valid = [acc_matrix[t, eval_point]
                             for t in range(eval_point + 1)]
                    avg_accs.append(np.mean(valid) if valid else 0)

                axes[0].plot(
                    range(1, num_tasks + 1), avg_accs,
                    marker='o', label=self.method_names.get(method, method)
                )

                # 遗忘曲线：第一个任务准确率随时间变化
                first_task_accs = task_accs.get('0', [])
                if first_task_accs:
                    axes[1].plot(
                        range(1, len(first_task_accs) + 1), first_task_accs,
                        marker='s', label=self.method_names.get(method, method)
                    )

            axes[0].set_xlabel('After Task #')
            axes[0].set_ylabel('Average Accuracy')
            axes[0].set_title(f'{dataset.upper()} - Average Accuracy Progression')
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            axes[1].set_xlabel('After Task #')
            axes[1].set_ylabel('Task 0 Accuracy')
            axes[1].set_title(f'{dataset.upper()} - Task 0 Forgetting')
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                self.output_dir / f'per_task_accuracy_{dataset}.png',
                dpi=150
            )
            plt.close()

    def plot_memory_ratio_sensitivity(self):
        """绘制核心集比例敏感性分析"""
        results_by_ratio = {}
        for r in self.results:
            key = (r['dataset'], r['method'])
            if key not in results_by_ratio:
                results_by_ratio[key] = {}
            ratio = r['memory_ratio']
            if ratio not in results_by_ratio[key]:
                results_by_ratio[key][ratio] = []
            results_by_ratio[key][ratio].append(r['summary']['average_accuracy'])

        datasets = sorted(set(k[0] for k in results_by_ratio.keys()))

        for dataset in datasets:
            fig, ax = plt.subplots(figsize=(8, 6))

            for method in ['random', 'greedy', 'csrel', 'bcsr', 'ensemble']:
                key = (dataset, method)
                if key not in results_by_ratio:
                    continue

                ratios = sorted(results_by_ratio[key].keys())
                accs_mean = [np.mean(results_by_ratio[key][r]) for r in ratios]
                accs_std = [np.std(results_by_ratio[key][r]) for r in ratios]

                ax.errorbar(
                    [r * 100 for r in ratios], accs_mean, yerr=accs_std,
                    marker='o', capsize=3,
                    label=self.method_names.get(method, method)
                )

            ax.set_xlabel('Coreset Ratio (%)')
            ax.set_ylabel('Average Accuracy')
            ax.set_title(f'{dataset.upper()} - Memory Budget Sensitivity')
            ax.legend()
            ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                self.output_dir / f'sensitivity_{dataset}.png',
                dpi=150
            )
            plt.close()

    def plot_time_comparison(self):
        """绘制耗时对比"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for dataset in sorted(set(r['dataset'] for r in self.results)):
            ds_results = [
                r for r in self.results
                if r['dataset'] == dataset and r.get('run_id', 0) == 0
            ]

            methods = [r['method'] for r in ds_results]
            select_times = [r['summary']['total_select_time'] for r in ds_results]
            train_times = [r['summary']['total_train_time'] for r in ds_results]
            labels = [self.method_names.get(m, m) for m in methods]

            x = np.arange(len(labels))
            width = 0.3

            d_idx = list(sorted(set(r['dataset'] for r in self.results))).index(dataset)
            offset = (d_idx - 0.5) * width

            axes[0].bar(x + offset, select_times, width, label=dataset.upper(), alpha=0.85)
            axes[1].bar(x + offset, train_times, width, label=dataset.upper(), alpha=0.85)

        for ax, title in zip(axes, ['Selection Time', 'Training Time']):
            ax.set_xlabel('Method')
            ax.set_ylabel('Time (seconds)')
            ax.set_title(title)
            ax.set_xticks(np.arange(5))
            ax.set_xticklabels([self.method_names.get(m, m)
                               for m in ['random', 'greedy', 'csrel', 'bcsr', 'ensemble']])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_comparison.png', dpi=150)
        plt.close()

    def plot_accuracy_heatmap(self):
        """绘制任务准确率热力图"""
        datasets = sorted(set(r['dataset'] for r in self.results))
        methods_order = ['random', 'greedy', 'csrel', 'bcsr', 'ensemble']

        for dataset in datasets:
            fig, axes = plt.subplots(1, len(methods_order), figsize=(20, 4))

            for i, method in enumerate(methods_order):
                results = [
                    r for r in self.results
                    if r['dataset'] == dataset
                    and r['method'] == method
                    and r.get('run_id', 0) == 0
                ]
                if not results:
                    continue

                r = results[0]
                num_tasks = r['num_tasks']
                task_accs = r['task_accuracies']

                # 构建矩阵 - 修复索引越界问题
                matrix = np.zeros((num_tasks, num_tasks))
                for tid in range(num_tasks):
                    accs = task_accs.get(str(tid), [])
                    start_col = max(0, num_tasks - len(accs))
                    for j, a in enumerate(accs):
                        col = start_col + j
                        if 0 <= col < num_tasks:
                            matrix[int(tid), col] = a

                sns.heatmap(
                    matrix, ax=axes[i], annot=True, fmt='.2f',
                    cmap='YlGn', vmin=0, vmax=1,
                    xticklabels=[f'A{T+1}' for T in range(num_tasks)],
                    yticklabels=[f'T{T}' for T in range(num_tasks)]
                )
                axes[i].set_title(self.method_names.get(method, method))
                axes[i].set_xlabel('Eval After')
                if i == 0:
                    axes[i].set_ylabel('Task')

            plt.suptitle(f'{dataset.upper()} - Task Accuracy Matrix', y=1.02)
            plt.tight_layout()
            plt.savefig(
                self.output_dir / f'heatmap_{dataset}.png',
                dpi=150, bbox_inches='tight'
            )
            plt.close()

    def export_latex_table(self):
        """导出 LaTeX 格式的结果表格"""
        rows = []
        for r in self.results:
            if r.get('run_id', 0) != 0:
                continue
            s = r['summary']
            mtype = self.method_types.get(r['method'], '')
            rows.append({
                'dataset': r['dataset'].upper(),
                'method': self.method_names.get(r['method'], r['method']),
                'type': mtype,
                'ratio': f"{r['memory_ratio']*100:.0f}\\%",
                'aa': s['average_accuracy'],
                'fm': s['forgetting_measure'],
                'test': s['final_test_accuracy'],
            })

        # 按数据集分组输出
        datasets = sorted(set(r['dataset'] for r in rows))
        latex_lines = []

        for dataset in datasets:
            ds_rows = [r for r in rows if r['dataset'] == dataset]

            # 找最优值
            best_aa = max(r['aa'] for r in ds_rows)
            best_fm = min(r['fm'] for r in ds_rows)

            for r in ds_rows:
                aa_str = f"\\textbf{{{r['aa']:.4f}}}" if r['aa'] == best_aa else f"{r['aa']:.4f}"
                fm_str = f"\\textbf{{{r['fm']:.4f}}}" if r['fm'] == best_fm else f"{r['fm']:.4f}"

                latex_lines.append(
                    f"{r['dataset']} & {r['method']} & {r['type']} & "
                    f"{r['ratio']} & {aa_str} & {fm_str} \\\\"
                )
            latex_lines.append("\\midrule")

        latex_text = "\n".join(latex_lines)
        latex_file = self.output_dir / 'results_table.tex'
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_text)

        print(f"LaTeX 表格保存至: {latex_file}")


def main():
    parser = argparse.ArgumentParser(description='实验结果分析')
    parser.add_argument(
        'results_path', type=str,
        help='实验结果 JSON 文件路径'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='图表输出目录'
    )

    args = parser.parse_args()

    analyzer = ResultAnalyzer(args.results_path, args.output)
    analyzer.generate_all()


if __name__ == '__main__':
    main()
