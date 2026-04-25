#!/usr/bin/env python
"""
结果分析脚本
从 JSON 结果文件生成可读的分析报告和图表
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_results(results_dir):
    """加载所有结果文件"""
    results_path = Path(results_dir)
    all_results = []

    # 递归查找所有 JSON 文件
    for json_file in results_path.rglob("comparison_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                all_results.append(data)
        except Exception as e:
            print(f"⚠️  无法加载 {json_file}: {e}")

    return all_results


def print_summary_table(all_results):
    """打印汇总表格"""
    if not all_results:
        print("❌ 没有找到结果文件")
        return

    # 组织数据
    table_data = defaultdict(lambda: defaultdict(list))

    for result in all_results:
        dataset = result.get('dataset', 'unknown')
        method = result.get('method', 'unknown')
        summary = result.get('summary', {})

        aa = summary.get('average_accuracy', 0)
        fm = summary.get('forgetting_measure', 0)
        select_time = summary.get('total_select_time', 0)
        train_time = summary.get('total_train_time', 0)

        table_data[dataset][method].append({
            'aa': aa,
            'fm': fm,
            'select_time': select_time,
            'train_time': train_time
        })

    # 打印表格
    print("\n" + "="*80)
    print("📊 实验结果汇总")
    print("="*80 + "\n")

    for dataset in sorted(table_data.keys()):
        print(f"{'='*80}")
        print(f"数据集: {dataset.upper()}")
        print(f"{'='*80}\n")

        # 表头
        print(f"{'方法':<12} {'平均准确率':<15} {'遗忘度量':<15} {'选择时间(s)':<15} {'训练时间(s)':<15}")
        print(f"{'-'*80}")

        # 数据行
        for method in ['random', 'greedy', 'csrel', 'bcsr']:
            if method not in table_data[dataset]:
                continue

            results_list = table_data[dataset][method]

            # 计算平均值和标准差
            aa_values = [r['aa'] for r in results_list]
            fm_values = [r['fm'] for r in results_list]
            select_times = [r['select_time'] for r in results_list]
            train_times = [r['train_time'] for r in results_list]

            if len(aa_values) > 1:
                import statistics
                aa_mean = statistics.mean(aa_values)
                aa_std = statistics.stdev(aa_values)
                fm_mean = statistics.mean(fm_values)
                fm_std = statistics.stdev(fm_values)

                aa_str = f"{aa_mean:.4f}±{aa_std:.4f}"
                fm_str = f"{fm_mean:.4f}±{fm_std:.4f}"
            else:
                aa_str = f"{aa_values[0]:.4f}"
                fm_str = f"{fm_values[0]:.4f}"

            select_time_str = f"{sum(select_times)/len(select_times):.1f}"
            train_time_str = f"{sum(train_times)/len(train_times):.1f}"

            print(f"{method:<12} {aa_str:<15} {fm_str:<15} {select_time_str:<15} {train_time_str:<15}")

        print()

    print("="*80 + "\n")


def generate_latex_table(all_results, output_file):
    """生成 LaTeX 表格"""
    if not all_results:
        return

    # 组织数据
    table_data = defaultdict(lambda: defaultdict(list))

    for result in all_results:
        dataset = result.get('dataset', 'unknown')
        method = result.get('method', 'unknown')
        summary = result.get('summary', {})

        aa = summary.get('average_accuracy', 0)
        fm = summary.get('forgetting_measure', 0)

        table_data[dataset][method].append({'aa': aa, 'fm': fm})

    # 生成 LaTeX
    latex_lines = []
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{核心集选择方法对比}")
    latex_lines.append("\\label{tab:coreset_comparison}")
    latex_lines.append("\\begin{tabular}{l|cc|cc}")
    latex_lines.append("\\hline")
    latex_lines.append("Method & \\multicolumn{2}{c|}{MNIST} & \\multicolumn{2}{c}{CIFAR-10} \\\\")
    latex_lines.append("         & AA & FM & AA & FM \\\\")
    latex_lines.append("\\hline")

    for method in ['random', 'greedy', 'csrel', 'bcsr']:
        latex_lines.append(f"\\textbf{{{method.capitalize()}}} ")

        # MNIST
        if 'mnist' in table_data and method in table_data['mnist']:
            aa_values = [r['aa'] for r in table_data['mnist'][method]]
            fm_values = [r['fm'] for r in table_data['mnist'][method]]

            if len(aa_values) > 1:
                import statistics
                aa_mean = statistics.mean(aa_values)
                aa_std = statistics.stdev(aa_values)
                fm_mean = statistics.mean(fm_values)
                fm_std = statistics.stdev(fm_values)
                latex_lines.append(f"& {aa_mean:.3f}\\pm{aa_std:.3f} & {fm_mean:.3f}\\pm{fm_std:.3f} ")
            else:
                latex_lines.append(f"& {aa_values[0]:.3f} & {fm_values[0]:.3f} ")
        else:
            latex_lines.append("& -- & -- ")

        # CIFAR-10
        if 'cifar10' in table_data and method in table_data['cifar10']:
            aa_values = [r['aa'] for r in table_data['cifar10'][method]]
            fm_values = [r['fm'] for r in table_data['cifar10'][method]]

            if len(aa_values) > 1:
                import statistics
                aa_mean = statistics.mean(aa_values)
                aa_std = statistics.stdev(aa_values)
                fm_mean = statistics.mean(fm_values)
                fm_std = statistics.stdev(fm_values)
                latex_lines.append(f"& {aa_mean:.3f}\\pm{aa_std:.3f} & {fm_mean:.3f}\\pm{fm_std:.3f}")
            else:
                latex_lines.append(f"& {aa_values[0]:.3f} & {fm_values[0]:.3f}")
        else:
            latex_lines.append("& -- & --")

        latex_lines.append("\\\\")

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    # 写入文件
    latex_content = "\n".join(latex_lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex_content)
        print(f"✅ LaTeX 表格已保存到: {output_file}")
    else:
        print("\n📄 LaTeX 表格:")
        print(latex_content)


def export_to_csv(all_results, output_file):
    """导出为 CSV 格式"""
    import csv

    if not all_results:
        return

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Dataset', 'Method', 'Average_Accuracy', 'Forgetting_Measure',
            'Selection_Time', 'Training_Time'
        ])

        for result in all_results:
            dataset = result.get('dataset', 'unknown')
            method = result.get('method', 'unknown')
            summary = result.get('summary', {})

            writer.writerow([
                dataset,
                method,
                f"{summary.get('average_accuracy', 0):.4f}",
                f"{summary.get('forgetting_measure', 0):.4f}",
                f"{summary.get('total_select_time', 0):.2f}",
                f"{summary.get('total_train_time', 0):.2f}"
            ])

    print(f"✅ CSV 文件已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="分析实验结果",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--input",
        type=str,
        default="results",
        help="结果目录"
    )

    parser.add_argument(
        "--latex",
        type=str,
        help="导出 LaTeX 表格到指定文件"
    )

    parser.add_argument(
        "--csv",
        type=str,
        help="导出 CSV 到指定文件"
    )

    args = parser.parse_args()

    # 加载结果
    print(f"📂 加载结果从: {args.input}")
    all_results = load_results(args.input)

    if not all_results:
        print("❌ 未找到任何结果文件")
        return

    print(f"✅ 成功加载 {len(all_results)} 个结果文件\n")

    # 打印汇总
    print_summary_table(all_results)

    # 导出 LaTeX
    if args.latex:
        generate_latex_table(all_results, args.latex)

    # 导出 CSV
    if args.csv:
        export_to_csv(all_results, args.csv)


if __name__ == "__main__":
    main()
