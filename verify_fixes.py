#!/usr/bin/env python
"""
验证所有 bug 修复是否正确应用
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_fix_1_create_combined_loader():
    """检查 _create_combined_loader 是否使用全局索引映射"""
    print("\n[修复 1] _create_combined_loader - 全局索引映射")

    with open('experiments/run_comparison.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查关键修复代码
    checks = [
        ('task_indices = prev_subset.indices', '✓ 提取任务索引'),
        ('global_indices = [task_indices[i] for i in indices', '✓ 映射到全局索引'),
        ('Subset(continual_dataset.full_dataset, global_indices)', '✓ 使用全局索引创建 Subset')
    ]

    all_passed = True
    for check_str, desc in checks:
        if check_str in content:
            print(f"  {desc}")
        else:
            print(f"  ✗ 缺少: {desc}")
            all_passed = False

    return all_passed


def check_fix_2_random_selector():
    """检查 RandomSelector 是否移除了错误的 merge 逻辑"""
    print("\n[修复 2] RandomSelector - 移除 merge 逻辑")

    with open('core/methods/random.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查移除的代码不存在
    if 'all_previous.extend(prev)' in content:
        print("  ✗ 仍然包含错误的 merge 逻辑")
        return False

    # 检查返回值正确
    if 'return selected_indices, weights' in content:
        print("  ✓ 正确返回选择结果")
        return True

    print("  ✗ 返回值格式不正确")
    return False


def check_fix_3_csrel_selector():
    """检查 CSReLSelector 是否修复了 ref_losses 缓存问题"""
    print("\n[修复 3] CSReLSelector - 修复 ref_losses 缓存")

    with open('core/methods/csrel.py', 'r', encoding='utf-8') as f:
        content = f.read()

    checks = [
        ('self.ref_num_samples = num_samples', '✓ 添加了 ref_num_samples 跟踪'),
        ('self.ref_num_samples != num_samples', '✓ 检测任务大小变化'),
        ('self.ref_num_samples = None', '✓ 初始化 ref_num_samples')
    ]

    all_passed = True
    for check_str, desc in checks:
        if check_str in content:
            print(f"  {desc}")
        else:
            print(f"  ✗ 缺少: {desc}")
            all_passed = False

    return all_passed


def check_fix_4_bcsr_selector():
    """检查 BCSRSelector 是否添加了批处理"""
    print("\n[修复 4] BCSRSelector - 批处理隐式梯度")

    with open('core/methods/bcsr.py', 'r', encoding='utf-8') as f:
        content = f.read()

    checks = [
        ('batch_size = 256', '✓ 使用批处理（batch_size=256）'),
        ('for i in range(0, n_train, batch_size):', '✓ 分批处理数据'),
        ('del grads_v', '✓ 清理中间变量')
    ]

    all_passed = True
    for check_str, desc in checks:
        if check_str in content:
            print(f"  {desc}")
        else:
            print(f"  ✗ 缺少: {desc}")
            all_passed = False

    return all_passed


def check_fix_5_run_colab():
    """检查 run_colab.py 是否修复了 KeyError"""
    print("\n[修复 5] run_colab.py - 修复 KeyError")

    with open('run_colab.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查修复
    if "runner.config['training'] = {" in content:
        print("  ✓ 正确初始化 training 配置")
        return True

    print("  ✗ 仍然有 KeyError 风险")
    return False


def check_fix_6_ensemble():
    """检查 Ensemble 是否修复了索引混合问题"""
    print("\n[修复 6] Ensemble - 修复索引混合")

    with open('core/methods/ensemble.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否移除了错误的 merge
    if 'all_previous.extend(prev)' in content:
        print("  ✗ 仍然包含错误的 merge 逻辑")
        return False

    # 检查是否有 _compute_reducible_losses
    if '_compute_reducible_losses' in content:
        print("  ✓ 实现了新的可约损失计算")
        return True

    print("  ✗ 缺少必要的实现")
    return False


def main():
    """运行所有检查"""
    print("="*60)
    print("🔍 Bug 修复验证工具")
    print("="*60)

    results = {
        '修复 1 (_create_combined_loader)': check_fix_1_create_combined_loader(),
        '修复 2 (RandomSelector)': check_fix_2_random_selector(),
        '修复 3 (CSReLSelector)': check_fix_3_csrel_selector(),
        '修复 4 (BCSRSelector)': check_fix_4_bcsr_selector(),
        '修复 5 (run_colab.py)': check_fix_5_run_colab(),
        '修复 6 (Ensemble)': check_fix_6_ensemble()
    }

    # 总结
    print("\n" + "="*60)
    print("📋 验证结果总结")
    print("="*60)

    all_passed = True
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n🎉 所有修复已正确应用！")
        print("你现在可以运行：")
        print("  • python run_quick.py")
        print("  • python run_colab.py")
        print("  • python scripts/run_experiments.py --quick")
        return 0
    else:
        print("\n⚠️  部分修复未应用，请检查失败的项目")
        return 1


if __name__ == "__main__":
    sys.exit(main())
