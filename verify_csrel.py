"""
验证 CSReL 实现是否符合规范
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import inspect
from core.methods.csrel import CSReLSelector


def verify_implementation():
    """验证 CSReL 实现的关键特性"""
    print("=" * 80)
    print("CSReL 实现规范验证")
    print("=" * 80)

    results = {}

    # 1. 检查 __init__ 参数
    print("\n1. 检查 __init__ 参数...")
    init_params = inspect.signature(CSReLSelector.__init__).parameters
    param_names = list(init_params.keys())

    # 检查是否有 temperature 参数（不应该有）
    has_temperature = 'temperature' in param_names
    results['无temperature参数'] = not has_temperature

    if has_temperature:
        print("   ❌ 发现 temperature 参数（应该移除）")
    else:
        print("   ✓ 已移除 temperature 参数")

    # 检查必要参数
    has_num_incremental_steps = 'num_incremental_steps' in param_names
    results['有num_incremental_steps参数'] = has_num_incremental_steps

    if has_num_incremental_steps:
        print("   ✓ 有 num_incremental_steps 参数")
    else:
        print("   ❌ 缺少 num_incremental_steps 参数")

    # 2. 检查关键方法
    print("\n2. 检查关键方法...")

    # 检查 _train_reference_model
    has_train_ref = hasattr(CSReLSelector, '_train_reference_model')
    results['有_train_reference_model方法'] = has_train_ref

    if has_train_ref:
        print("   ✓ 有 _train_reference_model 方法")
    else:
        print("   ❌ 缺少 _train_reference_model 方法")

    # 检查 _compute_losses
    has_compute_losses = hasattr(CSReLSelector, '_compute_losses')
    results['有_compute_losses方法'] = has_compute_losses

    if has_compute_losses:
        print("   ✓ 有 _compute_losses 方法")
    else:
        print("   ❌ 缺少 _compute_losses 方法")

    # 3. 检查源代码中的关键实现
    print("\n3. 检查源代码实现...")

    source = inspect.getsource(CSReLSelector.select_coreset)

    # 检查 Reducible Loss 计算公式
    has_rel_formula = 'cur_losses - self.ref_losses' in source
    results['正确计算Reducible Loss'] = has_rel_formula

    if has_rel_formula:
        print("   ✓ 正确计算 Reducible Loss = cur_losses - ref_losses")
    else:
        print("   ❌ Reducible Loss 计算公式不正确")

    # 检查是否使用 torch.topk
    has_topk = 'torch.topk' in source
    results['使用torch.topk选择'] = has_topk

    if has_topk:
        print("   ✓ 使用 torch.topk 进行选择")
    else:
        print("   ❌ 未使用 torch.topk（可能使用了随机采样）")

    # 检查增量式选择循环
    has_incremental_loop = 'for step in range(self.num_incremental_steps)' in source
    results['有增量式选择循环'] = has_incremental_loop

    if has_incremental_loop:
        print("   ✓ 有增量式选择循环")
    else:
        print("   ❌ 缺少增量式选择循环")

    # 检查是否避免了随机采样
    has_multinomial = 'torch.multinomial' in source or 'multinomial' in source
    results['避免随机采样'] = not has_multinomial

    if has_multinomial:
        print("   ❌ 仍然使用随机采样 (multinomial)")
    else:
        print("   ✓ 避免了随机采样")

    # 4. 检查参考模型训练
    print("\n4. 检查参考模型训练实现...")
    train_ref_source = inspect.getsource(CSReLSelector._train_reference_model)

    trains_on_full_data = 'all_data' in train_ref_source and 'all_targets' in train_ref_source
    results['参考模型在全量数据上训练'] = trains_on_full_data

    if trains_on_full_data:
        print("   ✓ 参考模型在全量数据上训练")
    else:
        print("   ❌ 参考模型未在全量数据上训练")

    # 5. 总结
    print("\n" + "=" * 80)
    print("验证结果总结")
    print("=" * 80)

    all_passed = all(results.values())

    for item, passed in results.items():
        status = "✓" if passed else "❌"
        print(f"{status} {item}")

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ 所有检查通过 - 实现符合规范")
    else:
        failed_count = sum(1 for v in results.values() if not v)
        print(f"❌ {failed_count} 项检查未通过")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    verify_implementation()
