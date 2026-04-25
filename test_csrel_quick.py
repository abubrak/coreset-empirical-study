"""
快速测试 CSReL 修复
"""
import torch
import torch.nn as nn
import sys

# 测试模型架构检查
def test_architecture_check():
    """测试模型架构检查是否正确"""
    print("\n" + "="*60)
    print("测试1: 模型架构检查 (_is_same_architecture)")
    print("="*60)

    from core.methods.csrel import CSReLSelector

    selector = CSReLSelector(memory_budget=100)

    # 创建两个相同类型的模型
    model1 = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )

    model2 = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )

    # 创建相同类型但不同形状的模型
    model3 = nn.Sequential(
        nn.Linear(10, 30),  # 不同的形状
        nn.ReLU(),
        nn.Linear(30, 10)
    )

    # 测试相同架构
    result1 = selector._is_same_architecture(model1, model2)
    print(f"\n相同架构模型检查: {result1}")
    assert result1 == True, "应该返回 True"

    # 测试不同形状
    result2 = selector._is_same_architecture(model1, model3)
    print(f"不同形状模型检查: {result2}")
    assert result2 == False, "应该返回 False (形状不同)"

    print("\n[PASS] 模型架构检查测试通过")
    return True


def test_class_balanced_selection():
    """测试类别平衡选择是否正确处理空类别"""
    print("\n" + "="*60)
    print("测试2: 类别平衡选择 (_select_class_balanced)")
    print("="*60)

    from core.methods.csrel import CSReLSelector

    selector = CSReLSelector(memory_budget=100)

    # 创建测试数据
    num_samples = 100
    num_classes = 5

    # 可约损失
    rel_losses = torch.randn(num_samples)

    # 标签 - 故意缺少类别2和3
    all_targets = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 4] + [0] * 88)
    all_targets = all_targets[:num_samples]

    # 掩码 - 假设都可用
    mask = torch.ones(num_samples, dtype=torch.bool)

    # 选择20个样本
    k = 20

    print(f"\n测试设置:")
    print(f"  总样本数: {num_samples}")
    print(f"  类别数: {num_classes}")
    print(f"  选择样本数: {k}")
    print(f"  实际存在的类别: {all_targets.unique().tolist()}")
    print(f"  缺失的类别: {[c for c in range(num_classes) if c not in all_targets.unique()]}")

    # 测试不应该崩溃
    try:
        selected = selector._select_class_balanced(rel_losses, all_targets, mask, k)

        print(f"\n选择结果:")
        print(f"  选择的样本数: {len(selected)}")
        print(f"  选择的样本索引: {sorted(selected)[:10]}...")

        # 验证选择的样本
        selected_targets = all_targets[selected]
        print(f"  选择样本的类别分布: {selected_targets.unique().tolist()}")

        assert len(selected) == k, f"应该选择 {k} 个样本,实际选择了 {len(selected)} 个"
        assert len(set(selected)) == len(selected), "不应该有重复的索引"

        print("\n[PASS] 类别平衡选择测试通过 (正确处理空类别)")
        return True

    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_csrel_pipeline():
    """测试完整的 CSReL 流程"""
    print("\n" + "="*60)
    print("测试3: 完整 CSReL 流程")
    print("="*60)

    try:
        from core.methods.csrel import CSReLSelector
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device('cpu')

        # 创建简单数据集
        num_samples = 500
        num_features = 20
        num_classes = 5

        X = torch.randn(num_samples, num_features)
        y = torch.randint(0, num_classes, (num_samples,))

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        # 创建简单模型
        model = nn.Sequential(
            nn.Linear(num_features, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes)
        ).to(device)

        # 创建 CSReL 选择器
        selector = CSReLSelector(
            memory_budget=50,
            device=device,
            num_incremental_steps=3,
            init_ratio=0.1,
            class_balanced=True
        )

        print(f"\n设置:")
        print(f"  数据集大小: {num_samples}")
        print(f"  特征数: {num_features}")
        print(f"  类别数: {num_classes}")
        print(f"  核心集大小: {selector.memory_budget}")

        # 运行选择
        print(f"\n运行 CSReL 选择...")
        selected_indices, weights = selector.select_coreset(
            dataset=dataloader,
            model=model,
            task_id=0
        )

        print(f"\n结果:")
        print(f"  选择的样本数: {len(selected_indices)}")
        print(f"  权重形状: {weights.shape}")
        print(f"  权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"  权重和: {weights.sum():.4f}")

        assert len(selected_indices) == selector.memory_budget, "选择的样本数应该等于 memory_budget"
        assert len(weights) == len(selected_indices), "权重数量应该等于选择的样本数"
        assert abs(weights.sum() - 1.0) < 1e-5, "权重和应该为 1"

        print("\n[PASS] 完整 CSReL 流程测试通过")
        return True

    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("CSReL 修复验证测试")
    print("="*60)

    results = []

    # 测试1: 模型架构检查
    results.append(("模型架构检查", test_architecture_check()))

    # 测试2: 类别平衡选择
    results.append(("类别平衡选择", test_class_balanced_selection()))

    # 测试3: 完整流程
    results.append(("完整 CSReL 流程", test_full_csrel_pipeline()))

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)

    print("\n" + "="*60)
    if all_passed:
        print("[SUCCESS] 所有测试通过!")
        print("="*60)
        return 0
    else:
        print("[ERROR] 部分测试失败")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
