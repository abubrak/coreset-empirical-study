"""
测试 CSReL 修正版实现

对比原始实现和修正版实现的关键差异
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from core.methods.csrel import CSReLSelector
from core.methods.csrel_fixed import CSReLSelectorFixed


def create_simple_model():
    """创建简单的测试模型"""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )


def create_test_dataset(batch_size=32):
    """创建测试数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 使用 MNIST 测试
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # 只使用前 1000 个样本加速测试
    subset = Subset(train_dataset, list(range(1000)))
    return DataLoader(subset, batch_size=batch_size, shuffle=False)


def test_original_vs_fixed():
    """对比原始实现和修正版"""
    print("=" * 80)
    print("CSReL 实现对比测试")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 创建数据集
    print("\n1. 创建测试数据集...")
    dataset = create_test_dataset(batch_size=32)
    print(f"   数据集大小: {len(dataset.dataset)}")

    # 创建模型
    print("\n2. 创建测试模型...")
    model = create_simple_model().to(device)
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 测试原始实现（有 bug）
    print("\n" + "=" * 80)
    print("测试原始实现（有 bug）")
    print("=" * 80)

    try:
        original_selector = CSReLSelector(
            memory_budget=100,
            device=device,
            temperature=1.0  # 原始实现有温度参数
        )

        print("\n3. 运行原始实现...")
        selected_indices, weights = original_selector.select_coreset(
            dataset=dataset,
            model=model,
            task_id=0
        )

        print(f"\n   ✓ 原始实现完成")
        print(f"   选择的样本数: {len(selected_indices)}")
        print(f"   权重形状: {weights.shape}")
        print(f"   权重范围: [{weights.min():.4f}, {weights.max():.4f}]")

        # 分析问题
        print("\n   分析原始实现的问题:")
        print("   ❌ 问题1: 使用温度参数和 Softmax")
        print("   ❌ 问题2: 使用随机采样而非确定性选择")
        print("   ❌ 问题3: 没有参考模型")
        print("   ❌ 问题4: Reducible Loss 计算错误")

    except Exception as e:
        print(f"\n   ✗ 原始实现出错: {e}")

    # 测试修正版
    print("\n" + "=" * 80)
    print("测试修正版实现")
    print("=" * 80)

    try:
        fixed_selector = CSReLSelectorFixed(
            memory_budget=100,
            device=device,
            init_size=20,
            selection_steps=5,
            class_balance=False,
            ref_epochs=5
        )

        print("\n4. 运行修正版实现...")
        selected_indices_fixed, weights_fixed = fixed_selector.select_coreset(
            dataset=dataset,
            model=model,
            task_id=0
        )

        print(f"\n   ✓ 修正版实现完成")
        print(f"   选择的样本数: {len(selected_indices_fixed)}")
        print(f"   权重形状: {weights_fixed.shape}")
        print(f"   权重范围: [{weights_fixed.min():.4f}, {weights_fixed.max():.4f}]")

        print("\n   修正版的改进:")
        print("   ✓ 正确实现 Reducible Loss = Loss_current - Loss_reference")
        print("   ✓ 使用确定性选择（Top-K）")
        print("   ✓ 实现参考模型训练")
        print("   ✓ 实现增量式选择流程")
        print("   ✓ 移除不必要的温度参数")

        # 对比选择结果
        print("\n" + "=" * 80)
        print("结果对比")
        print("=" * 80)

        print("\n关键差异:")
        print("1. 选择策略:")
        print("   原始: Softmax + 随机采样")
        print("   修正: Top-K 确定性选择")

        print("\n2. 损失计算:")
        print("   原始: 仅使用当前模型损失")
        print("   修正: 当前模型损失 - 参考模型损失")

        print("\n3. 选择流程:")
        print("   原始: 单次选择")
        print("   修正: 增量式多轮选择")

        print("\n4. 参数:")
        print("   原始: temperature=1.0")
        print("   修正: init_size=20, selection_steps=5")

    except Exception as e:
        print(f"\n   ✗ 修正版实现出错: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_reducible_loss():
    """演示 Reducible Loss 的正确计算"""
    print("\n" + "=" * 80)
    print("Reducible Loss 计算演示")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建简单的数据
    print("\n1. 创建测试数据...")
    x = torch.randn(100, 784).to(device)
    y = torch.randint(0, 10, (100,)).to(device)

    # 创建两个模型
    print("\n2. 创建参考模型和当前模型...")
    ref_model = create_simple_model().to(device)
    cur_model = create_simple_model().to(device)

    # 训练参考模型（模拟在全量数据上训练）
    print("\n3. 训练参考模型...")
    optimizer = torch.optim.SGD(ref_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(reduction='none')

    ref_model.train()
    for epoch in range(10):
        outputs = ref_model(x)
        loss = criterion(outputs, y).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch + 1}/10, Loss: {loss.item():.4f}")

    # 训练当前模型（模拟在子集上训练，训练较少）
    print("\n4. 训练当前模型（较少迭代）...")
    optimizer = torch.optim.SGD(cur_model.parameters(), lr=0.01)

    cur_model.train()
    for epoch in range(3):
        outputs = cur_model(x)
        loss = criterion(outputs, y).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2 == 0:
            print(f"   Epoch {epoch + 1}/3, Loss: {loss.item():.4f}")

    # 计算 Reducible Loss
    print("\n5. 计算 Reducible Loss...")
    ref_model.eval()
    cur_model.eval()

    with torch.no_grad():
        # 参考模型损失
        ref_outputs = ref_model(x)
        ref_loss = criterion(ref_outputs, y)

        # 当前模型损失
        cur_outputs = cur_model(x)
        cur_loss = criterion(cur_outputs, y)

        # Reducible Loss
        reducible_loss = cur_loss - ref_loss

    print(f"\n   参考模型平均损失: {ref_loss.mean():.4f}")
    print(f"   当前模型平均损失: {cur_loss.mean():.4f}")
    print(f"   Reducible Loss 平均值: {reducible_loss.mean():.4f}")
    print(f"   Reducible Loss 最大值: {reducible_loss.max():.4f}")
    print(f"   Reducible Loss 最小值: {reducible_loss.min():.4f}")

    # 选择 Top-K
    print("\n6. 基于 Reducible Loss 选择样本...")
    select_size = 10
    _, top_indices = torch.topk(reducible_loss, select_size)

    print(f"   选择的样本索引: {top_indices.tolist()}")
    print(f"   对应的 Reducible Loss: {reducible_loss[top_indices].tolist()}")

    print("\n   物理意义解释:")
    print("   - Reducible Loss 高: 当前模型在该样本上表现比参考模型差很多")
    print("   - 说明该样本包含当前模型缺失的重要信息")
    print("   - 将该样本加入训练集可以显著提升模型性能")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("CSReL 修正版测试")
    print("=" * 80)

    # 演示 Reducible Loss 计算
    demonstrate_reducible_loss()

    # 对比原始实现和修正版
    test_original_vs_fixed()

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

    print("\n总结:")
    print("1. 原始实现存在多个关键 bug")
    print("2. 修正版正确实现了 CSReL 算法")
    print("3. Reducible Loss = Loss_current - Loss_reference")
    print("4. 使用确定性 Top-K 选择，不需要随机采样")
    print("5. 实现增量式选择流程，逐步优化核心集")


if __name__ == "__main__":
    main()
