"""
单元测试：批次索引系统
测试全局批次索引计数器的正确性
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, TensorDataset
from core.coreset_base import _parse_batch, reset_batch_index_counter


def test_unique_indices():
    """测试索引在批次间是唯一的"""
    reset_batch_index_counter()

    # 创建虚拟数据集
    data = torch.randn(100, 10)
    labels = torch.randint(0, 5, (100,))
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)

    all_indices = []
    for batch in loader:
        _, _, indices = _parse_batch(batch)
        all_indices.extend(indices.tolist())

    # 验证所有索引都是唯一的
    assert len(all_indices) == len(set(all_indices)), \
        f"索引不唯一！总索引数: {len(all_indices)}, 唯一索引数: {len(set(all_indices))}"

    # 验证索引从0开始连续递增
    expected_indices = list(range(len(all_indices)))
    assert all_indices == expected_indices, \
        f"索引不连续！期望: {expected_indices}, 实际: {all_indices}"


def test_reset_function():
    """测试重置功能是否正常工作"""
    # 第一次运行
    reset_batch_index_counter()

    data = torch.randn(20, 10)
    labels = torch.randint(0, 5, (20,))
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)

    indices_first_run = []
    for batch in loader:
        _, _, indices = _parse_batch(batch)
        indices_first_run.extend(indices.tolist())

    # 第二次运行（重置后）
    reset_batch_index_counter()

    indices_second_run = []
    for batch in loader:
        _, _, indices = _parse_batch(batch)
        indices_second_run.extend(indices.tolist())

    # 两次运行的索引应该完全相同（都从0开始）
    assert indices_first_run == indices_second_run, \
        f"重置失败！第一次: {indices_first_run}, 第二次: {indices_second_run}"


def test_three_element_batch():
    """测试三元组批次格式 (x, y, idx) 的处理"""
    reset_batch_index_counter()

    # 创建包含索引的三元组批次
    data = torch.randn(10, 10)
    labels = torch.randint(0, 5, (10,))
    indices = torch.arange(100, 110)  # 自定义索引

    batch = (data, labels, indices)
    x, y, idx = _parse_batch(batch)

    # 验证返回的是原始索引，而不是生成的索引
    assert torch.equal(idx, indices), \
        f"三元组批次处理错误！期望索引: {indices.tolist()}, 实际: {idx.tolist()}"

    # 验证数据正确
    assert torch.equal(x, data), "数据不匹配"
    assert torch.equal(y, labels), "标签不匹配"


def test_two_element_batch():
    """测试二元组批次格式 (x, y) 的处理"""
    reset_batch_index_counter()

    # 创建不包含索引的二元组批次
    data = torch.randn(10, 10)
    labels = torch.randint(0, 5, (10,))

    batch = (data, labels)
    x, y, idx = _parse_batch(batch)

    # 验证生成了正确的索引（从0开始）
    expected_indices = torch.arange(0, 10)
    assert torch.equal(idx, expected_indices), \
        f"二元组批次索引生成错误！期望: {expected_indices.tolist()}, 实际: {idx.tolist()}"

    # 验证数据正确
    assert torch.equal(x, data), "数据不匹配"
    assert torch.equal(y, labels), "标签不匹配"


def test_multiple_batches_increasing_indices():
    """测试多个批次时索引递增"""
    reset_batch_index_counter()

    # 创建较大的数据集，包含多个批次
    data = torch.randn(30, 10)
    labels = torch.randint(0, 5, (30,))
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)

    batch_indices = []
    for batch in loader:
        _, _, indices = _parse_batch(batch)
        batch_indices.append(indices.tolist())

    # 验证每个批次的索引范围正确
    assert batch_indices[0] == list(range(0, 10)), \
        f"第1批次索引错误: {batch_indices[0]}"
    assert batch_indices[1] == list(range(10, 20)), \
        f"第2批次索引错误: {batch_indices[1]}"
    assert batch_indices[2] == list(range(20, 30)), \
        f"第3批次索引错误: {batch_indices[2]}"


if __name__ == '__main__':
    # 直接运行测试
    print("Running batch index system tests...")

    try:
        test_unique_indices()
        print("[PASS] test_unique_indices")
    except AssertionError as e:
        print(f"[FAIL] test_unique_indices: {e}")

    try:
        test_reset_function()
        print("[PASS] test_reset_function")
    except AssertionError as e:
        print(f"[FAIL] test_reset_function: {e}")

    try:
        test_three_element_batch()
        print("[PASS] test_three_element_batch")
    except AssertionError as e:
        print(f"[FAIL] test_three_element_batch: {e}")

    try:
        test_two_element_batch()
        print("[PASS] test_two_element_batch")
    except AssertionError as e:
        print(f"[FAIL] test_two_element_batch: {e}")

    try:
        test_multiple_batches_increasing_indices()
        print("[PASS] test_multiple_batches_increasing_indices")
    except AssertionError as e:
        print(f"[FAIL] test_multiple_batches_increasing_indices: {e}")

    print("\nAll tests completed!")
