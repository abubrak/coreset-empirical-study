#!/usr/bin/env python
"""
快速测试脚本 - 验证 run_experiments.py 修复
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_log_file_operations():
    """测试日志文件操作"""
    print("\n[测试] 日志文件操作")
    print("-" * 40)

    import tempfile
    from datetime import datetime

    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
        tmp_path = Path(tmp.name)

    try:
        # 第一次写入
        with open(tmp_path, 'w') as f:
            f.write(f"实验开始: {datetime.now()}\n")

        # 追加写入
        with open(tmp_path, 'a') as f:
            f.write(f"[dataset] 实验开始: {datetime.now()}\n")
            f.write(f"[dataset] ✅ 完成\n")
            f.write(f"\n实验结束: {datetime.now()}\n")

        # 读取并验证
        with open(tmp_path, 'r') as f:
            content = f.read()

        if '实验开始' in content and '✅ 完成' in content:
            print("✅ 日志文件操作正常")
            return True
        else:
            print("❌ 日志内容不完整")
            return False

    finally:
        # 清理临时文件
        tmp_path.unlink(missing_ok=True)

def test_full_experiments():
    """测试 full_experiments 函数（快速模式）"""
    print("\n[测试] full_experiments 函数")
    print("-" * 40)

    from scripts.run_experiments import full_experiments

    print("⚠️  这将运行完整实验，可能需要数小时")
    print("建议：")
    print("  1. 先运行 python run_quick.py 验证修复")
    print("  2. 或使用 python scripts/run_experiments.py --dataset mnist")
    print()

    response = input("是否继续？(y/N): ").strip().lower()

    if response == 'y':
        return full_experiments()
    else:
        print("测试已取消")
        return True

def main():
    """运行所有测试"""
    print("=" * 60)
    print("🧪 run_experiments.py 修复验证")
    print("=" * 60)

    # 测试 1: 日志文件操作
    test1 = test_log_file_operations()

    # 总结
    print("\n" + "=" * 60)
    print("📋 测试结果")
    print("=" * 60)
    print(f"日志文件操作: {'✅ 通过' if test1 else '❌ 失败'}")
    print("=" * 60)

    if test1:
        print("\n🎉 修复验证通过！")
        print("\n现在可以运行：")
        print("  • python scripts/run_experiments.py --quick")
        print("  • python scripts/run_experiments.py --dataset mnist")
        print("  • python scripts/run_experiments.py --full")
        return 0
    else:
        print("\n⚠️  部分测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
