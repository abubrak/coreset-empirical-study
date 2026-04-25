@echo off
REM 快速验证实验脚本 (Windows 版本)

echo =========================================
echo 🔍 快速验证实验
echo 目的: 验证所有 4 个 bug 已修复
echo 预计时间: 5-10 分钟
echo =========================================
echo.

REM 运行快速测试
echo 🚀 运行快速测试...
echo 数据集: MNIST
echo 方法: random, greedy, csrel, bcsr
echo 任务: 5 个
echo Epochs: 10
echo.

python run_quick.py

REM 检查结果
if %ERRORLEVEL% EQU 0 (
    echo.
    echo =========================================
    echo ✅ 所有测试通过！
    echo.
    echo 预期结果:
    echo   • random: 无 IndexError
    echo   • greedy: 无 IndexError
    echo   • csrel: 无 tensor size mismatch
    echo   • bcsr: 无 CUDA OOM
    echo.
    echo 结果文件: results\quick_test_*.json
    echo =========================================
) else (
    echo.
    echo =========================================
    echo ❌ 测试失败！
    echo 请检查错误信息并修复
    echo =========================================
    exit /b 1
)
