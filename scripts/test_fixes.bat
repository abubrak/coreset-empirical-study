@echo off
REM 快速测试脚本 - 验证所有修复 (Windows 版本)

echo =========================================
echo 🧪 测试所有 bug 修复
echo =========================================
echo.

echo 测试 1: 验证脚本参数传递
echo ---------------------------------------

python experiments/run_comparison.py ^
    --dataset mnist ^
    --method random ^
    --method greedy ^
    --task_type split ^
    --num_tasks_list 2 ^
    --memory_ratios 0.1 ^
    --num_runs 1

if %ERRORLEVEL% EQU 0 (
    echo ✅ 参数传递正常
) else (
    echo ❌ 参数传递失败
    exit /b 1
)

echo.
echo =========================================
echo ✅ 所有测试通过！
echo 你现在可以运行完整实验
echo =========================================
pause
