#!/bin/bash
#SBATCH -J prefix_sum_DEBUG  
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o ./outerr/debug_output_%j.log 
#SBATCH --mem=32G

set -e

# --- 准备工作 ---
echo "================================================="
echo "        Prefix Sum GPU - DEBUG MODE              "
echo "================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job started on $(hostname) at $(date)"
echo ""
mkdir -p outerr
mkdir -p my_outputs

# --- 编译 ---
echo "STEP 1: Compiling source code using Makefile..."
make

if [ ! -f ./prefix_sum ]; then
    echo "COMPILATION FAILED!"
    exit 1
fi
echo "Compilation successful."
echo ""

# --- 调试运行 ---
# 只运行失败的测试用例 '10.in'
TEST_CASE_FILE="testcases/10.in"
MY_OUTPUT_FILE="my_outputs/10.myout"

echo "STEP 2: Running the failing test case [10] in verbose mode..."
echo "Command: ./prefix_sum ${TEST_CASE_FILE}"
echo "------------------- PROGRAM OUTPUT START -------------------"

# 直接运行程序，不捕获时间和错误，让所有信息都打印到日志
./prefix_sum "${TEST_CASE_FILE}" > "${MY_OUTPUT_FILE}"

# 捕获退出码
EXIT_CODE=$?

echo "-------------------- PROGRAM OUTPUT END --------------------"
echo "Program exited with code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "## RESULT: PROGRAM FAILED! ##"
    echo "Please check the output above for error messages like 'Segmentation fault' or 'HIP Error'."
    echo "Common non-zero exit codes:"
    echo "  - 1: Usually a custom error from your code (e.g., file open failed)."
    echo "  - 139: Segmentation fault (SIGSEGV) -> classic memory access error."
    echo "  - Other: Could be a failed HIP call caught by your HIP_CHECK macro."
else
    echo "## RESULT: PROGRAM SUCCEEDED (Exit Code 0) ##"
    echo "If the script still fails, there might be an issue with the verifier script or file permissions."
fi

echo ""
echo "Job finished at $(date)"
echo "================================================="