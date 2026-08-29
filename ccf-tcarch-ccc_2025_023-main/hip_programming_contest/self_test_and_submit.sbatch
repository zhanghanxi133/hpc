#!/bin/bash
#SBATCH -J YourTeamName_SelfTest    # 作业名称，可以修改为自己的队名
#SBATCH --ntasks-per-node=1         # 每个节点运行 1 个任务
#SBATCH --cpus-per-task=8           # 每个任务使用 8 个 CPU核心
#SBATCH --gres=gpu:1                # 申请 1 张 GPU 卡
#SBATCH -o output_%j.log            # 将日志输出到 output_<job_id>.log 文件
#SBATCH --mem=32G

# --- Script Configuration ---
# C++ 编译器及参数
COMPILER="hipcc"
CXX_FLAGS=" "

# 源文件和可执行文件名
SOURCE_FILES="main.hip" # 如果有多个.cpp文件，用空格隔开
EXECUTABLE="main"

# 测试用例和输出结果的目录
TEST_CASE_DIR="testcases"
GOLDEN_DIR="testcases"
MY_OUTPUT_DIR="my_outputs" # 存放程序输出的临时目录

# 正确性验证脚本
VERIFIER="verify.py"

# --- Script Body ---

# 遇到任何错误立即退出
set -e

# 1. 准备工作
echo "================================================="
echo "               CCF TCARCH Self-Test              "
echo "================================================="
echo "Job started on $(hostname) at $(date)"
echo ""

# 清理旧的输出和可执行文件
rm -f ${EXECUTABLE}
rm -rf ${MY_OUTPUT_DIR}
mkdir -p ${MY_OUTPUT_DIR}

# 2. 编译代码
echo "STEP 1: Compiling source code..."
${COMPILER} ${CXX_FLAGS} -o ${EXECUTABLE} ${SOURCE_FILES}

# 检查编译是否成功
if [ ! -f ./${EXECUTABLE} ]; then
    echo "COMPILATION FAILED! Please check your code."
    exit 1
fi
echo "Compilation successful. Executable '${EXECUTABLE}' created."
echo ""

# 3. 初始化计时器和计数器
TOTAL_TIME=0.0
PASSED_COUNT=0
TEST_COUNT=$(ls -1q ${TEST_CASE_DIR}/*.in | wc -l)

echo "STEP 2: Running test cases..."
echo "Found ${TEST_COUNT} test cases in ${TEST_CASE_DIR}"
echo "-------------------------------------------------"

# 4. 循环执行所有测试用例
for input_file in ${TEST_CASE_DIR}/*.in; do
    # 从输入文件名派生出基础名和对应的输出文件名
    base_name=$(basename "${input_file}" .in)
    golden_output_file="${GOLDEN_DIR}/${base_name}.out"
    my_output_file="${MY_OUTPUT_DIR}/${base_name}.myout"

    echo -n "Running test [${base_name}]... "

    # 检查标准答案文件是否存在
    if [ ! -f "${golden_output_file}" ]; then
        echo "SKIPPED. Reason: Golden output file ${golden_output_file} not found."
        continue
    fi
    
    # 运行并计时。注意：这里的调用方式是 ./main <input_file>
    # 使用 /usr/bin/time 命令，并将时间输出 (stderr) 捕获到 EXEC_TIME 变量
    EXEC_TIME=$( { /usr/bin/time -f "%e" ./${EXECUTABLE} "${input_file}" > "${my_output_file}"; } 2>&1 )
    
    # 调用 Python 验证器进行比较
    if python3 ${VERIFIER} "${my_output_file}" "${golden_output_file}" >/dev/null 2>&1; then
        # 验证通过
        echo "PASS (${EXEC_TIME}s)"
        PASSED_COUNT=$((PASSED_COUNT + 1))
        # 使用 bc 工具进行浮点数加法
        TOTAL_TIME=$(echo "${TOTAL_TIME} + ${EXEC_TIME}" | bc)
    else
        # 验证失败
        echo "FAIL"
        echo "-------------------------------------------------"
        echo "ERROR: Output mismatch on test case [${base_name}]."
        echo "Your output is in: ${my_output_file}"
        echo "The correct output is in: ${golden_output_file}"
        echo "You can use 'diff -u ${golden_output_file} ${my_output_file}' to see the difference."
        echo "Aborting tests."
        exit 1
    fi
done

# 5. 输出最终总结
echo "-------------------------------------------------"
echo "FINAL RESULT: ALL TESTS PASSED!"
echo ""
echo "    Passed cases: ${PASSED_COUNT} / ${TEST_COUNT}"
echo "    Total execution time: ${TOTAL_TIME} seconds"
echo ""
echo "Job finished at $(date)"
echo "================================================="
