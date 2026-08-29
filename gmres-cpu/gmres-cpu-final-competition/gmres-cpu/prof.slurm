#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -J gmres_dcu_prof   # 作业名，方便识别
#SBATCH -p wzidnormal
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=dcu:1

# 加载环境模块
module purge
module load compiler/gcc/12.2.0
module load compiler/dtk/25.04

# 编译代码
make clean && make

# 检查编译是否成功
if [ ! -f ./gmres ]; then
    echo "Compilation failed, executable ./gmres not found."
    exit 1
fi

# 定义矩阵文件所在的目录
MATRIX_DIR="/work/home/xdzs2025_bhz/GMRES/sparseMatrixSet"
MTX_FILE="${MATRIX_DIR}/mtx1.bin"

# --- 性能分析部分 ---
echo "======================================================"
echo "Starting Performance Profiling for ${MTX_FILE}..."
echo "======================================================"

if [ -f "${MTX_FILE}" ]; then
    hipprof --hip-trace ./gmres "${MTX_FILE}"
else
    echo "Warning: Matrix file ${MTX_FILE} not found, skipping profiling."
fi

echo "======================================================"
echo "Profiling finished."
echo "Please check the generated hip-prof-*.db, result_*.csv, and result_*.json files."
echo "======================================================"
