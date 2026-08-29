#!/bin/bash
#SBATCH -o %j.out			    #指定作业的标准输出
#SBATCH -e %j.err			    #指定作业的错误输出
#SBATCH -J gmres_CPU			    #指定作业名称
#SBATCH -p wzidnormal			    #指定作业运行的队列
#SBATCH -t 00:30:00			    #指定作业最大运行120分钟
#SBATCH -N 1				    #指定节点数
#SBATCH --ntasks-per-node=1		    #指定每个节点的进程数
#SBATCH --ntasks-per-socket=1	            #指定每个Socket的进程数
#SBATCH --cpus-per-task=3		    #指定每个进程的CPU数，最大不超过3
#SBATCH --gres=dcu:1

module purge
module load compiler/gcc/12.2.0
module load compiler/dtk/25.04

make clean && make

if [ ! -f ./gmres ]; then
    echo "Compilation failed, executable ./gmres not found."
    exit 1
fi

# 定义矩阵文件所在的基础目录，需要修改为真实目录
MATRIX_DIR="/work/home/xdzs2025_bhz/GMRES/sparseMatrixSet"

for i in $(seq 1 80)
do
    MTX_FILE="${MATRIX_DIR}/mtx${i}.bin"
    
    if [ -f "$MTX_FILE" ]; then
        echo "================================================="
        echo "Running GMRES for mtx${i}.bin"
        echo "================================================="
        ./gmres "$MTX_FILE"
    else
        echo "Warning: Matrix file $MTX_FILE not found, skipping."
    fi
done

echo "All tests finished."
