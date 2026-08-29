#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -J gmres_check_arch   # 修改作业名，方便识别
#SBATCH -p wzidnormal
#SBATCH -t 00:10:00           # 只需要很短的时间
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=dcu:1

# 加载环境模块
module purge
module load compiler/gcc/12.2.0
module load compiler/dtk/25.04

# --- 查看硬件架构 ---
echo "======================================================"
echo "Checking DCU/GPU Hardware Architecture..."
echo "======================================================"

# 运行 rocminfo, 并用 grep 过滤出我们想要的信息
rocminfo | grep "Name:"

echo "======================================================"
echo "Architecture check finished."
echo "Please check the 'Name:' field above (e.g., gfx90a)."
echo "You can now use this value to set --offload-arch in your Makefile."
echo "======================================================"