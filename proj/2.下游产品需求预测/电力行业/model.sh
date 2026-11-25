#!/bin/bash
#SBATCH --job-name=demand_high        # 作业名称
#SBATCH --output=demand_high_output.txt      # 标准输出文件
#SBATCH --error=demand_high_error.txt       # 标准错误输出文件
#SBATCH --time=1000:00:00         # 预估作业运行时间（格式为 HH:MM:SS）
#SBATCH --ntasks=1              # 任务数
#SBATCH --cpus-per-task=24       # 每个任务使用的 CPU 核心数
#SBATCH --mem=200GB               # 内存大小

# 激活 Conda 环境
#source ~/software/anaconda3/etc/profile.d/conda.sh
#conda activate global-no2

# 你要执行的命令
python 全国及各省市下游预测_v4.5_全国_电力行业.py

