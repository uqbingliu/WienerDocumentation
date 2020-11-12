# WienerDocumentation
Monitering the nodes usages:
http://wiener.hpc.net.uq.edu.au/ganglia/?r=hour&cs=&ce=&m=load_one&s=by+name&c=Wiener&tab=m&vn=&hide-hf=false


Arguments:
-N: number of nodes
-w: spesifiy which node you want to use. Example: -w gpunode-1-14
--cpus-per-task: how many cpus you want to have for the task. Example: --cpus-per-task=4 (will use 4 cpus)
--mem-per-cpu: how much memery you want for each cpu. Example: --mem-per-cpu=30G (each cpu has 30G ram)
--gres: how many gpus you want for your job. Example: --gres=gpu:2 (will use 2 gpus)

Interactive mode:
srun -N 1 --cpus-per-task=2 --mem-per-cpu=30G  --gres=gpu:2 --pty bash

Submit job with sbatch:
#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=PPO_train2
#SBATCH -n 1
#SBATCH -w gpunode-1-14
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o tensor_out2.txt
#SBATCH -e tensor_erro2.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:2
#SBATCH --cpus-per-task=4

module load anaconda/3.6
source activate DeepLearning
module load cuda/10.0.130
module load gnu/5.4.0
module load mvapich2

srun python3 PPO_multi_gpu_train.py
