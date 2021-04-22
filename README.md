# Wiener Documentation
Monitering the nodes status: [http://wiener.hpc.net.uq.edu.au/ganglia/?r=hour&cs=&ce=&m=load_one&s=by+name&c=Wiener&tab=m&vn=&hide-hf=false](http://wiener.hpc.net.uq.edu.au/ganglia/?r=hour&cs=&ce=&m=load_one&s=by+name&c=Wiener&tab=m&vn=&hide-hf=false)


### Arguments:
`-N`: number of nodes.

`-w`: spesifiy which node you want to use. Example: -w gpunode-1-14

`--cpus-per-task`: how many cpus you want to have for the task. Example: --cpus-per-task=4 (will use 4 cpus)

`--mem-per-cpu`: how much memery you want for each cpu. Example: --mem-per-cpu=30G (each cpu has 30G ram)

`--gres`: how many gpus you want for your job. Example: `--gres=gpu:2` (will use 2 gpus), `--gres=gpu:tesla-smx2:2` (use 2 tesla GPU)

### Interactive mode: 
`srun -N 1 --cpus-per-task=2 --mem-per-cpu=30G  --gres=gpu:2 --pty bash` (1 node, 2 cpus, memory= 2 * 30g, 2 gpus.)

`srun -N 4 --ntasks-per-node=1 --cpus-per-task=2 --mem-per-cpu=30G --partition=gpu --gres=gpu:1 --pty bash` (4 node, 2 cpus per node, memory= 2 * 30g per node, 2 gpus.)

### Submit job with sbatch:
```
#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=PPO_train2
#SBATCH -n 1
#SBATCH -w gpunode-1-14
#SBATCH --time=48:00:00
#SBATCH --mail-user=xxx@xxx.com
#SBATCH --mail-type=END
#SBATCH --mem-per-cpu=30G
#SBATCH -o tensor_out2.txt
#SBATCH -e tensor_erro2.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:2
#SBATCH --cpus-per-task=4

module load anaconda/3.6
source activate deeplearning
module load cuda/10.0.130
module load gnu/5.4.0
module load mvapich2

srun python3 PPO_multi_gpu_train.py
```

### Monitoring your job:
`squeue`: list all current jobs of all users.

`squeue -u username <yourusername>`: list the jobs that belong to the user of username.

`watch -p -n 1 scontrol show jobid -dd 303438`: show the details of job with id 303438.

###
UQ RDM mounted folder: /afm02/Q3/Q3614/
