#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=pretrain_%j.log

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12355

module load singularity

srun singularity exec --nv --bind /project/c_gnn42/Cortex-SSL:/home/daniel docker://danielunyi42/pyg:latest python -u pretrain.py
