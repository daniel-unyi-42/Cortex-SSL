#!/bin/bash
#SBATCH --job-name=age_prediction
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=age_prediction_%j.log

module load singularity

srun singularity exec --nv --bind /project/c_gnn42/Cortex-SSL:/home/daniel docker://danielunyi42/pyg:latest python age_prediction.py --pretrained=True --frozen=True --num_labeled=70
