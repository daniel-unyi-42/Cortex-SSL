#!/bin/bash
#SBATCH --job-name=segmentation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=segmentation_%j.log

module load singularity

srun singularity exec --nv --bind /project/c_gnn42/Cortex-SSL:/home/daniel docker://danielunyi42/pyg:latest python segmentation.py --pretrained=True --frozen=False --num_labeled=7
