#!/bin/bash

#SBATCH --job-name=imClass
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1

module purge
module load  anaconda3/5.3.1
source /share/apps/anaconda3/5.3.1/etc/profile.d/conda.sh

module load cuda/10.1.105
module load cudnn/10.1v7.6.5.32

#conda deactivate
conda activate ./penv_gpu

echo "--Python binary:"
which python3

python3 imClass.py
