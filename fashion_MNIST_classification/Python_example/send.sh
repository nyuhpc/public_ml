#!/bin/bash

#SBATCH --job-name=imClass
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=4GB

module purge
module load  anaconda3/5.3.1
source /share/apps/anaconda3/5.3.1/etc/profile.d/conda.sh
#conda deactivate
conda activate ./penv

echo "--Python binary:"
which python3

python3 imClass.py
