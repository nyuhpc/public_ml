#!/bin/bash

#SBATCH --job-name=imClass
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1

module purge 
module load r/intel/3.6.0

module load cuda/10.0.130
module load cudnn/10.0v7.6.2.24

Rscript imClass.R

