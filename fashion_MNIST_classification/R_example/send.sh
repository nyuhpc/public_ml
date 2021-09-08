#!/bin/bash

#SBATCH --job-name=imClass
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=4GB

module purge 
module load r/intel/3.6.0

Rscript imClass.R
