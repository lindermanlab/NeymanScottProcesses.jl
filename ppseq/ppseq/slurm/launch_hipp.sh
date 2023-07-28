#!/bin/bash
#
#SBATCH --job-name=hippocampus
#
#SBATCH --time=180:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --chdir=/scratch/users/ahwillia/neymanscott/ppseq
#SBATCH --output=/scratch/users/ahwillia/neymanscott/ppseq/slurm/logs/hippocampus_slurm-%A_%a.out

echo "Working Directory = $(pwd)"

module load julia
srun julia launch_hipp_slurm.jl

