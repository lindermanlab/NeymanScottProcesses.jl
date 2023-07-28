#!/bin/bash
#
#SBATCH --job-name=warped
#
#SBATCH --time=60:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --chdir=/scratch/users/ahwillia/code/neymanscott/ppseq
#SBATCH --output=/scratch/users/ahwillia/code/neymanscott/ppseq/slurm/logs/warped_slurm-%A_%a.out

echo "Working Directory = $(pwd)"

module load julia
srun julia launch_warped_slurm.jl

