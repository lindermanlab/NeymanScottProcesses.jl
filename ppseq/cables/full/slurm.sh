#!/bin/bash
#
#SBATCH --job-name=cables_full
#
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G

datadir="/home/degleris/scratch/cables/"
echo "Working Directory = $(pwd)"
module load julia
echo "Julia loaded"

julia -t 4 /home/degleris/neymanscott/cables/full/run.jl $datadir
