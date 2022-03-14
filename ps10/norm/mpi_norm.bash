#!/bin/bash

#SBATCH --account=niac
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --mem-per-cpu=4096

mpirun ./mpi_norm.exe ${*}

