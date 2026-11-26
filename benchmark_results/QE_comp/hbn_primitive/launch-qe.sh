#!/bin/bash -l

#SBATCH --job-name=hbn_primitive
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=csstaff
#SBATCH --hint=nomultithread
#SBATCH --hint=exclusive
#SBATCH --no-requeue
#SBATCH --uenv=quantumespresso/v7.6:v2
#SBATCH --view=default

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_MALLOC_FALLBACK=1
export OMP_NUM_THREADS=1

ulimit -s unlimited

srun pw.x -in pw_comp.in > pw_comp.out
srun pw.x -in pw_perf.in > pw_perf.out
