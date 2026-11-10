#!/bin/bash -l

#SBATCH --job-name=dftk-benchmark
#SBATCH --time=24:00:00 
#SBATCH --nodes=1
#SBATCH --account=csstaff
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
#SBATCH --uenv=julia/26.3:v1
#SBATCH --view=juliaup

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_MALLOC_FALLBACK=1
export JULIA_LOAD_PATH=:

ulimit -s unlimited

export JULIA_CONDAPKG_OFFLINE=true
export JULIA_PYTHONCALL_EXE="/ritom/scratch/cscs/abussy/dftk_daint/GPUPaperBenchmarks.jl/.CondaPkg/.pixi/envs/default/bin/python"

sleep 90
srun --ntasks=1 julia --project=. -e 'using CUDA, PaperBenchmarks; @info "Precompilation done"'

export julia_threads=4
export nranks_per_calc=16
srun --ntasks=4 --cpu-bind=socket ./parallel_jobs.sh
