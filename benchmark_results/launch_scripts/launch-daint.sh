#!/bin/bash -l

#SBATCH --job-name=dftk-benchmark
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=csstaff
#SBATCH --hint=nomultithread
#SBATCH --hint=exclusive
#SBATCH --no-requeue
#SBATCH --uenv=julia/26.3:v1
#SBATCH --view=juliaup

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_MALLOC_FALLBACK=1
export JULIA_LOAD_PATH=:

ulimit -s unlimited

srun --ntasks=1 julia --project=. -e 'using CUDA, PaperBenchmarks; @info "Precompilation done"'

srun --output="GPU_1rank_1thread.out" --cpu-bind=socket ./mps-wrapper.sh julia --project=. scripts/run_benchmarks.jl --output_name="GPU_1rank_1thread.csv" --warmup=true --nrepeats=3 --architecture=CUDA --verbose=true --compute_stresses=false --convergence=maxiter --maxiter=10

#srun --cpu-bind=socket ./mps-wrapper.sh julia --project=. my_tests/aluminium.jl
