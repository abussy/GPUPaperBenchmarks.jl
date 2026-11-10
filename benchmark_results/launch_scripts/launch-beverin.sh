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
#SBATCH --partition=mi200

source ../setup_script

export MPICH_GPU_SUPPORT_ENABLED=1 
export MPICH_MALLOC_FALLBACK=1

ulimit -s unlimited

srun --ntasks=1 julia --project=. -e 'using AMDGPU, PaperBenchmarks; @info "Precompilation done"'

srun --output="GPU_1rank_1thread.out" bash -c '
julia --project=. scripts/run_benchmarks.jl --output_name="GPU_1rank_1thread.csv" --warmup=true --nrepeats=3 --architecture=AMDGPU --verbose=true --compute_stresses=false --convergence=maxiter --maxiter=10
'
