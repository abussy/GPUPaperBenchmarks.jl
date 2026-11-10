#!/bin/bash -l

calc_info="${nranks_per_calc}rank_${julia_threads}thread"
if !((nranks_per_calc > 0 && 72 % nranks_per_calc == 0)); then
    echo "Error: the number of ranks per calculation must be an integer divisor of 72 (# CPU cores per GPU)" >&2
    exit 1
fi
ncpus=$((288/$nranks_per_calc)) # do not modify this line

# Important note: with the --overlap option, each srun job will share resources. Therefore, we allocate all resources
# for each job, i.e. nranks_per_calc*ncpus==288, and distribute the work with different CPU and GPU masks.
if !((nranks_per_calc * ncpus == 288)); then
    echo "Error: all resources MUST be allocated for each calculation" >&2
    exit 1
fi



# CPU masks to ensure each calculation runs on a different set of CPU cores, close to a given GPU.
mask0="mask_cpu:0xffffffffffffffffff"                                                        # cores 0-71
mask1="mask_cpu:0xffffffffffffffffff000000000000000000"                                      # cores 72-143
mask2="mask_cpu:0xffffffffffffffffff000000000000000000000000000000000000"                    # cores 144-215
mask3="mask_cpu:0xffffffffffffffffff000000000000000000000000000000000000000000000000000000"  # cores 216-287

# mps-wrapper.sh starts the MPS daemons for each GPU, and creates directories for piping and logging.
mps_prefix="/tmp/$(id -un)/slurm-${SLURM_JOBID}.0/nvidia" # hardcoded stepid 0 (not exported by mps-wrapper.sh)

# Systems distributed with some load balancing
systems0="silicon_primitive,gaas_110_slab"
systems1="silicon_primitive,mos2_supercell,silicon_supercell,mos2_primitive,aluminium_primitive,zro2_primitive,hbn_primitive,mgo_primitive,iron_primitive"
systems2="silicon_primitive,iron_supercell,aluminium_supercell,copt_primitive,silicon_primitive"
systems3="silicon_primitive,zro2_supercell,gold_111_slab,mgo_supercell,hbn_supercell,copt_supercell"

# First calculation on GPU 0 and CPU cores 0-71
if [[ $SLURM_LOCALID -eq 0 ]]; then
    
    srun -u --overlap \
            --cpu-bind=$mask0,verbose \
            --ntasks=$nranks_per_calc \
            --cpus-per-task=$ncpus \
            --output="mod0_${calc_info}.out" \
            julia -t $julia_threads --project=. scripts/run_benchmarks.jl $systems0 \
            --output_name="mod0_${calc_info}.csv" --warmup=false --nrepeats=3 --verbose=true \
            --compute_stresses=false --convergence=maxiter --maxiter=10 \
            & j0=$!

    # wait for calculation to finish
    wait $j0
fi

# Second calculation on GPU 1 and CPU cores 72-143
if [[ $SLURM_LOCALID -eq 1 ]]; then
    sleep 60
    srun -u --overlap \
            --cpu-bind=$mask1,verbose \
            --ntasks=$nranks_per_calc \
            --cpus-per-task=$ncpus \
            --output="mod1_${calc_info}.out" \
            julia -t $julia_threads --project=. scripts/run_benchmarks.jl $systems1 \
            --output_name="mod1_${calc_info}.csv" --warmup=false --nrepeats=3 --verbose=true \
            --compute_stresses=false --convergence=maxiter --maxiter=10 \
            & j1=$!

    # wait for calculation to finish
    wait $j1
fi

# Third calculation on GPU 2 and CPU cores 144-215
if [[ $SLURM_LOCALID -eq 2 ]]; then
    sleep 120
    srun -u --overlap \
            --cpu-bind=$mask2,verbose \
            --ntasks=$nranks_per_calc \
            --cpus-per-task=$ncpus \
            --output="mod2_${calc_info}.out" \
            julia -t $julia_threads --project=. scripts/run_benchmarks.jl $systems2 \
            --output_name="mod2_${calc_info}.csv" --warmup=false --nrepeats=3 --verbose=true \
            --compute_stresses=false --convergence=maxiter --maxiter=10 \
            & j2=$!

    # wait for calculation to finish
    wait $j2
fi

# Fourth calculation on GPU 3 and CPU cores 216-287
if [[ $SLURM_LOCALID -eq 3 ]]; then
    sleep 180
    srun -u --overlap \
            --cpu-bind=$mask3,verbose \
            --ntasks=$nranks_per_calc \
            --cpus-per-task=$ncpus \
            --output="mod3_${calc_info}.out" \
            julia -t $julia_threads --project=. scripts/run_benchmarks.jl $systems3 \
            --output_name="mod3_${calc_info}.csv" --warmup=false --nrepeats=3 --verbose=true \
            --compute_stresses=false --convergence=maxiter --maxiter=10 \
            & j3=$!

    # wait for calculation to finish
    wait $j3
fi
