#!/bin/bash                                                                                          
# Example mps-wrapper.sh usage:                                                                      
# > srun [srun args] mps-wrapper.sh [cmd] [cmd args]                                                 

set -u                                                                                               

export CUDA_VISIBLE_DEVICES=$(( SLURM_LOCALID % 4 ))

exec "$@"
