#!/bin/bash
# Slurm script for training ITRA on Narval.
# Slurm GPU and memory are per node.
# DDP requires 1 task per GPU.

declare -A num_samples=( [3.]=16 [4.]=100 [5.]=1000 [6.]=100000 )
for version in 100 150 200 400 600 800
do
    for alpha in 3. 4. 5. 6.
    do
        for start_round in $(seq 0 10 100)
        do
            sbatch cedar_is_toy_diffusion.sh $alpha ${num_samples[$alpha]} $version $start_round
        done
    done
done
