#!/bin/bash
# Slurm script for training ITRA on Narval.
# Slurm GPU and memory are per node.
# DDP requires 1 task per GPU.

num_rounds=1
dim=32
declare -A params=( [8]=263193 [16]=1042225 [32]=4147809 )
declare -A num_samples=( [3.]=16 [4.]=100 [5.]=1000 [6.]=100000 )
# for version in 100 200 300 400 500 600 700 800 900
for version in $(seq 1 1 9)
do
    for alpha in 3. 4.
    do
        for start_round in $(seq 0 1 9)
        do
            sbatch is_toy_diffusion.sh $alpha ${num_samples[$alpha]} $version $start_round $num_rounds $dim ${params[$dim]}
        done
    done
done
