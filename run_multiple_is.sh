#!/bin/bash
# Slurm script for training ITRA on Narval.
# Slurm GPU and memory are per node.
# DDP requires 1 task per GPU.

# declare -A num_samples=( [3.]=370 [4.]=100 [5.]=10000 )
# for version in 1 4 7 10 13 16 19
# do
#     for alpha in 3. 4. 5.
#     do
#         sbatch ubcml_is_toy_diffusion.sh $alpha ${num_samples[$alpha]} $version
#     done
# done

num_rounds=10
dim=32
declare -A gaussian_params=( [8]=253673 [16]=1004369 [32]=3996833 )
declare -A bm_params=( [8]=263193 [16]=1042225 [32]=4147809 )
declare -A num_samples=( [3.]=16 [4.]=100 [5.]=1000 [6.]=100000 )
# for version in 100 200 300 400 500 600 700 800 900
# for version in $(seq 0 102400 2048000)
# for model in "VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_3996833_dim_32_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare4.7_v102400" "VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_3996833_dim_32_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.2_v1126400" "VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_3996833_dim_32_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.2_v2150400" "VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_3996833_dim_32_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.2_v3174400" "VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_3996833_dim_32_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.2_v4198400" "VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_3996833_dim_32_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.7_v5222400" "VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_3996833_dim_32_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.7_v6246400" "VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_3996833_dim_32_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.7_v7270400" "VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_3996833_dim_32_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.7_v8294400" "VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_3996833_dim_32_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.7_v9318400"
for model in "VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_3996833_dim_32_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.0_v204800" "VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_3996833_dim_32_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.0_v307200" "VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_3996833_dim_32_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.0_v409600" "VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_3996833_dim_32_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.2_v512000" "VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_3996833_dim_32_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.2_v614400" "VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_3996833_dim_32_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.2_v716800" "VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_3996833_dim_32_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.2_v819200" "VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_3996833_dim_32_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.2_v921600"
do
    for alpha in 3. 4.
    do
        for start_round in $(seq 0 10 90)
        do
	        pushd .
	        cd /ubc/cs/research/ubc_ml/jsefas/toy-diffusion 2> /dev/null
            round=$((start_round+num_rounds-1))
            python3 check_run.py --model $model --round $start_round --alpha $alpha 2> /dev/null
            popd
            if [[ $? -gt 0 ]]
            then
                sbatch ubcml_is_toy_diffusion.sh $alpha ${num_samples[$alpha]} $model $start_round $num_rounds $dim ${gaussian_params[$dim]}
            else
                echo skipping $model
            fi
        done
    done
done
