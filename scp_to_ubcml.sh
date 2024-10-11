#!/bin/bash

# scp from remote.cs.ubc.ca:/ubc/cs/home/j/jsefas to /ubc/cs/research/ubc_ml/jsefas/toy-diffusion/diffusion_models
declare -A gaussian_params=( [8]=253673 [16]=1004369 [32]=3996833 )
declare -A bm_params=( [8]=263193 [16]=1042225 [32]=4147809 )
for dim in 32 #8 16 32
do
    for version in $(seq 1 5 96)
    do 
        gaussian_model=VPSDEVelocitySampler_TemporalGaussianUnetAlpha_size_${gaussian_params[$dim]}_dim_${dim}_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.7_v${version}_epoch${version}0
	ssh -t jsefas@remote.cs.ubc.ca "scp $gaussian_model submit-ml:/ubc/cs/research/ubc_ml/jsefas/toy-diffusion/diffusion_models"
    
        bm_model=VPSDEVelocitySampler_TemporalUnetAlpha_size_${bm_params[$dim]}_dim_${dim}_BrownianMotionDiffExampleConfig_puncond_0.1_rare5.7_v${version}_epoch${version}0 
	ssh -t jsefas@remote.cs.ubc.ca "scp $bm_model submit-ml:/ubc/cs/research/ubc_ml/jsefas/toy-diffusion/diffusion_models"
    done
done
