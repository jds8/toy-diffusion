#!/bin/bash
# Slurm script for training ITRA on Narval.
# Slurm GPU and memory are per node.
# DDP requires 1 task per GPU.

#!/bin/bash
#SBATCH --mem=40000
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00-10:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --account=def-imitchel

echo "running"
export WANDB_MODE=online
export WANDB_CACHE_DIR=/home/jsefas1
export WANDB_DATA_DIR=/home/jsefas1
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
export WANDB_API_KEY=ccc990c8f025ef3aa8a73c9c6a5c7b14a6af95d8

cd /home/jsefas1
source venv/bin/activate
cd toy-diffusion

module load StdEnv/2020
module load StdEnv/2023
module load scipy-stack/2024a

# Match the gpu and nodes cfg to match those in the #SBATCH statements.
# do NOT iterate over condition_length
srun python3 toy_is.py \
  sampler=vpsde_velocity_sampler \
  example=brownian_motion_diff_example \
  example.sde_steps=104 \
  diffusion=temporal_unet_alpha \
  sampler.beta_schedule=CosineSchedule \
  sampler.diffusion_timesteps=5 \
  likelihood=brownian_motion_diff_tails_likelihood \
  likelihood.alpha=$1 \
  num_samples=$2 \
  guidance=ClassifierFree \
  sampler.guidance_coef=0. \
  cond=1 \
  num_rounds=100 \
  start_round=$4 \
  model_name=VPSDEVelocitySampler_TemporalUnetAlpha_BrownianMotionDiffExampleConfig_puncond_0.1_rare5.7_v$3_epoch$300 \
  # model_name=VPSDEVelocitySampler_TemporalIDK_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.7_v$3_epoch$300 \
