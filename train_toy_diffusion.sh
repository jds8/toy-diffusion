#!/bin/bash
# Slurm script for training ITRA on Narval.
# Slurm GPU and memory are per node.
# DDP requires 1 task per GPU.

#SBATCH --mem=64G
#SBATCH --partition=ubcml
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00

echo "running"
export WANDB_MODE=online
export WANDB_CACHE_DIR=/ubc/cs/research/ubc_ml/jsefas
export WANDB_DATA_DIR=/ubc/cs/research/ubc_ml/jsefas
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export WANDB_API_KEY=ccc990c8f025ef3aa8a73c9c6a5c7b14a6af95d8

cd /ubc/cs/research/ubc_ml/jsefas
source venv/bin/activate
cd toy-diffusion

# Match the gpu and nodes cfg to match those in the #SBATCH statements.
# do NOT iterate over condition_length
srun python3 toy_train.py \
  no_wandb=0 \
  batch_size=4096 \
  sampler=vpsde_velocity_sampler \
  example=student_t_example \
  example.nu=2.1 \
  diffusion=temporal_idk \
  sampler.beta_schedule=CosineSchedule \
  likelihood=gaussian_tails_likelihood \
  p_uncond=0.1 \
  likelihood.alpha=3. \
  # example.sde_steps=104 \
  # model_name=VPSDEVelocitySampler_TemporalUnetAlpha_BrownianMotionDiffExampleConfig_puncond_0.1_v651 \
