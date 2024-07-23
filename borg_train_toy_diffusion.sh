#!/bin/bash
# Slurm script for training ITRA on Narval.
# Slurm GPU and memory are per node.
# DDP requires 1 task per GPU.

#SBATCH --mem=64G
#SBATCH --partition=plai
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00

echo "running"
export WANDB_MODE=online
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export WANDB_API_KEY=ccc990c8f025ef3aa8a73c9c6a5c7b14a6af95d8
export IAI_CONTAINERS=/ubc/cs/research/plai-scratch/iai/containers
export WANDB_CACHE_DIR=/ubc/cs/research/plai-scratch/jsefas

cd /ubc/cs/research/fwood/jsefas/toy-diffusion

# Match the gpu and nodes cfg to match those in the #SBATCH statements.
# do NOT iterate over condition_length
srun /opt/singularity-3.9.2/bin/singularity exec --nv \
  -B $(pwd) -B $WANDB_CACHE_DIR -B /scratch-ssd $IAI_CONTAINERS/iai-driving-models.sif \
  python toy_train.py no_wandb=0 sampler=$2
  #python toy_train.py batch_size=512 lr=$1 no_wandb=0 max_gradient=-1. loss_fn=l2 sampler=$2 diffusion=$3 p_uncond=$4
