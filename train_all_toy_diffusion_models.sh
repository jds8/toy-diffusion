
# for sampler in base_xstart_sampler base_velocity_sampler base_epsilon_sampler
for sampler in vpsde_velocity_sampler vpsde_epsilon_sampler
do
    for lr in 0.0001
    do
        sbatch borg_train_toy_diffusion.sh $lr $sampler base_temporal_unet
    done
done
