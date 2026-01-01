We aim to better understand which training- and inference-
time hyperparameters improve the conditional density estimate of a diffusion
model when using the instantaneous change of variables (ICOV) formula and
probability flow ODE. We investigate the effects of the training set size, the
number of parameter updates during training, the value of ϵ chosen for in-
tegration of the ODEs, and the number of inference time samples used to
estimate the density. We train diffusion models on two datasets: a multivari-
ate Gaussian and a discretized Brownian motion. The probability
densities in these cases are angularly symmetric about their respective ori-
gins, which in the case of Brownian motion is the all-zeroes trajectory. The
training data also comprises a scalar α as well as a boolean value which
indicates whether or not the data point ”exits” |α|. Together these features
enable classifier-free guided training.

[[chi_curves.pdf]]

1) toy_train.py
   - Trains the diffusion model on Gaussian or Brownian motion data.
   - Implements classifier and classifier-free guidance
   - python3 toy_train.py   --config-name=continuous_train_config   no_wandb=1   batch_size=4096   \
sampler=vpsde_velocity_sampler   example=brownian_motion_diff_example   diffusion=temporal_unet_alpha  \
 sampler.beta_schedule=CosineSchedule   likelihood=brownian_motion_diff_tails_likelihood   p_uncond=0.1  \
 likelihood.alpha=3.   max_alpha=4.   use_fixed_dataset=1   example.sde_steps=3   diffusion.dim=120  \
 models_to_save='[102400,1024000,10240000,40960000,71680000,102400000,409600000,716800000,1024000000,4096000000,7168000000,10240000000,15000000000,20480000000,25000000000,30720000000,35000000000,40960000000]' \
 diffusion.dim_mults="[1,2]"

2) toy_sample.py
   - Samples from the diffusion model and produces probability density estimates.
   - Contains implementations of numerical integration schemes: Euler, Heun, RK4, and uses torchdiffeq implementation of DoPri
   - Implements Hutchinson-Skilling trace estimator for stochastic approximation of divergence term in continuity equation
   - Implements SDE and PFODE formulations for inference-time sampling
   - python3 toy_sample.py \                                                                                                                                   ─╯
  --config-name=continuous_sample_config \
  sampler=vpsde_velocity_sampler \
  example=brownian_motion_diff_example \
  sampler.diffusion_timesteps=20 \
  sampler.beta_schedule=CosineSchedule \
  diffusion=temporal_unet_alpha \
  model_name=VPSDEVelocitySampler_TemporalUnetAlpha_dim_120_BrownianMotionDiff3ExampleConfig_v10240000000 \
  sampler.guidance_coef=0 \
  likelihood=brownian_motion_diff_tails_likelihood \
  likelihood.alpha=0.5 \
  diffusion.dim=120 \
  num_samples=100 \
  guidance=ClassifierFree \
  cond=1 \
  compute_exact_trace=True \
  num_hutchinson_samples=1 \
  diffusion.dim_mults='[1,2]' \
  example.sde_steps=3 \
  num_sample_batches=1 \
  run_histogram_convergence=False

3) models/
   - toy_temporal.py contains model architecture definition
   - toy_sampler.py contains implementations of \epsilon- and v-parameterization training strategies

4) toy_likelihoods.py
   - Generates labels for conditional training

5) toy_train_config.py
   - Configuration file

6) compute_quadratures.py
   - Implements numerical quadrature code for Brownian motion density and tail integral computations

7) toy_train_comparison.py
   - Plots error versus training time across different checkpoints
   - python toy_train_comparison.py --config-name=continuous_multivariate_8_sample_config \
  sampler=vpsde_velocity_sampler example=multivariate_gaussian_example sampler.diffusion_timesteps=20 \
  sampler.beta_schedule=CosineSchedule diffusion=temporal_unet_alpha \
  model_name=VPSDEVelocitySampler_TemporalUnetAlpha_dim_32_MultivariateGaussian8ExampleConfig_v10240001048 \
  sampler.guidance_coef=0 likelihood=multivariate_gaussian_tails_likelihood likelihood.alpha=1 \
  diffusion.dim=32 num_samples=1000 guidance=ClassifierFree cond=-1 compute_exact_trace=True \
  num_hutchinson_samples=1 diffusion.dim_mults='[1,2,4,8]' debug=False run_histogram_convergence=False \
  trained_models='[VPSDEVelocitySampler_TemporalUnetAlpha_dim_32_MultivariateGaussian8ExampleConfig_v102400696,VPSDEVelocitySampler_TemporalUnetAlpha_dim_32_MultivariateGaussian8ExampleConfig_v1024003672,VPSDEVelocitySampler_TemporalUnetAlpha_dim_32_MultivariateGaussian8ExampleConfig_v10240001048]'

8) new_plot_conditional_bm.py
   - Generates gif of level curves of Brownian motion density with corresponding chi-transformed density
