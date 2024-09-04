#!/usr/bin/env python3

import logging
import warnings
import wandb
import os
import pathlib

from typing_extensions import Tuple

import hydra
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataclasses import dataclass

from toy_plot import SDE, Trajectories, integrate, score_function_heat_map, create_gif
from toy_train_config import TrainConfig, get_model_path, ExampleConfig, \
    GaussianExampleConfig, BrownianMotionExampleConfig, BrownianMotionDiffExampleConfig, \
    UniformExampleConfig, StudentTExampleConfig, StudentTDiffExampleConfig
from toy_configs import register_configs
from toy_likelihoods import traj_dist, Likelihood
from models.toy_temporal import TemporalTransformerUnet, TemporalUnet, \
    TemporalNNet, DiffusionModel, TemporalUnetAlpha
from models.toy_sampler import ForwardSample, AbstractSampler, \
    AbstractContinuousSampler, ForwardSample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVED_MODEL_DIR = 'diffusion_models'

def suppresswarning():
    warnings.warn("user", UserWarning)


class ModelInput:
    def __init__(self, x0_in, cond):
        self.x0_in = x0_in
        self.cond = cond

    def get_kept_x(self, kept_idx: torch.Tensor):
        x0_raw, x0 = self.x0_in
        kept_raw = x0_raw[kept_idx]
        kept_x0 = x0[kept_idx]
        kept_x = kept_raw, kept_x0
        return kept_x

    def update(self, kept_idx: torch.Tensor):
        kept_x = self.get_kept_x(kept_idx)
        return ModelInput(kept_x, self.cond[kept_idx])


class AlphaModelInput(ModelInput):
    def __init__(self, x0_in, cond, alpha):
        super().__init__(x0_in, cond)
        self.alpha = alpha

    def update(self, kept_idx: torch.Tensor):
        kept_x = self.get_kept_x(kept_idx)
        return AlphaModelInput(
            kept_x,
            self.cond[kept_idx],
            self.alpha[kept_idx]
        )


class ToyTrainer:
    def __init__(self, cfg: TrainConfig, diffusion_model):
        self.cfg = cfg
        self.diffusion_model = diffusion_model

        self.sampler = hydra.utils.instantiate(cfg.sampler)

        self.likelihood = hydra.utils.instantiate(cfg.likelihood)
        self.example = OmegaConf.to_object(cfg.example)

        self.diffusion_model = nn.DataParallel(diffusion_model).to(device)
        self.loss_fn = self.get_loss_fn()
        self.n_samples = torch.tensor([self.cfg.batch_size], device=device)
        self.end_time = torch.tensor(1., device=device)

        self.num_saves = 0
        self.rarity = torch.tensor(0.)

        self.dl_iter = None
        self.initialize_optimizer()

        if self.cfg.model_name:
            self.load_model()

    def load_model(self):
        model_path = get_model_path(self.cfg)
        try:
            # load softmax model
            print('attempting to load diffusion model: {}'.format(model_path))
            self.diffusion_model.module.load_state_dict(torch.load('{}'.format(model_path)))
            print('successfully loaded diffusion model')
        except Exception as e:
            print('FAILED to load model: {} because {}\ncreating it...'.format(model_path, e))

    def initialize_optimizer(self):
        self.optimizer = torch.optim.Adam(self.diffusion_model.module.parameters(), self.cfg.lr)
        self.num_steps = 0
        self.num_epochs = 0
        self.last_saved_epoch = 0

    def clip_gradients(self):
        nn.utils.clip_grad_norm_(self.diffusion_model.module.parameters(), self.cfg.max_gradient)

    def likelihood_weighting(self, model_output, forward_sample: ForwardSample):
        _, _, std = self.sampler.marginal_prob(
            x=torch.zeros_like(forward_sample.xt),
            t=forward_sample.t
        )
        score = self.sampler.get_sf_estimator(
            model_output,
            xt=forward_sample.xt,
            t=forward_sample.t
        )
        losses = (score + forward_sample.noise / std) ** 2  # score = -eps / std so we have *plus sign*
        g2 = self.sampler.sde(torch.zeros_like(model_output), forward_sample.t)[1] ** 2
        return (losses * g2).mean()

    def get_loss_fn(self):
        if self.cfg.loss_fn == 'likelihood_weighting':
            return self.likelihood_weighting
        else:
            if self.cfg.loss_fn == 'l1':
                loss_fn = torch.nn.L1Loss()
            elif self.cfg.loss_fn == 'l2':
                loss_fn = torch.nn.MSELoss()
            return lambda model_output, forward_sample: loss_fn(
                model_output,
                forward_sample.to_predict,
            )

    def log_artifact(self, saved_model_path, artifact_type):
        artifact = wandb.Artifact(artifact_type, type='model')
        artifact.add_file(saved_model_path)
        try:
            print('attempting to log model')
            wandb.log_artifact(artifact)
            print('successfully logged model')
        except Exception as e:
            print('failed to log model due to {}'.format(e))

    def delete_model(self, saved_model_path):
        print('removing {}'.format(saved_model_path))
        os.remove(saved_model_path)

    def _save_model(self):
        self.num_saves += 1
        self.last_saved_epoch = self.num_epochs
        rarity = '%.1f' % self.rarity
        saved_model_path = '{}_rare{}_v{}'.format(
            get_model_path(self.cfg),
            rarity,
            self.num_saves,
        )
        if self.last_saved_epoch:
            saved_model_path += '_epoch{}'.format(self.last_saved_epoch)
        try:
            pathlib.Path(SAVED_MODEL_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(self.diffusion_model.module.state_dict(), saved_model_path)
            print('saved model {}'.format(self.num_saves))
        except Exception as e:
            print('could not save model because {}'.format(e))
        return saved_model_path

    def should_save(self) -> bool:
        if self.cfg.use_fixed_dataset:
            return self.num_epochs % self.cfg.epochs_before_save == 0 and \
                   self.last_saved_epoch < self.num_epochs
        else:
            return self.num_steps % self.cfg.iterations_before_save == 0

    def train(self):
        while True:
            self.train_batch()
            if self.should_save():
                saved_model_path = self._save_model()
                if isinstance(self.example, GaussianExampleConfig):
                    # score_function_heat_map(
                    #     lambda x, time: self.diffusion_model(
                    #         x=x.reshape(-1, 1),
                    #         time=time.reshape(-1, 1),
                    #     ),
                    #     self.num_saves,
                    #     t_eps=1e-5,
                    #     mu=self.cfg.example.mu,
                    #     sigma=self.cfg.example.sigma,
                    # )
                    try:
                        score_function_heat_map(
                            lambda x, time: self.sampler.get_sf_estimator(
                                self.diffusion_model(
                                    x=x,
                                    time=time,
                                ),
                                xt=x.reshape(-1, 1, 1),
                                t=time.reshape(-1)
                            ),
                            self.num_saves,
                            t_eps=1e-5,
                            # mu=self.cfg.example.mu,  # not including mu and sigma due to standardization
                            # sigma=self.cfg.example.sigma,
                        )
                        # create_gif('figs/heat_maps', '{}_training_scores'.format(
                        #     OmegaConf.to_object(self.cfg.sampler).name()
                        # ))
                    except Exception as e:
                        print(e)
                        pass
                if not self.cfg.no_wandb:
                    self.log_artifact(saved_model_path, 'diffusion_model')
                    self.delete_model(saved_model_path)

    def train_batch(self):
        if self.cfg.use_fixed_dataset:
            x0 = self.get_batch()
        else:
            x0 = self.get_x0()
        loss = self.forward_process(x0)
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
            loss.backward()
            self.clip_gradients()
            self.optimizer.step()
            self.num_steps += 1
            try:
                if not self.cfg.no_wandb:
                    wandb.log({"train_loss": loss.detach()})
                    grads = []
                    for param in self.diffusion_model.parameters():
                        grads.append(param.grad.view(-1))
                    grads = torch.cat(grads)
                    grad_norm = grads.norm()
                    wandb.log({"train_grad_norm": grad_norm})
                else:
                    print("train_loss: {}".format(loss.detach()))
            except Exception as e:
                print(e)

    def viz_trajs(self, traj, end_time, idx, clf=True):
        import matplotlib.pyplot as plt
        full_state_pred = traj.detach().squeeze(0).cpu().numpy()

        plt.plot(torch.linspace(0, end_time, full_state_pred.shape[0]), full_state_pred, color='green')

        if idx % 100 == 0:
            plt.savefig('figs/train_{}.pdf'.format(idx))

        if clf:
            plt.clf()

    def get_x0(self):
        if type(self.example) == BrownianMotionDiffExampleConfig:
            x0 = torch.randn(
                self.cfg.batch_size,
                self.cfg.example.sde_steps-1,
                1,
                device=device
            )
            dt = self.end_time / (self.cfg.example.sde_steps-1)
            scaled_x0 = x0 * dt.sqrt()  # standardize data
            x0_raw = torch.cat([
                torch.zeros(self.cfg.batch_size, 1, 1, device=device),
                scaled_x0.cumsum(dim=1)
            ], dim=1)
        elif isinstance(self.example, GaussianExampleConfig):
            x0_raw = torch.randn(
                self.cfg.batch_size, 1, 1, device=device
            )
            x0 = x0_raw
        elif isinstance(self.example, UniformExampleConfig):
            x0_raw = torch.rand(
                self.cfg.batch_size, 1, 1, device=device
            )
            # E[logit(X)] = 0 if X is uniform(0, 1)
            logit_x0 = torch.logit(x0_raw)
            # Var[logit(X)] = pi^2/3 if X is uniform
            x0 = logit_x0 / (torch.pi / torch.tensor(3.).sqrt())
        elif isinstance(self.example, StudentTExampleConfig):
            x0_raw = dist.StudentT(self.cfg.example.nu).sample([
                self.cfg.batch_size, 1, 1,
            ]).to(device)
            scale = torch.tensor(35.9865)  # from dist.StudentT(1.5).sample([100000000]).var()
            if self.cfg.example.nu > 2.:
                scale = torch.tensor(self.cfg.example.nu / (self.cfg.example.nu - 2)).sqrt()
            x0 = x0_raw / scale
        elif isinstance(self.example, StudentTDiffExampleConfig):
            unscaled_x0 = dist.StudentT(self.cfg.example.nu).sample([
                self.cfg.batch_size,
                self.cfg.example.sde_steps-1,
                1,
            ]).to(device)
            scale = torch.tensor(self.cfg.example.nu / (self.cfg.example.nu - 2)).sqrt()
            x0 = unscaled_x0 / scale
            dt = self.end_time / (self.cfg.example.sde_steps-1)
            scaled_x0 = x0 * dt.sqrt()  # standardize data
            x0_raw = torch.cat([
                torch.zeros(self.cfg.batch_size, 1, 1, device=device),
                scaled_x0.cumsum(dim=1)
            ], dim=1)
        else:
            raise NotImplementedError
        return x0_raw, x0

    def set_dl_iter(self):
        dataset = torch.load('training_data.pt')
        dl = DataLoader(dataset, batch_size=self.cfg.batch_size)
        self.dl_iter = iter(dl)

    def get_batch(self):
        if self.dl_iter is None:
            self.set_dl_iter()
        try:
            x0_raw = next(self.dl_iter)
        except StopIteration:
            self.set_dl_iter()
            x0_raw = next(self.dl_iter)
        self.num_epochs += 1
        if type(self.example) == BrownianMotionDiffExampleConfig:
            x0_raw = x0_raw.to(device)
            dt = self.end_time / (self.cfg.example.sde_steps-1)
            x0 = x0_raw.diff(dim=1) / dt.sqrt()
        else:
            raise NotImplementedError
        return x0_raw, x0

    def upsample(self, x0_in, model_in, factor=3) -> Tuple[
            torch.Tensor,
            ModelInput,
]:
        _, x0 = x0_in
        num_rare = torch.maximum(model_in.cond.sum(), torch.tensor(1.))
        nonrare_idx = model_in.cond.squeeze().logical_not().to(float).nonzero()
        rare_idx = model_in.cond.squeeze().to(float).nonzero()
        kept_nonrare_idx = nonrare_idx[:(num_rare * factor).to(int)]
        kept_idx = torch.cat([rare_idx, kept_nonrare_idx]).squeeze()
        kept_model_in = model_in.update(kept_idx)
        return kept_model_in

    def get_cond_model_input(self, x0_in) -> ModelInput:
        raise NotImplementedError

    def get_uncond_model_input(self, x0_in) -> ModelInput:
        raise NotImplementedError

    def forward_process(self, x0_in):
        use_cond = torch.rand(1) > self.cfg.p_uncond
        if use_cond:
            model_in = self.get_cond_model_input(x0_in)
        else:
            model_in = self.get_uncond_model_input(x0_in)

        self.rarity = torch.maximum(self.rarity, self.likelihood.get_rarity(*x0_in).max())

        if self.cfg.upsample:
            model_in = self.upsample(x0_in, model_in)

        extras = {}
        if isinstance(self.example, GaussianExampleConfig):
            extras['mu'] = self.cfg.example.mu
            extras['sigma'] = self.cfg.example.sigma

        _, x0 = model_in.x0_in
        forward_sample_output = self.sampler.forward_sample(
            x_start=x0,
            extras=extras
        )

        model_output = self.get_model_output(forward_sample_output, model_in)

        loss = self.loss_fn(model_output, forward_sample_output)

        return loss


class ConditionTrainer(ToyTrainer):
    def analytical_gaussian_score(self, t, x):
        '''
        Compute the analytical marginal score of p_t for t in (0, 1)
        given the SDE formulation from Song et al. in the case that
        p_0 = N(mu_0, sigma_0) and p_1 = N(0, 1)
        '''
        pseudo_example = self.cfg.example.copy()
        pseudo_example['mu'] = 0.
        pseudo_example['sigma'] = 1.
        mean, _, std = self.sampler.analytical_marginal_prob(
            t=t,
            example=pseudo_example,
        )
        var = std ** 2
        score = (mean - x) / var
        return score

    def compare_score(self, x, time, model_output):
        if isinstance(self.example, GaussianExampleConfig) and \
           isinstance(self.sampler, AbstractContinuousSampler):
            true_sf = self.analytical_gaussian_score(t=time, x=x)
            sf_estimate = self.sampler.get_sf_estimator(model_output, xt=x, t=time)
            error = (true_sf.squeeze() - sf_estimate.detach().squeeze()).norm()
            if not self.cfg.no_wandb:
                wandb.log({"score error": error})
            else:
                print('score error: {}'.format(error))
        return



class ThresholdConditionTrainer(ConditionTrainer):
    def get_cond_model_input(self, x0_in) -> ModelInput:
        x0_raw, x0 = x0_in
        cond = self.likelihood.get_condition(x0_raw, x0).reshape(-1, 1)
        return ModelInput(x0_in, cond)

    def get_uncond_model_input(self, x0_in) -> ModelInput:
        return ModelInput(x0_in, None)

    def get_model_output(
            self,
            forward_sample_output: ForwardSample,
            model_in: ModelInput,
    ) -> torch.Tensor:
        return self.diffusion_model(
            x=forward_sample_output.xt,
            time=forward_sample_output.t,
            cond=model_in.cond,
        )


class AlphaConditionTrainer(ConditionTrainer):
    def get_cond_model_input(self, x0_in) -> AlphaModelInput:
        x0_raw, x0 = x0_in
        alphas = (torch.rand(self.cfg.batch_size) * 5).tile(x0_raw.shape[1], 1).T.to(x0_raw.device)
        self.likelihood.set_alpha(alphas)
        cond = self.likelihood.get_condition(
            x0_raw.squeeze(-1),
            x0.squeeze(-1),
        ).reshape(-1, 1)
        alpha = alphas[:, 0].reshape(-1, 1)
        return AlphaModelInput(x0_in, cond, alpha)

    def get_uncond_model_input(self, x0_in) -> AlphaModelInput:
        return AlphaModelInput(x0_in, None, None)

    def get_model_output(
            self,
            forward_sample_output: ForwardSample,
            alpha_model_in: AlphaModelInput,
    ) -> torch.Tensor:
        return self.diffusion_model(
            x=forward_sample_output.xt,
            time=forward_sample_output.t,
            cond=alpha_model_in.cond,
            alpha=alpha_model_in.alpha,
        )


class TrajectoryConditionTrainer(ToyTrainer):
    def get_model_output(
            self,
            forward_sample_output: ForwardSample,
            alpha_model_in: AlphaModelInput,
    ) -> torch.Tensor:
        return self.diffusion_model(
            x=forward_sample_output.xt,
            time=forward_sample_output.t,
            cond=alpha_model_in.cond,
            alpha=alpha_model_in.alpha,
        )

    def get_cond_model_input(self, x0_in) -> AlphaModelInput:
        x0_raw, x0 = x0_in
        alpha = x0_raw.max(dim=1).values.squeeze()
        sorted_alpha, sorted_idx = alpha.sort()
        sorted_x0_raw = x0_raw[sorted_idx]
        sorted_x0 = x0[sorted_idx]
        cond_x = sorted_x0[:self.n_samples//2]
        x_raw = sorted_x0_raw[self.n_samples//2:]
        x = sorted_x0[self.n_samples//2:]
        x_in = x_raw, x
        cond_alpha = sorted_alpha[self.n_samples//2:].unsqueeze(1)
        return AlphaModelInput(x_in, cond_x, cond_alpha)

    def get_uncond_model_input(self, x0_in) -> AlphaModelInput:
        return AlphaModelInput(x0_in, None, None)


@hydra.main(version_base=None, config_path="conf", config_name="continuous_train_config")
def train(cfg):
    logger = logging.getLogger("main")
    logger.info(f"CONFIG\n{OmegaConf.to_yaml(cfg)}")

    logger.info(f"NUM THREADS: {torch.get_num_threads()}\n")
    if not cfg.no_wandb:
        wandb.init(
            project="toy-diffusion",
            config=OmegaConf.to_container(cfg),
            settings=wandb.Settings(_service_wait=300)
        )

    cfg.max_gradient = cfg.max_gradient if cfg.max_gradient > 0. else float('inf')

    d_model = torch.tensor(1)
    diffusion_model = hydra.utils.instantiate(
        cfg.diffusion,
        d_model=d_model,
        device=device
    )
    if isinstance(diffusion_model, TemporalUnet):
        trainer = ThresholdConditionTrainer(cfg=cfg, diffusion_model=diffusion_model)
    elif isinstance(diffusion_model, TemporalUnetAlpha):
        trainer = AlphaConditionTrainer(cfg=cfg, diffusion_model=diffusion_model)
    elif isinstance(diffusion_model, TemporalTransformerUnet):
        trainer = TrajectoryConditionTrainer(cfg=cfg, diffusion_model=diffusion_model)
    else:
        trainer = ThresholdConditionTrainer(cfg=cfg, diffusion_model=diffusion_model)

    trainer.train()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_train_config", node=TrainConfig)
    register_configs()
    train()
