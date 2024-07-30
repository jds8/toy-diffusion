#!/usr/bin/env python3

import logging
import warnings
import wandb
import os
import pathlib

import hydra
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from toy_plot import SDE, Trajectories, integrate, score_function_heat_map, create_gif
from toy_train_config import TrainConfig, get_model_path, ExampleConfig, \
    GaussianExampleConfig, BrownianMotionExampleConfig, BrownianMotionDiffExampleConfig, \
    UniformExampleConfig
from toy_configs import register_configs
from toy_likelihoods import traj_dist, Likelihood
from models.toy_temporal import TemporalTransformerUnet, TemporalUnet, TemporalNNet, DiffusionModel
from models.toy_sampler import ForwardSample, AbstractSampler, AbstractContinuousSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVED_MODEL_DIR = 'diffusion_models'

def suppresswarning():
    warnings.warn("user", UserWarning)


class ToyTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

        d_model = torch.tensor(1)
        self.sampler = hydra.utils.instantiate(cfg.sampler)
        diffusion_model = hydra.utils.instantiate(
            cfg.diffusion,
            d_model=d_model,
            device=device
        )

        self.likelihood = hydra.utils.instantiate(cfg.likelihood)
        self.example = OmegaConf.to_object(cfg.example)

        self.diffusion_model = nn.DataParallel(diffusion_model).to(device)
        self.loss_fn = self.get_loss_fn()
        self.n_samples = torch.tensor([self.cfg.batch_size], device=device)
        self.end_time = torch.tensor(1., device=device)

        self.num_saves = 0

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
        saved_model_path = '{}_v{}'.format(get_model_path(self.cfg), self.num_saves)
        try:
            pathlib.Path(SAVED_MODEL_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(self.diffusion_model.module.state_dict(), saved_model_path)
            print('saved model {}'.format(self.num_saves))
        except Exception as e:
            print('could not save model because {}'.format(e))
        return saved_model_path

    def train(self):
        while True:
            self.train_batch()
            if self.num_steps % self.cfg.iterations_before_save == 0:
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


    def viz_trajs(self, traj, end_time, idx, clf=True):
        import matplotlib.pyplot as plt
        full_state_pred = traj.detach().squeeze(0).cpu().numpy()

        plt.plot(torch.linspace(0, end_time, full_state_pred.shape[0]), full_state_pred, color='green')

        if idx % 100 == 0:
            plt.savefig('figs/train_{}.pdf'.format(idx))

        if clf:
            plt.clf()

    def train_batch(self):
        raise NotImplementedError

    def get_x0(self):
        if isinstance(self.example, BrownianMotionExampleConfig):
            sde = SDE(
                self.cfg.example.sde_drift,
                self.cfg.example.sde_diffusion
            )
            trajs = integrate(
                sde,
                timesteps=self.cfg.example.sde_steps,
                end_time=self.end_time,
                n_samples=self.n_samples
            )
            x0 = trajs.W.unsqueeze(-1)
            if type(self.example) == BrownianMotionDiffExampleConfig:
                x0 = x0.diff(dim=1)
        elif isinstance(self.example, GaussianExampleConfig):
            x0 = torch.randn(
                self.cfg.batch_size, 1, 1, device=device
            )
        elif isinstance(self.example, UniformExampleConfig):
            x0_raw = torch.rand(
                self.cfg.batch_size, 1, 1, device=device
            )
            x0 = torch.logit(x0_raw)  # E[logit(X)] = 0 if X is uniform(0, 1)
            x0 /= (torch.pi / torch.tensor(3.).sqrt())  # Var[logit(X)] = pi^2/3 if X is uniform
        else:
            raise NotImplementedError
        return x0


class ConditionTrainer(ToyTrainer):
    def forward_process(self, x0):
        cond = self.likelihood.get_condition(x0).abs() if torch.rand(1) > self.cfg.p_uncond else torch.tensor(-1.)
        cond = cond.reshape(-1, 1)

        extras = {}
        if isinstance(self.example, GaussianExampleConfig):
            extras['mu'] = self.cfg.example.mu
            extras['sigma'] = self.cfg.example.sigma

        # # TODO: Remove
        # ts = torch.linspace(self.sampler.t_eps, 1, 1024)
        # mean, lmc, sigma = self.sampler.marginal_prob(
        #     torch.zeros_like(x0),
        #     ts,
        # )

        forward_sample_output = self.sampler.forward_sample(x_start=x0, extras=extras)

        model_output = self.diffusion_model(
            x=forward_sample_output.xt,
            time=forward_sample_output.t,
            cond=cond,
        )

        loss = self.loss_fn(model_output, forward_sample_output)

        # # TODO: The following computes the loss against the marginal score function
        # true_score=self.analytical_gaussian_score(
        #     forward_sample_output.t,
        #     forward_sample_output.xt
        # )
        # sf_estimator = self.sampler.get_sf_estimator(
        #     model_output,
        #     forward_sample_output.xt,
        #     forward_sample_output.t
        # )
        # loss = torch.nn.MSELoss()(sf_estimator, true_score)

        return loss

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

    def train_batch(self):
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


class TrajectoryConditionTrainer(ToyTrainer):
    def forward_transformer_process(self, x0, x_cond):
        cond_traj = x_cond if torch.rand(1) > self.cfg.p_uncond else None
        cond = self.traj_dist(x0, cond_traj).reshape(-1, 1) if cond_traj is not None else None
        xt, t, noise, to_predict = self.sampler.forward_sample(x_start=x0)

        model_output = self.diffusion_model(xt, t, cond_traj, cond)

        loss = self.loss_fn(model_output, to_predict)

        return loss

    def train_batch(self):
        xs = self.get_x0()
        x0 = xs[:self.n_samples//2]
        x_cond = xs[self.n_samples//2:]
        loss = self.forward_transformer_process(x0, x_cond)
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
            except Exception as e:
                print(e)


@hydra.main(version_base=None, config_path="conf", config_name="continuous_train_config")
def train(cfg):
    logger = logging.getLogger("main")
    logger.info(f"CONFIG\n{OmegaConf.to_yaml(cfg)}")

    if not cfg.no_wandb:
        wandb.init(
            project="toy-diffusion",
            config=OmegaConf.to_container(cfg)
        )

    cfg.max_gradient = cfg.max_gradient if cfg.max_gradient > 0. else float('inf')

    d_model = torch.tensor(1)
    diffusion_model = hydra.utils.instantiate(cfg.diffusion, d_model=d_model, device=device)
    if isinstance(diffusion_model, TemporalUnet):
        trainer = ConditionTrainer(cfg=cfg)
    elif isinstance(diffusion_model, TemporalTransformerUnet):
        trainer = TrajectoryConditionTrainer(cfg=cfg)
    else:
        trainer = ConditionTrainer(cfg=cfg)

    trainer.train()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_train_config", node=TrainConfig)
    register_configs()
    train()
