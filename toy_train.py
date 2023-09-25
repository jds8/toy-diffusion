#!/usr/bin/env python3

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

from toy_plot import SDE, Trajectories, integrate
from toy_train_config import TrainConfig, get_model_path, get_classifier_path
from toy_configs import register_configs
from models.toy_temporal import TemporalTransformerUnet, TemporalClassifier


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVED_MODEL_DIR = 'diffusion_models'

def suppresswarning():
    warnings.warn("user", UserWarning)


class ToyTrainer:
    def __init__(
            self,
            cfg,
            sampler,
            diffusion_model,
            likelihood,
            classifier=None,
    ):
        self.cfg = cfg
        self.sde = SDE(self.cfg.sde_drift, self.cfg.sde_diffusion)
        self.sampler = sampler
        self.diffusion_model = nn.DataParallel(diffusion_model).to(device)
        self.likelihood = likelihood
        self.classifier = classifier.to(device) if classifier is not None else nn.Parameter(torch.rand(3, 4))
        try:
            classifier_path = get_classifier_path(self.cfg)
            self.classifier.load_state_dict(torch.load('{}'.format(classifier_path)))
        except:
            print('Not loading classifier state_dict')
            pass
        self.loss_fn = self.get_loss_fn()
        self.n_samples = torch.tensor([self.cfg.batch_size], device=device)
        self.end_time = torch.tensor(1., device=device)

        self.iterations_before_save = 100
        self.num_saves = 0

        self.initialize_optimizer()

    def initialize_optimizer(self):
        self.optimizer = torch.optim.Adam(self.diffusion_model.module.parameters(), self.cfg.lr)
        try:
            self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), self.cfg.classifier_lr)
        except:
            self.classifier_optimizer = torch.optim.Adam([self.classifier], self.cfg.classifier_lr)
        self.num_steps = 0

    def clip_gradients(self):
        nn.utils.clip_grad_norm_(self.diffusion_model.module.parameters(), self.cfg.max_gradient)
        try:
            nn.utils.clip_grad_norm_(self.classifier.parameters(), self.cfg.max_gradient)
        except:
            nn.utils.clip_grad_norm_([self.classifier], self.cfg.max_gradient)

    def get_loss_fn(self):
        return {
            'l1': torch.nn.L1Loss(),
            'l2': torch.nn.MSELoss()
        }[self.cfg.loss_fn]

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
        classifier_path = '{}_classifier_v{}'.format(get_model_path(self.cfg), self.num_saves)
        try:
            pathlib.Path(SAVED_MODEL_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(self.diffusion_model.module.state_dict(), saved_model_path)
            if isinstance(self.classifier, nn.Parameter):
                torch.save(self.classifier, classifier_path)
            else:
                torch.save(self.classifier.state_dict(), classifier_path)
            print('saved model')
        except Exception as e:
            print('could not save model because {}'.format(e))
        return saved_model_path, classifier_path

    def train(self):
        while True:
            self.train_batch()
            if self.num_steps % self.iterations_before_save == 0:
                saved_model_path, classifier_path = self._save_model()
                if not self.cfg.no_wandb:
                    self.log_artifact(saved_model_path, 'diffusion_model')
                    self.log_artifact(classifier_path, 'classifier')
                    self.delete_model(saved_model_path)
                    self.delete_model(classifier_path)

    def viz_trajs(self, traj, end_time, idx, clf=True):
        import matplotlib.pyplot as plt
        full_state_pred = traj.detach().squeeze(0).cpu().numpy()

        plt.plot(torch.linspace(0, end_time, full_state_pred.shape[0]), full_state_pred, color='green')

        if idx % 100 == 0:
            plt.savefig('figs/train_{}.pdf'.format(idx))

        if clf:
            plt.clf()

    def train_batch(self):
        trajs = integrate(self.sde, timesteps=self.cfg.sde_steps, end_time=self.end_time, n_samples=self.n_samples)
        x0 = trajs.W.diff(dim=1).unsqueeze(-1)
        l2_loss, classifier_loss = self.forward_process(x0)
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
            self.classifier_optimizer.zero_grad()
            l2_loss.backward()
            classifier_loss.backward()
            self.clip_gradients()
            self.optimizer.step()
            self.classifier_optimizer.step()
            self.num_steps += 1
            try:
                if not self.cfg.no_wandb:
                    wandb.log({"train_loss": l2_loss.detach()})
                    wandb.log({"classifier_loss": classifier_loss.detach()})
                    grads = []
                    for param in self.diffusion_model.parameters():
                        grads.append(param.grad.view(-1))
                    grads = torch.cat(grads)
                    grad_norm = grads.norm()
                    wandb.log({"train_grad_norm": grad_norm})
                    grads = []
                    try:
                        for param in self.classifier.parameters():
                            grads.append(param.grad.view(-1))
                    except:
                        grads.append(self.classifier.grad.view(-1))
                    grads = torch.cat(grads)
                    grad_norm = grads.norm()
                    wandb.log({"classifier_grad_norm": grad_norm})
            except Exception as e:
                print(e)

    def forward_process(self, x0):
        cond = self.likelihood.get_condition(x0) if torch.rand(1) > self.cfg.p_uncond else torch.tensor(-1.)
        cond = cond.reshape(-1, 1)
        t = dist.Categorical(
            torch.ones(
                self.sampler.diffusion_timesteps,
                device=x0.device
            )
        ).sample([
            x0.shape[0]
        ])
        eps = torch.randn_like(x0)
        xt, to_predict = self.sampler.forward_sample(x_start=x0, t=t, noise=eps)

        model_output = self.diffusion_model(xt, t, cond)

        loss = self.loss_fn(model_output, to_predict)

        try:
            classifier_loss = self.get_classifier_loss(x0, xt, t)
        except Exception as e:
            print('Caught exception when getting classifier loss: {}'.format(e))
            classifier_loss = torch.zeros_like(loss, requires_grad=True)

        return loss, classifier_loss

    def get_classifier_loss(self, x0, xt, t):
        true_class = self.get_class(x0)
        predicted_unnormalized_logits = self.classifier(x0, t-t)
        predicted_class = nn.Softmax()(predicted_unnormalized_logits).argmax(dim=-1)
        training_accuracy = (true_class == predicted_class).to(float).mean()
        print('training accuracy: {}'.format(training_accuracy))
        if training_accuracy > 0.92:
            import pdb; pdb.set_trace()
        return F.cross_entropy(predicted_unnormalized_logits, true_class)

    def get_class(self, x0):
        final_state = x0.cumsum(dim=1)[:, -1, 0]
        # return torch.min(torch.abs(final_state).floor(), torch.tensor(self.classifier.num_classes, device=device)).to(torch.long)
        return (final_state > 0).to(torch.long)


@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def train(cfg):
    if not cfg.no_wandb:
        wandb.init(
            project="toy-diffusion",
            config=OmegaConf.to_container(cfg)
        )

    cfg.max_gradient = cfg.max_gradient if cfg.max_gradient > 0. else float('inf')

    d_model = torch.tensor(1)
    sampler = hydra.utils.instantiate(cfg.sampler)
    diffusion_model = hydra.utils.instantiate(cfg.diffusion, d_model=d_model, device=device)
    likelihood = hydra.utils.instantiate(cfg.likelihood)
    # classifier = None
    num_classes = 4
    classifier = TemporalClassifier(
        traj_length=1000,
        d_model=d_model,
        cond_dim=cfg.diffusion.cond_dim,
        num_classes=num_classes,
    )

    trainer = ToyTrainer(cfg=cfg, sampler=sampler, diffusion_model=diffusion_model, likelihood=likelihood, classifier=classifier)

    trainer.train()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="base_train_config", node=TrainConfig)
    register_configs()
    train()
