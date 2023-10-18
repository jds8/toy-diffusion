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
from models.toy_temporal import TemporalTransformerUnet, TemporalClassifier, NewTemporalClassifier


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVED_MODEL_DIR = 'diffusion_models'

def suppresswarning():
    warnings.warn("user", UserWarning)


class ToyTrainClassifier:
    def __init__(
            self,
            cfg,
            sampler,
            classifier,
    ):
        self.cfg = cfg

        self.sde = SDE(self.cfg.sde_drift, self.cfg.sde_diffusion)
        self.sampler = sampler
        self.classifier = classifier.to(device)

        self.n_samples = torch.tensor([self.cfg.batch_size], device=device)
        self.end_time = torch.tensor(1., device=device)

        self.iterations_before_save = 100
        self.num_saves = 0

        self.initialize_optimizer()

    def initialize_optimizer(self):
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), self.cfg.classifier_lr)
        self.num_steps = 0

    def clip_gradients(self):
        nn.utils.clip_grad_norm_(self.classifier.parameters(), self.cfg.max_gradient)

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
        classifier_path = '{}_classifier_v{}'.format(get_model_path(self.cfg), self.num_saves)
        try:
            pathlib.Path(SAVED_MODEL_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(self.classifier.state_dict(), classifier_path)
            print('saved model')
        except Exception as e:
            print('could not save model because {}'.format(e))
        return classifier_path

    def train(self):
        while True:
            self.train_batch()
            if self.num_steps % self.iterations_before_save == 0:
                classifier_path = self._save_model()
                if not self.cfg.no_wandb:
                    self.log_artifact(classifier_path, 'classifier')
                    self.delete_model(classifier_path)

    def train_batch(self):
        trajs = integrate(self.sde, timesteps=self.cfg.sde_steps, end_time=self.end_time, n_samples=self.n_samples)
        x0 = trajs.W.diff(dim=1).unsqueeze(-1)
        classifier_loss = self.forward_process(x0)
        if torch.is_grad_enabled():
            self.classifier_optimizer.zero_grad()
            classifier_loss.backward()
            self.clip_gradients()
            self.classifier_optimizer.step()
            self.num_steps += 1
            try:
                if not self.cfg.no_wandb:
                    wandb.log({"classifier_loss": classifier_loss.detach()})
                    grads = []
                    for param in self.classifier.parameters():
                        grads.append(param.grad.view(-1))
                    grads = torch.cat(grads)
                    grad_norm = grads.norm()
                    wandb.log({"classifier_grad_norm": grad_norm})
            except Exception as e:
                print(e)

    def forward_process(self, x0):
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

        classifier_loss = self.get_classifier_loss(x0, xt, t)
        return classifier_loss

    def get_classifier_loss(self, x0, xt, t):
        true_class = self.get_class(x0)
        predicted_unnormalized_logits = self.classifier(x0, t)
        predicted_class = nn.Softmax()(predicted_unnormalized_logits).argmax(dim=-1)
        train_acc = (true_class == predicted_class).to(float).mean()
        print('training accuracy: {}'.format(train_acc))
        return F.cross_entropy(predicted_unnormalized_logits, true_class)

    def get_class(self, x0):
        final_state = x0.cumsum(dim=1)[:, -1, 0]
        above_one = torch.max(final_state.floor(), torch.tensor(0., device=device))
        return torch.min(above_one, torch.tensor(self.classifier.num_classes-1, device=device)).to(torch.long)
        # return torch.min(torch.abs(final_state).floor(), torch.tensor(self.classifier.num_classes-1, device=device)).to(torch.long)
        # return (final_state > 0).to(torch.long)


@hydra.main(version_base=None, config_path="conf", config_name="discrete_train_config")
def train(cfg):
    if not cfg.no_wandb:
        wandb.init(
            project="toy-diffusion",
            config=OmegaConf.to_container(cfg)
        )

    cfg.max_gradient = cfg.max_gradient if cfg.max_gradient > 0. else float('inf')

    d_model = torch.tensor(1)
    sampler = hydra.utils.instantiate(cfg.sampler)
    num_classes = 4
    classifier = NewTemporalClassifier(
        traj_length=1000,
        d_model=d_model,
        cond_dim=cfg.diffusion.cond_dim,
        num_classes=num_classes,
    )

    trainer = ToyTrainClassifier(cfg=cfg, sampler=sampler, classifier=classifier)

    trainer.train()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="base_train_config", node=TrainConfig)
    register_configs()
    train()
