#!/usr/bin/env python3

import os
import re
import matplotlib.pyplot as plt

import torch


def plot_data(model_name, suffix):
    torch_suffix = '{}.pt'.format(suffix)
    pattern = re.compile(r'\[(\d+(\.\d*)?)\]')
    directory = 'figs/{}'.format(model_name)
    mse_data = []
    alphas = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(torch_suffix):
            data = torch.load(file_path)
            mse_data = torch.stack([mse_data, data]) if mse_data else data
            alpha = pattern.search(filename).group(1)
            alphas.append(alpha)
    means, stds = mse_data
    plt.errorbar(alphas, means, stds, ecolor='red')
    save_dir = 'figs/{}'.format(model_name)
    plt.savefig('{}/{}.pdf'.format(save_dir, suffix))

def plot_mse_llk(model_name):
    suffix = 'llk_stats'
    plot_data(model_name, suffix)

def plot_is_estimates(model_name):
    suffix = 'is_estimates'
    plot_data(model_name, suffix)

def plot_is_estimates2(
        alphas,
        target_is,
        diffusion_is,
        trues,
        model_name,
        suffix
):
    save_dir = 'figs/{}'.format(model_name)
    plt.plot(alphas, target_is, label='IS Estimate Against Target')
    plt.plot(
        alphas,
        diffusion_is,
        label='IS Estimate Against Unconditional Diffusion',
        color='green'
    )
    plt.plot(alphas, trues, label='True Tail Probability', color='red')
    plt.xlabel('Alpha')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig('{}/{}.pdf'.format(save_dir, suffix))

def plot_is_vs_alpha(model_name):
    target_suffix = 'target_is_stats.pt'
    diffusion_suffix = 'diffusion_is_stats.pt'
    true_suffix = 'tail_prob.pt'
    pattern = re.compile(r'(\d+(\.\d*)?)')
    directory = 'figs/{}'.format(model_name)
    target_is = []
    diffusion_is = []
    trues = []
    target_alphas = []
    diffusion_alphas = []
    true_alphas = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(target_suffix):
            data = torch.load(file_path)
            target_is.append(data[0])
            alpha = pattern.search(filename).group(1)
            target_alphas.append(float(alpha))
        elif filename.endswith(diffusion_suffix):
            data = torch.load(file_path)
            diffusion_is.append(data[0])
            alpha = pattern.search(filename).group(1)
            diffusion_alphas.append(float(alpha))
        elif filename.endswith(true_suffix):
            data = torch.load(file_path)
            trues.append(data)
            alpha = pattern.search(filename).group(1)
            true_alphas.append(float(alpha))
    sorted_target_alphas, idx = torch.sort(torch.tensor(target_alphas))
    target_is = torch.tensor(target_is)[idx]
    sorted_diffusion_alphas, idx = torch.sort(torch.tensor(diffusion_alphas))
    diffusion_is = torch.tensor(diffusion_is)[idx]
    sorted_true_alphas, idx = torch.sort(torch.tensor(true_alphas))
    trues = torch.tensor(trues)[idx]
    plot_is_estimates2(sorted_true_alphas, target_is, diffusion_is, trues, model_name, 'is_vs_alpha')


if __name__ == '__main__':
    model_name = 'VPSDEVelocitySampler_TemporalUnetAlpha_BrownianMotionDiffExampleConfig_puncond_0.1_v651_v1999'
    # plot_mse_llk(model_name)
    # plot_is_estimates(model_name)
    plot_is_vs_alpha(model_name)
