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


if __name__ == '__main__':
    model_name = 'VPSDEVelocitySampler_TemporalUnetAlpha_BrownianMotionDiffExampleConfig_puncond_0.1_v651_v1999'
    plot_mse_llk(model_name)
    plot_is_estimates(model_name)
