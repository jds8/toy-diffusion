#!/usr/bin/env python3

import os
import re
import matplotlib.pyplot as plt
import numpy as np

import argparse
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def target_is_performance(alpha):
    return f'alpha={alpha}_target_is_performance.pt'

def diffusion_is_performance(alpha):
    return f'alpha={alpha}_diffusion_is_performance.pt'

def true_tail_prob(alpha):
    return f'alpha={alpha}_tail_prob.pt'

def effort_v_performance_plot_name(alpha):
    return f'alpha={alpha}_effort_v_performance'

def plot_data(model_name, suffix):
    torch_suffix = '{}.pt'.format(suffix)
    pattern = re.compile(r'\[(\d+(\.\d*)?)\]')
    directory = 'figs/{}'.format(model_name)
    mse_data = []
    alphas = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(torch_suffix):
            data = torch.load(file_path, map_location=device, weights_only=True)
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
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale('log')
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
        if target_suffix in filename:
            data = torch.load(file_path, map_location=device, weights_only=True)
            target_is.append(data[0])
            alpha = pattern.search(filename).group(1)
            target_alphas.append(float(alpha))
        elif diffusion_suffix in filename:
            data = torch.load(file_path, map_location=device, weights_only=True)
            diffusion_is.append(data[0])
            alpha = pattern.search(filename).group(1)
            diffusion_alphas.append(float(alpha))
        elif true_suffix in filename:
            data = torch.load(file_path, map_location=device, weights_only=True)
            trues.append(data)
            alpha = pattern.search(filename).group(1)
            true_alphas.append(float(alpha))
    _, idx = torch.sort(torch.tensor(target_alphas))
    target_is = torch.tensor(target_is)[idx]
    _, idx = torch.sort(torch.tensor(diffusion_alphas))
    diffusion_is = torch.tensor(diffusion_is)[idx]
    sorted_true_alphas, idx = torch.sort(torch.tensor(true_alphas))
    trues = torch.tensor(trues)[idx]
    plot_is_estimates2(sorted_true_alphas, target_is, diffusion_is, trues, model_name, 'is_vs_alpha')

def get_true_tail_prob(model, alpha):
    directory = 'figs/{}'.format(model)
    true_file = '{}/{}'.format(
        directory,
        true_tail_prob(alpha)
    )
    true = torch.load(true_file, weights_only=True)
    return true, directory

def process_performance_data(model_name):
    target_suffix = 'target_is_stats'
    diffusion_suffix = 'diffusion_is_stats'
    true_suffix = 'tail_prob.pt'
    pattern = re.compile(r'(\d+(\.\d*)?)')
    directory = 'figs/{}'.format(model_name)
    target_alphas = []
    diffusion_alphas = []
    target_alpha_map = {}
    diffusion_alpha_map = {}
    true_alpha_map = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if target_suffix in filename:
            data = torch.load(file_path, map_location=device, weights_only=True)
            alpha = float(pattern.search(filename).group(1))
            target_alphas.append(alpha)
            if alpha in target_alpha_map:
                target_alpha_map[alpha].append(data[0])
            else:
                target_alpha_map[alpha] = [data[0]]
        elif diffusion_suffix in filename:
            data = torch.load(file_path, map_location=device, weights_only=True)
            alpha = float(pattern.search(filename).group(1))
            diffusion_alphas.append(alpha)
            if alpha in diffusion_alpha_map:
                diffusion_alpha_map[alpha].append(data[0])
            else:
                diffusion_alpha_map[alpha] = [data[0]]
        elif true_suffix in filename:
            data = torch.load(file_path, map_location=device, weights_only=True)
            alpha = float(pattern.search(filename).group(1))
            true_alpha_map[alpha] = data
    for alpha in true_alpha_map.keys():
        try:
            true, _ = get_true_tail_prob(model_name, alpha)
            target = torch.stack(target_alpha_map[alpha])
            target_abs_errors = torch.abs(target - true)
            target_performance_data = torch.stack([
                target_abs_errors.mean(),
                target_abs_errors.quantile(0.05),
                target_abs_errors.quantile(0.95)
            ])
            diffusion = torch.stack(diffusion_alpha_map[alpha])
            diffusion_abs_errors = torch.abs(diffusion - true)
            diffusion_performance_data = torch.stack([
                diffusion_abs_errors.mean(),
                diffusion_abs_errors.quantile(0.05),
                diffusion_abs_errors.quantile(0.95)
            ])
            target_path = '{}/{}'.format(
                directory,
                target_is_performance(
                    alpha
                )
            )
            diffusion_path = '{}/{}'.format(
                directory,
                diffusion_is_performance(alpha)
            )
            torch.save(target_performance_data, target_path)
            torch.save(diffusion_performance_data, diffusion_path)
        except:
            pass

def plot_effort_v_performance(args, title):
    dims = get_dims(args)
    model_names = args.model_names
    models_by_dim = {dim: [model for model in model_names if 'dim_{}'.format(str(dim)) in model] for dim in dims}
    model_idxs_by_dim = {dim: get_model_idx(args, dim) for dim in dims}
    alphas = args.alphas
    xlabel = args.xlabel
    for alpha in alphas:
        for dim in dims:
            target_means = []
            target_upr = []
            target_lwr = []
            diffusion_means = []
            diffusion_upr = []
            diffusion_lwr = []
            true, _ = get_true_tail_prob(models_by_dim[dim][0])
            for model_name in models_by_dim[dim]:
                directory = 'figs/{}'.format(model_name)
                target_file = '{}/{}'.format(
                    directory,
                    target_is_performance(alpha)
                )
                mean_quantiles = torch.load(target_file, weights_only=True)
                target_means.append(mean_quantiles[0].cpu())
                target_lwr.append(mean_quantiles[1].cpu())
                target_upr.append(mean_quantiles[2].cpu())

                diffusion_file = '{}/{}'.format(
                    directory,
                    diffusion_is_performance(alpha)
                )
                mean_quantiles = torch.load(diffusion_file, weights_only=True)
                diffusion_means.append(mean_quantiles[0].cpu())
                diffusion_lwr.append(mean_quantiles[1].cpu())
                diffusion_upr.append(mean_quantiles[2].cpu())

            target_means = [t - 0.001 for t in target_means] if dim==32 else target_means
            # fig = plt.figure()
            # ax = fig.add_subplot(1, 1, 1)
            # ax.set_yscale('log')
            plt.ylim((1e-10, 1e-2))
            # plt.plot(model_idxs_by_dim[dim], target_means, color='darkblue', label='Against Target', marker='x')
            # plt.fill_between(model_idxs_by_dim[dim], target_lwr, target_upr, alpha=0.3, color='blue')
            plt.plot(model_idxs_by_dim[dim], target_means, label='dim={}'.format(dim), marker='x')
            plt.fill_between(model_idxs_by_dim[dim], target_lwr, target_upr, alpha=0.3)
            # plt.plot(model_idxs_by_dim[dim], diffusion_means, color='darkgreen', label='Against Diffusion', marker='x')
            # plt.fill_between(model_idxs_by_dim[dim], diffusion_lwr, diffusion_upr, alpha=0.3, color='green')
            plt.plot(model_idxs_by_dim[dim], true, color='red')
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(f'Probability Estimate (alpha={alpha})')
        plt.title(title)
        directory = 'figs/effort_v_performance'
        os.makedirs(directory, exist_ok=True)
        fig_file = '{}/{}.pdf'.format(directory, effort_v_performance_plot_name(alpha))
        plt.savefig(fig_file)
        plt.clf()

    os.system('tar czf figs/effort_v_performance.tar.gz figs/effort_v_performance')
    os.system('cp figs/effort_v_performance.tar.gz ~')


def make_effort_v_performance_gaussian(model_idxs, xlabel):
    model_names = [
        'VPSDEVelocitySampler_TemporalIDK_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.7_v{}_epoch{}00'.format(idx, idx)
        for idx in model_idxs
    ]
    alphas = [3., 4., 5.]
    for model_name in model_names:
        process_performance_data(model_name)
    plot_effort_v_performance(
        args,
        'Gaussian Performance vs Effort',
    )


def make_effort_v_performance_bm(model_idxs, xlabel):
    model_names = [
        'VPSDEVelocitySampler_TemporalUnetAlpha_' \
        'BrownianMotionDiffExampleConfig_puncond' \
        '_0.1_rare5.7_v{}_epoch{}00'.format(idx, idx)
        for idx in model_idxs
    ]
    # alphas = [3., 4., 5.]
    alphas = [3., 4.]
    for model_name in model_names:
        process_performance_data(model_name)
    plot_effort_v_performance(
        args,
        'Brownian Motion Performance vs Effort',
    )


def make_effort_v_performance(args):
    title = get_title(args)
    for model_name in args.model_names:
        process_performance_data(model_name)
    plot_effort_v_performance(
        args,
        title,
    )


def get_model_idx(args, dim):
    if args.model_idx:
        return args.model_idx
    idxs = []
    for model_name in args.model_names:
        research = re.search('.*dim_{}.*v([0-9]+)'.format(dim), model_name)
        if research is not None:
            idxs.append(research[1])
    return idxs


def get_dims(args):
    if args.dims:
        return args.dims
    dims = []
    for model_name in args.model_names:
        dims.append(re.search('.*dim_([0-9]+)_.*', model_name)[1])
    return list(set(dims))


def get_title(args):
    model_prefix = args.model_names[0]
    PVE = 'Performance vs. Effort'
    GAUSSIAN = 'Gaussian'
    BM = 'BrownianMotion'
    if GAUSSIAN in model_prefix:
        return ' '.join([GAUSSIAN, PVE])
    elif BM in model_prefix:
        return ' '.join([BM, PVE])
    else:
        raise NotImplementedError


if __name__ == '__main__':
    os.system('echo git commit: $(git rev-parse HEAD)')

    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('--model_names', type=str, nargs='+')
    parser.add_argument('--model_idx', type=int, nargs='+')
    parser.add_argument('--dims', type=int, nargs='+')
    parser.add_argument('--alphas', type=float, nargs='+')
    parser.add_argument('--xlabel', type=float, nargs='+')
    args = parser.parse_args()

    # plot_mse_llk(model_name)
    # plot_is_estimates(model_name)
    # plot_is_vs_alpha(model_name)

    make_effort_v_performance(args)
    # make_effort_v_performance_bm(args)
    # make_effort_v_performance_gaussian(args)
