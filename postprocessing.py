#!/usr/bin/env python3

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import List

import argparse
import torch
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from toy_train_config import GaussianExampleConfig, MultivariateGaussianExampleConfig, BrownianMotionDiffExampleConfig, \
                             PostProcessingConfig, get_target
from toy_configs import register_configs
from toy_sample import ContinuousEvaluator
from toy_is import iterative_importance_estimate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def performance_v_samples(alpha):
    return f'alpha={alpha}_performance_v_samples'

def pct_all_saps_not_in_region(alpha):
    return f'alpha={alpha}_pct_all_saps_not_in_region.pt'

def target_is_performance(alpha):
    return f'alpha={alpha}_target_is_performance.pt'

def diffusion_is_performance(alpha):
    return f'alpha={alpha}_diffusion_is_performance.pt'

def true_tail_prob(alpha):
    return f'alpha={alpha}_tail_prob.pt'

def effort_v_performance_plot_name(alpha):
    return f'alpha={alpha}_effort_v_performance'

def pct_not_in_region_plot_name(alpha):
    return f'alpha={alpha}_pct_outside_region'

def plot_data(figs_dir, model_name, suffix):
    torch_suffix = '{}.pt'.format(suffix)
    pattern = re.compile(r'\[(\d+(\.\d*)?)\]')
    directory = '{}/{}'.format(figs_dir, model_name)
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
    save_dir = '{}/{}'.format(figs_dir, model_name)
    plt.savefig('{}/{}.pdf'.format(save_dir, suffix))

def plot_mse_llk(figs_dir, model_name):
    suffix = 'llk_stats'
    plot_data(figs_dir, model_name, suffix)

def plot_is_estimates(figs_dir, model_name):
    suffix = 'is_estimates'
    plot_data(figs_dir, model_name, suffix)

def plot_is_estimates2(
        alphas,
        target_is,
        diffusion_is,
        trues,
        figs_dir,
        model_name,
        suffix
):
    save_dir = '{}/{}'.format(figs_dir, model_name)
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

def plot_is_vs_alpha(figs_dir, model_name):
    target_suffix = 'target_is_stats.pt'
    diffusion_suffix = 'diffusion_is_stats.pt'
    true_suffix = 'tail_prob.pt'
    pattern = re.compile(r'(\d+(\.\d*)?)')
    directory = '{}/{}'.format(figs_dir, model_name)
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
    plot_is_estimates2(
        sorted_true_alphas,
        target_is,
        diffusion_is,
        trues,
        figs_dir,
        model_name,
        'is_vs_alpha'
    )

def get_true_tail_prob(directory, alpha):
    true_file = '{}/{}'.format(
        directory,
        true_tail_prob(alpha)
    )
    true = torch.load(true_file, weights_only=True)
    return true, directory

def process_performance_data(figs_dir, model_name, args):
    target_suffix = 'target_is_stats'
    diffusion_suffix = 'diffusion_is_stats'
    true_suffix = 'tail_prob.pt'
    pattern = re.compile(r'(\d+(\.\d*)?)')
    directory = '{}/{}'.format(figs_dir, model_name)
    target_alpha_map = {}
    diffusion_alpha_map = {}
    true_alpha_map = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if target_suffix in filename:
            data = torch.load(file_path, map_location=device, weights_only=True)
            alpha = float(pattern.search(filename).group(1))
            if alpha in target_alpha_map:
                target_alpha_map[alpha].append(data[0])
            else:
                target_alpha_map[alpha] = [data[0]]
        elif diffusion_suffix in filename:
            data = torch.load(file_path, map_location=device, weights_only=True)
            alpha = float(pattern.search(filename).group(1))
            if alpha in diffusion_alpha_map:
                diffusion_alpha_map[alpha].append(data[0])
            else:
                diffusion_alpha_map[alpha] = [data[0]]
        elif true_suffix in filename:
            data = torch.load(file_path, map_location=device, weights_only=True)
            alpha = float(pattern.search(filename).group(1))
            true_alpha_map[alpha] = data
    for alpha in args.alphas:
        true, _ = get_true_tail_prob(directory, alpha)
        target = torch.stack(target_alpha_map[alpha])
        target_rel_errors = torch.abs(target - true) / true
        target_performance_data = torch.stack([
            target_rel_errors.quantile(0.5),
            target_rel_errors.quantile(0.05),
            target_rel_errors.quantile(0.95)
        ])
        target_path = '{}/{}'.format(
            directory,
            target_is_performance(
                alpha
            )
        )
        torch.save(target_performance_data, target_path)
        if args.use_diffusion:
            diffusion = torch.stack(diffusion_alpha_map[alpha])
            diffusion_rel_errors = torch.abs(diffusion - true) / true
            diffusion_performance_data = torch.stack([
                diffusion_rel_errors.quantile(0.5),
                diffusion_rel_errors.quantile(0.05),
                diffusion_rel_errors.quantile(0.95)
            ])
            diffusion_path = '{}/{}'.format(
                directory,
                diffusion_is_performance(alpha)
            )
            torch.save(diffusion_performance_data, diffusion_path)

def old_plot_effort_v_performance(args, title, xlabel):
    dims = get_dims(args)
    model_names = args.model_names
    models_by_dim = {dim: [model for model in model_names if 'dim_{}'.format(str(dim)) in model] for dim in dims}
    model_idxs_by_dim = {dim: get_model_idx(args, dim) for dim in dims}
    alphas = args.alphas
    for alpha in alphas:
        for dim in dims:
            target_means = []
            target_upr = []
            target_lwr = []
            # diffusion_means = []
            # diffusion_upr = []
            # diffusion_lwr = []
            directory = f'{args.figs_dir}/{models_by_dim[dim][0]}'
            true, _ = get_true_tail_prob(directory, alpha)
            for model_name in models_by_dim[dim]:
                directory = '{}/{}'.format(args.figs_dir, model_name)
                target_file = '{}/{}'.format(
                    directory,
                    target_is_performance(alpha)
                )
                mean_quantiles = torch.load(target_file, weights_only=True)
                target_means.append(mean_quantiles[0].cpu())
                target_lwr.append(mean_quantiles[1].cpu())
                target_upr.append(mean_quantiles[2].cpu())

                # if args.use_diffusion:
                #     diffusion_file = '{}/{}'.format(
                #         directory,
                #         diffusion_is_performance(alpha)
                #     )
                #     mean_quantiles = torch.load(diffusion_file, weights_only=True)
                #     diffusion_means.append(mean_quantiles[0].cpu())
                #     diffusion_lwr.append(mean_quantiles[1].cpu())
                #     diffusion_upr.append(mean_quantiles[2].cpu())
            models_as_num = [int(x) for x in model_idxs_by_dim[dim]]
            plt.plot(
                models_as_num,
                target_means,
                label='params={}'.format(dim_to_param(dim, model_name)),
                marker='x'
            )
            plt.fill_between(models_as_num, target_lwr, target_upr, alpha=0.3)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel('Relative Error of Prob. Est.')
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.title(title+f' (alpha={alpha})')
        directory = '{}/effort_v_performance'.format(args.figs_dir)
        os.makedirs(directory, exist_ok=True)
        fig_file = '{}/{}.pdf'.format(directory, effort_v_performance_plot_name(alpha))
        plt.savefig(fig_file)
        plt.clf()
    return directory

def plot_effort_v_performance(args, title, xlabel):
    dims = get_dims(args)
    model_names = args.model_names
    run_type = 'Gaussian' if 'Gaussian' in model_names[0] else 'BrownianMotionDiff'
    models_by_dim = {dim: [model for model in model_names if 'dim_{}'.format(str(dim)) in model] for dim in dims}
    model_idxs_by_dim = {dim: get_model_idx(args, dim) for dim in dims}
    empirical_error = torch.load('empirical_errors.pt', weights_only=True)
    alphas = args.alphas
    for alpha in alphas:
        f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')
        for dim in dims:
            target_means = []
            target_upr = []
            target_lwr = []
            # diffusion_means = []
            # diffusion_upr = []
            # diffusion_lwr = []
            directory = f'{args.figs_dir}/{models_by_dim[dim][0]}'
            true, _ = get_true_tail_prob(directory, alpha)
            for model_name in models_by_dim[dim]:
                directory = '{}/{}'.format(args.figs_dir, model_name)
                target_file = '{}/{}'.format(
                    directory,
                    target_is_performance(alpha)
                )
                mean_quantiles = torch.load(target_file, weights_only=True)
                target_means.append(mean_quantiles[0].cpu())
                target_lwr.append(mean_quantiles[1].cpu())
                target_upr.append(mean_quantiles[2].cpu())

                # if args.use_diffusion:
                #     diffusion_file = '{}/{}'.format(
                #         directory,
                #         diffusion_is_performance(alpha)
                #     )
                #     mean_quantiles = torch.load(diffusion_file, weights_only=True)
                #     diffusion_means.append(mean_quantiles[0].cpu())
                #     diffusion_lwr.append(mean_quantiles[1].cpu())
                #     diffusion_upr.append(mean_quantiles[2].cpu())
            models_as_num = [int(x) for x in model_idxs_by_dim[dim]]
            ax.plot(
                models_as_num,
                target_means,
                label='params={}'.format(dim_to_param(dim, model_name)),
                marker='x'
            )
            ax.fill_between(models_as_num, target_lwr, target_upr, alpha=0.3)

        sap_error_pairs = [(sap, error) for sap, error in empirical_error[run_type][alpha].items()]
        model_idxs = [10, 100, 1000]
        for idx, (saps, error) in enumerate(sap_error_pairs):
            model_idx = models_as_num[-1] * model_idxs[idx]
            ax2.plot([model_idx, model_idx], [error[0], error[2]], alpha=0.3, color='red', linewidth=2.5)
            ax2.scatter(model_idx, error[1], marker='o', label=f'Empirical (N={saps})', color='red')

        ax.legend()
        ax2.legend()
        ax.set_xlabel(xlabel)
        f.supylabel('Relative Error of Prob. Est.')
        f.suptitle(title+f' (alpha={alpha})')

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax2.set_xscale('log')

        # Format as whole numbers
        ax.tick_params(axis='y', which='both', labelleft=True)

        # ax2.set_xlim(models_as_num[-1]*5, models_as_num[-1]*model_idxs[-1]*10)
        ax2.set_xticks([models_as_num[-1] * model_idx for model_idx in model_idxs])
        ax2.set_xticklabels(["N={}".format(pair[0]) for pair in sap_error_pairs])

        # hide the spines between ax and ax2
        ax.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        # plot break lines
        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((1-d, 1+d), (-d, +d), **kwargs)
        ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
        ax2.plot((-d, +d), (-d, +d), **kwargs)

        directory = '{}/effort_v_performance'.format(args.figs_dir)
        os.makedirs(directory, exist_ok=True)
        error_bar_file = '{}/{}.pdf'.format(
            directory,
            performance_v_samples(alpha)
        )
        plt.savefig(error_bar_file)
        plt.clf()
    return directory

def make_effort_v_performance_gaussian(model_idxs, xlabel, args):
    model_names = [
        'VPSDEVelocitySampler_TemporalIDK_GaussianExampleConfig_1.0_2.0_puncond_0.1_rare5.7_v{}_epoch{}00'.format(idx, idx)
        for idx in model_idxs
    ]
    alphas = [3., 4., 5.]
    for model_name in model_names:
        process_performance_data(args.figs_dir, model_name, args)
    plot_effort_v_performance(
        args,
        'Gaussian Performance vs Effort',
        xlabel
    )


def make_effort_v_performance_bm(model_idxs, xlabel, args):
    model_names = [
        'VPSDEVelocitySampler_TemporalUnetAlpha_' \
        'BrownianMotionDiffExampleConfig_puncond' \
        '_0.1_rare5.7_v{}_epoch{}00'.format(idx, idx)
        for idx in model_idxs
    ]
    # alphas = [3., 4., 5.]
    alphas = [3., 4.]
    for model_name in model_names:
        process_performance_data(args.figs_dir, model_name, args)
    plot_effort_v_performance(
        args,
        'Brownian Motion Performance vs Effort',
        xlabel
    )


def make_effort_v_performance(args):
    for model_name in args.model_names:
        process_performance_data(args.figs_dir, model_name, args)
        process_pct_saps_data(args.figs_dir, model_name)
    title = get_performance_v_effort_title(args)
    save_dir = plot_effort_v_performance(
        args,
        title,
        xlabel='Training Samples'
    )
    title = get_pct_not_in_region_title(args)
    try:
        save_dir = plot_pct_not_in_region(
            args,
            title,
            xlabel='Training Samples',
        )
    except Exception as e:
        print('Error: {}'.format(e))
        pass
    model_csv = ','.join(args.model_names)
    torch.save(model_csv, f'{save_dir}/models.csv')
    os.system('tar czf {}/effort_v_performance.tar.gz {}/effort_v_performance'.format(
        args.figs_dir,
        args.figs_dir
    ))
    os.system('cp {}/effort_v_performance.tar.gz ~'.format(args.figs_dir))


def get_model_idx(args, dim):
    if args.model_idx:
        return args.model_idx
    idxs = []
    for model_name in args.model_names:
        research = re.search('.*dim_{}.*v([0-9]+)'.format(dim), model_name)
        if research is not None:
            idxs.append(research[1])
    return idxs


def get_dims(args) -> List[int]:
    if args.dims:
        return [int(x) for x in args.dims]
    dims = []
    for model_name in args.model_names:
        dim = int(re.search('.*dim_([0-9]+)_.*', model_name)[1])
        dims.append(dim)
    return list(set(dims))


def get_title(args, title):
    model_prefix = args.model_names[0]
    GAUSSIAN = 'Gaussian'
    BM = 'BrownianMotion'
    if GAUSSIAN in model_prefix:
        return ' '.join([GAUSSIAN, title])
    elif BM in model_prefix:
        return ' '.join([BM, title])
    else:
        raise NotImplementedError


def get_performance_v_effort_title(args):
    PVE = 'Performance vs. Effort'
    return get_title(args, PVE)


def get_pct_not_in_region_title(args):
    POR = 'Pct. of Samples not in Tail'
    return get_title(args, POR)


def process_pct_saps_data(figs_dir, model_name):
    pct_saps_infix = 'pct_saps_not_in_region'
    pattern = re.compile(r'(\d+(\.\d*)?)')
    directory = '{}/{}'.format(figs_dir, model_name)
    pct_saps_map = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if pct_saps_infix in filename:
            data = torch.load(file_path, map_location=device, weights_only=True)
            alpha = float(pattern.search(filename).group(1))
            if alpha in pct_saps_map:
                pct_saps_map[alpha].append(data[0])
            else:
                pct_saps_map[alpha] = [data[0]]
    for alpha in pct_saps_map.keys():
        pct_saps = torch.stack(pct_saps_map[alpha])
        pct_all_saps_data = torch.stack([
            pct_saps.mean(),
            pct_saps.quantile(0.05),
            pct_saps.quantile(0.95),
        ])
        pct_saps_path = '{}/{}'.format(
            directory,
            pct_all_saps_not_in_region(alpha)
        )
        torch.save(pct_all_saps_data, pct_saps_path)

def dim_to_param(dim: int, model_name: str):
    gaussian_dict = {
        32: 3996833,
        40: 6238601,
        64: 15946049
    }
    bm_dict = {
        32: 4147809,
        40: 6474361,
        64: 16549057
    }
    if 'gaussian' in model_name.lower():
        return gaussian_dict[dim]
    return bm_dict[dim]

def plot_pct_not_in_region(args, title, xlabel):
    dims = get_dims(args)
    model_names = args.model_names
    models_by_dim = {dim: [model for model in model_names if 'dim_{}'.format(str(dim)) in model] for dim in dims}
    model_idxs_by_dim = {dim: get_model_idx(args, dim) for dim in dims}
    alphas = args.alphas
    for alpha in alphas:
        for dim in dims:
            pct_means = []
            pct_upr = []
            pct_lwr = []
            for model_name in models_by_dim[dim]:
                directory = '{}/{}'.format(args.figs_dir, model_name)
                target_file = '{}/{}'.format(
                    directory,
                    pct_all_saps_not_in_region(alpha)
                )
                mean_quantiles = torch.load(target_file, weights_only=True)
                pct_means.append(mean_quantiles[0].cpu())
                pct_lwr.append(mean_quantiles[1].cpu())
                pct_upr.append(mean_quantiles[2].cpu())
            models_as_num = [int(x) for x in model_idxs_by_dim[dim]]
            plt.plot(
                models_as_num,
                pct_means,
                label='params={}'.format(dim_to_param(dim, model_name), model_name),
                marker='x'
            )
            plt.fill_between(models_as_num, pct_lwr, pct_upr, alpha=0.3)
        plt.legend()
        plt.xlabel(xlabel)
        ax = plt.gca()
        ax.set_xscale('log')
        cfg_str = torch.load(f'{directory}/alpha={alpha}_config.txt', weights_only=True)
        pattern = re.compile('num_samples: ([0-9]+)')
        result = re.search(pattern, cfg_str)
        num_saps = int(result[1]) if result else 0
        plt.ylabel(f'Percentage out of {num_saps} Samples')
        plt.title(title+f' (alpha={alpha})')
        directory = '{}/effort_v_performance'.format(args.figs_dir)
        os.makedirs(directory, exist_ok=True)
        fig_file = '{}/{}.pdf'.format(directory, pct_not_in_region_plot_name(alpha))
        plt.savefig(fig_file)
        plt.clf()
    return directory

def get_round(filename) -> int:
    round_ = 'round_'
    output = filename[filename.find(round_)+len(round_):]
    suffix_idx = output.find('.')  # sometimes files end in '.pt'
    if suffix_idx > 0:
        output = output[:suffix_idx]  # if they do, then truncate the '.pt' part
    int_output = int(output)
    return int_output

def get_model_size(model) -> int:
    return int(model[model.find('_v')+2:])

def get_saps_raw(saps, cfg) -> torch.Tensor:
    omega_cfg = OmegaConf.to_object(cfg)
    if isinstance(omega_cfg.example, GaussianExampleConfig):
        return saps * cfg.example.sigma + cfg.example.mu
    elif isinstance(omega_cfg.example, MultivariateGaussianExampleConfig):
        torch_sigma = torch.tensor(omega_cfg.example.sigma)
        L = torch_sigma.cholesky()
        return torch.matmul(L, saps) + torch.tensor(omega_cfg.example.mu)
    elif isinstance(omega_cfg.example, BrownianMotionDiffExampleConfig):
        dt = torch.tensor(1. / saps.shape[1])
        scaled_saps = saps * dt.sqrt()
        saps_raw = torch.cat([
            torch.zeros(saps.shape[0], 1, 1, device=device),
            scaled_saps.cumsum(dim=1)
        ], dim=1)
        return saps_raw
    else:
        raise NotImplementedError

def make_performance_v_samples(cfg):
    """
    1) Load saps and log_qrobs data
    2) invoke target.log_prob(saps_raw) to get true_log_probs
    3) calculate importance estimate using iterative_importance_estimate from toy_is
    4) compute relative error
    5) add Naive MC error bars where applicable
    The plot shows IS error (with error bars) versus sample size.
    I also want to plot empirical error (with error bars) for comparison.
    The IS plots will be for a particular (the best?) model and the empirical plots
    will be for sample sizes 10x and 100x larger than the smallest IS sample size.
    """
    assert len(cfg.alphas) > 0 and len(cfg.samples) > 0
    directory = f'{cfg.figs_dir}/{cfg.model_name}'
    for alpha in cfg.alphas:
        # collect all sample data for each round
        sample_data = [None] * cfg.total_rounds
        sample_log_qrobs = [None] * cfg.total_rounds
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if re.search(f'alpha={alpha}_saps_[0-9]+_[0-9]+_round', file_path) is not None:
                data = torch.load(file_path, map_location=device, weights_only=True)
                rnd = get_round(file_path)
                if sample_data[rnd] is not None:
                    sample_data[rnd] = torch.cat([
                        sample_data[rnd],
                        data#.reshape(-1, 1, 1)
                    ])#.reshape(-1, 1, 1)
                else:
                    sample_data[rnd] = data#.reshape(-1, 1, 1)
            if re.search(f'alpha={alpha}_log_qrobs_[0-9]+_[0-9]+_round', file_path) is not None:
                data = torch.load(file_path, map_location=device, weights_only=True)
                rnd = get_round(file_path)
                if sample_log_qrobs[rnd] is not None:
                    sample_log_qrobs[rnd] = torch.cat([
                        sample_log_qrobs[rnd],
                        data#.reshape(-1, 1, 1)
                    ])#.reshape(-1, 1, 1)
                else:
                    try:
                        sample_log_qrobs[rnd] = data#.reshape(-1, 1, 1)
                    except Exception as e:
                        import pdb; pdb.set_trace()

        std = ContinuousEvaluator(cfg=cfg)
        target = get_target(std)
        target_rel_errors = [torch.tensor(0., device=device)] * cfg.total_rounds
        target_Ns = [torch.tensor(0, device=device)] * cfg.total_rounds
        num_saps_not_in_region_list = [torch.tensor(0., device=device)] * cfg.total_rounds
        test_fn = std.likelihood.get_condition
        quantile_map = {}

        omega_cfg = OmegaConf.to_object(cfg)
        if not isinstance(omega_cfg.example, BrownianMotionDiffExampleConfig):
            true, _ = get_true_tail_prob(directory, alpha)
            true = true.to('cpu')
        else:
            batch_size = 1690000
            x0 = torch.randn(
                batch_size,
                cfg.example.sde_steps-1,
                1,
            )
            dt = torch.tensor(1. / (cfg.example.sde_steps-1))
            scaled_x0 = x0 * dt.sqrt()  # standardize data
            sample_trajs = torch.cat([
                torch.zeros(batch_size, 1, 1),
                scaled_x0.cumsum(dim=1)
            ], dim=1)
            true = (sample_trajs > alpha).any(dim=1).to(sample_trajs.dtype).mean()

        for sample_idx, num_samples in enumerate([0]+cfg.samples[:-1]):
            # for each sample size, construct error bars
            for i, (data, all_log_qrobs) in enumerate(zip(sample_data, sample_log_qrobs)):
                if data is None:
                    continue
                # for each subsample of data, compute IS estimate
                saps = data[num_samples:cfg.samples[sample_idx]]
                saps_raw = get_saps_raw(saps, cfg)
                log_probs = target.log_prob(saps_raw).squeeze()
                log_qrobs = all_log_qrobs[num_samples:cfg.samples[sample_idx]]

                target_estimate, target_N, q_phis = \
                    iterative_importance_estimate(
                        test_fn=test_fn,
                        saps_raw=saps_raw,
                        saps=saps,
                        log_probs=log_probs,
                        log_qrobs=log_qrobs,
                        cur_expectation=target_rel_errors[i],
                        cur_N=target_Ns[i],
                    )
                target_rel_error = torch.abs(target_estimate - true) / true
                target_rel_errors[i] = target_rel_error
                target_Ns[i] = target_N
                num_saps_not_in_region_list[i] += (1-q_phis).sum()
            # construct error bar
            quantiles = torch.stack(target_rel_errors).to('cpu').quantile(
                torch.tensor([0.05, 0.5, 0.95])
            )
            print(quantiles)
            quantile_map[target_N.to('cpu')] = quantiles

        f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')

        sap_error_pairs = [(sap, error) for sap, error in quantile_map.items()]
        for saps, error in sap_error_pairs:
            ax.plot([saps, saps], [error[0], error[2]], alpha=0.3, color='blue', linewidth=2.5)
            ax.scatter(saps, error[1], marker='o', label=f'Diffusion (N={saps})', color='blue')

        empirical_error = torch.load('empirical_errors.pt', weights_only=True)
        run_type = 'Gaussian' if 'Gaussian' in cfg.model_name else 'BrownianMotionDiff'
        sap_error_pairs = [(sap, error) for sap, error in empirical_error[run_type][alpha].items()]
        for saps, error in sap_error_pairs:
            ax.plot([saps, saps], [error[0], error[2]], alpha=0.3, color='red', linewidth=2.5)
            ax.scatter(saps, error[1], marker='o', color='red')
            ax2.plot([saps, saps], [error[0], error[2]], alpha=0.3, color='red', linewidth=2.5)
            ax2.scatter(saps, error[1], marker='o', label=f'Empirical (N={saps})', color='red')

        ax.legend()
        ax2.legend()
        f.supxlabel('Monte Carlo Samples')
        f.supylabel('Relative Error of Estimate')
        model = 'Gaussian' if run_type == 'Gaussian' else 'Brownian Motion'
        f.suptitle(f'{model} Performance vs. Number of Samples\n(alpha={alpha}, dim={cfg.diffusion.dim})')

        ax2.set_xscale('log')
        # Format as whole numbers
        ax.tick_params(axis='y', which='both', labelleft=True)

        sorted_emp_saps = sorted(empirical_error[run_type][alpha].keys())
        max_q_sap = sorted(quantile_map.keys())[-1]
        min_emp_sap = min(sorted_emp_saps[0], max_q_sap)*1.5

        ax.set_xlim(0, max_q_sap*1.1)
        ax2.set_xlim(min_emp_sap, sorted_emp_saps[-1]*10)

        # hide the spines between ax and ax2
        ax.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.yaxis.tick_right()

        # plot break lines
        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((1-d, 1+d), (-d, +d), **kwargs)
        ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
        ax2.plot((-d, +d), (-d, +d), **kwargs)

        directory = '{}/effort_v_performance'.format(HydraConfig.get().run.dir)
        os.makedirs(directory, exist_ok=True)
        error_bar_file = '{}/{}.pdf'.format(
            directory,
            performance_v_samples(alpha)
        )
        plt.savefig(error_bar_file)
        plt.clf()


@hydra.main(version_base=None, config_path="conf", config_name="pp_config")
def main(cfg):
    make_effort_v_performance(cfg)
    # make_performance_v_samples(cfg)


if __name__ == '__main__':
    os.system('echo git commit: $(git rev-parse HEAD)')

    cs = ConfigStore.instance()
    cs.store(name="vpsde_pp_config", node=PostProcessingConfig)
    register_configs()

    with torch.no_grad():
        main()
