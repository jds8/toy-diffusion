#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from compute_quadratures import efficient_normalizing_constant_bm
import matplotlib.pyplot as plt
import argparse


def conditional_fn(data, args):
    if 'gaussian' in args.type.lower():
        return data.norm(dim=-1) > args.alpha
    elif 'brownian motion' in args.type.lower():
        return (data.abs() > args.alpha).any(dim=-1)
    else:
        raise NotImplementedError

def get_covariance(args):
    if 'gaussian' in args.type.lower():
        return torch.eye(2)
    elif 'brownian motion' in args.type.lower():
        return torch.tensor([[0.5, 0.], [0., 1.]])
    else:
        raise NotImplementedError

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=float)
    parser.add_argument('--type', type=str)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    x1 = torch.linspace(-2, 2, 500)
    x2 = torch.linspace(-2, 2, 500)
    xx, yy = torch.meshgrid(x1, x2, indexing='xy')
    data = torch.stack([xx, yy], dim=-1)
    covariance = get_covariance(args)
    numerator = torch.distributions.MultivariateNormal(torch.zeros(2), covariance).log_prob(data).exp()
    denom = efficient_normalizing_constant_bm(args.alpha)
    pdfs = numerator / denom
    pdfs = torch.where(conditional_fn(data, args), pdfs, torch.nan)
    plt.contourf(xx, yy, pdfs, cmap='viridis')
    plt.colorbar(label='PDF')
    ax = plt.gca()
    title_name = args.type.title()
    ax.set_title(rf'Conditional {title_name} Heatmap ($\alpha={args.alpha}$)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    name = args.type.lower().replace(' ', '_')
    plt.savefig(f'{name}_heatmap.pdf')

if __name__ == "__main__":
    main()
