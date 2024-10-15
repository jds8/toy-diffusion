#!/usr/bin/env python3

import sys
import os
import argparse


def check_run(args):
    model = args.model
    rnd = args.round
    alpha = args.alpha

    directory = 'figs/{}/'.format(model)
    diffusion_filename = 'alpha={}_diffusion_is_stats_round_{}.pt'.format(alpha, rnd)
    target_filename = 'alpha={}_target_is_stats_round_{}.pt'.format(alpha, rnd)
    files = os.listdir(directory)
    return diffusion_filename in files or target_filename in files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('--model', type=str)
    parser.add_argument('--round', type=int)
    parser.add_argument('--alpha', type=float)
    args = parser.parse_args()

    exit_code = not check_run(args)
    sys.exit(exit_code)
