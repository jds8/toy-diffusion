#!/usr/bin/env python3


import matplotlib.pyplot as plt
import torch


def analytical_gaussian_score(t, x, mu_0, sigma_0):
    '''
    Compute the analytical score p_t for t \in (0, 1)
    given the SDE formulation from Song et al. in the case that
    p_0 = N(mu_0, sigma_0) and p_1 = N(0, 1)
    '''
    beta0 = 0.1
    beta1 = 20.
    log_mean_coeff = -0.25 * t ** 2 * (beta1 - beta0) - 0.5 * t * beta0
    lmc = log_mean_coeff.repeat((log_mean_coeff.reshape(-1).shape[0],) + x.shape[1:])
    f = lmc.exp()
    g2 = (1 - (2. * lmc).exp())
    var = sigma_0 ** 2 * f ** 2 + g2
    score = (f * mu_0 - x) / var
    return score

def plt_gaussian_score(
        true_score,
        score_est,
        t: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor
):
    """
    true_score is true score function which takes as input x, t and outputs score
    score_est is score function estimate which takes as input x, t and outputs score
    t is time
    mu is mean of true Gaussian distribution
    sigma is std of true Gaussian distribution
    """

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('True Score Function Versus Estimate')

    x_vals = torch.linspace(mu-3*sigma, mu+3*sigma, 100)

    true_scores = true_score(x_vals, t)
    ax1.scatter(x_vals, true_scores, color='blue')

    # score_ests = score_est(x_vals, t)
    # ax2.scatter(x_vals, score_ests, color='red')

    plt.savefig('figs/score_functions_at_time_{}.pdf'.format(t))
    plt.clf()


if __name__ == "__main__":
    mu_0 = torch.tensor(1.)
    sigma_0 = torch.tensor(2.)

    true_score = lambda x, t: analytical_gaussian_score(t, x, mu_0, sigma_0)
    for t in torch.linspace(0., 1., 4):
        plt_gaussian_score(true_score, lambda x, t: 0., t, mu_0, sigma_0)
