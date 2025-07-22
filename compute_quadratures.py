#!/usr/bin/env python3

import os

import scipy.stats as stats
import scipy.integrate as integrate
import numpy as np
import torch

from typing import Callable, Optional

from importance_sampling import BrownianMotionDiffTarget

def normal_pdf(x, mean, var):
    """Probability density function of a normal distribution."""
    # scale specifies standard deviation
    return stats.norm.pdf(x, loc=mean, scale=np.sqrt(var))

def quadrature3(alpha):
   """Computes the probability of leaving [-alpha, alpha] using quadrature."""
   # First integral over X1 ~ N(0, 1/4)
   def inner_integral_x1(x1):
       # Fourth integral over X2 ~ N(X1, 1/4)
       def inner_integral_x2(x2):
           return normal_pdf(x2, x1, 1/2)
       result_x2, _ = integrate.quad(inner_integral_x2, -alpha, alpha)
       return normal_pdf(x1, 0, 1/2) * result_x2
   result_x1, _ = integrate.quad(inner_integral_x1, -alpha, alpha)
   return 1 - result_x1

def quadrature5(alpha):
   """Computes the probability of leaving [-alpha, alpha] using quadrature."""
   # First integral over X1 ~ N(0, 1/4)
   def inner_integral_x1(x1):
       # Second integral over X2 ~ N(X1, 1/4)
       def inner_integral_x2(x2):
           # Third integral over X3 ~ N(X2, 1/4)
           def inner_integral_x3(x3):
               # Fourth integral over X4 ~ N(X3, 1/4)
               def inner_integral_x4(x4):
                   return normal_pdf(x4, x3, 1/4)
               result_x4, _ = integrate.quad(inner_integral_x4, -alpha, alpha)
               return normal_pdf(x3, x2, 1/4) * result_x4
           result_x3, _ = integrate.quad(inner_integral_x3, -alpha, alpha)
           return normal_pdf(x2, x1, 1/4) * result_x3
       result_x2, _ = integrate.quad(inner_integral_x2, -alpha, alpha)
       return normal_pdf(x1, 0, 1/4) * result_x2

   result_x1, _ = integrate.quad(inner_integral_x1, -alpha, alpha)
   return 1 - result_x1

def pdf_3d_quadrature(p: torch.Tensor, alpha: torch.Tensor):
   # computes p(x_2||x_1| < \alpha)
   # p(x_2 | |x_1| < \alpha) = \int_{x_1\in [-\alpha, \alpha]} p(x_2 | x_1) p(x_1) dx_1

   # p(x_1, x_2 | |x_1| < \alpha, |x_2| < \alpha) =
   # p(x_2|x_1) * p(x_1) * 1(x_1,x_2 \in [-\alpha, \alpha]) / \int_{x_1,x_2 \in [-\alpha, \apha]}

   # p(\sqrt(x_1 ^ 2 + x_2 ^ 2) = p | |x_1| < alpha, |x_2| < alpha) =
   # \int_{x_1 \in [-p, p]} p(x_2=sqrt(p^2 - x_1^2|x_1) )p(x_1)dx_1

   assert p >= alpha
   def integrand_x1(x1):
      def integrand_x2(x2):
         return normal_pdf(x2, x1, 1/4) * normal_pdf(torch.sqrt(p - (x1**2 + x2**2)), x2, 1/4)
      boundary = torch.min(torch.sqrt(p - x1 ** 2), alpha)
      result_x2, _ = integrate.quad(integrand_x2, -boundary, boundary)
      return normal_pdf(x1, 0, 1/4) * result_x2
   boundary = p.sqrt()
   result, _ = integrate.quad(integrand_x1, -boundary, boundary)
   return result

def old_pdf_2d_quadrature_bm(p: np.ndarray, alpha: np.ndarray):
    """
    dX1**2 + dX2**2 = p for IID standard Gaussian dX1, dX2
    For a three step Brownian motion 0,X1,X2, the dXi
    represent deltas, so:
    dX1 = (X1 - 0) / dt.sqrt() = X1 / dt.sqrt()
    dX2 = (X2 - X1) / dt.sqrt()
    (0 + dX1 + dX2) = X1 / dt.sqrt() + (X2 - X1) / dt.sqrt() = X2 / dt.sqrt()
    where dt = 1/2
    The dx1 and dx2 in the code below represent dX1 and dX2
    """
    dt = np.array(1/2)
    def integrand_dx1(dx1):
        dx2 = np.sqrt(p**2 - dx1**2)
        x1 = dx1 * np.sqrt(dt)
        x2 = (dx1 + dx2) * np.sqrt(dt)
        if np.abs(x1) > alpha or np.abs(x2) > alpha:
            return 4 * p * normal_pdf(dx1, 0, 1) * normal_pdf(dx2, 0, 1) / dx2
        return 0.
    boundary = p
    result, _ = integrate.quad(integrand_dx1, 0., boundary)
    return result

def pdf_3d_quadrature_bm(p: float, alpha: float, num_pts=1000):
    dt = 1/3
    thetas = np.linspace(0, np.pi, num_pts).reshape(-1, 1)
    phis = np.linspace(0, 2 * np.pi, num_pts).reshape(1, -1)
    dx1 = p * np.sin(thetas) * np.cos(phis)  # x coord
    dx2 = p * np.sin(thetas) * np.sin(phis)  # y coord
    dx3 = p * np.cos(thetas)  # z coord

    x1 = dx1 * np.sqrt(dt)
    x2 = (dx1 + dx2) * np.sqrt(dt)
    x3 = (dx1 + dx2 + dx3) * np.sqrt(dt)

    phi_vals = normal_pdf(dx1, 0, 1) * normal_pdf(dx2, 0, 1) * normal_pdf(dx3, 0, 1)
    dtheta = thetas[1,0] - thetas[0,0]
    dphi = phis[0,1] - phis[0,0]
    weights = phi_vals * p ** 2 * np.sin(thetas) * dtheta * dphi

    if alpha == 0.:
        total_weight = 1.  # for alpha=0.
    elif alpha == 0.5:
        total_weight = 0.8060463160489418   # for alpha=0.5
    elif alpha == 1.:
        total_weight = 0.40119811854180637  # for alpha=1.
    elif alpha == 1.5:
        total_weight = 0.15720399570060603  # for alpha=1.5
    elif alpha == 2.0:
        total_weight = 0.050560932192670285  # for alpha=2.0
    else:
        raise NotImplementedError
    conditions = (np.abs(x1) > (alpha-1e-5)) | (np.abs(x2) > (alpha-1e-5)) | (np.abs(x3) > (alpha-1e-5))
    exit_weight = np.sum(weights[conditions])

    if exit_weight == total_weight == 0.:
        result = 0.
    else:
        result = exit_weight / total_weight
    return result

def pdf_2d_quadrature_bm(p: float, alpha: float, num_pts=1000):
    dt = 0.5
    thetas = np.linspace(0, 2 * np.pi, num_pts)
    dx1 = p * np.cos(thetas)
    dx2 = p * np.sin(thetas)

    x1 = dx1 * np.sqrt(dt)
    x2 = (dx1 + dx2) * np.sqrt(dt)

    phi_vals = normal_pdf(dx1, 0, 1) * normal_pdf(dx2, 0, 1)
    weights = phi_vals * p * (2 * np.pi / num_pts)  # arc length elements

    total_weight = BrownianMotionDiffTarget.bm_pdf(3)[alpha]
    exit_weight = np.sum(weights[(np.abs(x1) > (alpha-1e-5)) | (np.abs(x2) > (alpha-1e-5))])

    if exit_weight == total_weight == 0.:
        result = 0.
    else:
        result = exit_weight / total_weight
    return result

def get_quadrature_dir() -> str:
    dr = 'quadrature_data/'
    os.makedirs(dr, exist_ok=True)
    return dr

def get_quadrature_filename(sde_steps: int, num_abscissa: int, alpha: float) -> str:
    quad_dir = get_quadrature_dir()
    filename = f'{quad_dir}/sde_steps_{sde_steps}_num_abscissa_{num_abscissa}_alpha_{alpha:.1f}.pt'
    return filename

def cached(sde_steps: int, num_abscissa: int, alpha: float) -> Optional[torch.Tensor]:
    filename = get_quadrature_filename(sde_steps, num_abscissa, alpha)
    if os.path.isfile(filename):
        print('loading pdf from cache')
        return torch.load(filename)
    print('computing pdf using quadrature')
    return None

def get_2d_pdf(sde_steps: int, abscissa: torch.Tensor, alpha: float):
    cache = cached(sde_steps, abscissa.shape[0], alpha)
    if cache:
        return cache
    filename = get_quadrature_filename(sde_steps, abscissa.shape[0], alpha)
    if sde_steps == 3:
        pdf = [pdf_2d_quadrature_bm(a.cpu().item(), alpha) for a in abscissa]
        torch.save(pdf, filename)
    elif sde_steps == 4:
        pdf = [pdf_3d_quadrature_bm(a.cpu().item(), alpha) for a in abscissa]
        torch.save(pdf, filename)
    else:
        raise NotImplementedError
    return pdf

def estimate_integral(
    max_sample: np.ndarray,
    alpha: np.ndarray,
    quadrature_fn
):
    result, _ = integrate.quad(quadrature_fn, alpha, np.inf)
    return result

def plot_quadrature_vs_chi():
    import scipy
    q_values = np.linspace(0, 5, 100)
    alpha = 1.5
    numerical_pdf = [pdf_3d_quadrature_bm(q, alpha) for q in q_values]
    analytical_pdf = stats.chi(3).pdf(q_values)
    
    denom = integrate.simpson(numerical_pdf, x=q_values)
    print(denom)
    import matplotlib.pyplot as plt
    plt.plot(q_values, numerical_pdf, label="Numerical Quadrature")
    plt.plot(q_values, analytical_pdf, '--', label="Analytical Solution")
    plt.legend()
    plt.xlabel("q")
    plt.ylabel("PDF") 
    plt.title("Chi Distribution (2 DoF)")
    plt.show()

# def pdf_2d_quadrature(p: np.ndarray):
#    def integrand_x1(x1):
#       dx2 = np.sqrt(p - x1 ** 2)
#       x2_prob = normal_pdf(dx2, x1, 1/2) + normal_pdf(-dx2, x1, 1/2)
#       return normal_pdf(x1, 0, 1/2) * x2_prob / (2 * np.abs(dx2))
#    result, _ = integrate.quad(integrand_x1, -np.sqrt(p), np.sqrt(p))
#    return result


# def pdf_2d_quadrature(p: np.ndarray):
#    def integrand_x1(x1):
#       dx2 = np.sqrt(p - x1 ** 2)
#       x2_prob = normal_pdf(dx2, x1, 1/2) + normal_pdf(-x2, x1, 1/2)
#       return normal_pdf(x1, 0, 1/2) * x2_prob / (2 * np.abs(x2))
#    result, _ = integrate.quad(integrand_x1, -np.sqrt(p), np.sqrt(p))
#    return result


if __name__ == '__main__':
    # alphas = np.linspace(0.0, 3.5, 8)
    # # alphas = [np.array(0.)]
    # for alpha in alphas:
    #    str_alpha = str(alpha.item()).replace('.', '_')
    #    print(f'pdf_values_alpha_{str_alpha}', '= {')
    #    ps = np.linspace(alpha, 6., 250)
    #    # ps = np.linspace(alpha, 6., 39)
    #    for p in ps:
    #       print(f'{p}: {pdf_2d_quadrature(p, alpha)},')
    #       # print(f'{p}: {pdf_2d_quadrature(p)},')
    #    print('}')
    # for alpha in [0.5]:
    #     print(quadrature3(alpha))
    # for alpha in [0.5]:
    #     print(quadrature5(alpha))
    plot_quadrature_vs_chi()

     # pdf_values_alpha_2 = {
     #     2.0 :               0.1317,
     #     2.22 :               0.1203,
     #     2.44 :               0.1101,
     #     2.66 :               0.1007,
     #     2.88 :               0.0922,
     #     3.11 :               0.0844,
     #     3.33 :               0.0772,
     #     3.55 :               0.0707,
     #     3.77 :               0.0647,
     #     4.0 :               0.0593,
     #     4.22 :  0.05428856411218762,
     #     4.44 :  0.04971247250197807,
     #     4.66 :  0.04552178879974613,
     #     4.88 :   0.0416829874102561,
     #     5.11 : 0.038165123788538494,
     #     5.33 : 0.034939583168877006,
     #     5.55 :  0.03197996264978746,
     #     5.77 : 0.029261981231761425,
     #     6.0 : 0.026763452520150593
     # }
     # pdf_values_alpha_2_5 = {
     #     2.5: 0.10762911585203452,
     #     2.71875: 0.09861715450817077,
     #     2.9375: 0.09038505773953852,
     #     3.15625: 0.0828552989232785,
     #     3.375: 0.0759618325823267,
     #     3.59375: 0.06964721085594257,
     #     3.8125: 0.06386064152774146,
     #     4.03125: 0.058556677946002555,
     #     4.25: 0.05369431202944859,
     #     4.46875: 0.04923633297217918,
     #     4.6875: 0.0451488486092023,
     #     4.90625: 0.0414009149474499,
     #     5.125: 0.03796423546706243,
     #     5.34375: 0.03481290994609765,
     #     5.5625: 0.03192321382337604,
     #     5.78125: 0.02927340727791737,
     #     6.0: 0.026843566108258148
     # }
     # pdf_values_alpha_3 = {
     #     3.0: 0.0881651041354729,
     #     3.2142856121063232: 0.08096690648762274,
     #     3.4285714626312256: 0.0743639216502399,
     #     3.642857074737549: 0.06830391236130898,
     #     3.857142925262451: 0.06274040022544669,
     #     4.0714287757873535: 0.05763162814821908,
     #     4.285714149475098: 0.052939796085941536,
     #     4.5: 0.048630475567693,
     #     4.714285850524902: 0.044672264468216656,
     #     4.9285712242126465: 0.04103643122668797,
     #     5.142857074737549: 0.0376966235827229,
     #     5.357142925262451: 0.034628700705097555,
     #     5.5714287757873535: 0.03181050132941741,
     #     5.785714149475098: 0.02922168653506374,
     #     6.0: 0.026843566108258148
     # }
     # pdf_values_alpha_3_5 = {
     #     3.5: 0.0722859896948953,
     #     3.7083332538604736: 0.06655328790900723,
     #     3.9166667461395264: 0.06127736552902316,
     #     4.125: 0.05642098522943023,
     #     4.333333492279053: 0.05195025646917358,
     #     4.541666507720947: 0.04783426104550232,
     #     4.75: 0.044044646542324005,
     #     4.958333492279053: 0.04055543103902498,
     #     5.166666507720947: 0.03734274079763043,
     #     5.375: 0.03438460615848256,
     #     5.583333492279053: 0.03166084072125051,
     #     5.791666507720947: 0.029152865540142153,
     #     6.0: 0.026843566108258148
     # }
