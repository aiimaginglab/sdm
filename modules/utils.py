import os 
import sys
import math
import json
import random
import logging
import numpy as np

import torch

from tqdm import tqdm
from scipy import interpolate
from typing import Optional, Tuple


def set_seed(seed_value):
    """
    set seed for reproducibility. 
    """
    torch.manual_seed(seed_value)

    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) 

    np.random.seed(seed_value)
    random.seed(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_logger(save_path=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%m/%d %H:%M:%S')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if save_path:
        file_handler = logging.FileHandler(save_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def save_json(data, file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, allow_nan=True)

#----------------------------------------------------------------------------
# utils for adaptive solver selection

@torch.no_grad()
def determine_solver(sigma_t: torch.Tensor, uhat_fn, u_cut: float = 0.65):
    u = uhat_fn(sigma_t)
    is_early = (u >= u_cut)
    if is_early.ndim > 0:
        is_early = is_early.float().mean() >= 0.5
    return (dict(method='euler', n_corr=0) if is_early
            else dict(method='heun', n_corr=0))

@torch.no_grad()
def determine_solver_dxdt(
    kappa: float | torch.Tensor | None = None,
    tau_k: float = 1.0,
):
    def _to_scalar(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return float(x.mean().item())
        return float(x)

    k_val = _to_scalar(kappa)
    trigger = (k_val >= tau_k)

    method = "heun" if trigger else "euler"

    return dict(method=method, n_corr=0)

def compute_lambda(i: int, num_steps: int, lambda_schedule: str, solver_trigger: bool) -> float:
    """
    lambda = 1: Euler only
    lambda = 0: Heun only
    """
    if lambda_schedule == 'step':
        return 0.0 if solver_trigger else 1.0

    p = i / max(num_steps - 1, 1)
    a = 0.0; b = 1.0
    u = (p - a) / (b - a)

    # clamp to [0,1]
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)

    if lambda_schedule == 'linear':
        return 1.0 - u
    elif lambda_schedule == 'cosine':
        return 0.5 * (1.0 + math.cos(math.pi * u))
    else:
        raise ValueError(lambda_schedule)
    
#----------------------------------------------------------------------------
# util functions for computing first and second order derivatives 

def compute_D_sigma(net, x: torch.Tensor, t: torch.Tensor,
                    sigma, s, class_labels=None
    ) -> torch.Tensor:
    """
    Returns: D_sigma.

    method="jvp"  : torch.autograd.functional.jvp 
    method="fd"   : finite difference
    """

    dtype = next(net.parameters()).dtype
    device = next(net.parameters()).device

    x_in = x.detach().to(dtype=dtype, device=device) / s(t)
    sig  = sigma(t).detach().to(dtype=dtype, device=device).requires_grad_(True)

    with torch.enable_grad():
        def f(sig):
            return net(x_in, sig, class_labels).to(dtype)
        y, D_sig = torch.autograd.functional.jvp(
            f, (sig,), (torch.ones_like(sig),)
        )
        return D_sig.detach()

@torch.no_grad()
def compute_jacobian(
    net, x: torch.Tensor, t: torch.Tensor, sigma, s, class_labels,
    eps_theta: torch.Tensor
) -> torch.Tensor:
    """
    Returns: JD_eps.

    method="jvp" : torch.autograd.functional.jvp
    method="fd"  : finite difference
    """
    dtype  = next(net.parameters()).dtype
    device = next(net.parameters()).device

    x0   = x.detach().to(dtype=dtype, device=device)
    sig  = sigma(t).detach().to(dtype=dtype, device=device)
    epsT = eps_theta.detach().to(dtype=dtype, device=device)

    with torch.enable_grad():
        def f_x(xin):
            return net(xin / s(t), sig, class_labels).to(dtype)
        y, JD_eps = torch.autograd.functional.jvp(f_x, (x0,), (epsT,))
    return JD_eps.detach()

def per_sample_norm(x):
    return x.float().reshape(x.shape[0], -1).norm(dim=1)  # (B,)

def batch_mean_norm(x):
    return per_sample_norm(x).mean()  # scalar

#----------------------------------------------------------------------------
# Optimize timestep schedules based on cumulative score differences
# util functions for the paper "Score-Optimal Diffusion Models"

def build_sigmas(
    num_steps=40,
    sigma_min=0.002,
    sigma_max=80.0,
    schedule='edm',          # 'edm' | 'loglinear' | 'vp' | 've' | 'iddpm'
    rho=7.0,
    epsilon_s: float = 1e-3, 
    C_1: float = 0.001,      
    C_2: float = 0.008,      
    M: int = 1000,           
):
    """
    Returns:
        sigmas: shape [num_steps], np.array, sigmas[0] = sigma_min, sigmas[-1] = sigma_max
    """
    if schedule == 'edm':
        step_indices = np.arange(num_steps, dtype=np.float64)
        sigmas = (sigma_min ** (1/rho)
                          + step_indices / (num_steps - 1)
                            * (sigma_max ** (1/rho) - sigma_min ** (1/rho))) ** rho
    elif schedule == 'loglinear':
        log_sigmas = np.linspace(np.log(sigma_min), np.log(sigma_max), num_steps)
        sigmas = np.exp(log_sigmas)
    elif schedule == 'vp':
        vp_beta_d = 2 * (np.log(sigma_min**2 + 1) / epsilon_s - np.log(sigma_max**2 + 1)) / (epsilon_s - 1)
        vp_beta_min = np.log(sigma_max**2 + 1) - 0.5 * vp_beta_d
        def sigma_vp(t):
            return np.sqrt(np.exp(0.5 * vp_beta_d * (t**2) + vp_beta_min * t) - 1.0)
        t_vals = np.linspace(epsilon_s, 1.0, num_steps, dtype=np.float64)
        sigmas = sigma_vp(t_vals)
    elif schedule == 've':
        i = np.arange(num_steps, dtype=np.float64)
        t_vals = (sigma_min**2) * ((sigma_max**2 / sigma_min**2) ** (i / (num_steps - 1)))
        sigmas = np.sqrt(t_vals)
    elif schedule == 'iddpm':
        u = np.zeros(M + 1, dtype=np.float64)
        alpha_bar = lambda j: np.sin(0.5 * np.pi * j / M / (C_2 + 1)) ** 2

        for j in range(M, 0, -1):  # j = M, ..., 1
            u[j - 1] = np.sqrt(
                (u[j] ** 2 + 1.0) / max(alpha_bar(j - 1) / alpha_bar(j), C_1) - 1.0
            )

        mask = (u >= sigma_min) & (u <= sigma_max)
        u_filtered = u[mask]

        if len(u_filtered) == 0:
            raise RuntimeError("No IDDPM sigmas found in the range "
                               f"[{sigma_min}, {sigma_max}].")

        u_filtered = np.sort(u_filtered)

        if num_steps == 1:
            sigmas = np.array([u_filtered[0]], dtype=np.float64)
        else:
            idx = np.linspace(0, len(u_filtered) - 1, num_steps)
            sigmas = u_filtered[np.round(idx).astype(int)]
    else:
        raise ValueError(f"Unknown schedule type: {schedule}")

    return sigmas

@torch.no_grad()
def build_t_steps_from_sigmas(
    sigmas: torch.Tensor,
    *,
    net,
    device: torch.device,
    sigma_min: float,
    sigma_max: float,
    sigma_inv, 
) -> Tuple[torch.Tensor, int]:
    
    if not torch.is_tensor(sigmas):
        sigmas = torch.as_tensor(sigmas)
    sigmas = sigmas.to(device=device, dtype=torch.float64)

    if sigmas.numel() >= 2 and (sigmas[:-1] < sigmas[1:]).any():
        sigmas, _ = torch.sort(sigmas, descending=True)

    if sigmas.numel() > 0 and sigmas[-1].item() == 0.0:
        sigmas = sigmas[:-1]

    sigmas = sigmas.clamp(min=sigma_min, max=sigma_max)

    t_steps = sigma_inv(net.round_sigma(sigmas))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    num_steps = t_steps.numel() - 1
    return t_steps, num_steps

def get_time_transforms(discretization: str = 'edm', rho: float = 7.0):
    """
    discretization: 'edm', 'loglinear'
    """
    assert discretization in ['edm', 'loglinear']
    if discretization == 'edm':
        time_of_sigma  = lambda s: np.power(s, 1.0 / rho)
        sigma_of_time  = lambda t: np.power(np.asarray(t, np.float64), rho)
    else:
        time_of_sigma  = lambda s: np.log(s)
        sigma_of_time  = lambda t: np.exp(np.asarray(t, np.float64))
    return time_of_sigma, sigma_of_time

# calculate uhat: f_t2L represents t→Λ linear interpolation
def make_uhat_fn(f_t2L, sigma_min: float, sigma_max: float, discretization='edm', rho=7.0):
    """
    Returns: uhat(sigma_tensor: torch.Tensor) -> torch.Tensor in [0,1]
    """
    time_of_sigma, _ = get_time_transforms(discretization, rho)

    t_min = time_of_sigma(np.array([sigma_min], dtype=np.float64))[0]
    t_max = time_of_sigma(np.array([sigma_max], dtype=np.float64))[0]
    L_min = float(f_t2L(t_min))
    L_max = float(f_t2L(t_max))
    denom = max(L_max - L_min, 1e-20)

    def uhat(sigma_t: torch.Tensor) -> torch.Tensor:
        s_np = sigma_t.detach().cpu().double().numpy()
        t_np = time_of_sigma(s_np)
        L_np = f_t2L(t_np)                           # Λ(σ)
        u_np = (L_np - L_min) / denom
        return torch.as_tensor(u_np, dtype=sigma_t.dtype, device=sigma_t.device)
    return uhat

def compute_score_diffs_once(net, dataset_iterator, sigmas, batch_size, num_repeats=10, device='cuda'):

    score_diffs = np.zeros((num_repeats, len(sigmas)-1, batch_size))
    for repeat_idx in tqdm(range(num_repeats)):
        images, labels = next(dataset_iterator)
        images = images.to(device).to(torch.float32) / 127.5 - 1
        labels = labels.to(device)
        # log_sigmas = np.arange(np.log(min_sigma), np.log(max_sigma), 0.1)

        for i in range(len(sigmas)-1):
            left_sigma = sigmas[i]
            right_sigma = sigmas[i+1]

            # sample at the higher noise level
            right_noisy_images = images + torch.randn_like(images) * right_sigma

            left_sigmas_pt = torch.ones((batch_size,), device=device) * left_sigma
            right_sigmas_pt = torch.ones((batch_size,), device=device) * right_sigma

            right_to_left_cleaned_images = net(right_noisy_images, left_sigmas_pt, labels)
            right_to_right_cleaned_images = net(right_noisy_images, right_sigmas_pt, labels)

            right_to_left_score = (right_to_left_cleaned_images - right_noisy_images) / (left_sigmas_pt[:, None, None, None]**2)
            right_to_right_score = (right_to_right_cleaned_images - right_noisy_images) / (right_sigmas_pt[:, None, None, None]**2)

            score_diffs[repeat_idx, i, :] = torch.sum( right_sigmas_pt[:, None, None, None]**2 * (right_to_right_score - right_to_left_score)**2, dim=(1,2,3)).mean().sqrt().cpu().detach().numpy()

    avg_cum_sum = np.cumsum(np.mean(score_diffs, axis=0)[:, 0])
    return avg_cum_sum

def build_cos_sigmas(avg_cum_sum: np.ndarray,
                     sigmas_probe: np.ndarray,
                     num_steps: int = 40,
                     discretization: str = 'edm',
                     rho: float = 7.0,
                     sigma_min: float = None,
                     sigma_max: float = None,
                     append_zero: bool = True) -> (np.ndarray, interpolate.PchipInterpolator):
    """
    avg_cum_sum: shape [K]  (각 interval 비용의 누적 Λ; 예: np.cumsum(mean(score_diffs,axis=0)[:,0]))
    sigmas_probe: shape [K+1], 큰 -> 작 (초기 그리드; log-linear든 EDM이든 OK)
    반환: (sigmas_out[ num_steps (+1 if append_zero) ], f_t2L(t)=Λ 보간기)
    """
    avg_cum_sum = np.asarray(avg_cum_sum, dtype=np.float64)
    sigmas_probe = np.asarray(sigmas_probe, dtype=np.float64)
    assert len(sigmas_probe) == len(avg_cum_sum) + 1, "K+1 vs K 길이 불일치"

    # 1) Interpolation (t <-> lambda)
    # time_of_sigma, sigma_of_time = get_time_transforms(discretization, rho)
    time_of_sigma, sigma_of_time = get_time_transforms(discretization=discretization)
    t_vals = time_of_sigma(sigmas_probe[1:])
    f_L2t  = interpolate.PchipInterpolator(x=avg_cum_sum, y=t_vals)   # lambda -> t
    f_t2L  = interpolate.PchipInterpolator(x=t_vals,      y=avg_cum_sum)  # t -> lambda

    # 2) uniform lambda spacing
    linear_L  = np.linspace(0.0, avg_cum_sum[-1], num_steps, dtype=np.float64)
    optimized_t_vals    = f_L2t(linear_L)
    optimized_sigmas  = sigma_of_time(optimized_t_vals)

    # 3) sort, clamp
    optimized_sigmas = optimized_sigmas[::-1]
    if sigma_max is None: sigma_max = float(sigmas_probe[0])
    if sigma_min is None: sigma_min = float(sigmas_probe[-1])
    optimized_sigmas[0]  = max(sigma_max, optimized_sigmas[0])
    optimized_sigmas[-1] = min(sigma_min, optimized_sigmas[-1])

    # 4) add zero
    if append_zero:
        optimized_sigmas = np.concatenate([optimized_sigmas, [0.0]])

    return optimized_sigmas.astype(np.float32), f_t2L


#----------------------------------------------------------------------------
# utils for adaptive timestep scheduling

def find_bin_index(sigma_grid: torch.Tensor, sigma_cur: torch.Tensor) -> int:
    """
    sigma_grid: 1D, descending, includes sigma_min ... sigma_max (no 0 necessarily)
    returns i such that sigma_grid[i] >= sigma_cur > sigma_grid[i+1] (clamped)
    """
    sg = sigma_grid
    # clamp sigma_cur into range
    if sigma_cur >= sg[0]:
        return 0
    if sigma_cur <= sg[-1]:
        return sg.numel() - 2  # last valid bin
    for i in range(sg.numel() - 1):
        if sg[i] >= sigma_cur > sg[i + 1]:
            return i
    return sg.numel() - 2

@torch.no_grad()
def eval_dxdt(
    net,
    x: torch.Tensor,
    t: torch.Tensor,
    class_labels,
    sigma, sigma_deriv, s, s_deriv,
    metric: str = "dxdt",
):
    """
    Returns:
      drift term (dx/dt) at (x,t)
    """
    denoised = net(x / s(t), sigma(t), class_labels).to(torch.float64)
    d_cur = (sigma_deriv(t) / sigma(t) + s_deriv(t) / s(t)) * x \
          - sigma_deriv(t) * s(t) / sigma(t) * denoised
    return d_cur 

def resample_sigmas_uniform_cum_eta(
    sigma_path,             # shape (K+1,) decreasing, includes sigma_max ... 0
    eta_step,               # shape (K,) per-interval eta (between sigma_path[i] -> sigma_path[i+1])
    num_steps,              # desired number of steps (intervals). output len = N+1
    power=0.5,              
    q=0.0,
    append_zero=True,
    discretization="edm",   
    eps=1e-12,
):
    sigma_path = np.asarray(sigma_path, dtype=np.float64)
    eta_step   = np.asarray(eta_step,   dtype=np.float64)

    assert sigma_path.ndim == 1 and eta_step.ndim == 1
    assert len(sigma_path) == len(eta_step) + 1
    assert num_steps >= 2

    if sigma_path[-1] == 0:
        sigma_path = sigma_path[:-1]
        eta_step = eta_step[:-1]
        assert len(sigma_path) == len(eta_step) + 1

    # cost length ~ sqrt(L) in the paper; use eta as L
    length = np.maximum(eta_step, 0.0) ** power
    # guard: if all zero, fall back to uniform-in-index
    if not np.isfinite(length).all() or (length.sum() < eps):
        raise ValueError("Invalid eta_step: weights are all-zero or non-finite.")

    # weighted sum of eta_i
    sig = sigma_path[1:]
    sigma_max = float(sigma_path[0])
    g = ((sigma_max + eps)/(sig + eps)) ** q
    length = length * g

    cum = np.concatenate([[0.0], np.cumsum(length)])  # len K+1
    total = cum[-1]
    linear_L = np.linspace(0.0, total, num_steps)

    # 1) Interpolation (t <-> lambda)
    time_of_sigma, sigma_of_time = get_time_transforms(discretization=discretization)
    t_vals = np.asarray(time_of_sigma(sigma_path), dtype=np.float64) 

    # 2) uniform lambda spacing
    f_L2t = interpolate.PchipInterpolator(x=cum, y=t_vals)  # x strictly increasing, y increasing in typical schedules
    optimized_t_vals = f_L2t(linear_L)  
    optimized_sigmas  = sigma_of_time(optimized_t_vals)

    optimized_sigmas[0] = max(sigma_path[0], optimized_sigmas[0])
    optimized_sigmas[-1] = min(sigma_path[-1], optimized_sigmas[-1])

    # 4) add zero
    if append_zero:
        optimized_sigmas = np.concatenate([optimized_sigmas, [0.0]])

    return optimized_sigmas