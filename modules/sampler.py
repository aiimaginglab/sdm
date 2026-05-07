import os 
import sys 
import math
import logging
import numpy as np 
from typing import Optional, Tuple

import torch
from diffusers import DPMSolverMultistepScheduler as DefaultDPMSolver

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import per_sample_norm, batch_mean_norm, build_t_steps_from_sigmas, make_uhat_fn
from utils import determine_solver, determine_solver_dxdt, compute_lambda, compute_jacobian, compute_D_sigma
from utils import find_bin_index, eval_dxdt


#----------------------------------------------------------------------------
# baseline ode solvers

@torch.no_grad()
def euler_update(
    net, x_hat, t_hat, t_next, class_labels
):
    denoised = net(x_hat, t_hat, class_labels).to(torch.float64) 
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur
    return x_next


@torch.no_grad()
def heun_update(
    net, x_hat, t_hat, t_next, class_labels, idx, num_steps
):
    # euler update
    denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # heun update 
    if idx < num_steps - 1:
      denoised_next = net(x_next, t_next, class_labels).to(torch.float64)
      d_prime = (x_next - denoised_next) / t_next
      x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next

@torch.no_grad()
def ablation_euler_update(
    net, x_hat, t_hat, t_next, class_labels,
    sigma, sigma_deriv, s, s_deriv,
    return_dx_dt=False,
):
    h = t_next - t_hat
    denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
    d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat \
          - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
    x_next = x_hat + h * d_cur
    if return_dx_dt:
        return x_next, denoised, d_cur
    return x_next

@torch.no_grad()
def ablation_heun_update(
    net, x_hat, t_hat, t_next, class_labels,
    sigma, sigma_deriv, s, s_deriv,
    alpha=1.0,
    return_dx_dt=False,
):
    h = t_next - t_hat
    # 1st drift
    denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
    d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat \
          - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
    # euler update 
    x_prime = x_hat + alpha * h * d_cur
    t_prime = t_hat + alpha * h
    # 2nd drift 
    denoised_prime = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
    d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime \
            - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised_prime
    # trapezoid update
    x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)
    if return_dx_dt:
        return x_next, denoised, d_cur, d_prime
    return x_next


#----------------------------------------------------------------------------
# Baseline EDM sampler with customized ode solvers (euler, heun, dpm solver)

@torch.no_grad()
def dpm_solver_first_order_update(
    model_output, sample, sigma_s, sigma_t
):
    """
    First-order DPM-Solver++ update in EDM sigma parameterization.
    This is the bootstrap step for 2M.
    """
    lambda_t = -torch.log(sigma_t.clamp(min=1e-40))
    lambda_s = -torch.log(sigma_s.clamp(min=1e-40))
    h = lambda_t - lambda_s

    # alpha_t = alpha_s = 1 in EDM formulation
    x_t = (sigma_t / sigma_s) * sample - (torch.exp(-h) - 1.0) * model_output
    return x_t


@torch.no_grad()
def multistep_dpm_solver_second_order_update(
    model_output_list, sample, sigma_s1, sigma_s0, sigma_t, solver_type="midpoint"
):
    """
    EDM version of Diffusers' multistep_dpm_solver_second_order_update
    with alpha_t = 1, sigma_t = sigma.
    model_output_list[-1] : m0 (current denoised prediction at sigma_s0)
    model_output_list[-2] : m1 (previous denoised prediction at sigma_s1)
    """
    assert solver_type in ["midpoint", "heun"]

    m0, m1 = model_output_list[-1], model_output_list[-2]

    lambda_t  = -torch.log(sigma_t.clamp(min=1e-40))
    lambda_s0 = -torch.log(sigma_s0.clamp(min=1e-40))
    lambda_s1 = -torch.log(sigma_s1.clamp(min=1e-40))

    h   = lambda_t - lambda_s0
    h_0 = lambda_s0 - lambda_s1
    r0  = h_0 / h

    D0 = m0
    D1 = (1.0 / r0) * (m0 - m1)

    if solver_type == "midpoint":
        x_t = (
            (sigma_t / sigma_s0) * sample
            - (torch.exp(-h) - 1.0) * D0
            - 0.5 * (torch.exp(-h) - 1.0) * D1
        )
    else:  # heun
        x_t = (
            (sigma_t / sigma_s0) * sample
            - (torch.exp(-h) - 1.0) * D0
            + (((torch.exp(-h) - 1.0) / h) + 1.0) * D1
        )

    return x_t

@torch.no_grad()
def _ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='dpmsolver', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, sigmas: torch.Tensor | np.ndarray | None = None,
    solver_type="midpoint"
):
    assert solver in ['euler', 'heun', 'dpmsolver', 'unipc']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']
    assert solver_type in ['midpoint', 'heun']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # --- Build schedules & t_steps ---
    if sigmas is not None:
        # Import external sigmas
        t_steps, num_steps = build_t_steps_from_sigmas(
            sigmas, net=net, device=latents.device, sigma_min=sigma_min, sigma_max=sigma_max, sigma_inv=sigma_inv
        )
    else:
        # Define time steps in terms of noise level.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        if discretization == 'vp':
            orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
            sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
        elif discretization == 've':
            orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
            sigma_steps = ve_sigma(orig_t_steps)
        elif discretization == 'iddpm':
            u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
            alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
            for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
                u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
            u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
            sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
        else:
            assert discretization == 'edm'
            sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

        # Compute final time steps based on the corresponding noise levels.
        t_steps = sigma_inv(net.round_sigma(sigma_steps))
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))

    model_output_list = []
    sigma_history = []
    nfe = 0 
    
    # UniPC sampler
    unipc_state = None
    if solver == "unipc":
        unipc_state = UniPCSampler(
            solver_type="bh2",  
            lower_order_final=True,
            disable_corrector_steps=[0,1],
        )
        unipc_state.set_total_steps(len(t_steps) - 1)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        if solver == 'dpmsolver':
            logging.info(f"Step {i+1}/{num_steps}: Using DPM-Solver++(2M)...")

            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)

            model_output_list.append(denoised)
            sigma_history.append(sigma(t_hat))

            if len(model_output_list) > 2:
                model_output_list.pop(0)
                sigma_history.pop(0)

            if (i == 0) or (i == num_steps - 1):
                x_next = dpm_solver_first_order_update(
                    model_output=denoised,
                    sample=x_hat,
                    sigma_s=sigma(t_hat),
                    sigma_t=sigma(t_next),
                )
                nfe += 1
            else: 
                x_next = multistep_dpm_solver_second_order_update(
                    model_output_list=model_output_list,
                    sample=x_hat,
                    sigma_s1=sigma_history[-2],
                    sigma_s0=sigma_history[-1],
                    sigma_t=sigma(t_next),
                    solver_type=solver_type,
                )
                nfe += 2

        elif solver == 'unipc':
            logging.info(f"Step {i+1}/{len(t_steps)-1}: Using UniPC(2)...")

            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)

            x_next = unipc_state.step(
                model_output=denoised,
                sample=x_hat,
                sigma_cur=sigma(t_hat),
                sigma_next=sigma(t_next),
            )    
            nfe += 1
                    
        else: 
            if solver == 'euler' or i == num_steps - 1:
                logging.info(f"Step {i+1}/{num_steps}: Using Euler method...")
                x_next = ablation_euler_update(
                    net, x_hat, t_hat, t_next, class_labels,
                    sigma, sigma_deriv, s, s_deriv
                )
                nfe += 1 
            else:
                logging.info(f"Step {i+1}/{num_steps}: Using Heun method...")
                x_next = ablation_heun_update(
                    net, x_hat, t_hat, t_next, class_labels,
                    sigma, sigma_deriv, s, s_deriv, alpha=alpha
                )
                nfe += 1

    return x_next, nfe


@torch.no_grad()
def _ablation_edm2_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='dpmsolver', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, sigmas: torch.Tensor | np.ndarray | None = None,
    solver_type="midpoint",
    gnet=None, guidance=1.0,
):
    assert solver in ['euler', 'heun', 'dpmsolver', 'unipc']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']
    assert solver_type in ['midpoint', 'heun']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # # Adjust noise levels based on what's supported by the network.
    # sigma_min = max(sigma_min, net.sigma_min)
    # sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # --- Build schedules & t_steps ---
    if sigmas is not None:
        # Import external sigmas
        t_steps, num_steps = build_t_steps_from_sigmas(
            sigmas, net=net, device=latents.device, sigma_min=sigma_min, sigma_max=sigma_max, sigma_inv=sigma_inv
        )
    else:
        # Define time steps in terms of noise level.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        if discretization == 'vp':
            orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
            sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
        elif discretization == 've':
            orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
            sigma_steps = ve_sigma(orig_t_steps)
        elif discretization == 'iddpm':
            u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
            alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
            for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
                u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
            u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
            sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
        else:
            assert discretization == 'edm'
            sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

        # Compute final time steps based on the corresponding noise levels.
        t_steps = sigma_inv(sigma_steps) # sigma_inv(net.round_sigma(sigma_steps))
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    def denoise(x, t, class_labels):
        Dx = net(x, t, class_labels).to(torch.float64)
        if gnet is None or guidance == 1.0:
            return Dx
        ref_Dx = gnet(x, t, class_labels).to(torch.float64)
        return ref_Dx.lerp(Dx, guidance)

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))

    model_output_list = []
    sigma_history = []
    nfe = 0 
    
    # UniPC sampler
    unipc_state = None
    if solver == "unipc":
        unipc_state = UniPCSampler(
            solver_type="bh2", 
            lower_order_final=True,
            disable_corrector_steps=[0,1],
        )
        unipc_state.set_total_steps(len(t_steps) - 1)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(sigma(t_cur) + gamma * sigma(t_cur)) # sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        if solver == 'dpmsolver':
            logging.info(f"Step {i+1}/{num_steps}: Using DPM-Solver++(2M)...")

            denoised = denoise(x_hat, t_hat, class_labels) # denoised = net(x_hat, t_hat, class_labels).to(torch.float64)

            model_output_list.append(denoised)
            sigma_history.append(sigma(t_hat))

            if len(model_output_list) > 2:
                model_output_list.pop(0)
                sigma_history.pop(0)

            if (i == 0) or (i == num_steps - 1):
                x_next = dpm_solver_first_order_update(
                    model_output=denoised,
                    sample=x_hat,
                    sigma_s=sigma(t_hat),
                    sigma_t=sigma(t_next),
                )
                nfe += 1
            else: 
                x_next = multistep_dpm_solver_second_order_update(
                    model_output_list=model_output_list,
                    sample=x_hat,
                    sigma_s1=sigma_history[-2],
                    sigma_s0=sigma_history[-1],
                    sigma_t=sigma(t_next),
                    solver_type=solver_type,
                )
                nfe += 2

        elif solver == 'unipc':
            logging.info(f"Step {i+1}/{len(t_steps)-1}: Using UniPC(2)...")

            denoised = denoise(x_hat, t_hat, class_labels) # denoised = net(x_hat, t_hat, class_labels).to(torch.float64)

            x_next = unipc_state.step(
                model_output=denoised,
                sample=x_hat,
                sigma_cur=sigma(t_hat),
                sigma_next=sigma(t_next),
            )    
            nfe += 1
                    
        else: 
            if solver == 'euler' or i == num_steps - 1:
                logging.info(f"Step {i+1}/{num_steps}: Using Euler method...")
                x_next = ablation_euler_update(
                    net, x_hat, t_hat, t_next, class_labels,
                    sigma, sigma_deriv, s, s_deriv
                )
                nfe += 1 
            else:
                logging.info(f"Step {i+1}/{num_steps}: Using Heun method...")
                x_next = ablation_heun_update(
                    net, x_hat, t_hat, t_next, class_labels,
                    sigma, sigma_deriv, s, s_deriv, alpha=alpha
                )
                nfe += 1

    return x_next, nfe


#----------------------------------------------------------------------------
# Improved EDM sampler with adaptive solvers

@torch.no_grad()
def ablation_sampler_adaptive_solver(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, sigmas: torch.Tensor | np.ndarray | None = None,
    calc_curvature: bool = False,                  
    curvature_mode: str = 'proxy',                 # 'proxy' (1 NFE proxy) | 'heun_fd' (2 NFE true fd for d_next)
    eps: float = 1e-12,                            
    metric: str = 'dxdt',                          
    use_relative_metrics: bool = True,             # k_rel / k_abs
    u_cut=0.65, tau_k=1.0,
    lambda_schedule: str = 'step',                 # 'step' | 'linear' | 'cosine'
):
    """
    Behavior summary:
      - Always compute Euler drift once per step (1 NFE).
      - If metric='uhat': solver decision uses uhat (no extra NFE).
      - If metric='dxdt':
          * default decision uses kappa proxy from (d_cur - d_prev) / dt (1 NFE).
          * if calc_curvature=True and curvature_mode='heun_fd': compute d_next via Heun (extra 1 NFE)
            and log "true" fd curvature; solver decision can still use proxy or true fd—here we use true fd.
      - Update:
          * If decide Euler: x_next = x_euler (no extra NFE).
          * If decide Heun: do the 2nd eval and apply Heun correction (extra 1 NFE),
            unless curvature_mode='heun_fd' already computed it (reuse).
      - Last step: force Euler (matches original [Anonymous] behavior).
    """
    assert discretization in ['vp', 've', 'iddpm', 'edm', 'loglinear']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']
    assert metric in ['dxdt', 'uhat']
    assert curvature_mode in ['proxy', 'heun_fd']
    assert lambda_schedule in ['step', 'linear', 'cosine']


    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Define second derivative functions for VP & VE noise level schedules.
    def vp_sigma_second_deriv(beta_d, beta_min):
        def _sigma_ddot(t):
            sig = sigma(t)
            sigdot = sigma_deriv(t)
            B = beta_min + beta_d * t
            return 0.5 * beta_d * (sig + 1.0 / (sig + eps)) \
                 + 0.5 * B * sigdot * (1.0 - 1.0 / (sig**2 + eps))
        return _sigma_ddot
    ve_sigma_second_deriv = lambda t: -0.25 / (t**1.5 + eps)

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002, 'loglinear': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80, 'loglinear': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    if hasattr(net, "sigma_min") and hasattr(net, "sigma_max"):
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_second_deriv = vp_sigma_second_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_second_deriv = ve_sigma_second_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1.0
        sigma_second_deriv = lambda t: 0.0
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
        def s_second_deriv(t):
            sig     = sigma(t)
            sigdot  = sigma_deriv(t)
            sigddot = sigma_second_deriv(t)
            s_t     = s(t)
            ds_dsig   = -sig * (s_t**3)
            d2s_dsig2 = (2*sig**2 - 1.0) * (s_t**5)
            return d2s_dsig2 * (sigdot**2) + ds_dsig * sigddot
    else:
        assert scaling == 'none'
        s = lambda t: torch.ones_like(t) # 1
        s_deriv = lambda t: torch.zeros_like(t) # 0
        s_second_deriv = lambda t: torch.zeros_like(t) # 0

    # --- Build schedules & t_steps ---
    if sigmas is not None:
        # Load external sigmas
        t_steps, num_steps = build_t_steps_from_sigmas(
            sigmas, net=net, device=latents.device, sigma_min=sigma_min, sigma_max=sigma_max, sigma_inv=sigma_inv
        )
    else:
        # Define time steps in terms of noise level.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        if discretization == 'vp':
            orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
            sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
        elif discretization == 've':
            orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
            sigma_steps = ve_sigma(orig_t_steps)
        elif discretization == 'iddpm':
            u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
            alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
            for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
                u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
            u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
            sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
        elif discretization == 'edm':
            sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        else:
            assert discretization == 'loglinear'
            def make_loglinear_sigmas(sigma_min, sigma_max, num_steps, device):
                step = torch.linspace(0, 1, num_steps, device=device, dtype=torch.float64)
                log_min = math.log(float(sigma_min))
                log_max = math.log(float(sigma_max))
                logs = log_max + step * (log_min - log_max)
                return torch.exp(logs)
            sigma_steps = make_loglinear_sigmas(sigma_min, sigma_max, num_steps, latents.device)

        # Compute final time steps based on the corresponding noise levels.
        if hasattr(net, "round_sigma"):
            t_steps = sigma_inv(net.round_sigma(sigma_steps))
        else:
            t_steps = sigma_inv(sigma_steps)
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    
    # for dxdt proxy
    d_prev = None
    t_prev_hat = None

    gap_log = []  # logging for curvature analysis
    solver_trigger = False # step scheduler for Lambda function (euler -> heun)
    nfe = 0

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        if hasattr(net, "round_sigma"):
            t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        else:
            t_hat = sigma_inv(sigma(t_cur) + gamma * sigma(t_cur))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # ---- determine solver ----
        h = abs(t_next - t_hat) # + eps
        k_abs = None; k_rel = None

        x_euler, denoised, d_cur = ablation_euler_update(net, x_hat, t_hat, t_next, class_labels,
                                        sigma, sigma_deriv, s, s_deriv, return_dx_dt=True)
        
        if metric == 'uhat':
            uhat_fn = make_uhat_fn(f_t2L, sigma_min, sigma_max, discretization, rho=7.0)
            if lambda_schedule == "step":
                if not solver_trigger:
                    ops = determine_solver(t_hat, uhat_fn, u_cut=u_cut) 
                    if ops['method'] == 'heun':
                        solver_trigger = True
                else:
                    ops = {'method': 'heun'}
            else:
                # linear/cosine lambda schedule always uses both solvers
                ops = {'method': 'mixture'}

        elif metric == 'dxdt':
            # If calc_curvature=True and curvature_mode='heun_fd', compute "true" fd using Heun d_prime.
            # Else, use proxy := (d_cur - d_prev) / dt.
            
            # initialize 
            x_heun = None
            d_cur_h = None
            d_next = None

            if calc_curvature and curvature_mode == 'heun_fd' and (i < num_steps - 1):
                # true fd curvature (NFE = 2)
                x_heun, denoised_h, d_cur_h, d_next = ablation_heun_update(net, x_hat, t_hat, t_next, class_labels,
                                                            sigma, sigma_deriv, s, s_deriv, return_dx_dt=True)
                f_norm_h = per_sample_norm(d_cur_h)
                f_norm_b_h = f_norm_h.view(-1, *([1]*(d_cur_h.ndim-1)))
                
                if use_relative_metrics:
                    k_rel = float(batch_mean_norm((d_next - d_cur_h) / ((h + eps) * f_norm_b_h)).item())
                    kappa = k_rel
                else:
                    k_abs = float(batch_mean_norm((d_next - d_cur_h) / (h + eps)).item())
                    kappa = k_abs
            else: 
                # proxy: use previous drift (NFE = 1)
                if d_prev is None:
                    kappa = 0.0
                    k_rel = 0.0 if use_relative_metrics else None
                    k_abs = 0.0 if (not use_relative_metrics) else None
                else:
                    f_norm_prev = per_sample_norm(d_prev)   # (B,)
                    f_norm_b_prev = f_norm_prev.view(-1, *([1]*(d_prev.ndim-1)))   # (B,1,1,...)
                    h_proxy = abs(t_hat - t_prev_hat)
                    
                    if use_relative_metrics:
                        k_rel = float(batch_mean_norm((d_cur - d_prev) / ((h_proxy + eps) * f_norm_b_prev)).item())
                        kappa = k_rel
                    else:
                        k_abs = float(batch_mean_norm((d_cur - d_prev) / (h_proxy + eps)).item())
                        kappa = k_abs

            if lambda_schedule == "step":
                if not solver_trigger:
                    ops = determine_solver_dxdt(kappa=kappa, tau_k=tau_k)
                    if ops['method'] == 'heun':
                        solver_trigger = True  
                else: 
                    ops = {'method': 'heun'}
            else:
                # linear/cosine lambda schedule always uses both solvers
                ops = {'method': 'mixture'}

        # update lambda scheduler
        _lambda = compute_lambda(i, num_steps, lambda_schedule, solver_trigger)

        if i == num_steps - 1:
            _lambda = 1.0

        # update x (NFE < 2 for step schedule, NFE = 2 for linear/cosine schedule)
        if _lambda == 1.0: 
            # Euler only (no extra NFE)
            logging.info(f"Step {i+1}/{num_steps}: Using Euler method only ...")
            x_next = x_euler 
            nfe += 1
        else: 
            if not (calc_curvature and curvature_mode == 'heun_fd' and (x_heun is not None)):
                # compute heun solver 
                h_signed = (t_next - t_hat)
                x_prime = x_hat + alpha * h_signed * d_cur
                t_prime = t_hat + alpha * h_signed

                denoised_prime = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
                d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime \
                        - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised_prime

                x_heun = x_hat + h_signed * ((1 - 1/(2*alpha)) * d_cur + 1/(2*alpha) * d_prime)
                nfe += 2

            # update with Euler/Heun solver outputs
            logging.info(f"Step {i+1}/{num_steps}: Mixing Euler and Heun method with λ={_lambda:.4f}...")
            x_next = _lambda * x_euler + (1.0 - _lambda) * x_heun


        # general (EDM/VP/VE)
        if calc_curvature and metric == 'dxdt':
            # ---- eps_theta, JD_eps, D_sigma ----
            sig_hat = sigma(t_hat)                           # (B,)
            sig_b   = sig_hat.view(-1, *([1]*(x_hat.ndim-1)))
            s_hat   = s(t_hat)                               # (B,)
            s_b     = s_hat.view(-1, *([1]*(x_hat.ndim-1)))

            eps_theta = (x_hat - s_b * denoised) / (sig_b + 1e-12)

            JD_eps = compute_jacobian(
                net, x_hat, t_hat, sigma, s, class_labels, eps_theta
            )
            JD_D = compute_jacobian(
                net, x_hat, t_hat, sigma, s, class_labels, denoised
            )
            D_sig = compute_D_sigma(
                net, x_hat, t_hat, sigma, s, class_labels
            )

            # ---- first, second order derivaives ----
            sig_dot  = sigma_deriv(t_hat)             # (B,)
            s_dot    = s_deriv(t_hat)                 # (B,)
            sig_ddot = sigma_second_deriv(t_hat)      # (B,)
            s_ddot   = s_second_deriv(t_hat)          # (B,)

            # ---- PF-ODE second order derivation formula ----
            A = s_ddot / (s_hat + 1e-12)                                    # (B,)
            B = sig_ddot + 2.0 * sig_dot * (s_dot / (s_hat + 1e-12))
            C = - sig_dot * (s_dot + sig_dot * s_hat / (sig_hat + 1e-12))
            D = - sig_dot * (s_dot * s_hat / (sig_hat + 1e-12))
            E = - sig_dot * (sig_dot * s_hat / (sig_hat + 1e-12))

            A_b = A.view(-1, *([1]*(x_hat.ndim-1)))
            B_b = B.view(-1, *([1]*(x_hat.ndim-1)))
            C_b = C.view(-1, *([1]*(x_hat.ndim-1)))
            D_b = D.view(-1, *([1]*(x_hat.ndim-1)))
            E_b = E.view(-1, *([1]*(x_hat.ndim-1)))

            ddotx_closed_vec = (
                A_b * x_hat
                + B_b * eps_theta
                + C_b * JD_eps
                + D_b * JD_D
                + E_b * D_sig
            )

            if curvature_mode == 'heun_fd' and (d_next is not None):
                ddotx_fd_vec = (d_next - d_cur_h) / (h + 1e-12)
            else:
                # proxy curvature (one-step lag) : (d_cur - d_prev)/dt
                if d_prev is None:
                    ddotx_fd_vec = torch.zeros_like(d_cur)
                else:
                    dt = abs(t_hat - t_prev_hat)
                    ddotx_fd_vec = (d_cur - d_prev) / (dt + eps)

            Dtheta_norm       = float(batch_mean_norm(denoised).item())
            Dsigma_norm       = float(batch_mean_norm(D_sig).item())
            eps_theta_norm    = float(batch_mean_norm(eps_theta).item())
            JD_eps_norm       = float(batch_mean_norm(JD_eps).item())
            bracket_norm      = float(batch_mean_norm(JD_eps + D_sig).item())
            ddotx_closed_norm = float(batch_mean_norm(ddotx_closed_vec).item())
            ddotx_fd_norm     = float(batch_mean_norm(ddotx_fd_vec).item())
            ddotx_recon_err   = float(batch_mean_norm(ddotx_closed_vec - ddotx_fd_vec).item())

            x_minus_D_norm = float(batch_mean_norm(x_hat - denoised).item())   # ||x - Dθ||

            gap_log.append(
                {
                  "step": int(i),
                  "t_cur": float(t_cur), "t_next": float(t_next),
                  "sigma_cur": float(sigma(t_cur)), "sigma_next": float(sigma(t_next)),
                  "method": ops["method"], "h": float(h),
                  "lambda": float(_lambda),
                  "nfe": nfe,

                  # k_abs, k_rel
                  "k_abs": None if k_abs is None else float(k_abs),
                  "k_rel": None if k_rel is None else float(k_rel),

                  # batch mean norm 
                  "Dtheta_norm": Dtheta_norm,
                  "Dsigma_norm": Dsigma_norm,
                  "eps_theta_norm": eps_theta_norm,
                  "JD_eps_norm": JD_eps_norm,
                  "bracket_norm": bracket_norm,
                  "ddotx_closed_norm": ddotx_closed_norm,
                  "ddotx_fd_norm": ddotx_fd_norm,
                  "ddotx_recon_err_norm": ddotx_recon_err,

                  "x_minus_D_norm": x_minus_D_norm,
                  "sigma_hat": float(sig_hat),
                }
            )

        # update proxy state for next step
        if metric == 'dxdt':
            d_prev = d_cur.detach()
            t_prev_hat = t_hat.detach()

    return x_next, gap_log, nfe


#----------------------------------------------------------------------------
# Improved EDM sampler with adaptive solvers (DPM-Solver++/UniPC)

@torch.no_grad()
def ablation_sampler_adaptive_dpmsolver(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    sigmas: torch.Tensor | np.ndarray | None = None,
    calc_curvature: bool = False,
    eps: float = 1e-12,
    metric: str = 'dxdt',
    use_relative_metrics: bool = True,
    tau_k=1.0,
    lambda_schedule: str = 'step',
    solver_type: str = "midpoint",
    gnet=None, guidance=1.0,
):
    assert discretization in ['vp', 've', 'iddpm', 'edm', 'loglinear']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']
    assert metric in ['dxdt', 'uhat']
    assert lambda_schedule in ['step', 'linear', 'cosine']
    assert solver_type in ['midpoint', 'heun']

    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002, 'loglinear': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80, 'loglinear': 80}[discretization]

    if hasattr(net, "sigma_min") and hasattr(net, "sigma_max"):
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)

    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        sigma = lambda t: t
        sigma_deriv = lambda t: 1.0
        sigma_inv = lambda sigma: sigma

    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        s = lambda t: torch.ones_like(t)
        s_deriv = lambda t: torch.zeros_like(t)

    if sigmas is not None:
        t_steps, num_steps = build_t_steps_from_sigmas(
            sigmas, net=net, device=latents.device,
            sigma_min=sigma_min, sigma_max=sigma_max, sigma_inv=sigma_inv
        )
    else:
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        if discretization == 'vp':
            orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
            sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
        elif discretization == 've':
            orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
            sigma_steps = ve_sigma(orig_t_steps)
        elif discretization == 'iddpm':
            u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
            alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
            for j in torch.arange(M, 0, -1, device=latents.device):
                u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
            u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
            sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
        elif discretization == 'edm':
            sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        else:
            def make_loglinear_sigmas(sigma_min, sigma_max, num_steps, device):
                step = torch.linspace(0, 1, num_steps, device=device, dtype=torch.float64)
                log_min = math.log(float(sigma_min))
                log_max = math.log(float(sigma_max))
                logs = log_max + step * (log_min - log_max)
                return torch.exp(logs)
            sigma_steps = make_loglinear_sigmas(sigma_min, sigma_max, num_steps, latents.device)

        if hasattr(net, "round_sigma"):
            t_steps = sigma_inv(net.round_sigma(sigma_steps))
        else:
            t_steps = sigma_inv(sigma_steps)
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    def denoise(x, t):
        Dx = net(x, t, class_labels).to(torch.float64)
        if gnet is None or guidance == 1.0:
            return Dx
        ref_Dx = gnet(x, t, class_labels).to(torch.float64)
        return ref_Dx.lerp(Dx, guidance)

    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))

    d_prev = None
    t_prev_hat = None
    solver_trigger = False
    gap_log = []
    nfe = 0

    # solver history for multistep generation
    model_output_list = []
    sigma_history = []

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        if hasattr(net, "round_sigma"):
            t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        else:
            t_hat = sigma_inv(sigma(t_cur) + gamma * sigma(t_cur))
        x_hat = s(t_hat) / s(t_cur) * x_cur + \
            (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        h = abs(t_next - t_hat)
        k_abs = None
        k_rel = None

        # current denoised + drift
        denoised = denoise(x_hat, t_hat)
        d_cur = (x_hat - denoised) / sigma(t_hat)

        # candidate: 1M
        x_1m = dpm_solver_first_order_update(
            model_output=denoised,
            sample=x_hat,
            sigma_s=sigma(t_hat),
            sigma_t=sigma(t_next),
        )

        # prepare 2M history
        model_output_list.append(denoised)
        sigma_history.append(sigma(t_hat))
        if len(model_output_list) > 2:
            model_output_list.pop(0)
            sigma_history.pop(0)

        # candidate: 2M if available
        x_2m = None
        if len(model_output_list) >= 2 and i < num_steps - 1:
            x_2m = multistep_dpm_solver_second_order_update(
                model_output_list=model_output_list,
                sample=x_hat,
                sigma_s1=sigma_history[-2],
                sigma_s0=sigma_history[-1],
                sigma_t=sigma(t_next),
                solver_type=solver_type,
            )

        # curvature / solver selection
        if metric == 'dxdt':
            if d_prev is None:
                kappa = 0.0
                k_rel = 0.0 if use_relative_metrics else None
                k_abs = 0.0 if not use_relative_metrics else None
            else:
                f_norm_prev = per_sample_norm(d_prev)
                f_norm_b_prev = f_norm_prev.view(-1, *([1] * (d_prev.ndim - 1)))
                h_proxy = abs(t_hat - t_prev_hat)

                if use_relative_metrics:
                    k_rel = float(batch_mean_norm((d_cur - d_prev) / ((h_proxy + eps) * f_norm_b_prev)).item())
                    kappa = k_rel
                else:
                    k_abs = float(batch_mean_norm((d_cur - d_prev) / (h_proxy + eps)).item())
                    kappa = k_abs

            if lambda_schedule == "step":
                if not solver_trigger:
                    # first step or history unavailable => must use 1M
                    if x_2m is None:
                        ops = {'method': 'dpm1'}
                    else:
                        ops = determine_solver_dxdt(kappa=kappa, tau_k=tau_k)
                        if ops['method'] == 'heun':
                            solver_trigger = True
                    if x_2m is None:
                        _lambda = 1.0
                    else:
                        _lambda = compute_lambda(i, num_steps, lambda_schedule, solver_trigger)
                else:
                    ops = {'method': 'dpm2'}
                    _lambda = compute_lambda(i, num_steps, lambda_schedule, solver_trigger)
            else:
                ops = {'method': 'mixture'}
                _lambda = compute_lambda(i, num_steps, lambda_schedule, solver_trigger)
        else:
            raise NotImplementedError("uhat version not implemented for adaptive DPM-Solver++ yet")

        if i == num_steps - 1:
            _lambda = 1.0

        # final update
        if x_2m is None or _lambda == 1.0:
            logging.info(f"Step {i+1}/{num_steps}: Using DPM-Solver++(1M)...")
            x_next = x_1m
            nfe += 1
        else:
            logging.info(f"Step {i+1}/{num_steps}: Mixing DPM-Solver++(1M/2M) with λ={_lambda:.4f}...")
            x_next = _lambda * x_1m + (1.0 - _lambda) * x_2m
            # same convention as your existing sampler: 2M step counts as 2 NFE
            nfe += 2

        if calc_curvature:
            gap_log.append({
                "step": int(i),
                "t_cur": float(t_cur), "t_next": float(t_next),
                "sigma_cur": float(sigma(t_cur)), "sigma_next": float(sigma(t_next)),
                "method": ops["method"],
                "h": float(h),
                "lambda": float(_lambda),
                "nfe": nfe,
                "k_abs": None if k_abs is None else float(k_abs),
                "k_rel": None if k_rel is None else float(k_rel),
            })

        d_prev = d_cur.detach()
        t_prev_hat = t_hat.detach()

    return x_next, gap_log, nfe


#----------------------------------------------------------------------------
# Improved EDM sampler with adaptive scheduling

@torch.no_grad()
def ablation_sampler_adaptive_scheduling(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    discretization="edm", schedule="linear", scaling="none",
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, sigmas: torch.Tensor | np.ndarray | None = None,
    eta_min: float = 0.02,
    eta_max: float = 0.20,
    eta_p: float = 1.0,
    gamma_step: float = 0.95,       # safety factor for delta_t_next
    c_shrink: float = 0.5,          # shrinking factor for eta_trial > eta
    max_expand_bins: int = 2,       # how many bins to try to expand at most
    max_shrink_iters: int = 2,      # how many iters to try to shrink at most
    solver: str = "heun",           # 'euler' | 'heun'
    metric: str = "dxdt",           
    use_grid_snap: bool = True,     # snap sigma_next to EDM grid
    eps: float = 1e-12,
):
    assert solver in ["euler", "heun"]
    assert discretization in ["vp", "ve", "iddpm", "edm", "loglinear"]
    assert schedule in ["vp", "ve", "linear"]
    assert scaling in ["vp", "none"]
    assert metric in ["dxdt", "uhat"]

    device = latents.device

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {"vp": vp_def, "ve": 0.02, "iddpm": 0.002, "edm": 0.002, "loglinear": 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {"vp": vp_def, "ve": 100, "iddpm": 81, "edm": 80, "loglinear": 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    if hasattr(net, "sigma_min") and hasattr(net, "sigma_max"):
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define noise level schedule.
    if schedule == "vp":
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == "ve":
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        sigma = lambda t: t
        sigma_deriv = lambda t: 1.0
        sigma_inv = lambda sigma: sigma

    if scaling == "vp":
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        s = lambda t: torch.ones_like(t)
        s_deriv = lambda t: torch.zeros_like(t)

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    elif discretization == 'edm':
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    else:
        assert discretization == 'loglinear'
        def make_loglinear_sigmas(sigma_min, sigma_max, num_steps, device):
            step = torch.linspace(0, 1, num_steps, device=device, dtype=torch.float64)
            log_min = math.log(float(sigma_min))
            log_max = math.log(float(sigma_max))
            logs = log_max + step * (log_min - log_max)
            return torch.exp(logs)
        sigma_steps = make_loglinear_sigmas(sigma_min, sigma_max, num_steps, latents.device)

    if hasattr(net, "round_sigma"):
        sigma_grid = net.round_sigma(sigma_steps).to(torch.float64)
    else:
        sigma_grid = sigma_steps

    # --- eta scheduling: use per-step tolerance based on current sigma ---
    sigma_max = float(sigma_grid[0])

    def eta_scheduler(sig: torch.Tensor | float) -> float:
        sig_f = float(sig) if not torch.is_tensor(sig) else float(sig.item())
        if sigma_max <= 0:
            return float(eta_min)
        r = max(0.0, min(1.0, sig_f / sigma_max))
        return float(eta_min + (eta_max - eta_min) * (r ** eta_p))

    # --- init ---
    t_cur = sigma_inv(torch.as_tensor(sigma_grid[0], device=device, dtype=torch.float64))
    x_cur = latents.to(torch.float64) * (sigma(t_cur) * s(t_cur))

    nfe = 0
    states = []
    if hasattr(net, "round_sigma"):
        optimized_sigmas_iedm = [float(net.round_sigma(sigma(t_cur)))]
    else:
        optimized_sigmas_iedm = [float(sigma(t_cur))]

    # main loop: sigma_max to sigma_min 
    while True:

        # cache (to reduce NFE)
        cache = {
            "t_hat": None,
            "x_hat": None,
            "d_cur": None,
        }

        if hasattr(net, "round_sigma"):
            sig_cur = net.round_sigma(sigma(t_cur)).to(torch.float64)
        else:
            sig_cur = sigma(t_cur)
        if float(sig_cur) <= float(sigma_min) + 1e-15:
            break

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        if hasattr(net, "round_sigma"):
            t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        else:
            t_hat = sigma_inv(sigma(t_cur) + gamma * sigma(t_cur))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # drift at (x_hat, t_hat)
        d_hat = eval_dxdt(net, x_hat, t_hat, class_labels, sigma, sigma_deriv, s, s_deriv)
        nfe += 1
        cache["t_hat"] = t_hat
        cache["x_hat"] = x_hat
        cache["d_cur"] = d_hat

        if hasattr(net, "round_sigma"):
            sig_hat = net.round_sigma(sigma(t_hat)).to(torch.float64)
        else:
            sig_hat = sigma(t_hat)
        # --- eta scheduling: use per-step tolerance based on current sigma ---
        eta = eta_scheduler(sig_hat)
        
        # -------------------------
        # Step 1: choose initial "sigma_trial" as current EDM bin width
        # -------------------------
        idx = find_bin_index(sigma_grid, sig_hat)
        sig_next_bin = sigma_grid[min(idx + 1, sigma_grid.numel() - 1)]
        delta_sigma_init = float((sig_hat - sig_next_bin).clamp(min=0))
        delta_sigma_trial = min(float(sig_hat) - float(sigma_min), delta_sigma_init)
        
        # -------------------------
        # Steps 2-5: line search via bin expansion / shrink
        # -------------------------
        best_S = None
        best_delta_sigma = None
        best_delta_t = None 
        
        expand_iters, shrink_iters = 0, 0
        expand_flag, shrink_flag = False, False  

        # stopping criteria 
        prev_sig_trial = None

        while True:
            logging.info(f"delta_sigma_trial: {delta_sigma_trial}, expand_iters: {expand_iters}, shrink_iters: {shrink_iters}")
            sig_trial = torch.as_tensor(float(sig_hat) - delta_sigma_trial, device=device, dtype=torch.float64)
            sig_trial = sig_trial.clamp(min=float(sigma_min), max=float(sig_hat))
            if use_grid_snap:
                if hasattr(net, "round_sigma"):
                    sig_trial = net.round_sigma(sig_trial) 
            t_trial = sigma_inv(sig_trial)

            logging.info(f"sig_hat: {float(sig_hat)}, prev_sig_trial: {prev_sig_trial} -> sig_trial: {float(sig_trial)}")
            if (prev_sig_trial is not None) and (float(sig_trial) == float(prev_sig_trial)):
                break  
            prev_sig_trial = sig_trial

            h_hat = abs(float(t_trial - t_hat)) 

            # euler step
            x_trial = x_hat + (t_trial - t_hat) * d_hat

            # drift at (x_trial, t_trial)
            d_trial = eval_dxdt(net, x_trial, t_trial, class_labels, sigma, sigma_deriv, s, s_deriv)
            nfe += 1

            # compute S_hat at (x_trial, t_trial)
            S_hat = float(batch_mean_norm(d_trial - d_hat) / (h_hat + eps))

            # eta_trial = h_hat^2 * S_hat / 2
            eta_trial = 0.5 * (h_hat ** 2) * S_hat
            if eta_trial <= eta:
                best_S = S_hat
                best_delta_sigma = float(delta_sigma_trial)
                best_delta_t = h_hat

            # if eta_trial <= eta, search for larger bin exponentially: 1 -> 2 -> 4 -> ...
            if (eta_trial <= eta) and (expand_iters < max_expand_bins):
                if shrink_flag: 
                    break 
                # no more bins to expand
                if idx + 2**(expand_iters+1) > sigma_grid.numel() - 1: 
                    break  
                sig_next_bin = sigma_grid[min(idx + 2**(expand_iters+1), sigma_grid.numel() - 1)]
                new_delta_sigma = float((sig_hat - sig_next_bin).clamp(min=0))
                new_delta_sigma_trial = min(float(sig_hat) - float(sigma_min), new_delta_sigma)
                
                # stopping criteria: expanding doesn't increase delta (grid end or duplicates)
                if new_delta_sigma_trial <= float(delta_sigma_trial) + 1e-15:
                    break

                delta_sigma_trial = new_delta_sigma_trial
                expand_flag = True 
                expand_iters += 1
                continue

            # if eta_trial > eta, reduce step size by c_shrink
            if eta_trial > eta: 
                if shrink_iters < max_shrink_iters:
                    delta_sigma_trial = max(delta_sigma_trial * c_shrink, 1e-12)
                    shrink_flag = True
                    shrink_iters += 1
                    continue
                break

            break

        if best_S is None:
            best_S = S_hat
            best_delta_sigma = float(delta_sigma_trial)
            best_delta_t = h_hat  

        # -------------------------
        # Step 6: compute next step size with best S_hat
        # -------------------------
        delta_t_next = gamma_step * math.sqrt(2.0 * eta / max(best_S, 1e-30))
        # bounds
        dt_min = max(best_delta_t, 1e-12)
        t_min = sigma_inv(torch.as_tensor(float(sigma_min), device=device, dtype=torch.float64))
        dt_max = abs(float(t_min - t_hat))  # until sigma_min
        delta_t_next = min(dt_max, max(dt_min, delta_t_next))

        t_next = torch.clamp(t_hat - delta_t_next, min=float(t_min))

        if use_grid_snap:
            if hasattr(net, "round_sigma"):
                sig_target = net.round_sigma(sigma(t_next)).to(torch.float64)
            else:
                sig_target = sigma(t_next)
            sig_target = sig_target.clamp(min=float(sigma_min), max=float(sigma_max))

            candidates = sigma_grid[sigma_grid <= sig_target]
            sig_snap = candidates[0] if candidates.numel() > 0 else sigma_grid[-1]
            t_next = torch.clamp(sigma_inv(sig_snap), min=float(t_min))

        # -------------------------
        # Step 7: actual update (Euler or Heun)
        # -------------------------
        if solver == "euler":
            x_next = cache["x_hat"] + (t_next - cache["t_hat"]) * cache["d_cur"]
        else:
            x_prime = cache["x_hat"] + alpha * (t_next - cache["t_hat"]) * cache["d_cur"]
            d_prime = eval_dxdt(net, x_prime, t_next, class_labels, sigma, sigma_deriv, s, s_deriv)
            nfe += 1 
            x_next = cache["x_hat"] + (t_next - cache["t_hat"]) * ((1 - 1 / (2 * alpha)) * cache["d_cur"] + 1 / (2 * alpha) * d_prime)

        x_cur, t_cur = x_next, t_next
        if use_grid_snap: 
            if hasattr(net, "round_sigma"):
                optimized_sigmas_iedm.append(float(net.round_sigma(sigma(t_cur))))
            else:
                optimized_sigmas_iedm.append(float(sigma(t_cur)))
        else:
            optimized_sigmas_iedm.append(float(sigma(t_cur)))

        state = {
            "t_hat": float(t_hat), 
            "t_next": float(t_next), 
            "best_S": best_S if best_S is not None else float('nan'), 
            "delta_t_next": delta_t_next, 
            "best_eta": float(0.5 * (best_delta_t**2) * best_S) if best_S is not None else float('nan'),
            "expand_iters": expand_iters,
            "shrink_iters": shrink_iters,
            "eta": float(eta),
            "nfe": nfe,
        }        
        states.append(state)

    # clamp [sigma_max, sigma_min], append 0 
    optimized_sigmas_iedm[0] = float(sigma_max)
    optimized_sigmas_iedm[-1] = float(sigma_min)
    optimized_sigmas_iedm.append(0.0)

    return x_cur, optimized_sigmas_iedm, states


@torch.no_grad()
def calc_wasserstein_bound(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    discretization="edm", schedule="linear", scaling="none",
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, sigmas: torch.Tensor | np.ndarray | None = None,
    solver: str = "heun",
    eps: float = 1e-12,
):
    assert solver in ["euler", "heun"]
    assert discretization in ["vp", "ve", "iddpm", "edm", "loglinear"]
    assert schedule in ["vp", "ve", "linear"]
    assert scaling in ["vp", "none"]

    device = latents.device

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {"vp": vp_def, "ve": 0.02, "iddpm": 0.002, "edm": 0.002, "loglinear": 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {"vp": vp_def, "ve": 100, "iddpm": 81, "edm": 80, "loglinear": 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    if hasattr(net, "sigma_min") and hasattr(net, "sigma_max"):
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define noise level schedule.
    if schedule == "vp":
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == "ve":
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        sigma = lambda t: t
        sigma_deriv = lambda t: 1.0
        sigma_inv = lambda sigma: sigma

    if scaling == "vp":
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        s = lambda t: torch.ones_like(t)
        s_deriv = lambda t: torch.zeros_like(t)


    # --- Build schedules & t_steps ---
    if sigmas is not None:
        # Load external sigmas
        t_steps, num_steps = build_t_steps_from_sigmas(
            sigmas, net=net, device=latents.device, sigma_min=sigma_min, sigma_max=sigma_max, sigma_inv=sigma_inv
        )
    else:
        # Define time steps in terms of noise level.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        if discretization == 'vp':
            orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
            sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
        elif discretization == 've':
            orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
            sigma_steps = ve_sigma(orig_t_steps)
        elif discretization == 'iddpm':
            u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
            alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
            for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
                u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
            u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
            sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
        elif discretization == 'edm':
            sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        else:
            assert discretization == 'loglinear'
            def make_loglinear_sigmas(sigma_min, sigma_max, num_steps, device):
                step = torch.linspace(0, 1, num_steps, device=device, dtype=torch.float64)
                log_min = math.log(float(sigma_min))
                log_max = math.log(float(sigma_max))
                logs = log_max + step * (log_min - log_max)
                return torch.exp(logs)
            sigma_steps = make_loglinear_sigmas(sigma_min, sigma_max, num_steps, latents.device)

        # Compute final time steps based on the corresponding noise levels.
        if hasattr(net, "round_sigma"):
            t_steps = sigma_inv(net.round_sigma(sigma_steps))
        else:
            t_steps = sigma_inv(sigma_steps)
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))

    wasserstein_bound_log = {
        "eta_log": [],   
        "dt_log": [],
        "S_hat_log": []
    }

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        if hasattr(net, "round_sigma"):
            t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        else:
            t_hat = sigma_inv(sigma(t_cur) + gamma * sigma(t_cur))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # drift at (x_hat, t_hat)
        d_hat = eval_dxdt(net, x_hat, t_hat, class_labels, sigma, sigma_deriv, s, s_deriv)

        h_hat = abs(float(t_next - t_hat))
        x_euler = x_hat + (t_next - t_hat) * d_hat

        if float(t_next) == 0.0:
            wasserstein_bound_log["eta_log"].append(0.0)
            wasserstein_bound_log["S_hat_log"].append(0.0)
            wasserstein_bound_log["dt_log"].append(h_hat)
            
            # euler step only
            x_next = x_euler
            continue 

        # drift at (x_euler, t_next)
        d_next = eval_dxdt(net, x_euler, t_next, class_labels, sigma, sigma_deriv, s, s_deriv)

        # calculate eta
        S_hat = float(batch_mean_norm(d_next - d_hat) / (h_hat + eps))
        eta = 0.5 * (h_hat ** 2) * S_hat

        wasserstein_bound_log["eta_log"].append(float(eta))
        wasserstein_bound_log["S_hat_log"].append(float(S_hat))
        wasserstein_bound_log["dt_log"].append(float(h_hat))

        # step update
        if solver == "euler":
            x_next = x_euler
        else:
            x_prime = x_hat + alpha * (t_next - t_hat) * d_hat
            d_prime = eval_dxdt(net, x_prime, t_next, class_labels, sigma, sigma_deriv, s, s_deriv)
            x_next = x_hat + (t_next - t_hat) * ((1 - 1 / (2 * alpha)) * d_hat + 1 / (2 * alpha) * d_prime)

    return wasserstein_bound_log


#----------------------------------------------------------------------------
# UniPC sampler

class UniPCSampler:
    """
    EDM/VE-style UniPC(2) sampler state.

    Assumptions
    ----------
    - model_output is already x0 / denoised prediction from NVLabs EDM wrappers
      (VPPrecond / VEPrecond / EDMPrecond all expose net(x, sigma)->D_x-like output)
    - alpha_t = 1, sigma_t = sigma
    - deterministic only (do not mix with churn)
    """

    def __init__(
        self,
        solver_type="bh2",
        lower_order_final=True,
        disable_corrector_steps=None,
    ):
        assert solver_type in ["bh1", "bh2"]
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.disable_corrector_steps = set(disable_corrector_steps or [0])

        self.model_outputs = [None, None]   # previous, current
        self.sigma_list = [None, None]      # previous, current
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = 1
        self.step_index = 0
        self.total_steps = None

    def set_total_steps(self, total_steps):
        self.total_steps = total_steps

    @torch.no_grad()
    def convert_model_output(self, model_output):
        # prediction_type="sample", predict_x0=True
        return model_output.to(torch.float64)

    @torch.no_grad()
    def _sigma_to_alpha_sigma_t(self, sigma):
        # EDM / VE formulation
        alpha_t = torch.ones_like(sigma)
        sigma_t = sigma
        return alpha_t, sigma_t

    @torch.no_grad()
    def _compute_h(self, sigma_a, sigma_b):
        # lambda = log(alpha) - log(sigma) = -log(sigma) when alpha=1
        lambda_a = -torch.log(sigma_a.clamp(min=1e-40))
        lambda_b = -torch.log(sigma_b.clamp(min=1e-40))
        return lambda_a - lambda_b

    @torch.no_grad()
    def multistep_uni_p_bh_update(self, sample, sigma_next, order):
        """
        Predictor step (UniP), adapted from diffusers' multistep_uni_p_bh_update
        with alpha=1.
        """
        m0 = self.model_outputs[-1]
        sigma_s0 = self.sigma_list[-1]

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_next)
        alpha_s0, sigma_s0_eff = self._sigma_to_alpha_sigma_t(sigma_s0)

        h = self._compute_h(sigma_next, sigma_s0)
        device = sample.device

        # hh = -h when predict_x0=True
        hh = -h
        h_phi_1 = torch.expm1(hh)     # e^{-h} - 1
        h_phi_k = h_phi_1 / hh - 1

        if self.solver_type == "bh1":
            B_h = hh
        else:  # bh2
            B_h = torch.expm1(hh)

        # warmup / first-order
        if order == 1 or self.model_outputs[-2] is None:
            x_t = sigma_t / sigma_s0_eff * sample - alpha_t * h_phi_1 * m0
            return x_t.to(sample.dtype)

        # order-2 predictor
        rks = []
        D1s = []
        for i in range(1, order):
            sigma_si = self.sigma_list[-(i + 1)]
            mi = self.model_outputs[-(i + 1)]

            h_i = self._compute_h(sigma_si, sigma_s0)
            rk = h_i / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(torch.tensor(1.0, device=device, dtype=sample.dtype))
        rks = torch.stack([rk if torch.is_tensor(rk) else torch.tensor(rk, device=device, dtype=sample.dtype) for rk in rks])

        R = []
        b = []
        factorial_i = 1
        h_phi_k_local = h_phi_k
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k_local * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k_local = h_phi_k_local / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.stack([bi if torch.is_tensor(bi) else torch.tensor(bi, device=device, dtype=sample.dtype) for bi in b])

        D1s = torch.stack(D1s, dim=1)  # [B, K, C, H, W] after einsum broadcasting logic

        if order == 2:
            # same simplification as diffusers
            rhos_p = torch.tensor([0.5], dtype=sample.dtype, device=device)
        else:
            rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(sample.dtype)

        x_t_ = sigma_t / sigma_s0_eff * sample - alpha_t * h_phi_1 * m0
        pred_res = torch.einsum("k,bk...->b...", rhos_p, D1s)
        x_t = x_t_ - alpha_t * B_h * pred_res
        return x_t.to(sample.dtype)

    @torch.no_grad()
    def multistep_uni_c_bh_update(self, this_model_output, last_sample, this_sample, sigma_cur, order):
        """
        Corrector step (UniC), adapted from diffusers' multistep_uni_c_bh_update
        with alpha=1.
        """
        m0 = self.model_outputs[-1]
        sigma_s0 = self.sigma_list[-1]

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_cur)
        alpha_s0, sigma_s0_eff = self._sigma_to_alpha_sigma_t(sigma_s0)

        h = self._compute_h(sigma_cur, sigma_s0)
        device = this_sample.device

        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1

        if self.solver_type == "bh1":
            B_h = hh
        else:
            B_h = torch.expm1(hh)

        rks = []
        D1s = []
        for i in range(1, order):
            sigma_si = self.sigma_list[-(i + 1)]
            mi = self.model_outputs[-(i + 1)]

            h_i = self._compute_h(sigma_si, sigma_s0)
            rk = h_i / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(torch.tensor(1.0, device=device, dtype=this_sample.dtype))
        rks = torch.stack([rk if torch.is_tensor(rk) else torch.tensor(rk, device=device, dtype=this_sample.dtype) for rk in rks])

        R = []
        b = []
        factorial_i = 1
        h_phi_k_local = h_phi_k
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k_local * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k_local = h_phi_k_local / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.stack([bi if torch.is_tensor(bi) else torch.tensor(bi, device=device, dtype=this_sample.dtype) for bi in b])

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None

        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=this_sample.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(R, b).to(device).to(this_sample.dtype)

        x_t_ = sigma_t / sigma_s0_eff * last_sample - alpha_t * h_phi_1 * m0

        if D1s is not None:
            corr_res = torch.einsum("k,bk...->b...", rhos_c[:-1], D1s)
        else:
            corr_res = 0

        D1_t = this_model_output - m0
        x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        return x_t.to(this_sample.dtype)

    @torch.no_grad()
    def step(self, model_output, sample, sigma_cur, sigma_next):
        """
        sample      : x at current sigma_cur
        model_output: net(sample, t_cur, ...) -> denoised/x0-like
        sigma_cur   : current sigma
        sigma_next  : next sigma
        """
        if self.total_steps is None:
            raise ValueError("Call set_total_steps(...) before sampling.")

        model_output = self.convert_model_output(model_output)

        # same logic as diffusers: correct current sample first (except warmup/disabled steps)
        use_corrector = (
            self.step_index > 0
            and self.step_index not in self.disable_corrector_steps
            and self.last_sample is not None
        )

        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output,
                last_sample=self.last_sample,
                this_sample=sample,
                sigma_cur=sigma_cur,
                order=self.this_order,
            )

        # shift history
        self.model_outputs[0] = self.model_outputs[1]
        self.sigma_list[0] = self.sigma_list[1]
        self.model_outputs[1] = model_output
        self.sigma_list[1] = sigma_cur

        # lower-order warmup/final fallback
        remaining_steps = self.total_steps - self.step_index
        if self.lower_order_final:
            target_order = min(2, remaining_steps)
        else:
            target_order = 2
        self.this_order = min(target_order, self.lower_order_nums + 1)

        # predictor
        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            sample=sample,
            sigma_next=sigma_next,
            order=self.this_order,
        )

        if self.lower_order_nums < 2:
            self.lower_order_nums += 1

        self.step_index += 1
        return prev_sample
