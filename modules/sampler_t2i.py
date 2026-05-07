import math
import inspect
import logging
from sched import scheduler

import numpy as np 

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler as DefaultDPMSolver



@dataclass
class AdaptiveScheduleResult:
    optimized_sigmas: List[float]
    states: List[dict]
    latents: torch.Tensor

# basic helpers
def _scheduler_accepts_sigmas(scheduler) -> bool:
    return "sigmas" in inspect.signature(scheduler.set_timesteps).parameters

def _scheduler_accepts_timesteps(scheduler) -> bool:
    return "timesteps" in inspect.signature(scheduler.set_timesteps).parameters

def _batch_mean_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.reshape(x.shape[0], -1)
    return torch.sqrt(torch.clamp((x * x).mean(dim=1), min=eps)).mean()

def _find_bin_index_descending(sigma_grid: torch.Tensor, sigma_val: float) -> int:
    sigma_val = float(sigma_val)
    for i in range(len(sigma_grid) - 1):
        if float(sigma_grid[i]) >= sigma_val > float(sigma_grid[i + 1]):
            return i
    return len(sigma_grid) - 2

def _is_sdxl_pipe(pipe) -> bool:
    if isinstance(pipe, StableDiffusionXLPipeline):
        return True
    return False


# add support for setting custom timesteps/sigmas
class DPMSolverMultistepScheduler(DefaultDPMSolver):
    def set_timesteps(
        self,
        num_inference_steps=None,
        device=None,
        timesteps=None,
        sigmas=None,
    ):
        if timesteps is None and sigmas is None:
            super().set_timesteps(num_inference_steps=num_inference_steps, device=device)
            return

        if timesteps is not None and sigmas is not None:
            raise ValueError("Pass only one of `timesteps` or `sigmas`.")

        all_sigmas = np.array(((1 - self.alphas_cumprod.cpu().numpy()) / self.alphas_cumprod.cpu().numpy()) ** 0.5)

        # -----------------------------
        # Case 1: custom timesteps
        # -----------------------------
        if timesteps is not None:
            timesteps = np.array(timesteps, dtype=np.int64)

            self.timesteps = torch.tensor(timesteps[:-1], device=device, dtype=torch.int64)
            self.sigmas = torch.from_numpy(all_sigmas[timesteps]).to(torch.float32)

        # -----------------------------
        # Case 2: custom sigmas
        # -----------------------------
        if sigmas is not None:
            if isinstance(sigmas, torch.Tensor):
                sigmas = sigmas.detach().cpu().numpy()
            else:
                sigmas = np.array(sigmas, dtype=np.float32)

            # custom sigma -> nearest timestep index mapping
            log_all_sigmas = np.log(np.maximum(all_sigmas, 1e-12))
            # diffusers-style continuous interpolation
            log_sigmas = np.log(np.maximum(sigmas[:-1], 1e-12))   # remove the last timestep 0

            timestep_list = []
            for log_sigma in log_sigmas:
                dists = np.abs(log_all_sigmas - log_sigma)
                idx = int(np.argmin(dists))
                timestep_list.append(idx)

            self.timesteps = torch.tensor(timestep_list, device=device, dtype=torch.int64)
            self.sigmas = torch.from_numpy(sigmas).to(torch.float32)

        self.num_inference_steps = len(self.timesteps)

        self.model_outputs = [None] * self.config.solver_order
        self.lower_order_nums = 0

        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")


# prompt embeddings/latents/add_time_ids
@torch.no_grad()
def _encode_prompt(
    pipe,
    prompts,
    negative_prompts,
    guidance_scale: float,
    device,
):
    do_cfg = guidance_scale > 1.0

    # SDXL
    if _is_sdxl_pipe(pipe):
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompts,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )

        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        return prompt_embeds, pooled_prompt_embeds, do_cfg

    # SD 1.5
    else:
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompts,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompts,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )

        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        return prompt_embeds, None, do_cfg

def _get_add_time_ids(
    pipe,
    original_size,
    target_size,
    crops_coords_top_left,
    dtype,
    device,
    batch_size: int,
    do_cfg: bool,
):
    if not _is_sdxl_pipe(pipe):
        return None

    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    add_time_ids = add_time_ids.repeat(batch_size, 1)

    if do_cfg:
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    return add_time_ids

@torch.no_grad()
def _prepare_latents(
    pipe: StableDiffusionPipeline,
    batch_size: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    generator=None,
    latents: Optional[torch.Tensor] = None,
):
    if latents is None:
        shape = (
            batch_size,
            pipe.unet.config.in_channels,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        )
        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    latents = latents * pipe.scheduler.init_noise_sigma
    return latents


# sigma / timestep conversions
@torch.no_grad()
def _sigma_to_alpha_sigma_t_common(
    sigma: torch.Tensor,
    edm_style: bool = False,
):
    """
    edm_style=False:
        VP-style external sigma convention used by diffusers sigma schedulers
        alpha_t = 1 / sqrt(1 + sigma^2)
        sigma_t = sigma * alpha_t

    edm_style=True:
        EDM / VE-style sigma convention
        alpha_t = 1
        sigma_t = sigma
    """
    if edm_style:
        alpha_t = torch.ones_like(sigma)
        sigma_t = sigma
    else:
        alpha_t = 1.0 / torch.sqrt(1.0 + sigma**2)
        sigma_t = sigma * alpha_t
    return alpha_t, sigma_t

# reconstruct x0/compute drift from model output
@torch.no_grad()
def reconstruct_x0_from_model_output(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    sigma: float | torch.Tensor,
    prediction_type: str = "epsilon",
    edm_style: bool = False,
):
    """
    Reconstruct x0_pred from model output.
    """
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=sample.device, dtype=sample.dtype)

    while sigma.ndim < sample.ndim:
        sigma = sigma.unsqueeze(-1)

    alpha_t, sigma_t = _sigma_to_alpha_sigma_t_common(sigma, edm_style=edm_style)

    if prediction_type == "epsilon":
        x0_pred = (sample - sigma_t * model_output) / torch.clamp(alpha_t, min=1e-12)
    elif prediction_type == "sample":
        x0_pred = model_output
    elif prediction_type == "v_prediction":
        x0_pred = alpha_t * sample - sigma_t * model_output
    else:
        raise ValueError(f"Unsupported prediction_type: {prediction_type}")

    return x0_pred

@torch.no_grad()
def compute_drift_from_model_output(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    sigma: float | torch.Tensor,
    prediction_type: str = "epsilon",
    edm_style: bool = False,
    eps: float = 1e-12,
):
    """
    Drift proxy:
        d = (x - x0_pred) / sigma
    """
    x0_pred = reconstruct_x0_from_model_output(
        sample=sample,
        model_output=model_output,
        sigma=sigma,
        prediction_type=prediction_type,
        edm_style=edm_style,
    )

    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=sample.device, dtype=sample.dtype)

    while sigma.ndim < sample.ndim:
        sigma = sigma.unsqueeze(-1)

    drift = (sample - x0_pred) / torch.clamp(sigma, min=eps)
    return drift, x0_pred


# model prediction
@torch.no_grad()
def _predict_model_output_cfg(
    pipe,
    scheduler,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    do_cfg: bool,
    guidance_scale: float,
    pooled_prompt_embeds: torch.Tensor | None = None,
    add_time_ids: torch.Tensor | None = None,
):
    latent_model_input = latents
    if do_cfg:
        latent_model_input = torch.cat([latents, latents], dim=0)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    if _is_sdxl_pipe(pipe):
        model_output = pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            },
            return_dict=True,
        ).sample
    else:
        model_output = pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=True,
        ).sample

    if do_cfg:
        model_output_uncond, model_output_text = model_output.chunk(2)
        model_output = model_output_uncond + guidance_scale * (model_output_text - model_output_uncond)

    return model_output

#----------------------------------------------------------------------------
# T2I sampler with adaptive scheduling

@torch.no_grad()
def _clone_scheduler(
    pipe: StableDiffusionPipeline,
    device: torch.device,
    sigmas=None,
    timesteps=None,
):
    scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)

    if sigmas is not None and timesteps is not None:
        raise ValueError("Pass only one of sigmas or timesteps.")

    if sigmas is not None:
        if not _scheduler_accepts_sigmas(scheduler):
            raise ValueError(f"{scheduler.__class__.__name__} does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device)

    elif timesteps is not None:
        if not _scheduler_accepts_timesteps(scheduler):
            raise ValueError(f"{scheduler.__class__.__name__} does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device)

    else:
        raise ValueError("One of sigmas or timesteps must be provided.")

    return scheduler

@torch.no_grad()
def _one_sigma_probe_step(
    pipe,
    x_cur: torch.Tensor,
    sigma_cur: float,
    sigma_next: float,
    prompt_embeds: torch.Tensor,
    do_cfg: bool,
    guidance_scale: float,
    device,
    pooled_prompt_embeds: torch.Tensor | None = None,
    add_time_ids: torch.Tensor | None = None,
):
    scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)

    if not _scheduler_accepts_sigmas(scheduler):
        raise ValueError(f"{scheduler.__class__.__name__} does not support custom sigmas.")

    scheduler.set_timesteps(sigmas=[float(sigma_cur), float(sigma_next), 0.0], device=device)

    t_cur = scheduler.timesteps[0]
    x_in = x_cur.to(device=device, dtype=pipe.unet.dtype)

    model_out_cur = _predict_model_output_cfg(
        pipe=pipe,
        scheduler=scheduler,
        latents=x_in,
        timestep=t_cur,
        prompt_embeds=prompt_embeds,
        do_cfg=do_cfg,
        guidance_scale=guidance_scale,
        pooled_prompt_embeds=pooled_prompt_embeds,
        add_time_ids=add_time_ids,
    )

    x_next = scheduler.step(model_out_cur, t_cur, x_in, return_dict=True).prev_sample

    t_next = scheduler.timesteps[1]

    model_out_next = _predict_model_output_cfg(
        pipe=pipe,
        scheduler=scheduler,
        latents=x_next,
        timestep=t_next,
        prompt_embeds=prompt_embeds,
        do_cfg=do_cfg,
        guidance_scale=guidance_scale,
        pooled_prompt_embeds=pooled_prompt_embeds,
        add_time_ids=add_time_ids,
    )

    return x_next, model_out_cur, model_out_next

@torch.no_grad()
def adaptive_scheduling_t2i(
    pipe: StableDiffusionPipeline,
    latents: torch.Tensor,
    prompts: List[str],
    negative_prompts: List[str],
    num_steps: int = 10,
    height: int = 512,
    width: int = 512,
    guidance_scale: float = 7.5,
    generator=None,
    eta_min: float = 0.02,
    eta_max: float = 0.20,
    eta_p: float = 1.0,
    gamma_step: float = 0.95,
    c_shrink: float = 0.5,
    max_expand_bins: int = 2,
    max_shrink_iters: int = 2,
    use_grid_snap: bool = True,
    prediction_type: str = "epsilon",
    edm_style: bool = False,
    use_karras_sigmas: bool = True,
    use_ays_sigmas: bool = False,
    eps: float = 1e-12,
) -> AdaptiveScheduleResult:
    """
    Adaptive Scheduling for T2I.
      - use one-step scheduler probe
      - computes S_hat from drift 
    """
    device = pipe._execution_device
    dtype = pipe.unet.dtype

    prompt_embeds, pooled_prompt_embeds, do_cfg = _encode_prompt(
        pipe=pipe,
        prompts=prompts,
        negative_prompts=negative_prompts,
        guidance_scale=guidance_scale,
        device=device,
    )

    batch_size = len(prompts)
    x_cur = _prepare_latents(
        pipe=pipe,
        batch_size=batch_size,
        height=height,
        width=width,
        dtype=dtype,
        device=device,
        generator=generator,
        latents=latents,
    ).to(device=device, dtype=pipe.unet.dtype)

    add_time_ids = _get_add_time_ids(
        pipe=pipe,
        original_size=(height, width),
        target_size=(height, width),
        crops_coords_top_left=(0, 0),
        dtype=prompt_embeds.dtype,
        device=device,
        batch_size=len(prompts),
        do_cfg=do_cfg,
    )

    if use_karras_sigmas:
        scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    else:
        scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config) # base sigma grid from the current DPM-Solver++ scheduler config
    
    if not _scheduler_accepts_sigmas(scheduler):
        raise ValueError(
            f"{scheduler.__class__.__name__} does not support custom sigmas."
        )

    if use_ays_sigmas:
        scheduler.set_timesteps(sigmas=[14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029], device=device)
    else:
        scheduler.set_timesteps(num_inference_steps=num_steps, device=device)

    sigma_grid = scheduler.sigmas.detach().to(torch.float64).cpu()
    if float(sigma_grid[-1]) == 0.0:
        sigma_grid_main = sigma_grid[:-1]
    else:
        sigma_grid_main = sigma_grid

    logging.info(f"timesteps: {scheduler.timesteps.tolist()}")
    logging.info(f"sigmas: {scheduler.sigmas.tolist()}")

    sigma_max = float(sigma_grid_main[0])
    sigma_min = float(sigma_grid_main[-1])

    optimized_sigmas = [sigma_max]
    states = []
    nfe = 0

    sig_cur = sigma_max

    def eta_scheduler(sig: float) -> float:
        if sigma_max <= 0:
            return float(eta_min)
        r = max(0.0, min(1.0, float(sig) / sigma_max))
        return float(eta_min + (eta_max - eta_min) * (r ** eta_p))

    while sig_cur > sigma_min + 1e-15:
        idx = _find_bin_index_descending(sigma_grid_main, sig_cur)

        logging.info(f"sigma_grid_main: {[float(x) for x in sigma_grid_main.tolist()]}")
        logging.info(f"sig_cur={sig_cur}, idx={idx}, len_grid={len(sigma_grid_main)}")
        logging.info(f"grid[idx]={float(sigma_grid_main[idx])}, grid[idx+1]={float(sigma_grid_main[min(idx+1, len(sigma_grid_main)-1)])}")

        sig_next_bin = float(sigma_grid_main[min(idx + 1, len(sigma_grid_main) - 1)])
        delta_sigma_init = max(sig_cur - sig_next_bin, 0.0)
        delta_sigma_trial = min(sig_cur - sigma_min, delta_sigma_init)

        eta = eta_scheduler(sig_cur)

        best_S = None
        best_delta_sigma = None
        best_sig_trial = None

        expand_iters = 0
        shrink_iters = 0
        prev_sig_trial = None
        shrink_flag = False

        while True:
            logging.info(f"delta_sigma_trial: {delta_sigma_trial}, expand_iters: {expand_iters}, shrink_iters: {shrink_iters}")
            sig_trial = max(sigma_min, sig_cur - delta_sigma_trial)

            if use_grid_snap:
                candidates = sigma_grid_main[sigma_grid_main <= sig_trial]
                sig_trial = float(candidates[0]) if len(candidates) > 0 else sigma_min

            logging.info(f"sig_hat: {float(sig_cur)}, prev_sig_trial: {prev_sig_trial} -> sig_trial: {float(sig_trial)}")
            if prev_sig_trial is not None and abs(sig_trial - prev_sig_trial) < 1e-15:
                break
            prev_sig_trial = sig_trial

            # probe step
            x_trial, model_out_cur, model_out_trial = _one_sigma_probe_step(
                pipe=pipe,
                x_cur=x_cur,
                sigma_cur=sig_cur,
                sigma_next=sig_trial,
                prompt_embeds=prompt_embeds,
                do_cfg=do_cfg,
                guidance_scale=guidance_scale,
                device=device,
                pooled_prompt_embeds=pooled_prompt_embeds,
                add_time_ids=add_time_ids,
            )

            nfe += 2

            # drift-based S_hat
            drift_cur, _ = compute_drift_from_model_output(
                sample=x_cur,
                model_output=model_out_cur,
                sigma=sig_cur,
                prediction_type=prediction_type,
                edm_style=edm_style,
                eps=eps,
            )

            drift_trial, _ = compute_drift_from_model_output(
                sample=x_trial,
                model_output=model_out_trial,
                sigma=sig_trial,
                prediction_type=prediction_type,
                edm_style=edm_style,
                eps=eps,
            )

            S_hat = float(
                _batch_mean_norm(drift_trial - drift_cur) /
                max(abs(sig_trial - sig_cur), eps)
            )

            eta_trial = 0.5 * ((sig_cur - sig_trial) ** 2) * S_hat

            if eta_trial <= eta:
                best_S = S_hat
                best_delta_sigma = sig_cur - sig_trial
                best_sig_trial = sig_trial

            # expand
            if (eta_trial <= eta) and (expand_iters < max_expand_bins):
                if shrink_flag:
                    break
                jump = 2 ** (expand_iters + 1)
                if idx + jump > len(sigma_grid_main) - 1:
                    break

                sig_next_bin = float(sigma_grid_main[min(idx + jump, len(sigma_grid_main) - 1)])
                new_delta = min(sig_cur - sigma_min, max(sig_cur - sig_next_bin, 0.0))

                if new_delta <= delta_sigma_trial + 1e-15:
                    break

                delta_sigma_trial = new_delta
                expand_iters += 1
                continue

            # shrink
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
            best_delta_sigma = sig_cur - sig_trial
            best_sig_trial = sig_trial

        # next step
        delta_sigma_next = gamma_step * math.sqrt(2.0 * eta / max(best_S, 1e-30))
        delta_sigma_next = min(sig_cur - sigma_min, max(best_delta_sigma, delta_sigma_next))
        sig_next = max(sigma_min, sig_cur - delta_sigma_next)

        if use_grid_snap:
            candidates = sigma_grid_main[sigma_grid_main <= sig_next]
            sig_next = float(candidates[0]) if len(candidates) > 0 else sigma_min

        # commit one actual step on the search trajectory
        x_next, _, _ = _one_sigma_probe_step(
            pipe=pipe,
            x_cur=x_cur,
            sigma_cur=sig_cur,
            sigma_next=sig_next,
            prompt_embeds=prompt_embeds,
            do_cfg=do_cfg,
            guidance_scale=guidance_scale,
            device=device,
            pooled_prompt_embeds=pooled_prompt_embeds,
            add_time_ids=add_time_ids,
        )

        nfe += 2

        states.append(
            {
                "sigma_cur": float(sig_cur),
                "sigma_next": float(sig_next),
                "best_S": float(best_S),
                "best_delta_sigma": float(best_delta_sigma),
                "eta": float(eta),
                "expand_iters": int(expand_iters),
                "shrink_iters": int(shrink_iters),
                "nfe": int(nfe),
            }
        )

        x_cur, sig_cur = x_next, sig_next
        optimized_sigmas.append(float(sig_cur))

        if sig_cur <= sigma_min + 1e-15:
            break

    optimized_sigmas[0] = sigma_max
    optimized_sigmas[-1] = sigma_min
    optimized_sigmas.append(0.0)

    return AdaptiveScheduleResult(
        optimized_sigmas=optimized_sigmas,
        states=states,
        latents=x_cur,
    )


@torch.no_grad()
def calc_eta_distribution_t2i(
    pipe,
    latents: torch.Tensor,
    prompts,
    negative_prompts,
    num_steps: int = 10,
    guidance_scale: float = 7.5,
    sigmas=None,
    timesteps=None,
    prediction_type: str = "epsilon",
    edm_style: bool = False,
    eps: float = 1e-12,
):
    """
    Measure per-step eta distribution on a fixed T2I schedule.

    T2I counterpart of calc_wasserstein_bound(...) in sampler.py:
      - fixed baseline schedule (EDM / AYS / custom)
      - one-step Euler probe
      - drift-based S_hat
      - eta = 0.5 * (delta_sigma)^2 * S_hat
    """
    device = pipe._execution_device
    dtype = pipe.unet.dtype

    # latents: [B, 4, H/8, W/8]
    latent_h = latents.shape[-2]
    latent_w = latents.shape[-1]
    height = latent_h * pipe.vae_scale_factor
    width = latent_w * pipe.vae_scale_factor

    prompt_embeds, pooled_prompt_embeds, do_cfg = _encode_prompt(
        pipe=pipe,
        prompts=prompts,
        negative_prompts=negative_prompts,
        guidance_scale=guidance_scale,
        device=device,
    )

    # SDXL only; SD1.5 returns None
    add_time_ids = _get_add_time_ids(
        pipe=pipe,
        original_size=(height, width),
        target_size=(height, width),
        crops_coords_top_left=(0, 0),
        dtype=prompt_embeds.dtype,
        device=device,
        batch_size=len(prompts),
        do_cfg=do_cfg,
    )

    x_cur = latents.to(device=device, dtype=dtype)

    scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)

    if sigmas is not None and timesteps is not None:
        raise ValueError("Pass only one of `sigmas` or `timesteps`.")

    if sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device)
    elif timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device)
    else:
        scheduler.set_timesteps(num_inference_steps=num_steps, device=device)

    logging.info(f"timesteps: {scheduler.timesteps.tolist()}")
    logging.info(f"sigmas: {scheduler.sigmas.tolist()}")

    # scheduler.sigmas: length N+1
    sigma_seq = scheduler.sigmas.detach().to(torch.float64).cpu().numpy().tolist()
    timestep_seq = scheduler.timesteps.detach().cpu()

    eta_log = []
    S_hat_log = []
    dsigma_log = []
    sigma_cur_log = []
    sigma_next_log = []

    # iterate over actual denoising steps
    for i, t_cur in enumerate(timestep_seq):
        sigma_cur = float(sigma_seq[i])
        sigma_next = float(sigma_seq[i + 1])

        x_in = x_cur.to(device=device, dtype=dtype)

        # current model output
        model_out_cur = _predict_model_output_cfg(
            pipe=pipe,
            scheduler=scheduler,
            latents=x_in,
            timestep=t_cur,
            prompt_embeds=prompt_embeds,
            do_cfg=do_cfg,
            guidance_scale=guidance_scale,
            pooled_prompt_embeds=pooled_prompt_embeds,
            add_time_ids=add_time_ids,
        )

        # current drift
        drift_cur, _ = compute_drift_from_model_output(
            sample=x_in,
            model_output=model_out_cur,
            sigma=sigma_cur,
            prediction_type=prediction_type,
            edm_style=edm_style,
            eps=eps,
        )

        # one-step scheduler update
        x_next = scheduler.step(model_out_cur, t_cur, x_in, return_dict=True).prev_sample

        dsigma = abs(sigma_cur - sigma_next)

        if sigma_next <= eps:
            eta_log.append(0.0)
            S_hat_log.append(0.0)
            dsigma_log.append(float(dsigma))
            sigma_cur_log.append(float(sigma_cur))
            sigma_next_log.append(float(sigma_next))
            x_cur = x_next
            continue

        # evaluate next drift at the updated point
        t_next = timestep_seq[min(i + 1, len(timestep_seq) - 1)]
        model_out_next = _predict_model_output_cfg(
            pipe=pipe,
            scheduler=scheduler,
            latents=x_next,
            timestep=t_next,
            prompt_embeds=prompt_embeds,
            do_cfg=do_cfg,
            guidance_scale=guidance_scale,
            pooled_prompt_embeds=pooled_prompt_embeds,
            add_time_ids=add_time_ids,
        )

        drift_next, _ = compute_drift_from_model_output(
            sample=x_next,
            model_output=model_out_next,
            sigma=sigma_next,
            prediction_type=prediction_type,
            edm_style=edm_style,
            eps=eps,
        )

        S_hat = float(_batch_mean_norm(drift_next - drift_cur) / max(dsigma, eps))
        eta = 0.5 * (dsigma ** 2) * S_hat

        eta_log.append(float(eta))
        S_hat_log.append(float(S_hat))
        dsigma_log.append(float(dsigma))
        sigma_cur_log.append(float(sigma_cur))
        sigma_next_log.append(float(sigma_next))

        x_cur = x_next

    return {
        "eta_log": eta_log,
        "S_hat_log": S_hat_log,
        "dsigma_log": dsigma_log,
        "sigma_cur_log": sigma_cur_log,
        "sigma_next_log": sigma_next_log,
    }


#----------------------------------------------------------------------------
# Improved T2I sampler

@torch.no_grad()
def base_sampler_t2i(
    pipe: StableDiffusionPipeline,
    latents: torch.Tensor,
    prompts,
    negative_prompts,
    sigmas: Optional[List[float]] = None,
    timesteps: Optional[List[int]] = None,
    discretization: str | None = None,
    **kwargs,
):
    if sigmas is not None and timesteps is not None:
        raise ValueError("Pass only one of sigmas or timesteps.")

    if discretization == "edm":
        pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    
    return pipe(
        prompt=prompts,
        negative_prompt=negative_prompts,
        latents=latents,
        sigmas=sigmas,
        timesteps=timesteps,
        **kwargs,
    ).images