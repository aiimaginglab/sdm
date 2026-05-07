import os
import time
import pickle
import argparse
import logging

import numpy as np
from tqdm import trange

import torch
import torchvision.utils as vutils
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

from modules.datasets import COCODataset
from modules.utils import set_seed, set_logger, save_json, collate_fn_coco
from modules.visualizer import append_jsonl, save_coco_images
from modules.sampler_t2i import DPMSolverMultistepScheduler, adaptive_scheduling_t2i, base_sampler_t2i

"""Generate samples for Stable Diffusion 1.5, SDXL using EDM, AYS, SDM sampler"""

def main(args):
    # settings
    device = 'cuda'
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    set_seed(args.seed)

    # save directory 
    save_dir = os.path.join(args.save_dir, f"{args.dataset_name}/{args.discretization}/{args.exp}")
    os.makedirs(save_dir, exist_ok=True)

    # logger settings
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)

    logger_path = os.path.join(save_dir, f"logs/{args.exp}.log")
    os.makedirs(os.path.dirname(logger_path), exist_ok=True)
    logger = set_logger(logger_path)

    # load model/datasets
    if args.pretrained_model == "runwayml/stable-diffusion-v1-5" or args.pretrained_model == 'lykon/dreamshaper-8':
        pipe = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model,
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model,
            torch_dtype=torch.float16,
        ).to("cuda")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,    
        algorithm_type="dpmsolver++",
        solver_order=2, 
    )  

    net = pipe.unet
    vae = pipe.vae
    scheduler = pipe.scheduler

    logger.info("Successfully loaded Stable Diffusion 1.5")
    logger.info("Scheduler type: {}".format(type(scheduler).__name__))

    dataset = COCODataset(args.subset_json)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_coco,
    )

    logger.info("Successfully loaded dataset/dataloader")

    B = args.batch_size
    C, H, W = args.channel_size, args.image_size//args.downscale_factor, args.image_size//args.downscale_factor
    num_batches = (args.num_samples + B - 1) // B

    if args.load_optimized_sigmas:
        optimized_sigmas = np.load(os.path.join(BASE_PATH, f'{args.optimized_sigmas_save_dir}/{args.dataset_name}/{args.discretization}/{args.optimized_sigmas_exp}/optimized_schedules_{args.dataset_name}_{args.discretization}_steps_{args.num_steps}.npy'))
        logger.info(f"Loaded optimized sigmas: {optimized_sigmas}")
    else: 
        if args.discretization == "ays":
            if args.pretrained_model == "runwayml/stable-diffusion-v1-5"  or args.pretrained_model == 'lykon/dreamshaper-8':
                optimized_sigmas = [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]
            elif args.pretrained_model == "stabilityai/stable-diffusion-xl-base-1.0":
                optimized_sigmas = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]
        else:
            optimized_sigmas = None

    # samplers 
    sampler_map = {
        # dpm solver (2m)
        "dpm_solver": lambda latents, prompts, negative_prompts: base_sampler_t2i(
            pipe, 
            latents=latents,
            prompts=prompts,
            negative_prompts=negative_prompts,
            sigmas=optimized_sigmas,
            guidance_scale=args.guidance_scale,
            discretization=args.discretization,
            output_type="latent",
        ),

        # adaptive scheduling
        "sdm_scheduler": lambda latents, prompts, negative_prompts: adaptive_scheduling_t2i(
            pipe,
            latents=latents,
            prompts=prompts,
            negative_prompts=negative_prompts,
            num_steps=args.num_steps,
            height=args.image_size,
            width=args.image_size,
            guidance_scale=args.guidance_scale,
            eta_min=args.eta_min,
            eta_max=args.eta_max,
            eta_p=args.eta_p,
            gamma_step=args.gamma_step,
            c_shrink=args.c_shrink,
            max_expand_bins=args.max_expand_bins,
            max_shrink_iters=args.max_shrink_iters,
            use_grid_snap=args.use_grid_snap,
            prediction_type="epsilon",
            edm_style=False,
        )
    }

    assert args.sampler_name in sampler_map, f"Unknown sampler: {args.sampler_name}"
    sampler_fn = sampler_map[args.sampler_name]
    logger.info(f"Running sampler: {args.sampler_name}")

    for batch_idx, batch in enumerate(dataloader):

        start_idx = batch_idx * B
        if start_idx >= args.num_samples:
            break
     
        end_idx = min(start_idx + B, args.num_samples)
        this_batch = end_idx - start_idx

        prompts = batch["prompts"][:this_batch]
        negative_prompts = batch["negative_prompts"][:this_batch]
        file_names = batch["file_names"][:this_batch]

        logger.info(f"\nProcessing batch index {batch_idx}, generating {this_batch} samples...")

        batch_seeds = [args.prev_num_samples + start_idx + i for i in range(this_batch)]

        # fix seed
        all_latents = []
        for seed in batch_seeds:
            set_seed(seed)
            lat = torch.randn(1, C, H, W, device=device, dtype=pipe.unet.dtype)
            all_latents.append(lat)
        latents = torch.cat(all_latents, dim=0)

        with torch.no_grad():
            if args.sampler_name == "sdm_scheduler":
                x = sampler_fn(latents, prompts, negative_prompts)
                
                optimized_sigmas_sdm = x.optimized_sigmas 
                states = x.states
                x_latent = x.latents

                logger.info("optimized sigmas sdm: %s", optimized_sigmas_sdm)

            else: 
                x = sampler_fn(latents, prompts, negative_prompts)

        if args.sampler_name == "sdm_scheduler":
            save_json(states, os.path.join(save_dir, f'log-{args.exp}-batch{batch_idx}.json'))
            save_json(vars(args), os.path.join(save_dir, 'config.json'))

            optimized_num_steps = len(optimized_sigmas_sdm) - 1
            logger.info(f"Length of optimized_sigmas_iedm: {optimized_num_steps}")
            logger.info(f"    Optimized sigmas iedm: {optimized_sigmas_sdm}")
            np.save(os.path.join(save_dir, f"optimized_schedules_{args.dataset_name}_{args.discretization}_steps_{optimized_num_steps}.npy"), optimized_sigmas_sdm)

        if args.save_images:
            x_latent = x.to(device=pipe.vae.device)

            # SDXL decoder
            if args.pretrained_model == "stabilityai/stable-diffusion-xl-base-1.0":
                pipe.upcast_vae()
                x_latent = x_latent.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
            else:
                x_latent = x_latent.to(dtype=pipe.vae.dtype)

            x = pipe.vae.decode(
                x_latent / pipe.vae.config.scaling_factor,
                return_dict=False,
            )[0]

            save_coco_images(x, save_dir, batch_seeds, file_names, subdirs=False)

    logger.info(f"Done: {args.sampler_name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)

    # datasets/models
    parser.add_argument("--pretrained_model", type=str, help="pretrained t2i model name")
    parser.add_argument("--dataset_name", type=str, help="dataset name")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for dataloader")

    parser.add_argument("--channel_size", type=int, default=4, help="number of image channels")
    parser.add_argument("--image_size", type=int, default=512, help="image size")
    parser.add_argument("--downscale_factor", type=int, default=8, help="downscale factor for Stable Diffusion")

    # schedule 
    parser.add_argument("--num_steps", type=int, default=40, help="number of steps for given schedule")
    parser.add_argument("--sigma_min", type=float, default=0.002, help="minimum sigma for given schedule")
    parser.add_argument("--sigma_max", type=float, default=80, help="maximum sigma for given schedule")
    parser.add_argument("--discretization", type=str, help="discretization type")
    parser.add_argument("--schedule", type=str, default="vp", choices=['vp', 've', 'linear'], help="noise schedule type")
    parser.add_argument("--scaling", type=str, default="none", choices=['vp', 'none'], help="scaling type for noise schedule")
    parser.add_argument("--rho", type=float, default=7.0, help="rho parameter for edm discretization")

    # stochastic settings 
    parser.add_argument("--S_churn", type=float, default=0.0, help="S_churn parameter for adaptive solver")
    parser.add_argument("--S_min", type=float, default=0.0, help="S_min parameter for adaptive solver")
    parser.add_argument("--S_max", type=float, default=float("inf"), help="S_max parameter for adaptive solver")
    parser.add_argument("--S_noise", type=float, default=1.0, help="S_noise parameter for adaptive solver")
    
    # sampling 
    parser.add_argument("--exp", type=str, required=True, help="experiment name")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--prev_num_samples", type=int, default=0, help="number of previously generated samples")
    parser.add_argument("--num_samples", type=int, default=50000, help="number of samples to generate for FID")
    parser.add_argument("--sampler_name", type=str, default="adaptive", help="sampler name for ablation study")
    parser.add_argument("--save_dir", type=str, default="results/fid_results", help="directory to save model results")
    parser.add_argument("--solver", type=str, default="heun", help="ODE solver type")

    parser.add_argument("--guidance_scale", type=float, default=7.5, help="guidance scale for classifier-free guidance")
    parser.add_argument("--subset_json", type=str, required=True, help="Path to subset.json containing file_name/prompt pairs")

    # save images/calc curvature 
    parser.add_argument("--save_images", action='store_true', help="whether to save generated images")

    # load optimized_sigmas
    parser.add_argument("--optimized_sigmas_save_dir", type=str, default="results/checkpoints", help="directory to load optimized sigmas from")
    parser.add_argument("--optimized_sigmas_exp", type=str, default="", help="experiment name for optimized sigmas to load")
    parser.add_argument("--load_optimized_sigmas", action='store_true', help="whether to load optimized sigmas")

    # adaptive solver 
    parser.add_argument("--use_relative_metrics", action='store_false', help="whether to use relative metrics for adaptive solver")
    parser.add_argument("--tau_k", type=float, default=1.0, help="tolerance parameter for adaptive solver")
    parser.add_argument("--lambda_schedule", type=str, default="step", choices=['step', 'linear', 'cosine'], help="lambda scheduling type for adaptive solver")

    # adaptive scheduling 
    parser.add_argument("--eta_min", type=float, default=0.02, help="eta_min parameter for eta scheduling")
    parser.add_argument("--eta_max", type=float, default=0.20, help="eta_max parameter for eta scheduling")
    parser.add_argument("--eta_p", type=float, default=1.0, help="eta_p parameter for eta scheduling")    
    parser.add_argument("--gamma_step", type=float, default=0.95, help="safety factor for delta_sigma_trial")
    parser.add_argument("--c_shrink", type=float, default=0.5, help="shrinking factor for delta_sigma_trial")
    parser.add_argument("--max_expand_bins", type=int, default=2, help="maximum number of bins to try to expand at most")
    parser.add_argument("--max_shrink_iters", type=int, default=2, help="maximum number of iters to shrink at most")
    parser.add_argument("--use_grid_snap", action='store_true', help="whether to use grid snap for adaptive scheduling")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)