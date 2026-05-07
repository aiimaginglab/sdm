import os
import sys
import pickle
import argparse
import logging

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

import torch
import torchvision.utils as vutils
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

from modules.datasets import COCODataset
from modules.utils import set_seed, set_logger, save_json, collate_fn_coco, resample_sigmas_uniform_cum_eta
from modules.visualizer import append_jsonl, save_images
from modules.sampler_t2i import DPMSolverMultistepScheduler, adaptive_scheduling_t2i, calc_eta_distribution_t2i

"""eta-based N-step resampling algorithm for T2I models."""

def main(args):
    # settings
    device = 'cuda'
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    set_seed(args.seed)

    # save directory 
    if args.save_plot:
        save_dir = os.path.join(args.save_dir, f"{args.dataset_name}/{args.discretization}/{args.optimized_sigmas_exp}")
    elif args.n_step_resampling:
        save_dir = os.path.join(args.save_dir, f"{args.dataset_name}/{args.discretization}/{args.exp}")
    os.makedirs(save_dir, exist_ok=True)

    # logger settings
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)

    if args.save_plot:
        logger_path = os.path.join(save_dir, f"logs/save_plot_{args.optimized_sigmas_exp}.log")
    elif args.n_step_resampling:
        logger_path = os.path.join(save_dir, f"logs/{args.exp}.log")
    os.makedirs(os.path.dirname(logger_path), exist_ok=True)
    logger = set_logger(logger_path)

    # load model/datasets
    if args.pretrained_model == "runwayml/stable-diffusion-v1-5":
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
            if args.pretrained_model == "runwayml/stable-diffusion-v1-5":
                optimized_sigmas = [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]
            elif args.pretrained_model == "stabilityai/stable-diffusion-xl-base-1.0":
                optimized_sigmas = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]
        else:
            optimized_sigmas = None

    for batch_idx, batch in enumerate(dataloader):

        start_idx = batch_idx * B
        if start_idx >= args.num_samples:
            break

        end_idx = min(start_idx + B, args.num_samples)
        this_batch = end_idx - start_idx

        prompts = batch["prompts"]
        negative_prompts = batch["negative_prompts"]
        logger.info(f"\nProcessing batch index {batch_idx}, generating {this_batch} samples...")

        batch_seeds = [args.prev_num_samples + start_idx + i for i in range(this_batch)]

        # fix seed
        all_latents = []
        for seed in batch_seeds:
            set_seed(seed)
            lat = torch.randn(1, C, H, W, device=device, dtype=pipe.unet.dtype)
            all_latents.append(lat)
        latents = torch.cat(all_latents, dim=0)


        wasserstein_bound_log = calc_eta_distribution_t2i(
            pipe=pipe,
            latents=latents,
            prompts=prompts,
            negative_prompts=negative_prompts,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            sigmas=optimized_sigmas,
            prediction_type="epsilon",
            edm_style=False,
        )

    eta_log = wasserstein_bound_log["eta_log"] 


    if args.save_plot:
        plt.figure(figsize=(7, 4))
        plt.plot(np.arange(len(np.array(eta_log))), np.array(eta_log), label="eta value")
        plt.xlabel("step i")
        plt.ylabel("eta")
        plt.title("eta value over sampling steps")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "eta_value.png"), dpi=300, bbox_inches='tight')
        logger.info(f"Successfully saved plot to: {os.path.join(save_dir, 'eta_value.png')}")

    if args.n_step_resampling:
        # N-step resampling based on cumulative eta values
        resampled_sigmas_sdm = resample_sigmas_uniform_cum_eta(optimized_sigmas, eta_log, num_steps=args.resampled_num_steps, power=args.power, q=args.q)
        np.save(os.path.join(save_dir, f"optimized_schedules_{args.dataset_name}_{args.discretization}_steps_{args.resampled_num_steps}.npy"), resampled_sigmas_sdm)


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
    parser.add_argument("--exp", type=str, help="experiment name")
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
    parser.add_argument("--optimized_sigmas_save_dir", type=str, default="results/checkpoints_t2i", help="directory to load optimized sigmas from")
    parser.add_argument("--optimized_sigmas_exp", type=str, default="", help="experiment name for optimized sigmas to load")
    parser.add_argument("--load_optimized_sigmas", action='store_true', help="whether to load optimized sigmas")

    # n-step resampling 
    parser.add_argument("--save_plot", action='store_true', help="whether to save plot visualizing eta values")
    parser.add_argument("--n_step_resampling", action='store_true', help="whether to run N-step resampling algorithm")
    parser.add_argument("--resampled_num_steps", type=int, default=40, help="number of steps for given schedule")
    parser.add_argument("--power", type=float, default=0.5, help="geodesic arc length")
    parser.add_argument("--q", type=float, default=0.1, help="power value that determines the allocation of resampling steps")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)