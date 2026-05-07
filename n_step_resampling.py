import os 
import pickle 
import argparse
import logging

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

import torch 

from modules.utils import set_seed, set_logger, resample_sigmas_uniform_cum_eta
from modules.sampler import calc_wasserstein_bound

"""eta-based N-step resampling algorithm."""

def main(args):
    # settings
    device = 'cuda'
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    set_seed(args.seed)

    if args.save_plot:
        save_dir = os.path.join(args.save_dir, f"{args.dataset_name}/{args.discretization}/{args.optimized_sigmas_exp}")
    elif args.n_step_resampling:
        save_dir = os.path.join(args.save_dir, f"{args.dataset_name}/{args.discretization}/{args.exp}")
    os.makedirs(save_dir, exist_ok=True)

    if args.save_plot:
        logger_path = os.path.join(save_dir, f"logs/save_plot_{args.optimized_sigmas_exp}.log")
    elif args.n_step_resampling:
        logger_path = os.path.join(save_dir, f"logs/{args.exp}.log")
    os.makedirs(os.path.dirname(logger_path), exist_ok=True)
    logger = set_logger(logger_path)

    path_to_pretrained_model = os.path.join(BASE_PATH, "dnnlib/models")

    with open(os.path.join(path_to_pretrained_model, args.model_weights_pkl), 'rb') as f:
        data = pickle.load(f)

    logger.info("Successfully loaded pretrained model...")
    net = data['ema'].to(device)

    os.makedirs(args.save_dir, exist_ok=True)
    
    B = args.batch_size
    C, H, W = args.channel_size, args.image_size, args.image_size # CIFAR10 - 3, 32, 32 # FFHQ, AFHQv2 - 3, 64, 64
    num_batches = (args.num_samples + B - 1) // B

    if args.load_optimized_sigmas:
        optimized_sigmas_iedm = np.load(os.path.join(BASE_PATH, f'{args.optimized_sigmas_save_dir}/{args.dataset_name}/{args.discretization}/{args.optimized_sigmas_exp}/optimized_schedules_{args.dataset_name}_{args.discretization}_steps_{args.num_steps}.npy'))
        logger.info(f"Loaded optimized sigmas: {optimized_sigmas_iedm}")
    else:
        optimized_sigmas_iedm = None

    for batch_idx in trange(num_batches):
        start_idx = batch_idx * B
        end_idx = min(start_idx + B, args.num_samples)
        this_batch = end_idx - start_idx

        logger.info(f"\nProcessing batch index {batch_idx}, generating {this_batch} samples...")

        batch_seeds = [args.prev_num_samples + start_idx + i for i in range(this_batch)]

        # fix seed 
        all_latents = []
        for seed in batch_seeds:
            set_seed(seed)
            lat = torch.randn(1, C, H, W, device=device)
            all_latents.append(lat)
        latents = torch.cat(all_latents, dim=0)

        # class-conditional
        class_labels = None
        if args.dataset_name == 'imagenet': 
            logger.info("Using class-conditional generation for ImageNet...")
            ys = []
            for seed in batch_seeds:
                set_seed(seed)
                y = torch.randint(0, 1000, (1,), device=device)  # (1,)
                ys.append(y)

            y = torch.cat(ys, dim=0)  # (B,)
            class_labels = torch.zeros(latents.shape[0], 1000, device=device, dtype=torch.float32)
            class_labels.scatter_(1, y[:, None], 1.0)
            

        wasserstein_bound_log = calc_wasserstein_bound(
            net, latents, class_labels=None, randn_like=torch.randn_like,
            num_steps=args.num_steps, sigma_min=args.sigma_min, sigma_max=args.sigma_max, rho=args.rho, 
            solver='euler', discretization=args.discretization, S_churn=args.S_churn, S_min=args.S_min, S_max=args.S_max, S_noise=args.S_noise,
            sigmas=optimized_sigmas_iedm
        )

    eta_log = wasserstein_bound_log["eta_log"] 
    
    if args.save_plot:
        # save_dir = os.path.join(args.save_dir, f"{args.dataset_name}/{args.discretization}/{args.optimized_sigmas_exp}")
        # os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(7, 4))
        plt.plot(np.arange(len(eta_log)), eta_log, label="eta value")
        plt.xlabel("step i")
        plt.ylabel("eta")
        plt.title("eta value over sampling steps")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "eta_value.png"), dpi=300, bbox_inches='tight')
        logger.info("Successfully saved plot to:", os.path.join(save_dir, "eta_value.png"))

    if args.n_step_resampling:
        # save_dir = os.path.join(args.save_dir, f"{args.dataset_name}/{args.discretization}/{args.exp}")
        # os.makedirs(save_dir, exist_ok=True)

        # N-step resampling based on cumulative eta values
        resampled_sigmas_iedm = resample_sigmas_uniform_cum_eta(optimized_sigmas_iedm, eta_log, num_steps=args.resampled_num_steps, power=args.power, q=args.q)
        np.save(os.path.join(save_dir, f"optimized_schedules_{args.dataset_name}_{args.discretization}_steps_{args.resampled_num_steps}.npy"), resampled_sigmas_iedm)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)

    # datasets/models
    parser.add_argument("--model_weights_pkl", type=str, default='baseline/baseline-cifar10-32x32-uncond-vp.pkl',
                        choices=['baseline/baseline-cifar10-32x32-uncond-vp.pkl', 'edm/edm-cifar10-32x32-uncond-vp.pkl', 
                                'baseline/baseline-cifar10-32x32-cond-vp.pkl', 'edm/edm-cifar10-32x32-cond-vp.pkl', 
                                'baseline/baseline-imagnet-64x64-cond-adm.pkl', 'edm/edm-imagenet-64x64-cond-adm.pkl', 
                                'baseline/baseline-ffhq-64x64-uncond-vp.pkl', 'edm/edm-ffhq-64x64-uncond-vp.pkl',
                                'baseline/baseline-afhqv2-64x64-uncond-vp.pkl', 'edm/edm-afhqv2-64x64-uncond-vp.pkl',
                                'baseline/baseline-cifar10-32x32-uncond-ve.pkl', 'edm/edm-cifar10-32x32-uncond-ve.pkl', 
                                'baseline/baseline-cifar10-32x32-cond-ve.pkl', 'edm/edm-cifar10-32x32-cond-ve.pkl', 
                                'baseline/baseline-ffhq-64x64-uncond-ve.pkl', 'edm/edm-ffhq-64x64-uncond-ve.pkl',
                                'baseline/baseline-afhqv2-64x64-uncond-ve.pkl', 'edm/edm-afhqv2-64x64-uncond-ve.pkl'], 
                        help="pretrained model weights pickle file") 
    parser.add_argument("--dataset_path", type=str, default='cifar10-32x32.zip', 
                        choices=["cifar10-32x32.zip", 
                                "imagenet-64x64.zip", 
                                "ffhq-64x64.zip", 
                                "afhqv2-64x64.zip"], 
                        help="dataset path zip file")
    parser.add_argument("--dataset_name", type=str, default="cifar10", choices=['cifar10', 'imagenet', 'ffhq', 'afhqv2'], help="dataset name")
    
    parser.add_argument("--channel_size", type=int, default=3, help="number of image channels")
    parser.add_argument("--image_size", type=int, default=32, help="image size")

    # schedule 
    parser.add_argument("--num_steps", type=int, default=40, help="number of steps for given schedule")
    parser.add_argument("--sigma_min", type=float, default=0.002, help="minimum sigma for given schedule")
    parser.add_argument("--sigma_max", type=float, default=80, help="maximum sigma for given schedule")
    parser.add_argument("--discretization", type=str, default="edm", choices=['edm', 'loglinear'], help="discretization type")
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
    parser.add_argument("--save_dir", type=str, default="results/checkpoints", help="directory to save optimized schedules")
    parser.add_argument("--solver", type=str, default="heun", help="ODE solver type")

    # load optimized_sigmas
    parser.add_argument("--optimized_sigmas_save_dir", type=str, default="results/checkpoints", help="directory to load optimized sigmas from")
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