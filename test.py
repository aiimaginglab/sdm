import os
import pickle
import argparse
import logging
import numpy as np
from tqdm import trange

import torch
import torchvision.utils as vutils

from modules.utils import set_seed, set_logger, save_json
from modules.visualizer import append_jsonl, save_images
from modules.sampler import _ablation_sampler, ablation_sampler_adaptive_solver, ablation_sampler_adaptive_scheduling

"""Generate images for all samplers as an ablation study."""

def main(args):
    # settings
    device = 'cuda'
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    set_seed(args.seed)

    logger = set_logger()

    path_to_pretrained_model = os.path.join(BASE_PATH, "dnnlib/models")

    with open(os.path.join(path_to_pretrained_model, args.model_weights_pkl), 'rb') as f:
        data = pickle.load(f)

    logger.info("Successfully loaded pretrained model...")
    net = data['ema'].to(device)
    net.eval()

    os.makedirs(args.save_dir, exist_ok=True)

    logger.info(f"Experiment {args.exp} : Sampling {args.dataset_name} with base schedule {args.discretization}...")
    logger.info(f"Sampler: {args.sampler_name}, Number of steps: {args.num_steps}, Sigma min: {args.sigma_min}, Sigma max: {args.sigma_max}, Rho: {args.rho}")
    logger.info(f"Stochastic settings: S_churn: {args.S_churn}, S_min: {args.S_min}, S_max: {args.S_max}, S_noise: {args.S_noise}")

    B = args.batch_size
    C, H, W = args.channel_size, args.image_size, args.image_size
    num_batches = (args.num_samples + B - 1) // B

    if args.sampler_name == "sdm_solver" and args.lambda_schedule == "step":
        save_dir = os.path.join(args.save_dir, f"{args.dataset_name}/{args.discretization}/{args.exp}-tauk-{args.tau_k}")
    else: 
        save_dir = os.path.join(args.save_dir, f"{args.dataset_name}/{args.discretization}/{args.exp}")
    os.makedirs(save_dir, exist_ok=True)

    # calc curvature log
    jsonl_path = os.path.join(save_dir, "gap_log.jsonl")
    logger.info(f"Curvature log will be saved to: {jsonl_path}")

    if args.load_optimized_sigmas:
        optimized_sigmas = np.load(os.path.join(BASE_PATH, f'{args.optimized_sigmas_save_dir}/{args.dataset_name}/{args.discretization}/{args.optimized_sigmas_exp}/optimized_schedules_{args.dataset_name}_{args.discretization}_steps_{args.num_steps}.npy'))
        logger.info(f"Loaded optimized sigmas: {optimized_sigmas}")
    else: 
        optimized_sigmas = None

    # samplers 
    sampler_map = {
        # first-order ODE
        "euler": lambda latents, class_labels: _ablation_sampler(net, latents, class_labels, torch.randn_like, args.num_steps, args.sigma_min, args.sigma_max, 
                                                   args.rho, solver='euler', discretization=args.discretization, S_churn=args.S_churn, S_min=args.S_min, S_max=args.S_max, S_noise=args.S_noise),
        # second-order ODE
        "heun": lambda latents, class_labels: _ablation_sampler(net, latents, class_labels, torch.randn_like, args.num_steps, args.sigma_min, args.sigma_max, 
                                                    args.rho, solver='heun', discretization=args.discretization, S_churn=args.S_churn, S_min=args.S_min, S_max=args.S_max, S_noise=args.S_noise),
        # adaptive ODE 
        "sdm_solver": lambda latents, class_labels: ablation_sampler_adaptive_solver(net, latents, class_labels, torch.randn_like, args.num_steps, args.sigma_min, args.sigma_max, 
                                                        args.rho, discretization=args.discretization, S_churn=args.S_churn, S_min=args.S_min, S_max=args.S_max, S_noise=args.S_noise, sigmas=optimized_sigmas,
                                                        calc_curvature=args.calc_curvature, metric='dxdt', use_relative_metrics=args.use_relative_metrics, tau_k=args.tau_k, lambda_schedule=args.lambda_schedule),
        # adaptive scheduling
        "sdm_scheduler": lambda latents, class_labels: ablation_sampler_adaptive_scheduling(net, latents, class_labels, torch.randn_like, args.num_steps, args.sigma_min, args.sigma_max, 
                                                        args.rho, solver=args.solver, discretization=args.discretization, S_churn=args.S_churn, S_min=args.S_min, S_max=args.S_max, S_noise=args.S_noise,
                                                        eta_min=args.eta_min, eta_max=args.eta_max, eta_p=args.eta_p, use_grid_snap=args.use_grid_snap),
    }

    assert args.sampler_name in sampler_map, f"Unknown sampler: {args.sampler_name}"
    sampler_fn = sampler_map[args.sampler_name]
    logger.info(f"Running sampler: {args.sampler_name}")

    if args.sampler_name == "sdm_solver":
        logger.info(f"  |-- use_relative_metrics: {args.use_relative_metrics}, tau_k: {args.tau_k}")
    if args.sampler_name == "sdm_scheduler":
        logger.info(f"  |-- eta_min: {args.eta_min}, eta_max: {args.eta_max}, eta_p: {args.eta_p}, use_grid_snap: {args.use_grid_snap}")

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
        if args.dataset_name == 'imagenet' or (args.dataset_name == 'cifar10' and '-cond' in args.model_weights_pkl): 
            logger.info("Using class-conditional generation ...")
            
            if args.dataset_name == "imagenet":
                num_classes = 1000
            elif args.dataset_name == "cifar10":
                num_classes = 10
            
            ys = []
            for seed in batch_seeds:
                set_seed(seed)
                y = torch.randint(0, num_classes, (1,), device=device)  # (1,)
                ys.append(y)

            y = torch.cat(ys, dim=0)  # (B,)
            class_labels = torch.zeros(latents.shape[0], num_classes, device=device, dtype=torch.float32)
            class_labels.scatter_(1, y[:, None], 1.0)

        with torch.no_grad():
            if args.sampler_name == "sdm_solver":
                x, gap_log = sampler_fn(latents, class_labels)
            elif args.sampler_name == "sdm_scheduler":
                x, optimized_sigmas_iedm, states = sampler_fn(latents, class_labels)
            else: 
                x = sampler_fn(latents, class_labels)

        if args.sampler_name == "sdm_scheduler":
            save_json(states, os.path.join(save_dir, f'log-{args.exp}-batch{batch_idx}.json'))
            save_json(vars(args), os.path.join(save_dir, 'config.json'))

            optimized_num_steps = len(optimized_sigmas_iedm) - 1
            logger.info(f"Length of optimized_sigmas_iedm: {optimized_num_steps}")
            logger.info(f"    Optimized sigmas iedm: {optimized_sigmas_iedm}")
            np.save(os.path.join(save_dir, f"optimized_schedules_{args.dataset_name}_{args.discretization}_steps_{optimized_num_steps}.npy"), optimized_sigmas_iedm)

        if args.calc_curvature: 
            for rec in gap_log:
                rec["batch_size"] = int(this_batch)
            append_jsonl(jsonl_path, gap_log)

        if args.save_images:
            save_images(x, save_dir, batch_seeds, subdirs=True)

    if args.calc_curvature:
        logger.info(f"Done. JSONL saved to: {jsonl_path}")

    logger.info(f"Done: {args.sampler_name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)

    # datasets/models
    parser.add_argument("--model_weights_pkl", type=str, default='baseline/baseline-cifar10-32x32-uncond-vp.pkl',
                        choices=['baseline/baseline-cifar10-32x32-uncond-vp.pkl', 'edm/edm-cifar10-32x32-uncond-vp.pkl', 
                                'baseline/baseline-cifar10-32x32-cond-vp.pkl', 'edm/edm-cifar10-32x32-cond-vp.pkl', 
                                'baseline/baseline-imagenet-64x64-cond-adm.pkl', 'edm/edm-imagenet-64x64-cond-adm.pkl', 
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
    parser.add_argument("--exp", type=str, required=True, help="experiment name")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--prev_num_samples", type=int, default=0, help="number of previously generated samples")
    parser.add_argument("--num_samples", type=int, default=50000, help="number of samples to generate for FID")
    parser.add_argument("--sampler_name", type=str, default="adaptive", help="sampler name for ablation study")
    parser.add_argument("--save_dir", type=str, default="results/fid_results", help="directory to save model results")
    parser.add_argument("--solver", type=str, default="heun", help="ODE solver type")
    
    # save images/calc curvature 
    parser.add_argument("--save_images", action='store_true', help="whether to save generated images")
    parser.add_argument("--calc_curvature", action='store_true', help="whether to calculate curvature log")

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