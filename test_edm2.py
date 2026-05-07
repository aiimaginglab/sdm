import os
import time
import pickle
import argparse
import logging
import numpy as np
from tqdm import trange

import torch
import torchvision.utils as vutils

import dnnlib
from modules.utils import set_seed, set_logger, save_json
from modules.visualizer import append_jsonl, save_images
from modules.sampler import _ablation_edm2_sampler, ablation_sampler_adaptive_solver, ablation_sampler_adaptive_dpmsolver, ablation_sampler_adaptive_scheduling

import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "edm2"))


"""Generate images for all samplers as an ablation study."""

#----------------------------------------------------------------------------
# Configuration presets.

model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions'

config_presets = {
    'edm2-img512-xs-fid':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.135.pkl'),      # fid = 3.53
    'edm2-img512-xs-dino':             dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.200.pkl'),      # fd_dinov2 = 103.39
    'edm2-img512-s-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.130.pkl'),       # fid = 2.56
    'edm2-img512-s-dino':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.190.pkl'),       # fd_dinov2 = 68.64
    'edm2-img512-m-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.100.pkl'),       # fid = 2.25
    'edm2-img512-m-dino':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.155.pkl'),       # fd_dinov2 = 58.44
    'edm2-img512-l-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.085.pkl'),       # fid = 2.06
    'edm2-img512-l-dino':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.155.pkl'),       # fd_dinov2 = 52.25
    'edm2-img512-xl-fid':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.085.pkl'),      # fid = 1.96
    'edm2-img512-xl-dino':             dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.155.pkl'),      # fd_dinov2 = 45.96
    'edm2-img512-xxl-fid':             dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.070.pkl'),     # fid = 1.91
    'edm2-img512-xxl-dino':            dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.150.pkl'),     # fd_dinov2 = 42.84
    'edm2-img64-s-fid':                dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.075.pkl'),        # fid = 1.58
    'edm2-img64-m-fid':                dnnlib.EasyDict(net=f'{model_root}/edm2-img64-m-2147483-0.060.pkl'),        # fid = 1.43
    'edm2-img64-l-fid':                dnnlib.EasyDict(net=f'{model_root}/edm2-img64-l-1073741-0.040.pkl'),        # fid = 1.33
    'edm2-img64-xl-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img64-xl-0671088-0.040.pkl'),       # fid = 1.33
    'edm2-img512-xs-guid-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.045.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.045.pkl', guidance=1.40), # fid = 2.91
    'edm2-img512-xs-guid-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.150.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.150.pkl', guidance=1.70), # fd_dinov2 = 79.94
    'edm2-img512-s-guid-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.025.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.025.pkl', guidance=1.40), # fid = 2.23
    'edm2-img512-s-guid-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.085.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.085.pkl', guidance=1.90), # fd_dinov2 = 52.32
    'edm2-img512-m-guid-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.030.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.20), # fid = 2.01
    'edm2-img512-m-guid-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.015.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=2.00), # fd_dinov2 = 41.98
    'edm2-img512-l-guid-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.015.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.20), # fid = 1.88
    'edm2-img512-l-guid-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.035.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.035.pkl', guidance=1.70), # fd_dinov2 = 38.20
    'edm2-img512-xl-guid-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.020.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.020.pkl', guidance=1.20), # fid = 1.85
    'edm2-img512-xl-guid-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.030.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.70), # fd_dinov2 = 35.67
    'edm2-img512-xxl-guid-fid':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',      gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.20), # fid = 1.81
    'edm2-img512-xxl-guid-dino':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',      gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.70), # fd_dinov2 = 33.09
    'edm2-img512-s-autog-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.070.pkl',        gnet=f'{model_root}/edm2-img512-xs-0134217-0.125.pkl',        guidance=2.10), # fid = 1.34
    'edm2-img512-s-autog-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.120.pkl',        gnet=f'{model_root}/edm2-img512-xs-0134217-0.165.pkl',        guidance=2.45), # fd_dinov2 = 36.67
    'edm2-img512-xxl-autog-fid':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.075.pkl',      gnet=f'{model_root}/edm2-img512-m-0268435-0.155.pkl',         guidance=2.05), # fid = 1.25
    'edm2-img512-xxl-autog-dino':      dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.130.pkl',      gnet=f'{model_root}/edm2-img512-m-0268435-0.205.pkl',         guidance=2.30), # fd_dinov2 = 24.18
    'edm2-img512-s-uncond-autog-fid':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-uncond-2147483-0.070.pkl', gnet=f'{model_root}/edm2-img512-xs-uncond-0134217-0.110.pkl', guidance=2.85), # fid = 3.86
    'edm2-img512-s-uncond-autog-dino': dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-uncond-2147483-0.090.pkl', gnet=f'{model_root}/edm2-img512-xs-uncond-0134217-0.125.pkl', guidance=2.90), # fd_dinov2 = 90.39
    'edm2-img64-s-autog-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.045.pkl',         gnet=f'{model_root}/edm2-img64-xs-0134217-0.110.pkl',         guidance=1.70), # fid = 1.01
    'edm2-img64-s-autog-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.105.pkl',         gnet=f'{model_root}/edm2-img64-xs-0134217-0.175.pkl',         guidance=2.20), # fd_dinov2 = 31.85
}

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

    # calc curvature log
    jsonl_path = os.path.join(save_dir, "gap_log.jsonl")
    logger.info(f"Curvature log will be saved to: {jsonl_path}")

    # load model/datasets
    def load_edm2_model(preset, device, logger, verbose=True, encoder=None, encoder_batch_size=None):

        assert preset in config_presets, f"Invalid preset: {preset}"

        preset_cfg = config_presets[args.preset]
        net = preset_cfg.get("net")
        gnet = preset_cfg.get("gnet", None)
        guidance = preset_cfg.get("guidance", 1.0)

        # Load main network.
        if isinstance(net, str):
            if verbose:
                logger.info(f'Loading main network from {net} ...')
            if net.startswith("http://") or net.startswith("https://"):
                with dnnlib.util.open_url(net, verbose=verbose) as f:
                    data = pickle.load(f)
            else:
                with open(net, "rb") as f:
                    data = pickle.load(f)

            net = data['ema'].to(device)
            if encoder is None:
                encoder = data.get('encoder', None)
                if encoder is None:
                    encoder = dnnlib.util.construct_class_by_name(
                        class_name='training.encoders.StandardRGBEncoder'
                    )
        assert net is not None

        logger.info("Successfully loaded main network...")

        # Load guidance network. (not used yet in stage 1)
        if isinstance(gnet, str):
            if verbose:
                logger.info(f'Loading guiding network from {gnet} ...')
            if gnet.startswith("http://") or gnet.startswith("https://"):
                with dnnlib.util.open_url(gnet, verbose=verbose) as f:
                    gnet = pickle.load(f)['ema'].to(device)
            else:
                with open(gnet, "rb") as f:
                    gnet = pickle.load(f)['ema'].to(device)

        logger.info("Successfully loaded guidance network...")

        # Initialize encoder.
        assert encoder is not None
        if verbose:
            logger.info(f'Setting up {type(encoder).__name__}...')
        encoder.init(device)
        if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
            encoder.batch_size = encoder_batch_size

        logger.info("Successfully loaded encoder...")

        net.eval()
        if gnet is not None and hasattr(gnet, "eval"):
            gnet.eval()

        return net, gnet, encoder, guidance

    net, gnet, encoder, guidance = load_edm2_model(args.preset, device, logger)

    logger.info(f"Experiment {args.exp} : Sampling {args.dataset_name} with base schedule {args.discretization}...")
    logger.info(f"Sampler: {args.sampler_name}, Number of steps: {args.num_steps}, Sigma min: {args.sigma_min}, Sigma max: {args.sigma_max}, Rho: {args.rho}")
    logger.info(f"Stochastic settings: S_churn: {args.S_churn}, S_min: {args.S_min}, S_max: {args.S_max}, S_noise: {args.S_noise}")

    B = args.batch_size
    C, H, W = net.img_channels, net.img_resolution, net.img_resolution 
    num_batches = (args.num_samples - args.prev_num_samples + B - 1) // B # (args.num_samples + B - 1) // B

    if args.load_optimized_sigmas:
        optimized_sigmas = np.load(os.path.join(BASE_PATH, f'{args.optimized_sigmas_save_dir}/{args.dataset_name}/{args.discretization}/{args.optimized_sigmas_exp}/optimized_schedules_{args.dataset_name}_{args.discretization}_steps_{args.num_steps}.npy'))
        logger.info(f"Loaded optimized sigmas: {optimized_sigmas}")
    else: 
        optimized_sigmas = None

    # samplers 
    sampler_map = {
        # first-order ODE
        "euler": lambda latents, class_labels: _ablation_edm2_sampler(net, latents, class_labels, torch.randn_like, args.num_steps, args.sigma_min, args.sigma_max, 
                                                   args.rho, solver='euler', discretization=args.discretization, S_churn=args.S_churn, S_min=args.S_min, S_max=args.S_max, S_noise=args.S_noise, sigmas=optimized_sigmas),
        # second-order ODE
        "heun": lambda latents, class_labels: _ablation_edm2_sampler(net, latents, class_labels, torch.randn_like, args.num_steps, args.sigma_min, args.sigma_max, 
                                                    args.rho, solver='heun', discretization=args.discretization, S_churn=args.S_churn, S_min=args.S_min, S_max=args.S_max, S_noise=args.S_noise, sigmas=optimized_sigmas),
        # dpm solver (2m)
        "dpm_solver": lambda latents, class_labels: _ablation_edm2_sampler(net, latents, class_labels, torch.randn_like, args.num_steps, args.sigma_min, args.sigma_max,
                                                    args.rho, solver='dpmsolver', discretization=args.discretization, S_churn=args.S_churn, S_min=args.S_min, S_max=args.S_max, S_noise=args.S_noise, sigmas=optimized_sigmas, solver_type=args.dpm_solver_type),
        # unipc
        "unipc": lambda latents, class_labels: _ablation_edm2_sampler(net, latents, class_labels, torch.randn_like, args.num_steps, args.sigma_min, args.sigma_max,
                                                    args.rho, solver='unipc', discretization=args.discretization, S_churn=0.0, S_min=args.S_min, S_max=args.S_max, S_noise=args.S_noise, sigmas=optimized_sigmas),  
        # adaptive ODE 
        "sdm_solver": lambda latents, class_labels: ablation_sampler_adaptive_solver(net, latents, class_labels, torch.randn_like, args.num_steps, args.sigma_min, args.sigma_max, 
                                                        args.rho, discretization=args.discretization, S_churn=args.S_churn, S_min=args.S_min, S_max=args.S_max, S_noise=args.S_noise, sigmas=optimized_sigmas,
                                                        calc_curvature=args.calc_curvature, metric='dxdt', use_relative_metrics=args.use_relative_metrics, tau_k=args.tau_k, lambda_schedule=args.lambda_schedule),
        # adaptive dpm solver 
        "sdm_dpmsolver": lambda latents, class_labels: ablation_sampler_adaptive_dpmsolver(net, latents, class_labels, torch.randn_like, args.num_steps, args.sigma_min, args.sigma_max,
                                                            args.rho, discretization=args.discretization, S_churn=args.S_churn, S_min=args.S_min, S_max=args.S_max, S_noise=args.S_noise, sigmas=optimized_sigmas,
                                                            calc_curvature=args.calc_curvature, metric='dxdt', use_relative_metrics=args.use_relative_metrics, tau_k=args.tau_k, lambda_schedule=args.lambda_schedule,
                                                            solver_type=args.dpm_solver_type, gnet=gnet, guidance=guidance),
        
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
                if getattr(net, "label_dim", 0) > 0:
                    num_classes = net.label_dim
                else:
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
            if args.sampler_name == "sdm_solver" or args.sampler_name == "sdm_dpmsolver":
                x, gap_log, nfe = sampler_fn(latents, class_labels)
                logger.info(f"Total NFE: {nfe}")
                with open(os.path.join(save_dir, "nfe.txt"), "w") as f:
                    f.write(f"{nfe}\n")

            elif args.sampler_name == "sdm_scheduler":
                x, optimized_sigmas_iedm, states = sampler_fn(latents, class_labels)
                nfe = None
                
            else: 
                x, nfe = sampler_fn(latents, class_labels)
                logger.info(f"Total NFE: {nfe}")
                with open(os.path.join(save_dir, "nfe.txt"), "w") as f:
                    f.write(f"{nfe}\n")

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
            save_images(encoder.decode(x), save_dir, batch_seeds, subdirs=True, is_latent=True)

    if args.calc_curvature:
        logger.info(f"Done. JSONL saved to: {jsonl_path}")

    logger.info(f"Done: {args.sampler_name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)

    # datasets/models
    parser.add_argument("--preset", type=str, default=None, help="EDM2 preset name")
    parser.add_argument("--net", type=str, default=None, help="main network pickle path or URL")
    parser.add_argument("--gnet", type=str, default=None, help="guidance network pickle path or URL")
    parser.add_argument("--guidance", type=float, default=None, help="guidance scale")
    parser.add_argument("--model_weights_pkl", type=str, default=None, help="[deprecated] local pretrained model pkl")
        
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
    
    # dpm-solver++ (2M)
    parser.add_argument("--dpm_solver_type", type=str, default="midpoint", choices=["midpoint", "heun"], help="2nd-order update variant for DPM-Solver++(2M)")

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