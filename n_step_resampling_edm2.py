import os 
import pickle 
import argparse
import logging

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

import torch 

import dnnlib
from modules.utils import set_seed, set_logger, resample_sigmas_uniform_cum_eta
from modules.sampler import calc_wasserstein_bound

import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "edm2"))

"""eta-based N-step resampling algorithm."""

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

    # os.makedirs(args.save_dir, exist_ok=True)
    
    B = args.batch_size
    C, H, W = net.img_channels, net.img_resolution, net.img_resolution # args.channel_size, args.image_size, args.image_size
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