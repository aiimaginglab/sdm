import os
import pickle
import dnnlib
import argparse

import torch    
import numpy as np

from torch_utils import misc
from modules.utils import set_seed, build_sigmas, build_cos_sigmas, compute_score_diffs_once


#-----------------------------------------------------------------------------
# Re-implementation of the paper "Score-Optimal Diffusion Models"

def main(args):
    # settings
    device = 'cuda'
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    set_seed(args.seed)
    batch_size = args.batch_size

    # has_labels = False
    path_to_pretrained_model = os.path.join(BASE_PATH, "dnnlib/models")
    path_to_dataset = os.path.join(BASE_PATH, "dnnlib/datasets")

    with open(os.path.join(path_to_pretrained_model, args.model_weights_pkl), 'rb') as f:
        data = pickle.load(f)

    net = data['ema'].to(device)

    c = dnnlib.EasyDict()
    path_to_dataset = os.path.join(path_to_dataset, args.dataset_path)
    # c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=path_to_dataset, use_labels=False, xflip=False, cache=True)
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=path_to_dataset, use_labels=args.has_labels, xflip=False, cache=True)

    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=1, prefetch_factor=2)

    dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)

    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=0, num_replicas=1, seed=0)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=args.batch_size, **c.data_loader_kwargs))

    # check data 
    images, labels = next(dataset_iterator)
    images = images.to(device).to(torch.float32) / 127.5 - 1
    labels = labels.to(device)

    # initial schedule
    sigmas = build_sigmas(num_steps=args.num_steps, sigma_min=args.sigma_min, sigma_max=args.sigma_max, schedule=args.discretization, rho=args.rho)

    # calculate corrector optimized cost
    avg_cum_sum = compute_score_diffs_once(net, dataset_iterator, sigmas, batch_size, num_repeats=args.num_repeats, device=device)

    # optimized schedules
    optimized_sigmas, f_t2L = build_cos_sigmas(
        avg_cum_sum=avg_cum_sum,
        sigmas_probe=sigmas,                
        num_steps=args.num_steps,
        discretization=args.discretization,
        rho=args.rho,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        append_zero=True
    )

    os.makedirs(f'{args.save_dir}/{args.dataset_name}/{args.discretization}/{args.exp}', exist_ok=True)
    np.save(os.path.join(BASE_PATH, f'{args.save_dir}/{args.dataset_name}/{args.discretization}/{args.exp}/optimized_schedules_{args.dataset_name}_{args.discretization}_steps_{args.num_steps}.npy'), optimized_sigmas)
    with open(os.path.join(BASE_PATH, f'{args.save_dir}/{args.dataset_name}/{args.discretization}/{args.exp}/f_t2L_{args.dataset_name}_{args.discretization}_steps_{args.num_steps}.pkl'), 'wb') as f:
        pickle.dump(f_t2L, f)


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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataset_name", type=str, default="cifar10", choices=['cifar10', 'imagenet', 'ffhq', 'afhqv2'], help="dataset name")
    parser.add_argument("--has_labels", action="store_true", help="whether the dataset has labels or not")

    # schedule 
    parser.add_argument("--exp", type=str, required=True, help="experiment name")
    parser.add_argument("--num_repeats", type=int, default=10, help="number of repeats for calculating score difference")
    parser.add_argument("--num_steps", type=int, default=40, help="number of steps for optimized schedule")
    parser.add_argument("--sigma_min", type=float, default=0.002, help="minimum sigma for optimized schedule")
    parser.add_argument("--sigma_max", type=float, default=80, help="maximum sigma for optimized schedule")
    parser.add_argument("--discretization", type=str, default="edm", choices=['edm', 'loglinear'], help="discretization type")
    parser.add_argument("--rho", type=float, default=7.0, help="rho parameter for edm discretization")

    parser.add_argument("--save_dir", type=str, default="results/checkpoints", help="directory to save optimized schedules")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
