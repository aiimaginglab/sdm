# !/bin/bash

# ============== cifar10 - edm (vp) ============== #   
python test.py \
    --exp euler-cond-vp --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-cond-vp.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name euler --save_dir results/fid_results/images --num_samples 50000 \
    --save_images \
    --seed 42

python test.py \
    --exp euler-cos-cond-vp --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-cond-vp.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name euler_cos --save_dir results/fid_results/images --num_samples 50000 \
    --optimized_sigmas_exp cond-vp-bs128-repeats50 --load_optimized_sigmas \
    --save_images \
    --seed 42

python test.py \
    --exp heun-cond-vp --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-cond-vp.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name heun --save_dir results/fid_results/images --num_samples 50000 \
    --save_images \
    --seed 42

python test.py \
    --exp heun-cos-cond-vp --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-cond-vp.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name heun_cos --save_dir results/fid_results/images --num_samples 50000 \
    --optimized_sigmas_exp cond-vp-bs128-repeats50 --load_optimized_sigmas \
    --save_images \
    --seed 42

python test.py \
    --exp adaptive-cond-vp --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-cond-vp.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name adaptive --save_dir results/fid_results/images --num_samples 50000 \
    --tau_k 1e-4 \
    --save_images \
    --seed 42

python test.py \
    --exp adaptive-cos-cond-vp --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-cond-vp.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name adaptive_cos --save_dir results/fid_results/images --num_samples 50000 \
    --optimized_sigmas_exp cond-vp-bs128-repeats50 --load_optimized_sigmas \
    --tau_k 1e-4 \
    --save_images \
    --seed 42


# ============== cifar10 - edm (ve) ============== #   
python test.py \
    --exp euler-cond-ve --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-cond-ve.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name euler --save_dir results/fid_results/images --num_samples 50000 \
    --save_images \
    --seed 42

python test.py \
    --exp euler-cos-cond-ve --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-cond-ve.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name euler_cos --save_dir results/fid_results/images --num_samples 50000 \
    --optimized_sigmas_exp cond-ve-bs128-repeats50 --load_optimized_sigmas \
    --save_images \
    --seed 42

python test.py \
    --exp heun-cond-ve --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-cond-ve.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name heun --save_dir results/fid_results/images --num_samples 50000 \
    --save_images \
    --seed 42

python test.py \
    --exp heun-cos-cond-ve --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-cond-ve.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name heun_cos --save_dir results/fid_results/images --num_samples 50000 \
    --optimized_sigmas_exp cond-ve-bs128-repeats50 --load_optimized_sigmas \
    --save_images \
    --seed 42

python test.py \
    --exp adaptive-cond-ve --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-cond-ve.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name adaptive --save_dir results/fid_results/images --num_samples 50000 \
    --tau_k 1e-4 \
    --save_images \
    --seed 42

python test.py \
    --exp adaptive-cos-cond-ve --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-cond-ve.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name adaptive_cos --save_dir results/fid_results/images --num_samples 50000 \
    --optimized_sigmas_exp cond-ve-bs128-repeats50 --load_optimized_sigmas \
    --tau_k 1e-4 \
    --save_images \
    --seed 42
