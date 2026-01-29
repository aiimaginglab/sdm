# !/bin/bash

# adaptive solver parameters
SCHEDULE="step"
TAUK=2e-4

# ============== cifar10 - edm (vp) ============== #   
python test.py \
    --exp euler-uncond-vp --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-uncond-vp.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name euler --save_dir results/fid_results/images --num_samples 50000 \
    --save_images \
    --seed 42

python test.py \
    --exp euler-cos-uncond-vp --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-uncond-vp.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name euler --save_dir results/fid_results/images --num_samples 50000 \
    --optimized_sigmas_exp uncond-vp-bs128-repeats50 --load_optimized_sigmas \
    --save_images \
    --seed 42

python test.py \
    --exp heun-uncond-vp --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-uncond-vp.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name heun --save_dir results/fid_results/images --num_samples 50000 \
    --save_images \
    --seed 42

python test.py \
    --exp heun-cos-uncond-vp --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-uncond-vp.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name heun --save_dir results/fid_results/images --num_samples 50000 \
    --optimized_sigmas_exp uncond-vp-bs128-repeats50 --load_optimized_sigmas \
    --save_images \
    --seed 42

python test.py \
    --exp adaptive-uncond-vp --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-uncond-vp.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name sdm_solver --save_dir results/fid_results/images --num_samples 50000 \
    --tau_k $TAUK --lambda_schedule $SCHEDULE \
    --save_images \
    --seed 42


# ============== cifar10 - edm (ve) ============== #   
python test.py \
    --exp euler-uncond-ve --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-uncond-ve.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name euler --save_dir results/fid_results/images --num_samples 50000 \
    --save_images \
    --seed 42

python test.py \
    --exp euler-cos-uncond-ve --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-uncond-ve.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name euler --save_dir results/fid_results/images --num_samples 50000 \
    --optimized_sigmas_exp uncond-ve-bs128-repeats50 --load_optimized_sigmas \
    --save_images \
    --seed 42

python test.py \
    --exp heun-uncond-ve --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-uncond-ve.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name heun --save_dir results/fid_results/images --num_samples 50000 \
    --save_images \
    --seed 42

python test.py \
    --exp heun-cos-uncond-ve --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-uncond-ve.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name heun --save_dir results/fid_results/images --num_samples 50000 \
    --optimized_sigmas_exp uncond-ve-bs128-repeats50 --load_optimized_sigmas \
    --save_images \
    --seed 42

python test.py \
    --exp adaptive-uncond-ve --dataset_name cifar10 --batch_size 128 \
    --model_weights_pkl edm/edm-cifar10-32x32-uncond-ve.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name sdm_solver --save_dir results/fid_results/images --num_samples 50000 \
    --tau_k $TAUK --lambda_schedule $SCHEDULE \
    --save_images \
    --seed 42
