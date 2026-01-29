# !/bin/bash

# adaptive solver parameters
SCHEDULE="step"
TAUK=1e-4

# ============== imagenet ============== #   
python test.py \
    --exp euler-cond-adm --dataset_name imagenet --batch_size 128 \
    --model_weights_pkl edm/edm-imagenet-64x64-cond-adm.pkl \
    --channel_size 3 --image_size 64 \
    --num_steps 256 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --S_churn 40 --S_min 0.05 --S_max 50 --S_noise 1.003 \
    --sampler_name euler --save_dir results/fid_results/images --num_samples 50000 \
    --save_images \
    --seed 42

python test.py \
    --exp euler-cos-cond-adm --dataset_name imagenet --batch_size 128 \
    --model_weights_pkl edm/edm-imagenet-64x64-cond-adm.pkl \
    --channel_size 3 --image_size 64 \
    --num_steps 256 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --S_churn 40 --S_min 0.05 --S_max 50 --S_noise 1.003 \
    --sampler_name euler --save_dir results/fid_results/images --num_samples 50000 \
    --optimized_sigmas_exp cond-adm-bs128-repeats25 --load_optimized_sigmas \
    --save_images \
    --seed 42

python test.py \
    --exp heun-cond-adm --dataset_name imagenet --batch_size 128 \
    --model_weights_pkl edm/edm-imagenet-64x64-cond-adm.pkl \
    --channel_size 3 --image_size 64 \
    --num_steps 256 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --S_churn 40 --S_min 0.05 --S_max 50 --S_noise 1.003 \
    --sampler_name heun --save_dir results/fid_results/images --num_samples 50000 \
    --save_images \
    --seed 42

python test.py \
    --exp heun-cos-cond-adm --dataset_name imagenet --batch_size 128 \
    --model_weights_pkl edm/edm-imagenet-64x64-cond-adm.pkl \
    --channel_size 3 --image_size 64 \
    --num_steps 256 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --S_churn 40 --S_min 0.05 --S_max 50 --S_noise 1.003 \
    --sampler_name heun --save_dir results/fid_results/images --num_samples 50000 \
    --optimized_sigmas_exp cond-adm-bs128-repeats25 --load_optimized_sigmas \
    --save_images \
    --seed 42

python test.py \
    --exp adaptive-cond-adm --dataset_name imagenet --batch_size 128 \
    --model_weights_pkl edm/edm-imagenet-64x64-cond-adm.pkl \
    --channel_size 3 --image_size 64 \
    --num_steps 256 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --S_churn 40 --S_min 0.05 --S_max 50 --S_noise 1.003 \
    --sampler_name sdm_solver --save_dir results/fid_results/images --num_samples 50000 \
    --tau_k $TAUK --lambda_schedule $SCHEDULE \
    --save_images \
    --seed 42
