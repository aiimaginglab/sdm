# !/bin/bash

# ============== afhqv2 ============== #   
python optimize_schedules.py \
    --exp uncond-ve-bs128-repeats50 \
    --model_weights_pkl edm/edm-afhqv2-64x64-uncond-ve.pkl --dataset_path 'afhqv2-64x64.zip' \
    --dataset_name afhqv2 --batch_size 128 --num_repeats 50 \
    --num_steps 40 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --save_dir results/checkpoints \
    --seed 42

# ============== ffhq ============== #   
python optimize_schedules.py \
    --exp uncond-ve-bs128-repeats50 \
    --model_weights_pkl edm/edm-ffhq-64x64-uncond-ve.pkl --dataset_path 'ffhq-64x64.zip' \
    --dataset_name ffhq --batch_size 128 --num_repeats 50 \
    --num_steps 40 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --save_dir results/checkpoints \
    --seed 42

# ============== cifar10 ============== #   
python optimize_schedules.py \
    --exp uncond-ve-bs128-repeats50 \
    --model_weights_pkl edm/edm-cifar10-32x32-uncond-ve.pkl --dataset_path 'cifar10-32x32.zip' \
    --dataset_name cifar10 --batch_size 128 --num_repeats 50 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --save_dir results/checkpoints \
    --seed 42

# ============== cifar10 - conditional ============== #   
python optimize_schedules.py \
    --exp cond-vp-bs128-repeats50 \
    --model_weights_pkl edm/edm-cifar10-32x32-cond-vp.pkl --dataset_path 'cifar10-32x32.zip' \
    --dataset_name cifar10 --batch_size 128 --num_repeats 50 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --save_dir results/checkpoints \
    --has_labels \
    --seed 42

python optimize_schedules.py \
    --exp cond-ve-bs128-repeats50 \
    --model_weights_pkl edm/edm-cifar10-32x32-cond-ve.pkl --dataset_path 'cifar10-32x32.zip' \
    --dataset_name cifar10 --batch_size 128 --num_repeats 50 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --save_dir results/checkpoints \
    --has_labels \
    --seed 42

# ============== imagenet ============== #   
python optimize_schedules.py \
    --exp cond-adm-bs128-repeats25 \
    --model_weights_pkl edm/edm-imagenet-64x64-cond-adm.pkl --dataset_path 'imagenet-64x64.zip' \
    --dataset_name imagenet --batch_size 128 --num_repeats 25 \
    --num_steps 256 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --save_dir results/checkpoints \
    --has_labels \
    --seed 42