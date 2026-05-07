# !/bin/bash

SAVE_DIR=results/calc_curvature

# ============== afhqv2 ============== #  
DATASET=afhqv2 
EXP=sdm-schedule-uncond-vp-calc-curvature
python test.py \
    --exp $EXP --dataset_name $DATASET --batch_size 16 \
    --model_weights_pkl edm/edm-afhqv2-64x64-uncond-vp.pkl \
    --channel_size 3 --image_size 64 \
    --num_steps 40 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name sdm_solver --num_samples 16 --save_dir $SAVE_DIR \
    --optimized_sigmas_exp euler-uncond-vp-iedm-emin-0.02-emax-0.20-p-1.0-resampled-40-pow-0.5-q-0.25 --load_optimized_sigmas \
    --tau_k 0.0 \
    --calc_curvature \
    --seed 42

python calc_curvature.py \
    --jsonl_path $SAVE_DIR/$DATASET/edm/$EXP-tauk-0.0/gap_log.jsonl \
    --output_path $SAVE_DIR/$DATASET/edm/$EXP-tauk-0.0/curvature_plot.png

EXP=sdm-schedule-uncond-ve-calc-curvature
python test.py \
    --exp $EXP --dataset_name $DATASET --batch_size 16 \
    --model_weights_pkl edm/edm-afhqv2-64x64-uncond-ve.pkl \
    --channel_size 3 --image_size 64 \
    --num_steps 40 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name sdm_solver --num_samples 16 --save_dir $SAVE_DIR \
    --optimized_sigmas_exp euler-uncond-ve-iedm-emin-0.02-emax-0.20-p-1.0-resampled-40-pow-0.5-q-0.25 --load_optimized_sigmas \
    --tau_k 0.0 \
    --calc_curvature \
    --seed 42

python calc_curvature.py \
    --jsonl_path $SAVE_DIR/$DATASET/edm/$EXP-tauk-0.0/gap_log.jsonl \
    --output_path $SAVE_DIR/$DATASET/edm/$EXP-tauk-0.0/curvature_plot.png

# ============== ffhq ============== #
DATASET=ffhq
EXP=sdm-schedule-uncond-vp-calc-curvature
python test.py \
    --exp $EXP --dataset_name $DATASET --batch_size 16 \
    --model_weights_pkl edm/edm-ffhq-64x64-uncond-vp.pkl \
    --channel_size 3 --image_size 64 \
    --num_steps 40 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name sdm_solver --num_samples 16 --save_dir $SAVE_DIR \
    --optimized_sigmas_exp euler-uncond-vp-iedm-emin-0.02-emax-0.20-p-1.0-resampled-40-pow-0.5-q-0.25 --load_optimized_sigmas \
    --tau_k 0.0 \
    --calc_curvature \
    --seed 42

python calc_curvature.py \
    --jsonl_path $SAVE_DIR/$DATASET/edm/$EXP-tauk-0.0/gap_log.jsonl \
    --output_path $SAVE_DIR/$DATASET/edm/$EXP-tauk-0.0/curvature_plot.png

EXP=sdm-schedule-uncond-ve-calc-curvature
python test.py \
    --exp $EXP --dataset_name $DATASET --batch_size 16 \
    --model_weights_pkl edm/edm-ffhq-64x64-uncond-ve.pkl \
    --channel_size 3 --image_size 64 \
    --num_steps 40 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name sdm_solver --num_samples 16 --save_dir $SAVE_DIR \
    --optimized_sigmas_exp euler-uncond-ve-iedm-emin-0.02-emax-0.20-p-1.0-resampled-40-pow-0.5-q-0.25 --load_optimized_sigmas \
    --tau_k 0.0 \
    --calc_curvature \
    --seed 42

python calc_curvature.py \
    --jsonl_path $SAVE_DIR/$DATASET/edm/$EXP-tauk-0.0/gap_log.jsonl \
    --output_path $SAVE_DIR/$DATASET/edm/$EXP-tauk-0.0/curvature_plot.png

# ============== cifar10 ============== #   
DATASET=cifar10
EXP=sdm-schedule-uncond-vp-calc-curvature
python test.py \
    --exp $EXP --dataset_name $DATASET --batch_size 16 \
    --model_weights_pkl edm/edm-cifar10-32x32-uncond-vp.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name sdm_solver --num_samples 16 --save_dir $SAVE_DIR \
    --optimized_sigmas_exp euler-uncond-vp-iedm-emin-0.01-emax-0.40-p-1.0-resampled-18-pow-0.5-q-0.1 --load_optimized_sigmas \
    --tau_k 0.0 \
    --calc_curvature \
    --seed 42

python calc_curvature.py \
    --jsonl_path $SAVE_DIR/$DATASET/edm/$EXP-tauk-0.0/gap_log.jsonl \
    --output_path $SAVE_DIR/$DATASET/edm/$EXP-tauk-0.0/curvature_plot.png

EXP=sdm-schedule-uncond-ve-calc-curvature
python test.py \
    --exp $EXP --dataset_name $DATASET --batch_size 16 \
    --model_weights_pkl edm/edm-cifar10-32x32-uncond-ve.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name sdm_solver --num_samples 16 --save_dir $SAVE_DIR \
    --optimized_sigmas_exp euler-uncond-ve-iedm-emin-0.01-emax-0.40-p-1.0-resampled-18-pow-0.5-q-0.25 --load_optimized_sigmas \
    --tau_k 0.0 \
    --calc_curvature \
    --seed 42

python calc_curvature.py \
    --jsonl_path $SAVE_DIR/$DATASET/edm/$EXP-tauk-0.0/gap_log.jsonl \
    --output_path $SAVE_DIR/$DATASET/edm/$EXP-tauk-0.0/curvature_plot.png

EXP=sdm-schedule-cond-vp-calc-curvature
python test.py \
    --exp $EXP --dataset_name $DATASET --batch_size 16 \
    --model_weights_pkl edm/edm-cifar10-32x32-cond-vp.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name sdm_solver --num_samples 16 --save_dir $SAVE_DIR \
    --optimized_sigmas_exp euler-cond-vp-sdm-emin-0.01-emax-0.40-p-1.0-resampled-18-pow-0.5-q-0.1 --load_optimized_sigmas \
    --tau_k 0.0 \
    --calc_curvature \
    --seed 42

python calc_curvature.py \
    --jsonl_path $SAVE_DIR/$DATASET/edm/$EXP-tauk-0.0/gap_log.jsonl \
    --output_path $SAVE_DIR/$DATASET/edm/$EXP-tauk-0.0/curvature_plot.png

EXP=sdm-schedule-cond-ve-calc-curvature
python test.py \
    --exp $EXP --dataset_name $DATASET --batch_size 16 \
    --model_weights_pkl edm/edm-cifar10-32x32-cond-ve.pkl \
    --channel_size 3 --image_size 32 \
    --num_steps 18 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --sampler_name sdm_solver --num_samples 16 --save_dir $SAVE_DIR \
    --optimized_sigmas_exp euler-cond-ve-iedm-emin-0.02-emax-0.10-p-1.0-resampled-18-pow-0.5-q-0.25 --load_optimized_sigmas \
    --tau_k 0.0 \
    --calc_curvature \
    --seed 42

python calc_curvature.py \
    --jsonl_path $SAVE_DIR/$DATASET/edm/$EXP-tauk-0.0/gap_log.jsonl \
    --output_path $SAVE_DIR/$DATASET/edm/$EXP-tauk-0.0/curvature_plot.png

# ============== imagenet ============== #   
python test.py \
    --exp $EXP --dataset_name $DATASET --batch_size 16 \
    --model_weights_pkl edm/edm-imagenet-64x64-cond-adm.pkl \
    --channel_size 3 --image_size 64 \
    --num_steps 256 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
    --S_churn 40 --S_min 0.05 --S_max 50 --S_noise 1.003 \
    --sampler_name adaptive --num_samples 16 --save_dir $SAVE_DIR \
    --tau_k 0.0 \
    --calc_curvature \
    --seed 42

python calc_curvature.py \
    --jsonl_path $SAVE_DIR/$DATASET/edm/$EXP-tauk-0.0/gap_log.jsonl \
    --output_path $SAVE_DIR/$DATASET/edm/$EXP-tauk-0.0/curvature_plot.png