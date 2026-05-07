# !/bin/bash

DATASET="afhqv2"
SOLVER="euler"
# MODEL_TYPE="vp" # "vp" / "ve"

IMG=64

RESAMPLED_NUM_STEPS=40
POWER=0.5

ETA_MIN="0.02"
ETA_MAX="0.20"
ETA_P="1.0"
Q="0.25"

for MODEL_TYPE in uncond-vp uncond-ve; do

    EXP="${SOLVER}-${MODEL_TYPE}-iedm-emin-${ETA_MIN}-emax-${ETA_MAX}-p-${ETA_P}"
    N_STEP_EXP="${EXP}-resampled-${RESAMPLED_NUM_STEPS}-pow-${POWER}-q-${Q}"
    
    python test.py \
        --exp dpm-solver-${MODEL_TYPE} --dataset_name ${DATASET} --batch_size 128 \
        --model_weights_pkl edm/edm-${DATASET}-${IMG}x${IMG}-${MODEL_TYPE}.pkl \
        --channel_size 3 --image_size ${IMG} \
        --num_steps ${RESAMPLED_NUM_STEPS} --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
        --sampler_name dpm_solver --save_dir results/fid_results/images --num_samples 50000 \
        --save_images \
        --seed 42
    torchrun --standalone --nproc_per_node=1 fid.py calc \
      --images "results/fid_results/images/${DATASET}/edm/dpm-solver-${MODEL_TYPE}" \
      --ref "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/${DATASET}-${IMG}x${IMG}.npz"

    python test.py \
        --exp dpm-solver-sdm-schedule-${MODEL_TYPE} --dataset_name ${DATASET} --batch_size 128 \
        --model_weights_pkl edm/edm-${DATASET}-${IMG}x${IMG}-${MODEL_TYPE}.pkl \
        --channel_size 3 --image_size ${IMG} \
        --num_steps ${RESAMPLED_NUM_STEPS} --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
        --sampler_name dpm_solver --save_dir results/fid_results/images --num_samples 50000 \
        --optimized_sigmas_exp ${N_STEP_EXP} --load_optimized_sigmas \
        --save_images \
        --seed 42
    torchrun --standalone --nproc_per_node=1 fid.py calc \
      --images "results/fid_results/images/${DATASET}/edm/dpm-solver-sdm-schedule-${MODEL_TYPE}" \
      --ref "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/${DATASET}-${IMG}x${IMG}.npz"
done
