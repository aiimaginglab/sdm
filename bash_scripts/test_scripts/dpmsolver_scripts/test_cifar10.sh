# !/bin/bash

DATASET="cifar10"
SOLVER="euler"
IMG=32
# MODEL_TYPE="vp" # "vp" / "ve"

RESAMPLED_NUM_STEPS=18
POWER=0.5

# ETA_MIN="0.02"
# ETA_MAX="0.20"
# ETA_P="1.0"
# Q="0.25"

for MODEL_TYPE in uncond-vp uncond-ve cond-vp cond-ve; do

    if [[ "$MODEL_TYPE" == "uncond-vp" ]]; then
        ETA_MIN="0.01"
        ETA_MAX="0.40"
        ETA_P="1.0"
        Q="0.1"
    elif [[ "$MODEL_TYPE" == "uncond-ve" ]]; then
        ETA_MIN="0.01"
        ETA_MAX="0.40"
        ETA_P="1.0"
        Q="0.25"
    elif [[ "$MODEL_TYPE" == "cond-vp" ]]; then
        ETA_MIN="0.01"              
        ETA_MAX="0.40"  
        ETA_P="1.0"
        Q="0.1"
    elif [[ "$MODEL_TYPE" == "cond-ve" ]]; then
        ETA_MIN="0.02"
        ETA_MAX="0.10"      
        ETA_P="1.0"
        Q="0.25"
    fi

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
