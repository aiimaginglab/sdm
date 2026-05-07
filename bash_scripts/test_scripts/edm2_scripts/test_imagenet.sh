# !/bin/bash

DATASET="imagenet"
SOLVER="euler"
# MODEL_TYPE="vp" # "vp" / "ve"

IMG=512 # 64

PRESETS=("edm2-img512-m-fid") # ("edm2-img512-s-fid" "edm2-img512-m-fid" "edm2-img512-l-fid") # "edm2-img512-xl-fid")
# PRESET="edm2-img512-s-fid"

RESAMPLED_NUM_STEPS=32
POWER=0.5

ETA_MIN="0.1" # "0.001"
ETA_MAX="1.0" # "0.01"
ETA_P="1.0"
Q="0.25"

for PRESET in "${PRESETS[@]}"; do

    EXP="${SOLVER}-${PRESET}-sdm-emin-${ETA_MIN}-emax-${ETA_MAX}-p-${ETA_P}"
    N_STEP_EXP="${EXP}-resampled-${RESAMPLED_NUM_STEPS}-pow-${POWER}-q-${Q}"
    
    # dpm solver
    python test_edm2.py \
        --exp dpm-solver-${PRESET} --dataset_name ${DATASET} --batch_size 128 \
        --preset ${PRESET} \
        --num_steps ${RESAMPLED_NUM_STEPS} --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
        --sampler_name dpm_solver --save_dir results/fid_results/images --num_samples 50000 \
        --save_images \
        --seed 42
    torchrun --standalone --nproc_per_node=1 edm2/calculate_metrics.py calc \
      --images "results/fid_results/images/${DATASET}/edm/dpm-solver-${PRESET}" \
      --ref "https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl" \
      --metrics fid

    # unipc
    python test_edm2.py \
        --exp unipc-${PRESET} --dataset_name ${DATASET} --batch_size 128 \
        --preset ${PRESET} \
        --num_steps ${RESAMPLED_NUM_STEPS} --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
        --sampler_name unipc --save_dir results/fid_results/images --num_samples 50000 \
        --save_images \
        --seed 42
    torchrun --standalone --nproc_per_node=1 edm2/calculate_metrics.py calc \
      --images "results/fid_results/images/${DATASET}/edm/unipc-${PRESET}" \
      --ref "https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl" \
      --metrics fid

    # sdm solver (dpm solver) 
    SCHEDULE="step"
    for TAUK in 2e-5 5e-5 0.0001; do 
        # adaptive solver
        python test_edm2.py \
            --exp sdm-dpmsolver-${PRESET}-tauk-${TAUK} --dataset_name ${DATASET} --batch_size 128 \
            --preset ${PRESET} \
            --num_steps ${RESAMPLED_NUM_STEPS} --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
            --sampler_name sdm_dpmsolver --save_dir results/fid_results/images --num_samples 50000 \
            --tau_k $TAUK --lambda_schedule $SCHEDULE \
            --save_images \
            --seed 42
        torchrun --standalone --nproc_per_node=1 edm2/calculate_metrics.py calc \
        --images "results/fid_results/images/${DATASET}/edm/sdm-dpmsolver-${PRESET}-tauk-${TAUK}" \
        --ref "https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl" \
        --metrics fid
    done

    # sdm schedule + dpm solver
    python test_edm2.py \
        --exp dpm-solver-sdm-schedule-${PRESET} --dataset_name ${DATASET} --batch_size 128 \
        --preset ${PRESET} \
        --num_steps ${RESAMPLED_NUM_STEPS} --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
        --sampler_name dpm_solver --save_dir results/fid_results/images --num_samples 50000 \
        --optimized_sigmas_exp ${N_STEP_EXP} --load_optimized_sigmas \
        --save_images \
        --seed 42
    torchrun --standalone --nproc_per_node=1 edm2/calculate_metrics.py calc \
      --images "results/fid_results/images/${DATASET}/edm/dpm-solver-sdm-schedule-${PRESET}" \
      --ref "https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl" \
      --metrics fid

    # sdm schedule + unipc
    python test_edm2.py \
        --exp unipc-sdm-schedule-${PRESET} --dataset_name ${DATASET} --batch_size 128 \
        --preset ${PRESET} \
        --num_steps ${RESAMPLED_NUM_STEPS} --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
        --sampler_name unipc --save_dir results/fid_results/images --num_samples 50000 \
        --optimized_sigmas_exp ${N_STEP_EXP} --load_optimized_sigmas \
        --save_images \
        --seed 42
    torchrun --standalone --nproc_per_node=1 edm2/calculate_metrics.py calc \
      --images "results/fid_results/images/${DATASET}/edm/unipc-sdm-schedule-${PRESET}" \
      --ref "https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl" \
      --metrics fid

done
