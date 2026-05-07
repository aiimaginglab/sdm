# !/bin/bash

SAVE_DIR=results/calc_curvature

# ============== imagenet ============== #   
SOLVER="euler"
DATASET=imagenet
PRESETS=("edm2-img512-s-fid" "edm2-img512-m-fid" "edm2-img512-l-fid")
# PRESET=edm2-img512-m-fid

RESAMPLED_NUM_STEPS=32
POWER=0.5

ETA_MIN="0.1" # "0.001"
ETA_MAX="1.0" # "0.01"
ETA_P="1.0"
Q="0.25"

for PRESET in "${PRESETS[@]}"; do
    
    EXP=${PRESET}-calc-curvature

    python test_edm2.py \
        --exp $EXP --dataset_name $DATASET --batch_size 16 \
        --preset ${PRESET} \
        --channel_size 3 --image_size 512 \
        --num_steps 32 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
        --sampler_name sdm_solver --num_samples 16 --save_dir $SAVE_DIR \
        --tau_k 0.0 \
        --calc_curvature \
        --seed 42

    python calc_curvature.py \
        --jsonl_path $SAVE_DIR/$DATASET/edm/$EXP/gap_log.jsonl \
        --output_path $SAVE_DIR/$DATASET/edm/$EXP/curvature_plot.png

    EXP="${SOLVER}-${PRESET}-sdm-emin-${ETA_MIN}-emax-${ETA_MAX}-p-${ETA_P}"
    N_STEP_EXP="${EXP}-resampled-${RESAMPLED_NUM_STEPS}-pow-${POWER}-q-${Q}"

    EXP=sdm-schedule-${PRESET}-calc-curvature
    python test_edm2.py \
        --exp $EXP --dataset_name $DATASET --batch_size 16 \
        --preset ${PRESET} \
        --channel_size 3 --image_size 512 \
        --num_steps 32 --discretization edm --sigma_min 0.002 --sigma_max 80 --rho 7.0 \
        --sampler_name sdm_solver --num_samples 16 --save_dir $SAVE_DIR \
        --optimized_sigmas_exp ${N_STEP_EXP} --load_optimized_sigmas \
        --tau_k 0.0 \
        --calc_curvature \
        --seed 42

    python calc_curvature.py \
        --jsonl_path $SAVE_DIR/$DATASET/edm/$EXP/gap_log.jsonl \
        --output_path $SAVE_DIR/$DATASET/edm/$EXP/curvature_plot.png
done