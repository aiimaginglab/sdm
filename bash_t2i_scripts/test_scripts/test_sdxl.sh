# !/bin/bash

# ============== T2I ============== #  
DATASET="coco"
DATASET_DIR="/workspace/improved-edm/prompts_t2i/sdxl/prompts.json"
MODEL_TYPE="sdxl"
PRETRAINED_MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

ROOT_SAVE_DIR="results/checkpoints_t2i"
IMAGE_SAVE_DIR="results/fid_results_t2i/images"

SOLVER="dpmsolver"

CHANNEL=4
IMG=1024
DOWNSCALE_FACTOR=8

SIGMA_MIN=0.002
SIGMA_MAX=80
RHO=7.0

BATCH=2 
SEED=42

NUM_SAMPLES_FID=10000

RESAMPLED_NUM_STEPS=10
POWER=0.5

ETA_MIN="0.02"
ETA_MAX="0.20"
ETA_P="1.0"
Q="0.25"

GUIDANCE_SCALE=5.0

# ============== FID / CLIP ============== #  
GEN_DIR="results/fid_results_t2i/images"
REF_DIR="/home/flow/data/coco/coco_val10k"
    
# baseline
EXP="baseline_sdxl_cfg_${GUIDANCE_SCALE}"
DISCRETIZATION="default"
python test_t2i.py \
    --exp ${EXP} --dataset_name "${DATASET}" --batch_size "${BATCH}" \
    --pretrained_model ${PRETRAINED_MODEL_NAME} --guidance_scale ${GUIDANCE_SCALE} --subset_json "${DATASET_DIR}" \
    --channel_size "${CHANNEL}" --image_size "${IMG}" --downscale_factor "${DOWNSCALE_FACTOR}" \
    --num_steps "${RESAMPLED_NUM_STEPS}" --discretization "${DISCRETIZATION}" --sigma_min "${SIGMA_MIN}" --sigma_max "${SIGMA_MAX}" --rho "${RHO}" \
    --sampler_name dpm_solver --save_dir "${IMAGE_SAVE_DIR}" --num_samples "${NUM_SAMPLES_FID}" \
    --save_images \
    --seed "${SEED}"
python evaluate_t2i.py \
  --gen_dir "$GEN_DIR/$DATASET/$DISCRETIZATION/$EXP" \
  --ref_dir "$REF_DIR/real_images_1024" \
  --subset_json "$REF_DIR/subset.json"

# edm schedule
EXP="edm_sdxl_cfg_${GUIDANCE_SCALE}"
DISCRETIZATION="edm"
python test_t2i.py \
    --exp "${EXP}" --dataset_name "${DATASET}" --batch_size "${BATCH}" \
    --pretrained_model ${PRETRAINED_MODEL_NAME} --guidance_scale ${GUIDANCE_SCALE} --subset_json "${DATASET_DIR}" \
    --channel_size "${CHANNEL}" --image_size "${IMG}" --downscale_factor "${DOWNSCALE_FACTOR}" \
    --num_steps "${RESAMPLED_NUM_STEPS}" --discretization "${DISCRETIZATION}" --sigma_min "${SIGMA_MIN}" --sigma_max "${SIGMA_MAX}" --rho "${RHO}" \
    --sampler_name dpm_solver --save_dir "${IMAGE_SAVE_DIR}" --num_samples "${NUM_SAMPLES_FID}" \
    --save_images \
    --seed "${SEED}"
python evaluate_t2i.py \
  --gen_dir "$GEN_DIR/$DATASET/$DISCRETIZATION/$EXP" \
  --ref_dir "$REF_DIR/real_images_1024" \
  --subset_json "$REF_DIR/subset.json"

# ays schedule
EXP="ays_sdxl_cfg_${GUIDANCE_SCALE}"
DISCRETIZATION="ays"
python test_t2i.py \
    --exp "${EXP}" --dataset_name "${DATASET}" --batch_size "${BATCH}" \
    --pretrained_model ${PRETRAINED_MODEL_NAME} --guidance_scale ${GUIDANCE_SCALE} --subset_json "${DATASET_DIR}" \
    --channel_size "${CHANNEL}" --image_size "${IMG}" --downscale_factor "${DOWNSCALE_FACTOR}" \
    --num_steps "${RESAMPLED_NUM_STEPS}" --discretization "${DISCRETIZATION}" --sigma_min "${SIGMA_MIN}" --sigma_max "${SIGMA_MAX}" --rho "${RHO}" \
    --sampler_name dpm_solver --save_dir "${IMAGE_SAVE_DIR}" --num_samples "${NUM_SAMPLES_FID}" \
    --save_images \
    --seed "${SEED}"
python evaluate_t2i.py \
  --gen_dir "$GEN_DIR/$DATASET/$DISCRETIZATION/$EXP" \
  --ref_dir "$REF_DIR/real_images_1024" \
  --subset_json "$REF_DIR/subset.json"

# sdm schedule
EXP="${SOLVER}-${MODEL_TYPE}-edm-based-sdm-emin-${ETA_MIN}-emax-${ETA_MAX}-p-${ETA_P}_cfg_${GUIDANCE_SCALE}"
N_STEP_EXP="${EXP}-resampled-${RESAMPLED_NUM_STEPS}-pow-${POWER}-q-${Q}"
DISCRETIZATION="sdm"

python test_t2i.py \
    --exp "${N_STEP_EXP}" --dataset_name "${DATASET}" --batch_size "${BATCH}" \
    --pretrained_model ${PRETRAINED_MODEL_NAME} --guidance_scale ${GUIDANCE_SCALE} --subset_json "${DATASET_DIR}" \
    --channel_size "${CHANNEL}" --image_size "${IMG}" --downscale_factor "${DOWNSCALE_FACTOR}" \
    --num_steps "${RESAMPLED_NUM_STEPS}" --discretization "${DISCRETIZATION}" --sigma_min "${SIGMA_MIN}" --sigma_max "${SIGMA_MAX}" --rho "${RHO}" \
    --sampler_name dpm_solver --save_dir "${IMAGE_SAVE_DIR}" --num_samples "${NUM_SAMPLES_FID}" \
    --optimized_sigmas_save_dir "${ROOT_SAVE_DIR}" --optimized_sigmas_exp "${N_STEP_EXP}" --load_optimized_sigmas \
    --save_images \
    --seed "${SEED}"
python evaluate_t2i.py \
  --gen_dir "$GEN_DIR/$DATASET/$DISCRETIZATION/$N_STEP_EXP" \
  --ref_dir "$REF_DIR/real_images_1024" \
  --subset_json "$REF_DIR/subset.json"
