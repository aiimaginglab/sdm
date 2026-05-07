# !/bin/bash

# ============== imagenet ============== #  
DATASET="imagenet"
DISCRETIZATION="edm"

ROOT_SAVE_DIR="results/checkpoints"
IMAGE_SAVE_DIR="results/fid_results/images"

SOLVER="euler" # "euler" / "heun"
# SAMPLER_NAME="sdm_solver" # "euler_cos" / "heun_cos"

PRESET="${PRESET:-edm2-img512-s-fid}"
CHANNEL=3
IMG=512 # 64

SIGMA_MIN=0.002
SIGMA_MAX=80
RHO=7.0

BATCH=128
SEED=42

NUM_SAMPLES_SCHEDULE=128
NUM_SAMPLES_FID=50000

RESAMPLED_NUM_STEPS=32 # 256
POWER=0.5

ETA_MIN="0.001"
ETA_MAX="0.01"
ETA_P="1.0"
Q="0.25"


exp_dir () {
  local exp="$1"
  echo "${ROOT_SAVE_DIR}/${DATASET}/${DISCRETIZATION}/${exp}"
}

extract_steps_from_exp () {
  local exp="$1"
  local dir
  dir="$(exp_dir "$exp")"

  echo "DEBUG dir=$dir" >&2
  ls -l "$dir" >&2

  local fname
  fname="$(find "$dir" -maxdepth 1 -type f -name "optimized_schedules_${DATASET}_${DISCRETIZATION}_steps_*.npy" | head -n 1)"

  if [[ -z "$fname" ]]; then
    echo "ERROR: no optimized schedule found in $dir" >&2
    exit 1
  fi

  local count
  count="$(find "$dir" -maxdepth 1 -type f -name "optimized_schedules_${DATASET}_${DISCRETIZATION}_steps_*.npy" | wc -l | tr -d ' ')"
  if [[ "$count" != "1" ]]; then
    echo "ERROR: expected exactly one optimized schedule in $dir, found $count" >&2
    find "$dir" -maxdepth 1 -type f -name "optimized_schedules_${DATASET}_${DISCRETIZATION}_steps_*.npy" >&2
    exit 1
  fi

  local base
  base="$(basename "$fname")"

  local steps
  steps="$(echo "$base" | sed -n 's/.*_steps_\([0-9][0-9]*\)\.npy/\1/p')"
  if [[ -z "$steps" ]]; then
    echo "ERROR: failed to parse optimized_num_steps from $base" >&2
    exit 1
  fi

  echo "$steps"
}



EXP="${SOLVER}-${PRESET}-sdm-emin-${ETA_MIN}-emax-${ETA_MAX}-p-${ETA_P}"
N_STEP_EXP="${EXP}-resampled-${RESAMPLED_NUM_STEPS}-pow-${POWER}-q-${Q}"

# # (1) initial eta-based scheduling 
# python test_edm2.py \
#     --exp "${EXP}" --dataset_name "${DATASET}" --batch_size "${BATCH}" \
#     --preset ${PRESET} \
#     --num_steps "${RESAMPLED_NUM_STEPS}" --discretization "${DISCRETIZATION}" --sigma_min "${SIGMA_MIN}" --sigma_max "${SIGMA_MAX}" --rho "${RHO}" \
#     --sampler_name sdm_scheduler --solver "${SOLVER}" --num_samples "${NUM_SAMPLES_SCHEDULE}" --save_dir "${ROOT_SAVE_DIR}" \
#     --eta_min "${ETA_MIN}" --eta_max "${ETA_MAX}" --eta_p "${ETA_P}" \
#     --seed "${SEED}"

# # (1.5) parse optimized_num_steps from EXP dir
# NUM_STEPS_GRID=$(extract_steps_from_exp "${EXP}")
# echo "[${EXP}] optimized_num_steps = ${NUM_STEPS_GRID}"

# (2) visualize eta values for initial generated timestep schedules 
python n_step_resampling_edm2.py \
    --dataset_name "${DATASET}" --batch_size "${BATCH}" \
    --preset ${PRESET} \
    --num_steps 32 --discretization "${DISCRETIZATION}" --sigma_min "${SIGMA_MIN}" --sigma_max "${SIGMA_MAX}" --rho "${RHO}" \
    --solver "${SOLVER}" --num_samples "${NUM_SAMPLES_SCHEDULE}" --save_dir "${ROOT_SAVE_DIR}" \
    --optimized_sigmas_exp "edm" \
    --save_plot \
    --seed "${SEED}"

# # (3) n-step resampling -> saves resampled schedule under N_STEP_EXP dir
# python n_step_resampling_edm2.py \
#     --exp "${N_STEP_EXP}" --dataset_name "${DATASET}" --batch_size "${BATCH}" \
#     --preset ${PRESET} \
#     --num_steps "${NUM_STEPS_GRID}" --discretization "${DISCRETIZATION}" --sigma_min "${SIGMA_MIN}" --sigma_max "${SIGMA_MAX}" --rho "${RHO}" \
#     --solver "${SOLVER}" --num_samples "${NUM_SAMPLES_SCHEDULE}" --save_dir "${ROOT_SAVE_DIR}" \
#     --optimized_sigmas_save_dir "${ROOT_SAVE_DIR}" --optimized_sigmas_exp "${EXP}" --load_optimized_sigmas \
#     --n_step_resampling \
#     --resampled_num_steps "${RESAMPLED_NUM_STEPS}" --power "${POWER}" --q "${Q}" \
#     --seed "${SEED}"

# # (4) (optional) visualize eta values for resampled schedule
# python n_step_resampling_edm2.py \
#     --dataset_name "${DATASET}" --batch_size "${BATCH}" \
#     --preset ${PRESET} \
#     --num_steps "${RESAMPLED_NUM_STEPS}" --discretization "${DISCRETIZATION}" --sigma_min "${SIGMA_MIN}" --sigma_max "${SIGMA_MAX}" --rho "${RHO}" \
#     --solver "${SOLVER}" --num_samples "${NUM_SAMPLES_SCHEDULE}" --save_dir "${ROOT_SAVE_DIR}" \
#     --optimized_sigmas_save_dir "${ROOT_SAVE_DIR}" --optimized_sigmas_exp "${N_STEP_EXP}" --load_optimized_sigmas \
#     --save_plot \
#     --seed "${SEED}"

# # ======================================= #
# # Euler 
# # ======================================= #
# SAMPLER_NAME="euler" # "euler_cos" / "heun_cos"

# # (5) generate images for FID using the resampled schedule
# python test_edm2.py \
#     --exp "${N_STEP_EXP}" --dataset_name "${DATASET}" --batch_size "${BATCH}" \
#     --preset ${PRESET} \
#     --num_steps "${RESAMPLED_NUM_STEPS}" --discretization "${DISCRETIZATION}" --sigma_min "${SIGMA_MIN}" --sigma_max "${SIGMA_MAX}" --rho "${RHO}" \
#     --sampler_name "${SAMPLER_NAME}" --save_dir "${IMAGE_SAVE_DIR}" --num_samples "${NUM_SAMPLES_FID}" \
#     --optimized_sigmas_exp "${N_STEP_EXP}" --load_optimized_sigmas \
#     --save_images \
#     --seed "${SEED}"

# # (6) FID calc
# torchrun --standalone --nproc_per_node=1 edm2/calculate_metrics.py calc \
#   --images "results/fid_results/images/${DATASET}/edm/dpm-solver-${PRESET}" \
#   --ref "https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl" \
#   --metrics fid

# # ======================================= #
# # Heun 
# # ======================================= #
# SAMPLER_NAME="heun" # "euler_cos" / "heun_cos"

# # (5) generate images for FID using the resampled schedule - euler
# python test.py \
#     --exp "${N_STEP_EXP/euler/heun}" --dataset_name "${DATASET}" --batch_size "${BATCH}" \
#     --model_weights_pkl "${MODEL_PKL}" \
#     --channel_size "${CHANNEL}" --image_size "${IMG}" \
#     --num_steps "${RESAMPLED_NUM_STEPS}" --discretization "${DISCRETIZATION}" --sigma_min "${SIGMA_MIN}" --sigma_max "${SIGMA_MAX}" --rho "${RHO}" \
#     --S_churn 40 --S_min 0.05 --S_max 50 --S_noise 1.003 \
#     --sampler_name "${SAMPLER_NAME}" --save_dir "${IMAGE_SAVE_DIR}" --num_samples "${NUM_SAMPLES_FID}" \
#     --optimized_sigmas_exp "${N_STEP_EXP}" --load_optimized_sigmas \
#     --save_images \
#     --seed "${SEED}"

# # (6) FID calc
# torchrun --standalone --nproc_per_node=1 fid.py calc \
#     --images "${IMAGE_SAVE_DIR}/${DATASET}/${DISCRETIZATION}/${N_STEP_EXP/euler/heun}" \
#     --ref "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz"

# # ======================================= #
# # Adaptive solver
# # ======================================= #
# SAMPLER_NAME="sdm_solver" # "euler_cos" / "heun_cos"
# SCHEDULE="step"
# if [[ "$MODEL_TYPE" == "vp" ]]; then
#     TAUK="0.0002"
# elif [[ "$MODEL_TYPE" == "ve" ]]; then
#     TAUK="0.001"
# fi

# # (5) generate images for FID using the resampled schedule - euler
# python test.py \
#     --exp "${N_STEP_EXP/euler/adaptive}" --dataset_name "${DATASET}" --batch_size "${BATCH}" \
#     --model_weights_pkl "${MODEL_PKL}" \
#     --channel_size "${CHANNEL}" --image_size "${IMG}" \
#     --num_steps "${RESAMPLED_NUM_STEPS}" --discretization "${DISCRETIZATION}" --sigma_min "${SIGMA_MIN}" --sigma_max "${SIGMA_MAX}" --rho "${RHO}" \
#     --S_churn 40 --S_min 0.05 --S_max 50 --S_noise 1.003 \
#     --sampler_name "${SAMPLER_NAME}" --save_dir "${IMAGE_SAVE_DIR}" --num_samples "${NUM_SAMPLES_FID}" \
#     --optimized_sigmas_exp "${N_STEP_EXP}" --load_optimized_sigmas \
#     --tau_k "${TAUK}" --lambda_schedule "${SCHEDULE}" \
#     --save_images \
#     --seed "${SEED}"

# # (6) FID calc
# torchrun --standalone --nproc_per_node=1 fid.py calc \
#     --images "${IMAGE_SAVE_DIR}/${DATASET}/${DISCRETIZATION}/${N_STEP_EXP/euler/adaptive}" \
#     --ref "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz"
