# !/bin/bash
IMAGES_DIR="results/fid_results/images" # "results/fid_results/images"
ZIP_DIR="results/fid_results/zip"
EXP="euler-uncond-vp" # "euler-uncond-vp" # "euler-cos-uncond-vp" "heun-uncond-vp" "heun-cos-uncond-vp" "adaptive-uncond-vp" "adaptive-cos-uncond-vp"

# ============== afhqv2 ============== #  
torchrun --standalone --nproc_per_node=1 fid.py calc --images=$IMAGES_DIR/afhqv2/edm/$EXP \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz

# ============== ffhq ============== # 
torchrun --standalone --nproc_per_node=1 fid.py calc --images=$IMAGES_DIR/ffhq/edm/$EXP \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz

# ============== cifar10 ============== #  
torchrun --standalone --nproc_per_node=1 fid.py calc --images=$IMAGES_DIR/cifar10/edm/$EXP \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

# ============== imagenet (edm) ============== # 
torchrun --standalone --nproc_per_node=1 fid.py calc --images=$IMAGES_DIR/imagenet/edm/$EXP \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz

# ============== imagenet (baseline) ============== # 
torchrun --standalone --nproc_per_node=1 fid.py calc --images=$IMAGES_DIR/imagenet/edm/$EXP \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64-baseline.npz	