
#!/usr/bin/env bash
set -e

# GPU assignments
GPUS=(
    7
    7 
    )

# Corresponding model directories
MODEL_DIRS=(
  "opencood/logs/skylink_intermediate_v2xvit_seg/default__2025_05_10_14_38_09/"
  "opencood/logs/skylink_intermediate_where2com_seg/default__2025_05_10_14_48_27"
)

# Evaluation epochs for each model
EPOCHS=(
    15
    5
    )

# Loop over indices of the arrays
for i in "${!MODEL_DIRS[@]}"; do
  export CUDA_VISIBLE_DEVICES="${GPUS[i]}"
  python opencood/tools/inference_skylink_seg.py \
    --model_dir "${MODEL_DIRS[i]}" \
    --save_vis 1 \
    --eval_epoch "${EPOCHS[i]}" > \
    ${MODEL_DIRS[i]}/inference.log 2>&1 &
done



