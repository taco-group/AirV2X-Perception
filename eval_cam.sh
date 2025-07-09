# CUDA_VISIBLE_DEVICES=5 \
# python opencood/tools/inference_multi_scenario.py \
#     --model_dir opencood/logs/skylink_intermediate_v2xvit/default__2025_05_07_14_15_09/ \
#     --save_vis 1 \
#     --eval_epoch 19 \
#     --save_pred > \
#     opencood/logs/skylink_intermediate_v2xvit/default__2025_05_07_14_15_09/inference.log 2>&1 &

# CUDA_VISIBLE_DEVICES=5 \
# python opencood/tools/inference.py \
#     --model_dir opencood/logs/skylink_intermediate_cobevt/default__2025_05_08_15_37_39 \
#     --save_vis 1 \
#     --eval_epoch 11 \
#     --save_pred > \
#     opencood/logs/skylink_intermediate_cobevt/default__2025_05_08_15_37_39/inference.log 2>&1 &

# CUDA_VISIBLE_DEVICES=5 \
# python opencood/tools/inference.py \
#     --model_dir opencood/logs/skylink_intermediate_sicp/default__2025_05_09_14_26_37 \
#     --save_vis 1 \
#     --eval_epoch 20 \
#     --save_pred

# CUDA_VISIBLE_DEVICES=3 \
# python opencood/tools/inference.py \
#     --model_dir opencood/logs/skylink_intermediate_when2com/default__2025_05_08_11_23_33 \
#     --save_vis 1 \
#     --eval_epoch 20 \
#     --save_pred

# CUDA_VISIBLE_DEVICES=3 \
# python opencood/tools/inference.py \
#     --model_dir opencood/logs/skylink_intermediate_where2com/default__2025_05_07_14_15_16 \
#     --save_vis 1 \
#     --eval_epoch 12 \
#     --save_pred

# CUDA_VISIBLE_DEVICES=3 \
# python opencood/tools/inference.py \
#     --model_dir opencood/logs/skylink_intermediate_where2com/default__2025_05_08_15_33_09 \
#     --save_vis 1 \
#     --eval_epoch 14 \
#     --save_pred


#!/usr/bin/env bash
set -e

# GPU assignments
GPUS=(
    7
    6 
    )

# Corresponding model directories
MODEL_DIRS=(
  "opencood/logs/skylink_intermediate_where2com_camera/default__2025_05_07_14_15_33"
  "opencood/logs/skylink_intermediate_v2xvit_camera/default__2025_05_07_14_15_29"
)

# Evaluation epochs for each model
EPOCHS=(
    19 
    15
    )

# Loop over indices of the arrays
for i in "${!MODEL_DIRS[@]}"; do
  export CUDA_VISIBLE_DEVICES="${GPUS[i]}"
  python opencood/tools/inference_multi_scenario.py \
    --model_dir "${MODEL_DIRS[i]}" \
    --save_vis 1 \
    --eval_epoch "${EPOCHS[i]}" \
    --save_pred > \
    ${MODEL_DIRS[i]}/inference.log 2>&1 &
done



