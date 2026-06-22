#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: $0 vision|tactile}"

REPO="/data/mingxinwang/openpi-univtac"
PY="/data/environment/miniconda3/envs/openpi/bin/python"
RUN_ID="tabero_object_spatial_success_only_20260614"
LOGDIR="$REPO/logs/tabero_training_20260614"

cd "$REPO"
mkdir -p "$LOGDIR"

export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export HF_LEROBOT_HOME="$REPO/.cache/lerobot"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"

run_train() {
  local config="$1"
  local exp_name="$2"
  local repo_id="$3"
  local data_root="$4"
  local assets_dir="$5"
  local asset_id="$6"
  local params_path="$7"

  "$PY" scripts/train.py "$config" \
    --exp-name "$exp_name" \
    --data.repo-id "$repo_id" \
    --data.root "$data_root" \
    --data.assets.assets-dir "$assets_dir" \
    --data.assets.asset-id "$asset_id" \
    --weight-loader.params-path "$params_path" \
    --fsdp-devices 4 \
    --batch-size 256 \
    --num-workers 8 \
    --num-train-steps 30000 \
    --save-interval 1000 \
    --log-interval 10 \
    --overwrite
}

case "$MODE" in
  vision)
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    run_train \
      pi0_lora_vision_tabero \
      tabero_object_spatial_success_only_pi0_vision_h50_bs256_4gpu_20260614 \
      local/tabero_vision_object_spatial_success_only_20260614 \
      "$REPO/.cache/lerobot/local/tabero_vision_object_spatial_success_only_20260614" \
      "$REPO/assets/$RUN_ID/pi0_vision" \
      tabero_vision_pi0_h50 \
      /data/mingxinwang/openpi/.cache/openpi/openpi-assets/checkpoints/pi0_base/params

    run_train \
      pi05_lora_vision_tabero \
      tabero_object_spatial_success_only_pi05_vision_h50_bs256_4gpu_20260614 \
      local/tabero_vision_object_spatial_success_only_20260614 \
      "$REPO/.cache/lerobot/local/tabero_vision_object_spatial_success_only_20260614" \
      "$REPO/assets/$RUN_ID/pi05_vision" \
      tabero_vision_pi05_h50 \
      /data/mingxinwang/openpi-univtac/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
    ;;

  tactile)
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    run_train \
      pi0_lora_tacall_tabero \
      tabero_object_spatial_success_only_pi0_tactile_h50_bs256_4gpu_20260614 \
      local/tabero_tactile_object_spatial_success_only_20260614 \
      "$REPO/.cache/lerobot/local/tabero_tactile_object_spatial_success_only_20260614" \
      "$REPO/assets/$RUN_ID/pi0_tactile" \
      tabero_tactile_pi0_h50 \
      /data/mingxinwang/openpi/.cache/openpi/openpi-assets/checkpoints/pi0_base/params

    run_train \
      pi05_lora_tacall_tabero \
      tabero_object_spatial_success_only_pi05_tactile_h50_bs256_4gpu_20260614 \
      local/tabero_tactile_object_spatial_success_only_20260614 \
      "$REPO/.cache/lerobot/local/tabero_tactile_object_spatial_success_only_20260614" \
      "$REPO/assets/$RUN_ID/pi05_tactile" \
      tabero_tactile_pi05_h50 \
      /data/mingxinwang/openpi-univtac/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
    ;;

  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
