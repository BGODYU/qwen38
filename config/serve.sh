#!/usr/bin/env bash
# =============================================================================
#  Qwen3.8-27B-INT8-w8a8 on K100-AI x2 — SGLang 0.5.12 serving script
#  与本次性能测试使用的配置完全一致，未做任何改动。
# =============================================================================
set -euo pipefail

MODEL_SRC="/data1/models/Qwen3.8-27B-Channel-INT8-w8a8"
MODEL=/tmp/q38-target-model-eval-v2
TARGET_ROOT=/data/qwen38-27b-k100ai-int8-opt
PATCH=$TARGET_ROOT/runtime_patch_tp2
ROOT=/data1/code/qwen38

mkdir -p "$ROOT/logs" "$MODEL"

for required in \
  "$MODEL_SRC/config.json" \
  "$MODEL_SRC/tokenizer.json" \
  "$MODEL_SRC/chat_template.jinja" \
  "$PATCH/sitecustomize.py" \
  "$TARGET_ROOT/cache/tp2" \
  "$TARGET_ROOT/scripts/launch_sglang_require_sitecustomize.py"; do
  test -e "$required" || { echo "Missing required file: $required" >&2; exit 1; }
done

for required_env in \
  TRITON_JSON_DIR \
  PYTHONPATH \
  SGLANG_REQUIRED_SITECUSTOMIZE_PREFIX \
  SGLANG_Q38_TP2_COMPACT_HEAD_M1 \
  SGLANG_Q38_NATIVE_OUT_GEMV_SO; do
  test -n "${!required_env:-}" || { echo "Missing required environment: $required_env" >&2; exit 1; }
done

for file in "$MODEL_SRC"/*; do
  test -e "$file" && ln -sfn "$file" "$MODEL/$(basename "$file")"
done
ln -sfn /opt/qwen38-k100ai/model_overrides/preprocessor_config.json "$MODEL/preprocessor_config.json"
ln -sfn /opt/qwen38-k100ai/model_overrides/video_preprocessor_config.json "$MODEL/video_preprocessor_config.json"

printf '%s\n' "$$" > "$ROOT/logs/sglang_tp2_v2.pid"

HIP_VISIBLE_DEVICES=0,1 exec python3 -u "$TARGET_ROOT/scripts/launch_sglang_require_sitecustomize.py" \
  --model-path "$MODEL" \
  --random-seed 0 \
  --served-model-name Qwen3.8-27B-INT8-TP2-Eval \
  --chat-template "$MODEL_SRC/chat_template.jinja" \
  --reasoning-parser qwen3 \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --tp-size 2 \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --page-size 64 \
  --mamba-scheduler-strategy extra_buffer \
  --max-mamba-cache-size 80 \
  --cuda-graph-bs 1 2 4 8 16 \
  --disable-piecewise-cuda-graph \
  --context-length 65536 \
  --mem-fraction-static 0.88 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 16384 \
  --pack-paged-kv-to-varlen auto \
  --pack-paged-kv-to-varlen-min-q-tokens 8192 \
  --pack-paged-kv-to-varlen-min-kv-tokens 8192 \
  --max-running-requests 16 \
  --enable-metrics
