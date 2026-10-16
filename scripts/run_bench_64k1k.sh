#!/usr/bin/env bash
# =============================================================================
#  64K-in / 1K-out concurrency sweep: C1 / C8
#  官方 benchmark.py 口径：2 轮预热 + 3 轮测量，requests-per-round = concurrency
#  SLA: TTFT < 30s 且 TPOT < 100ms  -> 预期 SLA 最大并发 = C1
# =============================================================================
set -uo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
MODEL="${MODEL:-Qwen3.8-27B-INT8-TP2-Eval}"
OUT_ROOT="${OUT_ROOT:-result/raw}"
BENCH="$(dirname "$0")/benchmark.py"

cd "$(dirname "$0")/.." || exit 1

for c in 1 8; do
  echo "=========================================="
  echo "[64K/1K] concurrency=${c}"
  echo "=========================================="
  python3 -u "$BENCH" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --output-dir "${OUT_ROOT}/64k-1k-c${c}-$(date +%Y%m%d-%H%M%S)" \
    --input-len 65536 --output-len 1024 \
    --concurrency "$c" --requests-per-round "$c" \
    --warmup-rounds 2 --measure-rounds 3 --timeout 3600
done

echo "=========================================="
echo "64K/1K 扫描完成，汇总："
python3 tools/aggregate.py --result-root "$OUT_ROOT" --out-dir result
