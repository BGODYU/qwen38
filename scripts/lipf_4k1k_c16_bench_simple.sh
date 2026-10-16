#!/usr/bin/env bash
# =============================================================================
#  lipf 比赛 v2 — 4K/1K × c=16，单次 sglang.bench_serving，不叠加 max(8, 2×N) 抽样
#  与 lipf_4k1k_c16_bench.sh 的差异：
#    1. 不做 WARM + MEAS phase 区分（沿用 dsv4 模板每 bs 单跑）
#    2. 每轮请求数 = bs（按主人意图，不擅自追加抽样）
#    3. --num-prompts 16 == --max-concurrency 16
#  其他：sglang.bench_serving + CSV 13 列 + grep/awk 提取 + awk 衍生列
# =============================================================================
set -uo pipefail

# 1. 基础配置（与 lipf_4k1k_c16_bench.sh 一致）
export ALLREDUCE_STREAM_WITH_COMPUTE=1
SERVED_MODEL="Qwen3.8-27B-INT8-TP2-Eval"
model="${SERVED_MODEL}"
tp=2
port=30000

time=$(date "+%m%d-%H%M")
HERE="$(cd "$(dirname "$0")" && pwd)"
logpath="${HERE}/../lipf-logs/"
if [ ! -d "${logpath}" ]; then
    mkdir -p "${logpath}"
fi
all_log="${logpath}/lipf-simple-${model}-tp${tp}_${time}.csv"

# 2. 测试矩阵：单组 4K/1K，单档 c=16
INPUT_LEN=4096
OUTPUT_LEN=1024
CONCURRENCY=16
NUM_PROMPTS=16          # = concurrency，主人意图：不叠加 max(8, 2×N) 抽样

# 3. CSV 标题行（13 列：phase/round/input/output/bs + 5 指标 + 3 衍生列）
echo "phase,round,input_len,output_len,bs,TTFT_mean(ms),TPOT_mean(ms),ITL_mean(ms),GenerateThroughput(tokens/s),TotalThroughput(tokens/s),Mean E2E Latency(ms),OutputThroughputPerBS(tokens/s),DecodeThroughputPerBS(tokens/s)" > "$all_log"

# 4. 等服务就绪
echo "=========================================="
echo "lipf-simple v2 比赛压测开始"
echo "模型: ${model}  TP=${tp}  端口: ${port}"
echo "矩阵: in=${INPUT_LEN} / out=${OUTPUT_LEN} / c=${CONCURRENCY} / promp=${NUM_PROMPTS}"
echo "结果 CSV: ${all_log}"
echo "=========================================="

for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
        break
    fi
    echo "waiting for http://127.0.0.1:${port}/v1/models ... ($i/60)"
    sleep 2
done

# 5. 两轮：WARM 烧 kernel + MAIN 取结果（沿用 dsv4 模板参数，仅在主测前多跑一轮预热）
warm_log_file="${logpath}/lipf-simple-${model}-tp${tp}-${time}-WARM-in${INPUT_LEN}-out${OUTPUT_LEN}-bs${CONCURRENCY}.log"
log_file="${logpath}/lipf-simple-${model}-tp${tp}-${time}-MAIN-in${INPUT_LEN}-out${OUTPUT_LEN}-bs${CONCURRENCY}.log"

# Round 1：WARM 预热（cuda graph capture + kernel 全 cold→hot，结果丢弃不写 csv）
echo "=========================================="
echo "[WARM] bs=${CONCURRENCY}, prompt=${INPUT_LEN}, completion=${OUTPUT_LEN}  ← 预热，丢弃结果"
echo "=========================================="

python -m sglang.bench_serving \
    --backend sglang \
    --model "/tmp/q38-target-model-eval-v2" \
    --tokenizer "/models/models/Qwen3.8-27B-Channel-INT8-w8a8" \
    --served-model-name "${SERVED_MODEL}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --dataset-name random-ids \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --random-range-ratio 1 \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${CONCURRENCY}" \
    --request-rate inf 2>&1 | tee "${warm_log_file}"

# Round 2：MAIN 正式测量（kernel/cache 已稳，本轮结果写 csv）
echo "=========================================="
echo "[MAIN] bs=${CONCURRENCY}, prompt=${INPUT_LEN}, completion=${OUTPUT_LEN}  ← 正式，写 CSV"
echo "=========================================="

python -m sglang.bench_serving \
    --backend sglang \
    --model "/tmp/q38-target-model-eval-v2" \
    --tokenizer "/models/models/Qwen3.8-27B-Channel-INT8-w8a8" \
    --served-model-name "${SERVED_MODEL}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --dataset-name random-ids \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --random-range-ratio 1 \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${CONCURRENCY}" \
    --request-rate inf 2>&1 | tee "${log_file}"

# 6. 提取指标（沿用主人 dsv4 模板的 grep + awk {print $NF} 写法）
Output_token_throughput=$(grep -a "^Output token throughput" "${log_file}" | awk '{print $NF}')
Total_Token_throughput=$(grep -a "^Total token throughput"   "${log_file}" | awk '{print $NF}')
Mean_TTFT=$(grep -a "^Mean TTFT" "${log_file}" | awk '{print $NF}')
Mean_TPOT=$(grep -a "^Mean TPOT" "${log_file}" | awk '{print $NF}')
Mean_ITL=$(grep -a "^Mean ITL"   "${log_file}" | awk '{print $NF}')

: "${Output_token_throughput:=N/A}"
: "${Total_Token_throughput:=N/A}"
: "${Mean_TTFT:=N/A}"
: "${Mean_TPOT:=N/A}"
: "${Mean_ITL:=N/A}"

# 7. 衍生列（awk N/A 兜底，与原 lipf 一致）
Mean_E2E=$(awk -v t="${Mean_TTFT}" -v p="${Mean_TPOT}" -v n="${OUTPUT_LEN}" \
    'BEGIN{ if (t=="N/A" || p=="N/A") {print "N/A"} else {printf "%.2f", t + (n-1)*p} }')
OutputThroughputPerBS=$(awk -v g="${Output_token_throughput}" -v b="${CONCURRENCY}" \
    'BEGIN{ if (g=="N/A" || b==0) {print "N/A"} else {printf "%.2f", g/b} }')
DecodeThroughputPerBS=$(awk -v p="${Mean_TPOT}" \
    'BEGIN{ if (p=="N/A" || p==0) {print "N/A"} else {printf "%.2f", 1000/p} }')

# 8. 写 CSV
printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "MAIN" "1" "${INPUT_LEN}" "${OUTPUT_LEN}" "${CONCURRENCY}" \
    "${Mean_TTFT}" "${Mean_TPOT}" "${Mean_ITL}" \
    "${Output_token_throughput}" "${Total_Token_throughput}" \
    "${Mean_E2E}" "${OutputThroughputPerBS}" "${DecodeThroughputPerBS}" >> "${all_log}"

echo "=========================================="
echo "全部测试完成！结果已保存至: ${all_log}"
echo "[MAIN] Mean TTFT=${Mean_TTFT} ms | Mean TPOT=${Mean_TPOT} ms | Output throughput=${Output_token_throughput} tok/s"
echo "=========================================="
cat "${all_log}"
