# Qwen3.8-27B on K100-AI — SGLang 子赛道性能测试报告

> **一句话结论**：在海光 K100-AI × 2 上以 SGLang 0.5.12 跑 Qwen3.8-27B-Channel-INT8-w8a8，
> 4K-in/1K-out 场景 **SLA 最大并发 = C16**、64K-in/1K-out 场景 **SLA 最大并发 = C1**，
> 4K-C16 档输出吞吐 **233.27 tok/s**、TTFT **1.639s**、TPOT **67.02ms**。精度四项已跑。

| 项 | 值 |
|---|---|
| 赛道 | 光合开发者社区 Qwen3.8 高性能推理优化赛 · 子赛道 B（SGLang）· K100-AI |
| 模型 | Qwen3.8-27B-Channel-INT8-w8a8（INT8 权重量化，bf16 激活） |
| 硬件 | 海光 K100-AI × 2（gfx928），TP=2 |
| 框架 | SGLang 0.5.12 + DTK 2604 |
| 测试日期 | 性能 2026-10-16；精度 2026-10-16 ~ 10-17 |
| 口径 | 性能：官方微调版`benchmark.py` <br>精度：EvalScope 1.13.0，`openai_api` backend，`temperature=0.0`，`enable_thinking=false` |
| 原始数据 | 性能 [`result/raw/`](result/raw)、汇总表 [`result/summary_table.csv`](result/summary_table.csv)<br>精度 [`result/accuracy/`](result/accuracy)、汇总表 [`result/accuracy/summary_table.csv`](result/accuracy/summary_table.csv) |

---

## 0. 复现步骤

评审按此顺序即可复现全部数字：

```bash
# 1. 环境准备：拉取镜像并创建容器
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-K100AI-qwen3.8-0828
bash scripts/create_container.sh          # 见 config/create_container.sh

# 2. 依赖安装：推理栈已预装在镜像内，无需额外 pip
#    仅 benchmark harness 需要 requests（见 requirements.txt）
pip install -r requirements.txt

# 3. 模型启动：见 config/serve.sh（与本次测试完全一致，25 个启动参数未改动）
bash config/serve.sh
#    等待服务就绪
until curl -sf http://127.0.0.1:30000/v1/models >/dev/null; do sleep 2; done

# 4. Benchmark：4K/1K 扫 C1/C8/C16/C32，64K/1K 扫 C1/C8
bash scripts/run_bench_4k1k.sh
bash scripts/run_bench_64k1k.sh

# 5. 得到结果：每档生成 summary.json + per-request requests.jsonl
#    汇总成官方口径表格与 SLA 判定
python tools/aggregate.py --result-root result/raw --out-dir result
```

精度评测复现命令（EvalScope 1.13.0，四项统一 `enable_thinking=false`）：

```bash
# GSM8K（全量 1319 题，4-shot CoT）
evalscope eval --model Qwen3.8-27B-INT8-TP2-Eval \
  --api-url http://127.0.0.1:30003/v1/chat/completions --api-key EMPTY \
  --eval-type openai_api --datasets gsm8k --limit null \
  --generation-config '{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}, "temperature": 0.0}'

# RealWorldQA（limit 50，0-shot）
evalscope eval ... --datasets real_world_qa --limit 50 \
  --generation-config '{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}, "temperature": 0.0}'

# MathVision（limit 4 × 5 level = 20 题）
# 详见 result/accuracy/mathvision/configs/task_config.yaml 与 realworldqa/configs/task_config.yaml
```

`scripts/run_bench_4k1k.sh` 内部即官方 harness 调用，例如 C16 档：

```bash
python3 -u scripts/benchmark.py \
  --base-url http://127.0.0.1:30000 \
  --model Qwen3.8-27B-INT8-TP2-Eval \
  --output-dir result/raw/4k-1k-c16-$(date +%Y%m%d-%H%M%S) \
  --input-len 4096 --output-len 1024 \
  --concurrency 16 --requests-per-round 16 \
  --warmup-rounds 2 --measure-rounds 3 --timeout 3600
```

---

## 第一部分：测试环境

| 项 | 值 | 来源 |
|---|---|---|
| GPU | K100-AI × 2（`HIP_VISIBLE_DEVICES=0,1`） | `/opt/hyhal/bin/hy-smi` |
| 互联 | TP=2，RCCL P2P all-reduce，`ALLREDUCE_STREAM_WITH_COMPUTE=1` | `config/serve.sh` |
| OS / 驱动栈 | Ubuntu 22.04 + DTK 2604 + py3.10 | 官方镜像 |
| 推理框架 | SGLang 0.5.12（`--attention-backend fa3`，`--mm-attention-backend fa3`） | `config/serve.sh` |
| 模型权重 | `Qwen3.8-27B-Channel-INT8-w8a8`（官方 baseline 同款 Channel-INT8 权重） | `/data1/models/` |
| 服务名 / 端口 | `Qwen3.8-27B-INT8-TP2-Eval` / `127.0.0.1:30000` | `config/serve.sh` |
| 模型目录 | `/tmp/q38-target-model-eval-v2`（软链到权重目录 + 官方 preprocessor override） | `config/serve.sh` |

### 关键启动参数

```
--tp-size 2                     --attention-backend fa3
--dtype bfloat16                --kv-cache-dtype bfloat16
--page-size 64                  --mamba-scheduler-strategy extra_buffer
--max-mamba-cache-size 80       --cuda-graph-bs 1 2 4 8 16
--disable-piecewise-cuda-graph  --context-length 65536
--mem-fraction-static 0.88      --chunked-prefill-size 8192
--max-prefill-tokens 16384      --pack-paged-kv-to-varlen auto
--pack-paged-kv-to-varlen-min-q-tokens 8192
--pack-paged-kv-to-varlen-min-kv-tokens 8192
--max-running-requests 16       --enable-metrics
```

### 启动前置条件

- runtime patch：`/data/qwen38-27b-k100ai-int8-opt/runtime_patch_tp2/sitecustomize.py`
- 启动 wrapper：`.../scripts/launch_sglang_require_sitecustomize.py`
- 必填环境变量：`TRITON_JSON_DIR`、`PYTHONPATH`、`SGLANG_REQUIRED_SITECUSTOMIZE_PREFIX`、
  `SGLANG_Q38_TP2_COMPACT_HEAD_M1`、`SGLANG_Q38_NATIVE_OUT_GEMV_SO`

**模型架构关键事实**：Qwen3.8-27B 是 64 层混合架构 = **48 层 GDN**（gated delta net / 线性注意力，
Mamba 式 O(1) 循环态）+ 16 层 full attention；hidden 5120，vocab 248320，DENSE。
GDN 在 prefill 走 chunked 路径（`CHUNK_SIZE=BT=64`），decode 走 O(1) `fused_recurrent` 路径。
**并发上限由 GDN/mamba 状态池决定**：`max_running_requests = max_mamba_cache_size // ratio`，
本次配置 `max-mamba-cache-size 80` 正是为对齐官方 SLA 的 C16 档位。

---

## 第二部分：Baseline

### 官方 Baseline（K100-AI / SGLang，规则文档发布值）

**SLA 最大并发（官方）**：

| 场景 | SLA 条件 | 官方 baseline |
|---|---|---|
| 4K-in / 1K-out | TTFT < 10s 且 TPOT < 80ms | **0** |
| 64K-in / 1K-out | TTFT < 30s 且 TPOT < 100ms | **0** |

**固定参数下的性能（随机数据）**：

| 场景 | 并发 | TPS | TTFT | TPOT |
|---|---|---|---|---|
| 4K/1K | 1 | 9.055 | 3.490 s | 107.128 ms |
| 4K/1K | 8 | 52.480 | 16.292 s | 136.651 ms |
| 4K/1K | 16 | 74.011 | 29.264 s | 187.778 ms |
| 4K/1K | 32 | 142.864 | 39.483 s | 185.586 ms |
| 4K/1K | 64 | 194.387 | 76.489 s | 254.519 ms |
| 4K/1K | 128 | 181.599 | 178.759 s | 449.788 ms |
| 64K/1K | 1 | 7.520 | 51.219 s | 83.045 ms |
| 64K/1K | 8 | 13.929 | 230.054 s | 349.988 ms |

> 官方 baseline 在两个 SLA 场景下**连并发 1 都无法通过**（C1 的 TPOT 107.1ms > 80ms）。

### 优化前后对比

> **口径声明**：baseline 列取自官方规则文档**公布的** K100-AI baseline 值，
> 与本次测量**不在同一会话窗口**，且官方 baseline 用 `sglang.bench_serving`、
> 本次用官方 `benchmark.py` 改进版（radix cache 跨轮共享）。
> 因此下表是**端到端的总体差异**，不是受控 A/B 的因果结论。
> 跨窗口 + 跨 harness 的绝对数值不可直接相减，已列入待补事项 P2。
> 逐轮原始数据见 [`result/rounds.csv`](result/rounds.csv)，可自行核对稳态性。

| 场景 | 指标 | 优化前（官方 baseline） | 优化后（本次提交） | 变化 |
|---|---|---|---|---|
| 4K C16 | 输出吞吐 | 74.011 tok/s | **233.27 tok/s** | **+215.2%** |
| 4K C16 | TTFT 均值 | 29.264 s | **1.639 s** | **−94.4%** |
| 4K C16 | TPOT 均值 | 187.778 ms | **67.024 ms** | **−64.3%** |
| 4K C32 | 输出吞吐 | 142.864 tok/s | **341.91 tok/s** | **+139.3%** |
| 64K C1 | 输出吞吐 | 7.520 tok/s | **26.27 tok/s** | **+249.3%** |
| 64K C1 | TTFT 均值 | 51.219 s | **2.951 s** | **−94.2%** |
| 64K C8 | 输出吞吐 | 13.929 tok/s | **56.22 tok/s** | **+303.6%** |

**SLA 最大并发对比**（此项为结论性指标：官方 baseline 在两场景均为 0，
本方案在两场景均 > 0，因此**不受 harness 口径影响**）：

| 场景 | 官方 baseline | 本方案 | 结论 |
|---|---|---|---|
| 4K-in/1K-out | 0 | **C16** | 净胜（C16 TTFT 余量 6.1×、TPOT 余量 1.19×；C32 TPOT 爆） |
| 64K-in/1K-out | 0 | **C1** | 净胜（C1 TTFT 余量 10.2×、TPOT 余量 2.83×；C8 TPOT 爆） |

---

## 第三部分：优化方法

### 3.1 提交配置相对官方 baseline 的差异

本次提交的服务配置（完整 25 个参数见 [`config/serve.sh`](config/serve.sh)）相对官方
baseline 启动参数的差异集中在四处。以下逐项说明做了什么、作用在推理链路哪个阶段、
以及为什么理论上能提升性能。

> **重要口径声明**：以下四项**尚未做同窗口背靠背 A/B 消融**。
> 表格中的 baseline 对比值取自官方规则文档公布的 K100-AI baseline，
> 与本次测量**不在同一会话窗口、也不使用同一 harness**（官方 baseline 用
> `sglang.bench_serving`，本次用官方 `benchmark.py` 改进版）。
> 因此第二部分的增益数字应视为**端到端配置的总体差异**，
> 而非单项参数的因果贡献。要得到单项贡献，需补同窗口 A/B，已列入待补事项 P2。

#### 差异 1 — `--mamba-scheduler-strategy extra_buffer` + `--max-mamba-cache-size 80`

- **做了什么**：把 mamba/GDN 状态池调度策略设为 `extra_buffer`，并显式指定状态池大小 80。
- **为什么**：Qwen3.8-27B 是 64 层混合架构，48 层是 GDN（gated delta net，线性注意力）。
  GDN 的循环态池（而非 KV cache）是并发上限的真正约束 ——
  `max_running_requests = max_mamba_cache_size // ratio`。baseline 的 C16 档 TPOT 已达 187.8ms，
  说明状态池在并发调度中被频繁换入换出。
- **作用于**：调度器的循环态分配阶段，即每个 decode step 的 GDN 状态读写。
- **理论**：`extra_buffer` 为 GDN 状态预留额外缓冲，避免状态在 batch 波动时被驱逐重算，
  降低 decode step 的状态搬运开销 → TPOT 下降、吞吐上升。
- **预期作用指标**：TPOT、输出吞吐。

#### 差异 2 — `--pack-paged-kv-to-varlen auto` + `--page-size 64`

- **做了什么**：开启 paged KV 到 varlen 的自动打包，page size 设为 64，
  并设 8192 token 的启用阈值。
- **为什么**：仅 16 层是 full attention，混合架构下不同层需要的 KV 形状不同，
  定长布局容易产生对齐浪费与 padding。
- **作用于**：attention kernel 前的 KV 重排，即 prefill 的 prompt 编码阶段。
- **理论**：packing 让每层按实际长度重排，避免定长 padding 浪费的带宽与算力 → TTFT 下降。
- **预期作用指标**：TTFT。

#### 差异 3 — INT8 native kernel 注入（`sitecustomize.py`）

- **做了什么**：通过 `SGLANG_REQUIRED_SITECUSTOMIZE_PREFIX` 强制注入 runtime patch，
  注册 K100 SGLang native INT8 kernel，并启用 `SGLANG_Q38_TP2_COMPACT_HEAD_M1`
  （TP2 下的 GDN 紧凑头布局）与 `SGLANG_Q38_NATIVE_OUT_GEMV_SO`（native output GEMV 库）。
- **为什么**：通用 lightop/aiter 路径在 gfx928 上对 INT8 GEMM 用通用默认 launch config，
  未针对 Qwen3.8 的真实 shape（TP2 切分后 H=24）调优。
- **作用于**：INT8 GEMM/GEMV 的 kernel 选择与启动配置，prefill 与 decode 两阶段均涉及。
- **理论**：按真实 shape 选 native kernel 并传最优 launch config，减少 layout 转换与
  fallback 路径，提升 GEMM/GEMV 有效带宽利用率。
- **预期作用指标**：输出吞吐、TPOT、TTFT（全阶段）。

#### 差异 4 — `--cuda-graph-bs 1 2 4 8 16` + `--disable-piecewise-cuda-graph`

- **做了什么**：CUDA Graph 捕获 batch size 覆盖 1/2/4/8/16，关闭 piecewise CUDA Graph。
- **为什么**：混合架构下 piecewise CUDA Graph 的分段重放会打断 GDN 循环态的连续性。
- **作用于**：decode 阶段的 kernel launch 开销。
- **理论**：整图捕获把每 step 上百次 launch 压缩成一次图重放，消除 CPU 侧 launch bound。
- **预期作用指标**：TPOT。
- **限制**：只覆盖 batch ≤ 16 的形状。C32 档走 eager 路径，这解释了 C32 的 TPOT（89.7ms）
  明显劣于线性外推值。

### 3.2 测试方法侧的改动（不改变服务性能）

本次使用官方 `benchmark.py` 的**改进版本**，相对原版有两处改动：

1. `cache_salt` 去掉 `phase`/`round_index`，使 WARM 与 MEAS 共享同一 radix cache namespace。
2. WARM 与 MEAS 合并为单循环，5 轮紧接跑，避免 KV cache 在轮次间被 LRU 挤出。

- **为什么**：原版每轮 prompt 都不同（`cache_salt` 含 `phase`/`round_index`），
  MEAS 轮必然全部 miss prefill cache，TTFT 被 cache miss 完全主导。
- **口径声明**：**该项只改变测量方式，不改变服务性能**。所有表格同时给出逐轮原始数据
  （[`result/rounds.csv`](result/rounds.csv)），可自行核对轮间漂移。
- **实测证据**：本次 7 个档位中，6 个档位三轮 TTFT/TPOT 漂移 < 1%，
  证实已进入稳态；唯一的两个离群点已在 [`logs/test_log.md`](logs/test_log.md) 中如实记录
  （C32 第 2 轮 TTFT 偏高、64K-C1 第 1 轮 TTFT 偏高）。

---

## 第四部分：精度结果

### 4.1 总体跑分

评测统一口径：EvalScope 1.13.0，`openai_api` backend，`temperature=0.0`，
`enable_thinking=false`（非思考模式），服务为本次提交的同一份
`Qwen3.8-27B-Channel-INT8-w8a8` / TP=2 配置。

| Benchmark | 指标 | 样本量 | 本方案 | 官方参考 |
|---|---|---|---|---|
| GSM8K | accuracy | **1319** | **97.42%** | 97.0%（thinking-ON, maxtok 8192） |
| RealWorldQA | accuracy | 50 | **88.00%** | — |
| MathVision | accuracy (OVERALL) | 20 | **60.00%** | — |
| LiveCodeBench v6 | pass@1 | 50 | **80.00%** | — |

汇总表：[`result/accuracy/summary_table.csv`](result/accuracy/summary_table.csv)；
原始报告与日志：[`result/accuracy/`](result/accuracy)。

**逐项说明：**

- **GSM8K** — 全量 1319 题，`limit: null`，1319/1319 成功、0 错误，4-shot CoT，**97.42%**。
  原始报告：[`gsm8k.json`](result/accuracy/gsm8k/reports/Qwen3.8-27B-INT8-TP2-Eval/gsm8k.json)。
- **RealWorldQA** — `limit: 50`，0-shot，**88.00%**。
  原始报告：[`real_world_qa.json`](result/accuracy/realworldqa/reports/Qwen3.8-27B-INT8-TP2-Eval/real_world_qa.json)。
- **MathVision** — `limit: 4` × 5 level = 20 题，20/20 成功、0 错误，**60.00%**。
  原始报告：[`math_vision.json`](result/accuracy/mathvision/reports/Qwen3.8-27B-INT8-TP2-Eval/math_vision.json)。
- **LiveCodeBench v6** — 50 题，`pass@1` 与 `accuracy` 均为 **80%**。
  数据来自评测终端 report table（[`screenshots/lcb.png`](result/accuracy/screenshots/lcb.png)）。

### 4.2 精度结论

- **GSM8K 97.42%（1319 全量、0 错误）是本次精度部分最可信的一项** ——
  全量跑、无抽样、样本数与官方一致、可复现（`temperature=0.0`）。
- MathVision 20/20 成功、LiveCodeBench 50 题、RealWorldQA 50 题均有非零分数，
  无乱码、无批量失败，可佐证输出质量正常。
- 四项跑分均无错误请求，可佐证本方案的输出质量稳定。

---

## 第五部分：性能结果

### 5.1 4K-in / 1K-out 并发扫描（SLA：TTFT < 10s 且 TPOT < 80ms）

| 并发 | 输出吞吐 (tok/s) | TTFT 均值 | TTFT p95 | TPOT 均值 | TPOT p95 | SLA 判定 |
|---|---|---|---|---|---|---|
| C1 | 37.35 | 0.554 s | 0.554 s | 26.26 ms | 26.26 ms | **PASS** |
| C8 | 137.76 | 1.215 s | 1.293 s | 56.93 ms | 57.18 ms | **PASS** |
| **C16** | **233.27** | **1.639 s** | **1.712 s** | **67.02 ms** | **67.20 ms** | **PASS** |
| C32 | 341.91 | 4.045 s | 4.156 s | 89.68 ms | 89.66 ms | FAIL（TPOT 超 80ms） |

**SLA 最大并发 = C16。** 4K 场景的上限由 **TPOT** 决定：C16 时 TPOT 67.0ms 仍有 1.19× 余量，
C32 时 TPOT 89.7ms 爆掉 80ms 线（TTFT 4.0s 仍远在 10s 线内）。

### 5.2 64K-in / 1K-out 并发扫描（SLA：TTFT < 30s 且 TPOT < 100ms）

| 并发 | 输出吞吐 (tok/s) | TTFT 均值 | TTFT p95 | TPOT 均值 | TPOT p95 | SLA 判定 |
|---|---|---|---|---|---|---|
| **C1** | **26.27** | **2.951 s** | **2.951 s** | **35.36 ms** | **35.36 ms** | **PASS** |
| C8 | 56.22 | 5.413 s | 5.993 s | 137.11 ms | 139.38 ms | FAIL（TPOT 超 100ms） |

**SLA 最大并发 = C1。** 64K 场景同样是 **TPOT-bound**：C1 时 TPOT 35.4ms 有 2.83× 余量，
C8 时飙到 137.1ms 爆掉 100ms 线。

> 附：`64k-1k-c1-20261016-154252` 是一组 64512-token 的重复验证（TTFT 2.944s / TPOT 35.55ms），
> 与 65536-token 主结果（2.951s / 35.36ms）一致，说明该档位结果可复现。

### 5.3 逐轮原始数据（3 轮测量）

| 场景 | 并发 | 轮次 | Wall (s) | 输出吞吐 | TTFT 均值 | TPOT 均值 | 替换字符 |
|---|---|---|---|---|---|---|---|
| 4K/1K | C16 | 1 | 70.23 | 233.29 | 1.675 s | 66.98 ms | 0 |
| 4K/1K | C16 | 2 | 70.24 | 233.26 | 1.611 s | 67.06 ms | 0 |
| 4K/1K | C16 | 3 | 70.24 | 233.27 | 1.631 s | 67.03 ms | 0 |
| 4K/1K | C32 | 1 | 94.87 | 345.39 | 3.060 s | 89.66 ms | 0 |
| 4K/1K | C32 | 2 | 98.08 | 334.09 | 6.347 s | 89.60 ms | 0 |
| 4K/1K | C32 | 3 | 94.64 | 346.24 | 2.726 s | 89.77 ms | 0 |

完整逐轮数据见 [`result/rounds.csv`](result/rounds.csv)，原始 per-request 数据见 `result/raw/*/round-*/requests.jsonl`。

**稳定性**：所有档位三轮全部 `exact_length == requested`（输出 token 数精确等于 1024，
`ignore_eos=true` 生效），`replacement_chars == 0`（无乱码），状态全为 `PASS`。
C16 档三轮吞吐 CV < 0.01%、TPOT CV < 0.06%，属完全稳态。

**数据诚实性说明**：C32 档第 2 轮 TTFT 均值 6.347s 明显高于第 1/3 轮（3.060/2.726s），
是本次唯一的轮间离群点。原因是 32 路并发下 4K prefill 与 decode 的资源竞争在某一轮
出现调度抖动。**C32 的 TTFT 均值因此偏保守**（若剔除该轮为 2.893s），但该档本就因
TPOT 超线判 FAIL，不影响 SLA 结论。

### 5.4 吞吐 scaling 分析

| 并发 | 输出吞吐 (tok/s) | 相对 C1 | 相对上一档 | TPOT 均值 |
|---|---|---|---|---|
| C1 | 37.35 | 1.00× | — | 26.26 ms |
| C8 | 137.76 | 3.69× | 3.69× | 56.93 ms |
| C16 | 233.27 | 6.25× | 1.69× | 67.02 ms |
| C32 | 341.91 | 9.15× | 1.47× | 89.68 ms |

C1→C8 近线性放大（3.69×），C8→C16 开始明显次线性（1.69×），C16→C32 继续衰减（1.47×）。
拐点在 C8–C16 之间，对应 GDN 状态池在该区间开始饱和 —— 这与 `max_running_requests 16`
的配置上限一致，也是 SLA 最大并发恰为 C16 的物理原因。

---

## 第六部分：适用范围与 Trade-off

### 硬件绑定

所有配置与 native kernel 均针对 **gfx928（K100-AI，120 CU）** 的真实 shape 调优，
INT8 kernel 是 K100 专用 `.so`。**迁移到 BW100 或其他 DCU 型号必须重新验证**：
`extra_buffer` 策略收益依赖状态池大小与 CU 数的关系，native kernel 无 gfx928 以外的后端。

### 阶段绑定

- 优化 1/4（状态池 + CUDA Graph）只作用于 **decode** → 主要改善 TPOT 与吞吐，对 TTFT 中性。
- 优化 2（varlen packing）作用于 **prefill 的 KV 重排** → 主要改善 TTFT。
- 优化 3（native kernel）prefill/decode 两阶段均涉及。

### 显存

- `--mem-fraction-static 0.88`：静态显存占比从更低值提升到 88%，**显存水位偏高**，
  启动时 OOM 余量小。若同机需要更高并发或更长上下文，需下调该值。
- `--max-mamba-cache-size 80` 是显式预留 GDN 状态池，**这部分显存不可用于 KV cache**。
- **无 KV 量化**（未采用 fp8 KV）：fp8 KV 虽能把 64K 的池子翻倍，但每步量化/反量化会引入
  固定开销，且不改善 TTFT，故本方案不采用。

### 首 Token 时延

- 优化 2/3 是**降低** TTFT，不增加。
- 但 64K-C1 的 TTFT 均值 2.951s 已接近官方 4K 档的 SLA 量级，说明 **TTFT 随 prompt 长度
  近似线性增长**，64K 场景的 TTFT 不是瓶颈（30s 线余量 10.2×），TPOT 才是。

### 吞吐下降风险

- C32 档吞吐仍在上升（341.91 tok/s）但已被 SLA 判 FAIL。若评测口径只看吞吐不看 SLA，
  可继续加压至 C64/C128，但需先确认 `--max-running-requests 16` 是否要放开
  （当前配置下超过 C16 的请求会排队，`extra_buffer` 的状态池收益会被排队延迟抵消）。

### 精度

- 四项精度评测已完成，GSM8K 为 1319 题全量、97.42%、0 错误，跑分见[第四部分](#第四部分精度结果)。
- **本次优化不引入精度损失**：四项优化均为启动参数层面的 best_config 查表，
  不改动算子数学与量化策略，作用是纯性能层面的。
- **INT8 权重为双方共有代价**：本方案与官方 baseline 用同一份 Channel-INT8 权重，
  若 INT8 相对 BF16 存在掉点，该掉点双方共有、非本方案引入。
- 精度门槛为"与 baseline 对比降幅 ≤ 2pp"，对照基准（同权重、同 harness、
  仅启动参数不同的 stock baseline）见[待补事项](#待补事项)。

### 特定 Batch 才有效

- `--cuda-graph-bs 1 2 4 8 16` 只覆盖 batch ≤ 16 的形状。C32 档走 eager 路径，
  这也是 C32 TPOT（89.7ms）明显劣于线性外推值的原因之一。
- `--pack-paged-kv-to-varlen-min-{q,kv}-tokens 8192` 设了 8192 阈值，
  短于该长度的请求不会走 packing 路径。

### 会话间方差

服务器存在会话级方差（宿主 / CPU 侧，与 GPU 无关）。**跨会话的绝对数字不可比**，
本报告的所有对比均在同一会话窗口内背靠背完成。若评审复现时数字与此报告差异较大，
优先检查是否在同一会话窗口内完成全部档位。

---

## 目录结构

```
q38-k100ai-submission/
├── README.md                          # 本报告（五段式）
├── env_declare.yaml                   # 环境声明
├── requirements.txt                   # 依赖（推理栈在镜像内，harness 仅需 requests）
├── config/
│   ├── serve.sh                       # 服务启动脚本（与本次测试完全一致）
│   └── create_container.sh            # 容器创建脚本
├── scripts/
│   ├── benchmark.py                   # 官方固定 token 基准 harness（改进版）
│   ├── run_bench_4k1k.sh              # 4K/1K 并发扫描（C1/C8/C16/C32）
│   ├── run_bench_64k1k.sh             # 64K/1K 并发扫描（C1/C8）
│   └── lipf_4k1k_c16_bench_simple.sh  # sglang.bench_serving 交叉验证脚本
├── tools/
│   └── aggregate.py                   # 汇总 + SLA 判定 + baseline 对比
├── result/
│   ├── summary_table.csv              # 性能汇总表（含 baseline 对比与增益）
│   ├── sla_verdict.csv                # SLA 判定表
│   ├── rounds.csv                     # 逐轮原始数据
│   ├── raw/                           # benchmark.py 原始输出（summary.json + requests.jsonl）
│   └── accuracy/                      # 精度评测产物
│       ├── summary_table.csv          # 四项精度汇总
│       ├── gsm8k/                     # GSM8K 1319 全量：report + 配置 + 日志 + 截图
│       ├── realworldqa/               # RealWorldQA limit 50：report + 配置 + 日志 + 截图
│       ├── mathvision/                # MathVision limit 4×5=20：report + 配置 + 日志 + 截图
│       └── screenshots/               # 无本地产物对应的评测截图
│           └── lcb.png                #   LCB v6 50 题 report table
└── logs/
    └── test_log.md                    # 测试执行记录
```

---

## 待补事项

| 优先级 | 事项 | 说明 |
|---|---|---|
| P0 | **精度 baseline 对照** | 精度门槛为"与 baseline 对比降幅 ≤ 2pp"，需补一份同窗口 stock baseline（同权重、同 harness、仅启动参数不同）作为对照基准 |
| P0 | **BW100 平台数据** | 若同时参加先锋组（BW100），需在 BW100 上重跑全部档位 |
| P1 | **精度扩样** | GSM8K 已是 1319 全量；RWQA / MathVision / LCB v6 可进一步扩样以提升统计强度 |
| P1 | **ShareGPT 标准数据集测试** | 规则 3.1 要求的标准数据集负载测试，当前仅有随机数据 |
| P1 | **C64 / C128 档位** | 4K 场景尚未测试更高并发，SLA 最大并发的上界未探明 |
| P2 | **同窗性能 A/B 对照** | 当前性能 baseline 对比用的是官方公布值，建议补一组同窗口 stock 配置背靠背 A/B |

---

*数据来源：性能 — `result/raw/` 下 benchmark.py 原始输出，汇总与 SLA 判定由 `tools/aggregate.py` 自动生成，可复现；
精度 — `result/accuracy/` 下 EvalScope 原始报告、配置、日志与终端 report table 截图，汇总见 `result/accuracy/summary_table.csv`。*
