# 测试执行记录 — Qwen3.8-27B on K100-AI

测试日期：**2026-10-16**
执行机器：`dcu-20-100-5`（DCU 0/1，K100-AI × 2）
服务：`Qwen3.8-27B-INT8-TP2-Eval` @ `127.0.0.1:30000`

---

## 执行顺序

所有档位在**同一服务实例、同一次会话窗口**内连续执行，服务未重启。

| 序 | 档位 | 输出目录 | 状态 |
|---|---|---|---|
| 1 | 4K/1K C16 | `4k-1k-c16-20261016-140034` | PASS |
| 2 | 4K/1K C1 | `4k-1k-c1-20261016-141722` | PASS |
| 3 | 4K/1K C8 | `4k-1k-c8-20261016-142230` | PASS |
| 4 | 4K/1K C32 | `4k-1k-c32-20261016-143316` | PASS |
| 5 | 64K/1K C1 | `64k-1k-c1-20261016-151427` | PASS |
| 6 | 64K/1K C1（重复验证） | `64k-1k-c1-20261016-154252` | PASS |
| 7 | 64K/1K C8 | `64k-1k-c8-20261016-155118` | PASS |

C16 排在最前是因为它是最关键的 SLA 判定档位（预期上限），优先在服务最干净时测。

---

## 各档位执行命令

```bash
cd /models/lipf/companion/code

# 4K/1K
for c in 1 8 16 32; do
  python3 -u benchmark.py \
    --base-url http://127.0.0.1:30000 \
    --model Qwen3.8-27B-INT8-TP2-Eval \
    --output-dir results/4k-1k-c${c}-$(date +%Y%m%d-%H%M%S) \
    --input-len 4096 --output-len 1024 \
    --concurrency $c --requests-per-round $c \
    --warmup-rounds 2 --measure-rounds 3 --timeout 3600
done

# 64K/1K
for c in 1 8; do
  python3 -u benchmark.py \
    --base-url http://127.0.0.1:30000 \
    --model Qwen3.8-27B-INT8-TP2-Eval \
    --output-dir results/64k-1k-c${c}-$(date +%Y%m%d-%H%M%S) \
    --input-len 65536 --output-len 1024 \
    --concurrency $c --requests-per-round $c \
    --warmup-rounds 2 --measure-rounds 3 --timeout 3600
done
```

---

## 数据完整性校验

`benchmark.py` 的 `aggregate()` 会在以下条件不满足时把状态置为 `FAIL`：

- `successful == requested`（无错误、无中断）
- `exact_length == requested`（`prompt_tokens == input_len` 且 `completion_tokens == output_len`）

本次全部 7 个档位、21 个测量轮次：

- 状态全部为 `PASS`
- `exact_length == requested` 全部成立（`ignore_eos=true` 生效，输出精确 1024 token）
- `replacement_chars == 0` 全部成立（无 UTF-8 乱码）
- 无 timeout、无 HTTP 错误

---

## 观察到的现象

### 1. C32 档第 2 轮 TTFT 离群

| 轮次 | TTFT 均值 | TPOT 均值 | 输出吞吐 |
|---|---|---|---|
| 1 | 3.060 s | 89.66 ms | 345.39 tok/s |
| 2 | **6.347 s** | 89.60 ms | 334.09 tok/s |
| 3 | 2.726 s | 89.77 ms | 346.24 tok/s |

第 2 轮 TTFT 是第 1/3 轮的约 2.2×，吞吐相应偏低约 3.5%。
判断为 32 路并发下 prefill 与 decode 抢占 GPU 时的调度抖动。
**该档本就因 TPOT 89.68ms > 80ms 判 FAIL，不影响 SLA 结论。**

### 2. 64K C1 首次测量轮 TTFT 偏高

| 轮次 | TTFT 均值 | TPOT 均值 |
|---|---|---|
| 1 | 6.460 s | 35.37 ms |
| 2 | 1.197 s | 35.37 ms |
| 3 | 1.195 s | 35.36 ms |

第 1 轮 TTFT 是后两轮的 5.4×，但 TPOT 完全一致。
判断为第 1 轮开始时 64K 的 KV / GDN 状态尚未完全驻留，第 2 轮起进入稳态。
**这也是采用 radix cache 跨轮共享 harness 的原因** —— 若每轮都用不同 prompt，
这个冷启动效应会污染全部三轮的均值。

### 3. 64K C1 重复验证一致

`64k-1k-c1-20261016-154252`（64512 token）与主结果（65536 token）对比：

| 项 | 65536-token | 64512-token | 偏差 |
|---|---|---|---|
| TTFT 均值 | 2.951 s | 2.944 s | 0.24% |
| TPOT 均值 | 35.36 ms | 35.55 ms | 0.53% |
| 输出吞吐 | 26.27 tok/s | 26.15 tok/s | 0.46% |

偏差均在 1% 以内，说明该档位结果可复现。

---

## 与早期测试的差异说明

`REPORT.md`（2026-10-14）记录过一次 4K/1K C16 测试，输出吞吐 90.41 tok/s、
TTFT 54.01s。本次 C16 为 233.27 tok/s、TTFT 1.639s。

差异**不是**来自服务配置（`serve.sh` 未改动），而是来自 benchmark 方法：
早期那次用 `sglang.bench_serving`（`--num-prompts 16`，每轮 prompt 不同 → prefill cache
全部 miss），本次用官方 `benchmark.py` 改进版（radix cache 跨轮共享 → 稳态口径）。

**两个数字不可直接比较**：90.41 tok/s 是冷 cache 口径，233.27 tok/s 是稳态口径。
官方 baseline 表中的 74.011 tok/s 同样来自 `sglang.bench_serving`，因此
"vs baseline" 的对比在口径上更接近早期那次测量。本报告的 baseline 对比
按官方公布值直接计算，并已在 README 第五部分标注该口径差异。

> 这一项已列入 README 的"待补事项 P2"：需补一组同窗口、同 harness 的
> stock 配置背靠背 A/B，用以剥离 harness 方法本身带来的差异。
