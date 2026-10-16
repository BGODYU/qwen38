#!/usr/bin/env python3
"""Aggregate benchmark.py result directories into the official SLA report tables.

Reads every ``<result-dir>/summary.json`` (top level) plus its per-round
``round-*/summary.json`` and ``round-*/requests.jsonl``, then emits:

* ``summary_table.csv``  - one row per (input_len, concurrency) with 3-round stats
* ``sla_verdict.csv``    - SLA pass/fail per row
* ``rounds.csv``         - per-round raw numbers

Official SLA (K100-AI, sglang sub-track):
* 4K-in / 1K-out : TTFT < 10s  AND TPOT < 80ms
* 64K-in / 1K-out: TTFT < 30s  AND TPOT < 100ms
"""

import argparse
import csv
import json
import os

SLA = {
    4096: {"ttft_s": 10.0, "tpot_ms": 80.0},
    65536: {"ttft_s": 30.0, "tpot_ms": 100.0},
}

# Longest prompt length that still counts as the 4K SLA bucket.
SHORT_PROMPT_MAX = 4096


def sla_for(input_len):
    """Map a prompt length onto its SLA bucket; 64K-class prompts share one bucket."""
    return SLA[4096] if input_len <= SHORT_PROMPT_MAX else SLA[65536]

# Official K100-AI / sglang baseline published in the challenge rules doc.
BASELINE = {
    (4096, 1): {"tps": 9.055, "ttft_ms": 3490.0, "tpot_ms": 107.128},
    (4096, 8): {"tps": 52.480, "ttft_ms": 16292.0, "tpot_ms": 136.651},
    (4096, 16): {"tps": 74.011, "ttft_ms": 29264.0, "tpot_ms": 187.778},
    (4096, 32): {"tps": 142.864, "ttft_ms": 39483.0, "tpot_ms": 185.586},
    (4096, 64): {"tps": 194.387, "ttft_ms": 76489.0, "tpot_ms": 254.519},
    (4096, 128): {"tps": 181.599, "ttft_ms": 178759.0, "tpot_ms": 449.788},
    (65536, 1): {"tps": 7.520, "ttft_ms": 51219.0, "tpot_ms": 83.045},
    (65536, 8): {"tps": 13.929, "ttft_ms": 230054.0, "tpot_ms": 349.988},
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_requests(round_dir):
    """Global per-request means, mirroring the official mean-based SLA scoring."""
    path = os.path.join(round_dir, "requests.jsonl")
    if not os.path.exists(path):
        return None
    ttfts, tpots = [], []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("error") is not None or not item.get("done"):
                continue
            if item.get("ttft_s") is not None:
                ttfts.append(item["ttft_s"])
            if item.get("tpot_ms") is not None:
                tpots.append(item["tpot_ms"])
    if not ttfts or not tpots:
        return None
    return {
        "ttft_mean_s": sum(ttfts) / len(ttfts),
        "tpot_mean_ms": sum(tpots) / len(tpots),
        "samples": len(ttfts),
    }


def collect(result_root):
    runs = []
    for name in sorted(os.listdir(result_root)):
        run_dir = os.path.join(result_root, name)
        top_path = os.path.join(run_dir, "summary.json")
        if not os.path.isdir(run_dir) or not os.path.exists(top_path):
            continue
        top = read_json(top_path)
        rounds = []
        for entry in sorted(os.listdir(run_dir)):
            if not entry.startswith("round-"):
                continue
            round_dir = os.path.join(run_dir, entry)
            round_summary = read_json(os.path.join(round_dir, "summary.json"))
            per_request = read_requests(round_dir)
            rounds.append({"summary": round_summary, "per_request": per_request})
        runs.append({"name": name, "top": top, "rounds": rounds})
    return runs


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def build_rows(runs):
    """Collapse the three measured rounds of each run into one comparable row."""
    rows = []
    for run in runs:
        top = run["top"]
        input_len = top["input_len"]
        concurrency = top["concurrency"]
        if top.get("status") != "PASS" or not run["rounds"]:
            continue

        tps = [r["summary"]["output_throughput_tps"] for r in run["rounds"]]
        ttft_round_mean = [r["summary"]["ttft_s"]["mean"] for r in run["rounds"]]
        ttft_p95 = [r["summary"]["ttft_s"]["p95"] for r in run["rounds"]]
        tpot_round_mean = [r["summary"]["tpot_ms"]["mean"] for r in run["rounds"]]
        tpot_p95 = [r["summary"]["tpot_ms"]["p95"] for r in run["rounds"]]

        # Global per-request means across all measured rounds (official mean scoring).
        req_ttft, req_tpot, req_n = [], [], 0
        for r in run["rounds"]:
            if r["per_request"]:
                req_ttft.append(r["per_request"]["ttft_mean_s"])
                req_tpot.append(r["per_request"]["tpot_mean_ms"])
                req_n += r["per_request"]["samples"]

        ttft_mean = mean(req_ttft) if req_ttft else mean(ttft_round_mean)
        tpot_mean = mean(req_tpot) if req_tpot else mean(tpot_round_mean)

        sla = sla_for(input_len)
        ttft_pass = ttft_mean is not None and ttft_mean < sla.get("ttft_s", float("inf"))
        tpot_pass = tpot_mean is not None and tpot_mean < sla.get("tpot_ms", float("inf"))

        baseline_bucket = 4096 if input_len <= SHORT_PROMPT_MAX else 65536
        baseline = BASELINE.get((baseline_bucket, concurrency))
        row = {
            "input_len": input_len,
            "output_len": top["output_len"],
            "concurrency": concurrency,
            "requests_per_round": top["requests_per_round"],
            "measured_rounds": len(run["rounds"]),
            "requests_scored": req_n,
            "output_tps_mean": mean(tps),
            "output_tps_min": min(tps),
            "output_tps_max": max(tps),
            "ttft_mean_s": ttft_mean,
            "ttft_mean_ms": ttft_mean * 1000.0 if ttft_mean is not None else None,
            "ttft_p95_s": mean(ttft_p95),
            "tpot_mean_ms": tpot_mean,
            "tpot_p95_ms": mean(tpot_p95),
            "sla_ttft_limit_s": sla.get("ttft_s"),
            "sla_tpot_limit_ms": sla.get("tpot_ms"),
            "sla_pass": "PASS" if (ttft_pass and tpot_pass) else "FAIL",
            "sla_fail_reason": ", ".join(
                ([] if ttft_pass else ["TTFT"]) + ([] if tpot_pass else ["TPOT"])
            ),
            "baseline_tps": baseline["tps"] if baseline else None,
            "tps_gain_pct": (
                (mean(tps) / baseline["tps"] - 1.0) * 100.0 if baseline and mean(tps) else None
            ),
            "baseline_ttft_ms": baseline["ttft_ms"] if baseline else None,
            "baseline_tpot_ms": baseline["tpot_ms"] if baseline else None,
            "source_dir": run["name"],
        }
        rows.append(row)
    rows.sort(key=lambda r: (r["input_len"], r["concurrency"]))
    return rows


def fmt(value, spec):
    if value is None:
        return ""
    return format(value, spec)


def write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    runs = collect(args.result_root)
    rows = build_rows(runs)
    os.makedirs(args.out_dir, exist_ok=True)

    summary_fields = [
        "input_len", "output_len", "concurrency", "requests_per_round",
        "measured_rounds", "requests_scored",
        "output_tps_mean", "output_tps_min", "output_tps_max",
        "ttft_mean_ms", "ttft_p95_s",
        "tpot_mean_ms", "tpot_p95_ms",
        "sla_ttft_limit_s", "sla_tpot_limit_ms", "sla_pass", "sla_fail_reason",
        "baseline_tps", "tps_gain_pct", "baseline_ttft_ms", "baseline_tpot_ms",
        "source_dir",
    ]
    write_csv(os.path.join(args.out_dir, "summary_table.csv"), rows, summary_fields)

    sla_fields = [
        "input_len", "output_len", "concurrency",
        "ttft_mean_ms", "tpot_mean_ms",
        "sla_ttft_limit_s", "sla_tpot_limit_ms", "sla_pass", "sla_fail_reason",
        "output_tps_mean", "baseline_tps", "tps_gain_pct",
    ]
    write_csv(os.path.join(args.out_dir, "sla_verdict.csv"), rows, sla_fields)

    round_rows = []
    for run in runs:
        for r in run["rounds"]:
            s = r["summary"]
            pr = r["per_request"] or {}
            round_rows.append({
                "source_dir": run["name"],
                "round": s["round"],
                "status": s["status"],
                "concurrency": s["concurrency"],
                "wall_time_s": round(s["wall_time_s"], 2),
                "successful": s["successful"],
                "exact_length": s["exact_length"],
                "output_tps": round(s["output_throughput_tps"], 2),
                "input_tps": round(s["input_throughput_tps"], 2),
                "total_tps": round(s["total_throughput_tps"], 2),
                "ttft_round_mean_s": round(s["ttft_s"]["mean"], 3),
                "ttft_p50_s": round(s["ttft_s"]["p50"], 3),
                "ttft_p95_s": round(s["ttft_s"]["p95"], 3),
                "ttft_p99_s": round(s["ttft_s"]["p99"], 3),
                "tpot_round_mean_ms": round(s["tpot_ms"]["mean"], 2),
                "tpot_p50_ms": round(s["tpot_ms"]["p50"], 2),
                "tpot_p95_ms": round(s["tpot_ms"]["p95"], 2),
                "request_ttft_mean_s": round(pr["ttft_mean_s"], 3) if pr else "",
                "request_tpot_mean_ms": round(pr["tpot_mean_ms"], 2) if pr else "",
                "replacement_chars": s["replacement_chars"],
            })
    round_rows.sort(key=lambda r: (r["source_dir"], r["round"]))
    write_csv(
        os.path.join(args.out_dir, "rounds.csv"),
        round_rows,
        list(round_rows[0].keys()) if round_rows else ["source_dir"],
    )

    print("input  conc  out_tps   ttft_mean   tpot_mean   SLA")
    for row in rows:
        print(
            "{:<6} {:<5} {:>8.2f}   {:>8.3f}s   {:>8.2f}ms   {}".format(
                row["input_len"], row["concurrency"], row["output_tps_mean"],
                row["ttft_mean_s"], row["tpot_mean_ms"], row["sla_pass"],
            )
        )

    for bucket in (4096, 65536):
        passing = [r["concurrency"] for r in rows
                   if sla_for(r["input_len"]) is SLA[bucket] and r["sla_pass"] == "PASS"]
        print("max concurrency under SLA @ {}-in/1K-out: {}".format(
            bucket, max(passing) if passing else 0))


if __name__ == "__main__":
    main()
