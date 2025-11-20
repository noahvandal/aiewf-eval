#!/usr/bin/env python3
import json
import sys
from pathlib import Path

runs = [
    ("runs/20251119T214129", "gpt-5.1", 30),
    ("runs/20251119T215548", "gemini-3-pro-preview", 30),
    ("runs/20251119T220013", "gpt-5", 30),
    ("runs/20251119T220401", "gpt-5.1-chat-latest", 30),
    ("runs/20251119T220645", "gpt-4o", 30),
    ("runs/20251119T221106", "gemini-2.5-flash", 30),
    ("runs/20251119T221253", "qwen/qwen3-30b-a3b-instruct-2507", 29),
    ("runs/20251119T221722", "gpt-4.1-mini", 30),
    ("runs/20251119T222013", "gpt-4.1", 30),
    ("runs/20251119T222411", "openai/gpt-oss-120b", 30),
    ("runs/20251119T222727", "qwen/qwen3-235b-a22b-2507", 29),
    ("runs/20251119T223646", "gpt-4o-mini", 30),
    ("runs/20251119T224016", "gpt-5-mini", 13),
    ("runs/20251119T224154", "meta-llama/llama-3.3-70b-instruct", 30),
    ("runs/20251119T224447", "us.amazon.nova-pro-v1:0", 5),
]

results = []
for run_dir, model_name, turns in runs:
    transcript_path = Path(run_dir) / "transcript.jsonl"
    if not transcript_path.exists():
        continue
    
    ttfts = []
    with open(transcript_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("llm_ttft_ms") is not None and data["llm_ttft_ms"] > 0:
                ttfts.append(data["llm_ttft_ms"])
    
    if ttfts:
        ttfts_sorted = sorted(ttfts)
        median = ttfts_sorted[len(ttfts_sorted) // 2]
        avg = sum(ttfts) / len(ttfts)
        results.append({
            "model": model_name,
            "turns": turns,
            "median_ms": median,
            "avg_ms": int(avg),
            "min_ms": min(ttfts),
            "max_ms": max(ttfts),
            "samples": len(ttfts)
        })

# Sort by median TTFT
results.sort(key=lambda x: x["median_ms"])

print(f"{'Model':<45} {'Turns':<7} {'Median':<10} {'Avg':<10} {'Min':<10} {'Max':<10}")
print("=" * 100)
for r in results:
    status = "✓" if r["turns"] == 30 else f"⚠ {r['turns']}/30"
    print(f"{r['model']:<45} {status:<7} {r['median_ms']:>6}ms   {r['avg_ms']:>6}ms   {r['min_ms']:>6}ms   {r['max_ms']:>6}ms")
