import os
import json
import time
import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-5",
    "gpt-5-mini",
]


def newest_run_dir(runs_dir: Path, since_epoch: float | None = None) -> Path | None:
    if not runs_dir.exists():
        return None
    candidates = []
    for p in runs_dir.iterdir():
        if p.is_dir():
            mtime = p.stat().st_mtime
            if since_epoch is None or mtime >= since_epoch - 1.0:
                candidates.append((mtime, p))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def run_eval_for_model(model: str, env: dict) -> Path:
    print(f"\n=== Running conversation for model: {model} ===")
    start = time.time()
    env2 = env.copy()
    env2["OPENAI_MODEL"] = model
    subprocess.run([sys.executable, "-u", "convo-test.py"], check=True, env=env2)
    run_dir = newest_run_dir(Path("runs"), start)
    if not run_dir:
        raise RuntimeError("Could not locate newly created runs/<timestamp> directory")
    print(f"New run dir: {run_dir}")
    return run_dir


def judge_run(run_dir: Path, env: dict) -> dict:
    print(f"Judging run: {run_dir}")
    subprocess.run([sys.executable, "-u", "judge_transcript_alt.py", str(run_dir)], check=True, env=env)
    alt_summary_path = run_dir / "alt_summary.json"
    if not alt_summary_path.exists():
        raise RuntimeError(f"Missing {alt_summary_path}")
    summary = json.loads(alt_summary_path.read_text())
    return summary


def sample_turns(run_dir: Path, indices: list[int]) -> list[dict]:
    alt_path = run_dir / "alt_judged.jsonl"
    tx_path = run_dir / "transcript.jsonl"
    if not alt_path.exists() or not tx_path.exists():
        return []
    alt = [json.loads(l) for l in alt_path.open()]
    alt_by_turn = {r["turn"]: r for r in alt}
    out = []
    for idx in indices:
        r = alt_by_turn.get(idx)
        if not r:
            continue
        out.append(
            {
                "turn": idx,
                "user": r.get("user_text", "")[:120],
                "assistant": r.get("assistant_text", "")[:180],
                "scores": r.get("scores", {}),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Run evals for multiple models and score")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated model list (default: {','.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--samples",
        default="0,1,7,20,24",
        help="Comma-separated turn indexes to include in sampling output",
    )
    args = parser.parse_args()

    env = os.environ.copy()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sample_idxs = [int(x) for x in args.samples.split(",") if x.strip().isdigit()]

    results = []
    for model in models:
        rd = run_eval_for_model(model, env)
        summary = judge_run(rd, env)
        samples = sample_turns(rd, sample_idxs)
        results.append({"run_dir": str(rd), "model": model, "summary": summary, "samples": samples})

    print("\n=== Aggregate Results ===")
    for r in results:
        s = r["summary"].get("alt_passes", {})
        print(
            f"{r['model']}: IF={s.get('instruction_following','?')}/30, KG={s.get('kb_grounding','?')}/30, run={r['run_dir']}"
        )
        for sm in r["samples"]:
            print(
                f"  turn {sm['turn']:>2}: IF={sm['scores'].get('instruction_following')}, KG={sm['scores'].get('kb_grounding')} | {sm['user']}"
            )


if __name__ == "__main__":
    main()
