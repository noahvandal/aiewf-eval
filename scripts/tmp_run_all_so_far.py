import json
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# Hard-code the run directories you want to judge.
# Edit this list as needed.
RUN_DIRS = [
    "runs/20250909T040459",
    "runs/20250909T142801",
    "runs/20250909T144524",
]

# Max number of parallel judge processes
MAX_PROCS = min(4, len(RUN_DIRS)) or 1


def has_complete_transcript(run_dir: Path) -> bool:
    tx = run_dir / "transcript.jsonl"
    if not tx.exists():
        return False
    try:
        # Consider complete if it has >= 30 lines or last record has turn >= 29
        line_count = 0
        last = None
        with tx.open() as f:
            for line in f:
                line_count += 1
                last = line
        if line_count >= 30:
            return True
        if last:
            try:
                rec = json.loads(last)
                if int(rec.get("turn", -1)) >= 29:
                    return True
            except Exception:
                pass
    except Exception:
        return False
    return False


def judge_run(run_dir: Path) -> dict:
    env = os.environ.copy()
    cmd = [sys.executable, "judge_transcript_alt.py", str(run_dir)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    summary_path = run_dir / "alt_summary.json"
    result = {
        "run_dir": str(run_dir),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "summary": None,
    }
    try:
        if summary_path.exists():
            result["summary"] = json.loads(summary_path.read_text())
    except Exception as e:
        result["stderr"] += f"\nFailed to read summary: {e}"
    return result


def main():
    root = Path.cwd()
    todo = []
    for d in RUN_DIRS:
        p = root / d
        if not p.exists():
            print(f"SKIP (missing): {p}")
            continue
        if not has_complete_transcript(p):
            print(f"SKIP (incomplete transcript): {p}")
            continue
        todo.append(p)

    if not todo:
        print("Nothing to judge. Edit RUN_DIRS in this script if needed.")
        return

    print(f"Judging {len(todo)} run(s) in parallel (max_procs={MAX_PROCS})…\n")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_PROCS) as ex:
        futs = {ex.submit(judge_run, p): p for p in todo}
        for fut in as_completed(futs):
            results.append(fut.result())

    print("\n=== Results ===")
    for r in results:
        print(f"\nRun: {r['run_dir']}")
        if r["summary"]:
            s = r["summary"]
            passes = s.get("alt_passes", {})
            print(f"model={s.get('model_name')} turns={s.get('turns_scored')} judge={s.get('judge_version')}")
            print(
                f"tool={passes.get('tool_use_correct')}/30 IF={passes.get('instruction_following')}/30 KG={passes.get('kb_grounding')}/30"
            )
        else:
            print(f"returncode={r['returncode']}")
            if r["stdout"]:
                print(r["stdout"])
            if r["stderr"]:
                print(r["stderr"])


if __name__ == "__main__":
    main()

