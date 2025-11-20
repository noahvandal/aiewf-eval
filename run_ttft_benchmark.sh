#!/bin/bash
# Run conversation tests for all models to collect TTFT metrics
# Stops on first error for debugging

set -e  # Exit on error
set -o pipefail  # Catch errors in pipes

PROGRESS_DOC="TTFT_BENCHMARK_PROGRESS.md"
MODELS=(
    "gemini-3-pro-preview"
    "gpt-5"
    "gpt-5.1-chat-latest"
    "gpt-4o"
    "gemini-2.5-flash"
    "qwen/qwen3-30b-a3b-instruct-2507"
    "gpt-4.1-mini"
    "gpt-4.1"
    "openai/gpt-oss-120b"
    "qwen/qwen3-235b-a22b-2507"
    "gpt-4o-mini"
    "gpt-5-mini"
    "meta-llama/llama-3.3-70b-instruct"
    "us.amazon.nova-pro-v1:0"
)

update_progress() {
    local model=$1
    local status=$2
    local run_dir=$3
    local avg_ttft=$4
    local min_ttft=$5
    local max_ttft=$6

    # Use sed to update the status in the progress document
    # This is a simple approach - in production you might use a more robust method
    echo "Updating progress for $model: $status"
}

echo "Starting TTFT benchmark for ${#MODELS[@]} models"
echo "================================================"
echo "Progress will be logged to: $PROGRESS_DOC"
echo ""

MODEL_NUM=2  # Starting from 2 since GPT-5.1 is #1

for model in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "Model #$MODEL_NUM: $model"
    echo "========================================"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Run the conversation test - will exit on error due to set -e
    uv run convo-test.py --model "$model"

    # Get the latest run directory
    RUN_DIR=$(ls -td runs/*/ | head -1 | sed 's:/$::')
    TURNS=$(wc -l < "$RUN_DIR/transcript.jsonl" 2>/dev/null || echo "0")

    echo ""
    echo "✓ Completed: $TURNS turns recorded"
    echo "Run directory: $RUN_DIR"

    # Calculate TTFT statistics
    if [ -f "$RUN_DIR/transcript.jsonl" ]; then
        STATS=$(cat "$RUN_DIR/transcript.jsonl" | jq -r '.llm_ttft_ms // 0' | awk '{
            sum+=$1;
            count++;
            if(NR==1){min=max=$1}
            if($1<min){min=$1}
            if($1>max){max=$1}
        } END {
            if(count>0) {
                printf "avg=%d min=%d max=%d", int(sum/count), min, max
            } else {
                print "avg=0 min=0 max=0"
            }
        }')

        AVG_TTFT=$(echo $STATS | sed 's/.*avg=\([0-9]*\).*/\1/')
        MIN_TTFT=$(echo $STATS | sed 's/.*min=\([0-9]*\).*/\1/')
        MAX_TTFT=$(echo $STATS | sed 's/.*max=\([0-9]*\).*/\1/')

        echo ""
        echo "TTFT Metrics:"
        echo "  Average: ${AVG_TTFT}ms"
        echo "  Min: ${MIN_TTFT}ms"
        echo "  Max: ${MAX_TTFT}ms"
    fi

    echo ""
    echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"

    MODEL_NUM=$((MODEL_NUM + 1))
done

echo ""
echo "================================================"
echo "TTFT benchmark complete!"
echo "All results stored in runs/ directory"
echo "See $PROGRESS_DOC for detailed results"
