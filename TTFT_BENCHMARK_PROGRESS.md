# TTFT Benchmark Progress

**Started**: 2025-11-19
**Purpose**: Collect Time To First Token (TTFT) metrics for all conversation test models

## Benchmark Configuration

- **Turns per conversation**: 30
- **Test script**: `convo-test.py`
- **Metric captured**: `llm_ttft_ms` (LLM Time To First Token in milliseconds)
- **Total models**: 14

## Implementation Details

- **Field added**: `llm_ttft_ms` in transcript.jsonl
- **Implementation**: convo-test.py:30, 108, 119, 133-137, 154-156, 168, 475-479
- **Metrics source**: Pipecat `TTFBMetricsData` filtered for LLM processors only
- **First TTFT per turn**: Only captures the first TTFB event per turn

---

## Completed Runs

### 1. GPT-5.1 ✓

**Model**: `gpt-5.1`
**Run Directory**: `runs/20251119T214129/`
**Completed**: 2025-11-19 16:47
**Status**: ✅ Success

**TTFT Metrics**:
- Average: 2,533ms (~2.5s)
- Min: 0ms (Turn 0 - edge case)
- Max: 9,058ms (~9s, Turn 3)
- Turns: 30/30

**Notes**:
- Turn 0 shows 0ms TTFT (likely first turn doesn't emit TTFB metrics properly)
- Turns 3-5 had higher TTFT (8-9s) - possibly due to longer context/reasoning
- Consistent TTFT of 1.3-2.8s for most turns

---

## Pending Runs

### 2. gemini-3-pro-preview
**Status**: ⏳ Queued

### 3. gpt-5
**Status**: ⏳ Queued

### 4. gpt-5.1-chat-latest
**Status**: ⏳ Queued

### 5. gpt-4o
**Status**: ⏳ Queued

### 6. gemini-2.5-flash
**Status**: ⏳ Queued

### 7. qwen/qwen3-30b-a3b-instruct-2507
**Status**: ⏳ Queued

### 8. gpt-4.1-mini
**Status**: ⏳ Queued

### 9. gpt-4.1
**Status**: ⏳ Queued

### 10. openai/gpt-oss-120b
**Status**: ⏳ Queued

### 11. qwen/qwen3-235b-a22b-2507
**Status**: ⏳ Queued

### 12. gpt-4o-mini
**Status**: ⏳ Queued

### 13. gpt-5-mini
**Status**: ⏳ Queued

### 14. meta-llama/llama-3.3-70b-instruct
**Status**: ⏳ Queued

### 15. us.amazon.nova-pro-v1:0
**Status**: ⏳ Queued

---

## Summary Statistics

**Completed**: 1/14 (7%)
**Failed**: 0
**In Progress**: 0
**Queued**: 13

---

## Next Steps

1. Run sequential benchmark for remaining 13 models
2. Analyze TTFT patterns across models
3. Correlate TTFT with model performance scores
4. Generate comparative analysis report
