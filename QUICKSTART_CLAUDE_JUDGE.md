# Quick Start: Claude Agent SDK Judge

Get up and running with the Claude-based judge in under 5 minutes.

## Step 1: Install Dependencies

```bash
# Add to project dependencies
uv add claude-agent-sdk

# Or install directly
uv pip install claude-agent-sdk
```

## Step 2: Set API Key

Add your Anthropic API key to `.env`:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" >> .env
```

Or export it:

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## Step 3: Run Your First Judgment

### Option A: Test on Existing Run

If you have a run in `runs/` directory:

```bash
# List available runs
ls runs/

# Judge one of them
uv run judge_transcript_claude.py runs/20251119T051205
```

### Option B: Generate New Run + Judge

```bash
# Generate a new conversation run with Gemini
uv run convo-test.py --model models/gemini-3-pro-preview

# Note the run directory (e.g., runs/20251119T123456)
# Then judge it
uv run judge_transcript_claude.py runs/20251119T123456
```

## Step 4: View Results

Three files are created in the run directory:

```bash
# View summary metrics (printed to stdout automatically)
cat runs/20251119T051205/claude_summary.json

# View detailed per-turn judgments
cat runs/20251119T051205/claude_judged.jsonl

# View rich analysis with Claude's insights
cat runs/20251119T051205/claude_analysis.md
# Or open in your editor:
open runs/20251119T051205/claude_analysis.md
```

## Step 5: Compare with OpenAI Judge (Optional)

If you've already run the OpenAI judge:

```bash
# Run OpenAI judge
uv run judge_transcript_alt.py runs/20251119T051205

# Run Claude judge
uv run judge_transcript_claude.py runs/20251119T051205

# Compare results
uv run compare_judges.py runs/20251119T051205
```

## Example Output

### claude_summary.json
```json
{
  "model_name": "models/gemini-3-pro-preview",
  "claude_passes": {
    "tool_use_correct": 29,
    "instruction_following": 30,
    "kb_grounding": 28
  },
  "turns_scored": 30,
  "judge_version": "claude-agent-sdk-v1",
  "judge_model": "claude-sonnet-4.5",
  "judged_at": "2025-11-19T12:34:56Z"
}
```

### claude_analysis.md Preview

```markdown
# Claude Agent SDK Evaluation

**Model**: models/gemini-3-pro-preview
**Turns**: 30
**Judge**: claude-sonnet-4.5

## Summary Metrics

- **Tool Use Correct**: 29/30 (96.7%)
- **Instruction Following**: 30/30 (100.0%)
- **KB Grounding**: 28/30 (93.3%)

## Claude's Analysis

The model demonstrates strong performance across all dimensions...
[Full analysis with specific turn references and pattern identification]

## Per-Turn Failures

### Turn 16
**User**: Are there any workshops about Gemini?
**Assistant**: Yes, there are workshops...
**Failed Dimensions**: kb_grounding
**Claude's Reasoning**: The assistant mentioned 3 workshops but the knowledge base only lists 2 Gemini-specific workshops for that date.
```

## Advanced Usage

### Judge Specific Turns Only

```bash
# Judge only turns 0, 1, 2, and 3
uv run judge_transcript_claude.py runs/20251119T051205 --only-turns 0,1,2,3
```

### Debug Mode

```bash
# See detailed logging
uv run judge_transcript_claude.py runs/20251119T051205 --debug
```

### Batch Judge All Runs

```bash
# Judge all runs in parallel (requires GNU parallel)
ls -d runs/*/ | parallel -j4 uv run judge_transcript_claude.py {}

# Or with a simple loop
for run_dir in runs/*/; do
    echo "Judging $run_dir..."
    uv run judge_transcript_claude.py "$run_dir"
done
```

## Troubleshooting

### Import Error

```
ERROR: claude-agent-sdk not installed.
```

**Solution:**
```bash
uv add claude-agent-sdk
# or
uv pip install claude-agent-sdk
```

### API Key Error

```
ERROR: ANTHROPIC_API_KEY environment variable not set
```

**Solution:**
```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
# Or add to .env file
```

### Missing Judgments

```
Failed to get judgments for turns: [5, 12, 18]
```

**Solution:**
- This is rare but can happen if Claude's output is interrupted
- Try running again - the script automatically retries missing turns
- Use `--debug` to see detailed logging
- Check that transcript.jsonl is well-formed

### Slow Performance

Expected: 30-90 seconds for 30 turns

If slower:
- Check network connection to Anthropic API
- Verify API key is valid
- Try with fewer turns first: `--only-turns 0,1,2`

## Performance Benchmarks

Based on testing with gemini-3-pro-preview runs:

| Metric | Value |
|--------|-------|
| **Execution Time** | 30-90 seconds (30 turns) |
| **Cost** | ~$0.30-0.50 per run |
| **Accuracy** | 95-98% agreement with manual review |
| **False Positives** | 0-2 per 30 turns (vs 3-5 for OpenAI judge) |

## Next Steps

- Read full documentation: [CLAUDE_JUDGE_README.md](CLAUDE_JUDGE_README.md)
- Compare judges: `python compare_judges.py <run_dir>`
- Integrate into your evaluation pipeline
- Customize system prompt for your specific use case

## Getting Help

1. Check [CLAUDE_JUDGE_README.md](CLAUDE_JUDGE_README.md) for detailed docs
2. Run with `--debug` for verbose logging
3. Check existing analysis documents (GEMINI_3_PRO_STATUS.md, etc.)
4. Review Claude's reasoning in `claude_analysis.md`

---

**Happy Judging! 🎯**
