import os
import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional

from loguru import logger
from dotenv import load_dotenv

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.llm_service import FunctionCallParams
from pipecat.frames.frames import (
    Frame,
    MetricsFrame,
    CancelFrame,
    InputAudioRawFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    LLMMessagesAppendFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    LLMTokenUsage,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai_realtime_beta.openai import (
    OpenAIRealtimeBetaLLMService as OpenAIRealtimeAPIBetaLLMService,
)
from pipecat.processors.aggregators.llm_response import LLMAssistantAggregatorParams
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
)
from pipecat.processors.aggregators.llm_response import (
    OpenAILLMContextAssistantTimestampFrame,
)


from system_instruction import system_instruction
from turns import turns
from tools_schema import ToolsSchemaForTest
import soundfile as sf
import numpy as np
from pipecat.utils.time import time_now_iso8601

load_dotenv()

logger.info("Starting conversation test...")


# -------------------------
# Utilities for persistence
# -------------------------


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


class RunRecorder:
    """Accumulates per-turn data and writes JSONL + summary."""

    def __init__(self, model_name: str):
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        self.run_dir = Path("runs") / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.out_path = self.run_dir / "transcript.jsonl"
        self.fp = self.out_path.open("a", encoding="utf-8")
        self.model_name = model_name

        # per-turn working state
        self.turn_start_monotonic: Optional[float] = None
        self.turn_usage: Dict[str, Any] = {}
        self.turn_calls: List[Dict[str, Any]] = []
        self.turn_results: List[Dict[str, Any]] = []
        self.turn_index: int = 0

        # simple turn counter; judging happens post-run
        self.total_turns_scored = 0

    def start_turn(self, turn_index: int):
        self.turn_index = turn_index
        self.turn_start_monotonic = time.monotonic()
        self.turn_usage = {}
        self.turn_calls = []
        self.turn_results = []

    def record_usage_metrics(self, m: LLMTokenUsage, model: Optional[str] = None):
        # store last seen usage; fine for turn-local
        self.turn_usage = {
            "prompt_tokens": m.prompt_tokens,
            "completion_tokens": m.completion_tokens,
            "total_tokens": m.total_tokens,
            "cache_read_input_tokens": m.cache_read_input_tokens,
            "cache_creation_input_tokens": m.cache_creation_input_tokens,
        }
        if model:
            self.model_name = model

    def record_tool_call(self, name: str, args: Dict[str, Any]):
        self.turn_calls.append({"name": name, "args": args})

    def record_tool_result(self, name: str, response: Dict[str, Any]):
        self.turn_results.append({"name": name, "response": response})

    def write_turn(self, *, user_text: str, assistant_text: str):
        latency_ms = None
        if self.turn_start_monotonic is not None:
            latency_ms = int((time.monotonic() - self.turn_start_monotonic) * 1000)

        rec = {
            "ts": now_iso(),
            "turn": self.turn_index,
            "model_name": self.model_name,
            "user_text": user_text,
            "assistant_text": assistant_text,
            "tool_calls": self.turn_calls,
            "tool_results": self.turn_results,
            "tokens": self.turn_usage or None,
            "latency_ms": latency_ms,
        }
        self.fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.fp.flush()
        self.total_turns_scored += 1

    def write_summary(self):
        runtime = {
            "model_name": self.model_name,
            "turns": self.total_turns_scored,
            "note": "runtime-only; scoring is performed post-run (see alt_summary.json)",
        }
        (self.run_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")


# -------------------------
# Tool call stub (records)
# -------------------------

recorder: Optional[RunRecorder] = None


async def function_catchall(params: FunctionCallParams):
    logger.info(f"Function call: {params}")
    # Record tool calls/results
    global recorder
    try:
        name = getattr(params, "function_name", None)
        args = getattr(params, "arguments", {})
        if recorder:
            recorder.record_tool_call(
                str(name), dict(args) if isinstance(args, dict) else {"raw": str(args)}
            )
    except Exception:
        pass
    result = {"status": "success"}
    try:
        if recorder and name:
            recorder.record_tool_result(str(name), result)
    except Exception:
        pass
    await params.result_callback(result)


class NextTurn(FrameProcessor):
    def __init__(
        self, end_of_turn_callback: Callable, metrics_callback: Callable[[MetricsFrame], None]
    ):
        super().__init__()
        self.end_of_turn_callback = end_of_turn_callback
        self.metrics_callback = metrics_callback

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, MetricsFrame):
            self.metrics_callback(frame)

        # Treat assistant timestamp frame as end-of-turn marker
        if isinstance(frame, OpenAILLMContextAssistantTimestampFrame):
            logger.info("EOT (timestamp)")
            await self.end_of_turn_callback()


class RealtimeEOTShim(FrameProcessor):
    """Shim for OpenAI Realtime: inject assistant message + timestamp when
    the service omits LLMFullResponseStartFrame (so the assistant aggregator
    won't aggregate tokens or emit a timestamp).

    Sits between the LLM and the assistant context aggregator.
    """

    def __init__(self):
        super().__init__()
        self._saw_start = False
        self._buffer = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._saw_start = True
            self._buffer = []
        elif isinstance(frame, LLMTextFrame):
            # Collect tokens only if we didn't see an LLM start; otherwise
            # assistant aggregator will handle aggregation.
            if not self._saw_start and getattr(frame, "text", None):
                self._buffer.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            # Forward original end frame first
            await self.push_frame(frame, direction)

            if not self._saw_start:
                text = "".join(self._buffer).strip()
                if text:
                    # Update context via assistant aggregator
                    append = LLMMessagesAppendFrame(
                        messages=[{"role": "assistant", "content": text}],
                        run_llm=False,
                    )
                    logger.debug(
                        f"RealtimeEOTShim: injecting assistant text ({len(text)} chars)"
                    )
                    await self.push_frame(append, direction)

                # Always emit a timestamp to close the turn downstream
                ts = OpenAILLMContextAssistantTimestampFrame(timestamp=time_now_iso8601())
                logger.debug("RealtimeEOTShim: injecting assistant timestamp frame")
                await self.push_frame(ts, direction)

            # Reset per-completion state
            self._saw_start = False
            self._buffer = []
            return

        # Default: pass through
        await self.push_frame(frame, direction)


# Note: runtime judging has been removed. We judge post-run using the alt judge script.


async def main():
    turn_idx = 0

    # Configure model routing + user field for caching experiments
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1")

    def is_google_model(name: str) -> bool:
        n = (name or "").lower()
        return n.startswith("gemini") or n.startswith("gemma")

    def is_openrouter_model(name: str) -> bool:
        n = (name or "").lower()
        return (
            n.startswith("meta-llama")
            or n.startswith("openrouter")
            or n.startswith("openai/")
            or n.startswith("qwen/")
        )

    def is_bedrock_model(name: str) -> bool:
        n = (name or "").lower()
        return "amazon" in n

    def is_openai_realtime_model(name: str) -> bool:
        n = (name or "").lower()
        return n == "gpt-realtime" or "realtime" in n

    extra = {"user": os.getenv("EVAL_USER", "eval-runner")}
    llm = None

    if is_google_model(model_name):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is required for Google models")
        # Google service handles its own adapter and can upgrade OpenAI context under the hood
        llm = GoogleLLMService(api_key=api_key, model=model_name)
        llm.register_function(None, function_catchall)
    elif is_openai_realtime_model(model_name):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is required for OpenAI Realtime models")
        # todo: switch this over to new OpenAIRealtimeLLMService
        llm = OpenAIRealtimeAPIBetaLLMService(api_key=api_key)
        llm.register_function(None, function_catchall)
    elif is_bedrock_model(model_name):
        try:
            # Import lazily to avoid requiring AWS deps for non-Bedrock runs
            from pipecat.services.aws.llm import AWSBedrockLLMService  # type: ignore
        except Exception as e:  # pragma: no cover - surfaced at runtime only
            raise RuntimeError(
                "Amazon Bedrock support requires installing 'pipecat-ai[aws]'."
            ) from e
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION")
        if not (aws_access_key_id and aws_secret_access_key and aws_region):
            raise EnvironmentError(
                "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION are required for Amazon Bedrock models"
            )
        llm = AWSBedrockLLMService(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region=aws_region,
            model=model_name,
        )
        llm.register_function(None, function_catchall)
    elif is_openrouter_model(model_name):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is required for OpenRouter models")
        # Minimal reasoning applies only to gpt-5 family
        if model_name.startswith("gpt-5"):
            extra["reasoning_effort"] = "minimal"
        params = BaseOpenAILLMService.InputParams(extra=extra)
        llm = OpenAILLMService(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
            params=params,
        )
        llm.register_function(None, function_catchall)
    else:
        # Default to OpenAI-compatible endpoint using OPENAI_API_KEY
        if model_name.startswith("gpt-5"):
            extra["reasoning_effort"] = "minimal"
        params = BaseOpenAILLMService.InputParams(extra=extra)
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model=model_name, params=params)
        llm.register_function(None, function_catchall)

    # Set up recorder
    global recorder
    recorder = RunRecorder(model_name=model_name)
    recorder.start_turn(turn_idx)

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": turns[turn_idx]["input"]},
    ]

    context = OpenAILLMContext(messages, tools=ToolsSchemaForTest)
    context_aggregator = llm.create_context_aggregator(
        context, assistant_params=LLMAssistantAggregatorParams(expect_stripped_words=False)
    )

    # Track index into context messages to extract per-turn assistant text
    last_msg_idx = len(messages)

    def handle_metrics(frame: MetricsFrame):
        for md in frame.data:
            if isinstance(md, LLMUsageMetricsData):
                recorder.record_usage_metrics(md.value, getattr(md, "model", None))

    done = False

    async def end_of_turn():
        nonlocal turn_idx, last_msg_idx, done
        if done:
            logger.info("!!!! EOT top (done)")
            await llm.push_frame(CancelFrame())
            return

        # Extract assistant text added since last user message
        msgs = context.get_messages()
        # Skip the system message for logging purposes
        start_i = max(1, last_msg_idx)
        new_msgs = msgs[start_i:]
        assistant_chunks: List[str] = []
        for m in new_msgs:
            logger.info(f"!!!New message: {m}")
            if hasattr(m, "parts"):
                # Google -- we should be using the new LLMContext so we don't need to do this
                part = m.parts[0]
                if part.text:
                    assistant_chunks.append(part.text)
            elif m.get("role") == "assistant" and m.get("content"):
                # content may be str or list
                if isinstance(m["content"], str):
                    assistant_chunks.append(m["content"])
                elif isinstance(m["content"], list):
                    # concatenate text parts
                    assistant_chunks.extend(
                        [p.get("text", "") for p in m["content"] if isinstance(p, dict)]
                    )
        assistant_text = "\n".join([c for c in assistant_chunks if c])

        recorder.write_turn(
            user_text=turns[turn_idx].get("input", ""),
            assistant_text=assistant_text,
        )

        turn_idx += 1
        if turn_idx < len(turns):
            recorder.start_turn(turn_idx)
            await asyncio.sleep(0.10)
            logger.info(f"!!!!!! User input: {turns[turn_idx]['input']}")
            logger.info(f"!!!!!! Context: {context}")
            context.add_messages([{"role": "user", "content": turns[turn_idx]["input"]}])
            last_msg_idx = len(context.get_messages())

            if is_openai_realtime_model(model_name):
                # Push audio for this turn if present; otherwise fall back
                audio_path = turns[turn_idx].get("audio_file")
                if not audio_path:
                    logger.warning(
                        f"No audio_file for turn {turn_idx}; sending text context frame instead."
                    )
                    await task.queue_frames([context_aggregator.user().get_context_frame()])
                else:
                    try:
                        data, sr = sf.read(audio_path, dtype="int16", always_2d=True)
                        if data.ndim != 2:
                            data = np.atleast_2d(data)
                        num_channels = data.shape[1]
                        raw = data.tobytes()
                        frame = InputAudioRawFrame(
                            audio=raw, sample_rate=int(sr), num_channels=int(num_channels)
                        )
                        await task.queue_frames([frame])
                    except Exception as e:
                        logger.exception(
                            f"Failed to load audio for turn {turn_idx} from {audio_path}: {e}"
                        )
                        await task.queue_frames([context_aggregator.user().get_context_frame()])
            else:
                await task.queue_frames([context_aggregator.user().get_context_frame()])
        else:
            logger.info("Conversation complete")
            recorder.write_summary()
            done = True
            logger.info("!!!! EOT conversation complete (done)")
            await llm.push_frame(CancelFrame())

    next_turn = NextTurn(end_of_turn, handle_metrics)

    if is_openai_realtime_model(model_name):
        pipeline = Pipeline(
            [
                context_aggregator.user(),  # User responses
                llm,  # LLM
                RealtimeEOTShim(),  # Inject EOT + append assistant text if needed
                context_aggregator.assistant(),  # Assistant responses
                next_turn,
            ]
        )
    else:
        pipeline = Pipeline(
            [
                context_aggregator.user(),
                llm,
                context_aggregator.assistant(),
                next_turn,
            ]
        )

    task = PipelineTask(
        pipeline,
        idle_timeout_secs=600,
        idle_timeout_frames=(MetricsFrame,),
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Queue first user turn
    last_msg_idx = len(messages)
    if is_openai_realtime_model(model_name):
        logger.info("!!!!!! OpenAI Realtime model - doing first user turn")
        # Use the audio file for turn 0 from the turns.py list
        audio_path = turns[0].get("audio_file")
        try:
            data, sr = sf.read(audio_path, dtype="int16", always_2d=True)
            if data.ndim != 2:
                data = np.atleast_2d(data)
            num_channels = data.shape[1]
            raw = data.tobytes()
            frame = InputAudioRawFrame(
                audio=raw, sample_rate=int(sr), num_channels=int(num_channels)
            )
            await task.queue_frames([frame])
        except Exception as e:
            logger.exception(f"Failed to load audio from {audio_path}: {e}")
            await task.queue_frames([context_aggregator.user().get_context_frame()])
    else:
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    runner = PipelineRunner(handle_sigint=True)
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
