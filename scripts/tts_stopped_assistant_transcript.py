from typing import List

from pipecat.frames.frames import (
    Frame,
    TTSTextFrame,
    TTSStoppedFrame,
    LLMTextFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.transcript_processor import AssistantTranscriptProcessor
from pipecat.utils.time import time_now_iso8601


class TTSStoppedAssistantTranscriptProcessor(AssistantTranscriptProcessor):
    """Assistant transcript shim that flushes on end-of-response.

    - Aggregates TTSTextFrame fragments (AUDIO modality) and LLMTextFrame fragments (TEXT modality).
    - Emits a single assistant TranscriptionUpdateFrame when either TTSStoppedFrame (audio) or
      LLMFullResponseEndFrame (text) arrives.
    - Avoids default flush triggers from AssistantTranscriptProcessor.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Bypass AssistantTranscriptProcessor.process_frame to avoid its default
        # flush triggers. Call the base FrameProcessor implementation directly.
        await FrameProcessor.process_frame(self, frame, direction)

        if isinstance(frame, (TTSTextFrame, LLMTextFrame)):
            if not getattr(self, "_aggregation_start_time", None):
                self._aggregation_start_time = time_now_iso8601()
            # AssistantTranscriptProcessor defines _current_text_parts
            self._current_text_parts.append(frame.text)
            await self.push_frame(frame, direction)
        elif isinstance(frame, (TTSStoppedFrame, LLMFullResponseEndFrame)):
            # Flush aggregated text on audio stop or text response end
            await self._emit_aggregated_text()
            await self.push_frame(frame, direction)
        else:
            # Forward everything else without flushing
            await self.push_frame(frame, direction)
