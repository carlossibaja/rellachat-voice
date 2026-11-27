"""Per-call usage the API bills and costs (T8): LLM tokens, STT seconds, TTS characters.

``CallUsage`` aggregates one call's usage into the ``usage`` field of the
``ended`` callback::

    {"llm": [{model, input_tokens, output_tokens, cache_read_input_tokens,
              cache_creation_input_tokens}], "stt_seconds": float, "tts_characters": int}

Sources:

* LLM tokens: the ``LLMUsageMetricsData`` Pipecat's AnthropicLLMService emits
  in ``MetricsFrame``s (``enable_usage_metrics=True``), collected by
  ``UsageObserver``; plus the summaries' own Anthropic responses
  (``add_anthropic_usage``). Anthropic reports ``input_tokens`` net of the
  prompt cache, so input and cache counts add up without double counting.
* TTS characters: the ``TTSUsageMetricsData`` CartesiaTTSService emits.
* STT seconds: counted by ``SttAudioMeter`` from the caller audio frames that
  reach the STT (see ``add_input_audio``).

Collecting usage must never break a call: every method swallows bad data.
"""

from collections.abc import Callable
from typing import Any

from loguru import logger
from pipecat.frames.frames import Frame, InputAudioRawFrame, MetricsFrame
from pipecat.metrics.metrics import LLMUsageMetricsData, TTSUsageMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _count(value: Any) -> int:
    """A non-negative token/character count; anything else counts as 0."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(int(value), 0)


class CallUsage:
    """What one call consumed, aggregated per LLM model."""

    def __init__(self, default_model: str | None = None):
        self.default_model = default_model
        self._llm: dict[str, dict[str, int]] = {}
        self._stt_seconds = 0.0
        self._tts_characters = 0

    def add_llm(
        self,
        model: str | None,
        *,
        input_tokens: Any = 0,
        output_tokens: Any = 0,
        cache_read_input_tokens: Any = 0,
        cache_creation_input_tokens: Any = 0,
    ) -> None:
        counts = {
            "input_tokens": _count(input_tokens),
            "output_tokens": _count(output_tokens),
            "cache_read_input_tokens": _count(cache_read_input_tokens),
            "cache_creation_input_tokens": _count(cache_creation_input_tokens),
        }
        if not any(counts.values()):
            return
        name = model or self.default_model or "unknown"
        totals = self._llm.setdefault(name, dict.fromkeys(TOKEN_FIELDS, 0))
        for field, value in counts.items():
            totals[field] += value

    def add_metrics(self, data: Any) -> None:
        """One Pipecat metric. STT usage metrics are ignored: we meter the audio ourselves."""
        try:
            if isinstance(data, LLMUsageMetricsData):
                tokens = data.value
                self.add_llm(
                    data.model,
                    input_tokens=tokens.prompt_tokens,
                    output_tokens=tokens.completion_tokens,
                    cache_read_input_tokens=tokens.cache_read_input_tokens,
                    cache_creation_input_tokens=tokens.cache_creation_input_tokens,
                )
            elif isinstance(data, TTSUsageMetricsData):
                self._tts_characters += _count(data.value)
        except Exception as e:
            logger.debug(f"Ignoring usage metric {data!r}: {e}")

    def add_anthropic_usage(self, model: str | None, usage: Any) -> None:
        """The ``usage`` of an Anthropic Messages API response (e.g. a summary)."""
        if usage is None:
            return
        try:
            self.add_llm(
                model,
                input_tokens=getattr(usage, "input_tokens", 0),
                output_tokens=getattr(usage, "output_tokens", 0),
                cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0),
                cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0),
            )
        except Exception as e:
            logger.debug(f"Ignoring Anthropic usage {usage!r}: {e}")

    def add_input_audio(self, frame: InputAudioRawFrame) -> None:
        """Seconds of caller audio (16-bit PCM) streamed to the STT.

        Pipecat 1.17's CartesiaSTTService does emit ``STTUsageMetricsData``, but
        only on final transcripts plus a last flush when the pipeline is
        cancelled; a hang-up cancels the pipeline and tears the observers down,
        so that tail would be lost. Counting the frames synchronously as they
        enter the STT survives hang-ups. Streaming STT is billed by streamed
        audio (silence included), which is what reaches the STT while the caller
        talks to the AI; audio on hold or bridged to an advisor never does.
        """
        try:
            bytes_per_second = frame.sample_rate * frame.num_channels * 2
            if bytes_per_second > 0:
                self._stt_seconds += len(frame.audio) / bytes_per_second
        except Exception as e:
            logger.debug(f"Ignoring audio frame for usage: {e}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "llm": [{"model": model, **counts} for model, counts in self._llm.items()],
            "stt_seconds": round(self._stt_seconds, 3),
            "tts_characters": self._tts_characters,
        }


class UsageObserver(BaseObserver):
    """Feeds the usage carried by the pipeline's ``MetricsFrame``s into a ``CallUsage``."""

    def __init__(self, usage: CallUsage, **kwargs):
        # A metrics frame passed along by later processors is counted once
        super().__init__(observe_every_push=False, **kwargs)
        self._usage = usage

    async def on_push_frame(self, data: FramePushed):
        if isinstance(data.frame, MetricsFrame):
            for metrics in data.frame.data:
                self._usage.add_metrics(metrics)


class SttAudioMeter(FrameProcessor):
    """Pass-through placed right before the STT; meters the caller audio it receives."""

    def __init__(self, on_audio: Callable[[InputAudioRawFrame], None], **kwargs):
        super().__init__(**kwargs)
        self._on_audio = on_audio

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            self._on_audio(frame)
        await self.push_frame(frame, direction)
