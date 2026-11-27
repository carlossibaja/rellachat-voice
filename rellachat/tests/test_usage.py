"""Call usage reported to the API for billing (T8): LLM tokens, STT seconds, TTS characters."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from pipecat.frames.frames import InputAudioRawFrame, MetricsFrame, TextFrame
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    STTUsage,
    STTUsageMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import run_test

from rellachat import api_client, summarizer
from rellachat.api_client import RellachatApi
from rellachat.usage import CallUsage, SttAudioMeter, UsageObserver


def llm_metrics(model, prompt, completion, cache_read=None, cache_creation=None):
    return LLMUsageMetricsData(
        processor="AnthropicLLMService#0",
        model=model,
        value=LLMTokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion + (cache_read or 0) + (cache_creation or 0),
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
        ),
    )


def pushed(frame):
    return FramePushed(
        source=None, destination=None, frame=frame, direction=FrameDirection.DOWNSTREAM, timestamp=0
    )


def test_an_empty_call_reports_zero_usage():
    assert CallUsage().to_payload() == {"llm": [], "stt_seconds": 0.0, "tts_characters": 0}


def test_llm_usage_is_aggregated_per_model():
    usage = CallUsage()
    usage.add_metrics(llm_metrics("claude-haiku-5-5", 1200, 40, cache_read=800, cache_creation=300))
    usage.add_metrics(llm_metrics("claude-haiku-5-5", 1500, 60, cache_read=1100))
    usage.add_metrics(llm_metrics("claude-sonnet-5", 900, 120))

    assert usage.to_payload()["llm"] == [
        {
            "model": "claude-haiku-5-5",
            "input_tokens": 2700,
            "output_tokens": 100,
            "cache_read_input_tokens": 1900,
            "cache_creation_input_tokens": 300,
        },
        {
            "model": "claude-sonnet-5",
            "input_tokens": 900,
            "output_tokens": 120,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    ]


def test_llm_usage_without_a_model_goes_under_the_default_model():
    usage = CallUsage(default_model="claude-haiku-5-5")
    usage.add_metrics(llm_metrics(None, 10, 5))

    assert usage.to_payload()["llm"][0]["model"] == "claude-haiku-5-5"


def test_tts_characters_and_stt_metrics_are_summed_and_other_metrics_ignored():
    usage = CallUsage()
    for characters in (42, 18):
        usage.add_metrics(
            TTSUsageMetricsData(processor="CartesiaTTSService#0", model="sonic-3", value=characters)
        )
    usage.add_metrics(TTFBMetricsData(processor="CartesiaTTSService#0", value=0.3))
    # Pipecat's own STT usage is not what we bill (see CallUsage.add_input_audio)
    usage.add_metrics(
        STTUsageMetricsData(processor="CartesiaSTTService#0", value=STTUsage(audio_seconds=9.0))
    )

    payload = usage.to_payload()
    assert payload["tts_characters"] == 60
    assert payload["stt_seconds"] == 0.0
    assert payload["llm"] == []


def test_stt_seconds_come_from_the_caller_audio_sent_to_the_stt():
    usage = CallUsage()
    # 0.5 s of 16 kHz mono 16-bit audio, then 0.25 s of 48 kHz stereo
    for size, rate, channels in ((16000, 16000, 1), (48000, 48000, 2)):
        usage.add_input_audio(
            InputAudioRawFrame(audio=b"\x00" * size, sample_rate=rate, num_channels=channels)
        )

    assert usage.to_payload()["stt_seconds"] == pytest.approx(0.75)


def test_the_meter_passes_the_audio_on_to_the_stt_and_counts_it_in_a_pipeline():
    usage = CallUsage()
    audio = InputAudioRawFrame(audio=b"\x00" * 32000, sample_rate=16000, num_channels=1)
    metrics = MetricsFrame(data=[llm_metrics("claude-haiku-5-5", 70, 7)])

    asyncio.run(
        run_test(
            SttAudioMeter(usage.add_input_audio),
            frames_to_send=[audio, metrics],
            expected_down_frames=[InputAudioRawFrame, MetricsFrame],
            observers=[UsageObserver(usage)],
        )
    )

    payload = usage.to_payload()
    assert payload["stt_seconds"] == pytest.approx(1.0)
    assert payload["llm"][0]["input_tokens"] == 70


def test_the_observer_collects_metrics_frames_and_ignores_other_frames():
    usage = CallUsage()
    observer = UsageObserver(usage)

    frame = MetricsFrame(
        data=[
            llm_metrics("claude-haiku-5-5", 100, 20),
            TTSUsageMetricsData(processor="CartesiaTTSService#0", value=30),
        ]
    )
    asyncio.run(observer.on_push_frame(pushed(frame)))
    asyncio.run(observer.on_push_frame(pushed(TextFrame(text="hola"))))

    payload = usage.to_payload()
    assert payload["llm"][0]["input_tokens"] == 100
    assert payload["tts_characters"] == 30


def test_bad_usage_data_never_raises():
    usage = CallUsage()
    usage.add_metrics(SimpleNamespace(value="garbage"))
    usage.add_input_audio(InputAudioRawFrame(audio=b"\x00" * 10, sample_rate=0, num_channels=1))
    usage.add_anthropic_usage("claude-haiku-5-5", None)
    usage.add_anthropic_usage("claude-haiku-5-5", SimpleNamespace(input_tokens="x"))

    assert usage.to_payload() == {"llm": [], "stt_seconds": 0.0, "tts_characters": 0}


class FakeAnthropic:
    """Stands in for AsyncAnthropic; replies with ``response`` (or raises it)."""

    response: object = None

    def __init__(self, api_key):
        self.messages = self

    async def create(self, **kwargs):
        if isinstance(FakeAnthropic.response, Exception):
            raise FakeAnthropic.response
        return FakeAnthropic.response


TRANSCRIPT = [{"role": "user", "text": "Hola, quiero una cita", "at": "2026-11-27T10:00:00Z"}]


def test_the_end_of_call_summary_adds_its_tokens_under_its_model(monkeypatch):
    monkeypatch.setattr(summarizer, "AsyncAnthropic", FakeAnthropic)
    FakeAnthropic.response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="El cliente pidió una cita.")],
        usage=SimpleNamespace(
            input_tokens=350,
            output_tokens=45,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=0,
        ),
    )
    usage = CallUsage()
    usage.add_metrics(llm_metrics("claude-haiku-5-5", 1000, 50, cache_read=200))

    summary = asyncio.run(
        summarizer.summarize_call("key", "claude-haiku-5-5", TRANSCRIPT, usage=usage)
    )

    assert summary == "El cliente pidió una cita."
    assert usage.to_payload()["llm"] == [
        {
            "model": "claude-haiku-5-5",
            "input_tokens": 1350,
            "output_tokens": 95,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 0,
        }
    ]


def test_a_failed_summary_adds_no_usage(monkeypatch):
    monkeypatch.setattr(summarizer, "AsyncAnthropic", FakeAnthropic)
    FakeAnthropic.response = RuntimeError("overloaded")
    usage = CallUsage()

    summary = asyncio.run(
        summarizer.summarize_call("key", "claude-haiku-5-5", TRANSCRIPT, usage=usage)
    )

    assert summary == ""
    assert usage.to_payload()["llm"] == []


class FakeClient:
    calls: list[dict] = []

    def __init__(self, timeout):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json, headers):
        FakeClient.calls.append({"url": url, "json": json})
        return httpx.Response(200, json={"ok": True})


@pytest.fixture
def fake_http(monkeypatch):
    FakeClient.calls = []
    monkeypatch.setattr(api_client.httpx, "AsyncClient", FakeClient)
    return FakeClient


def test_ended_carries_the_call_usage(fake_http):
    usage = {"llm": [], "stt_seconds": 12.5, "tts_characters": 300}

    api = RellachatApi("http://api.test/api", "s")

    ok = asyncio.run(api.ended("u-1", [], "Resumen", 30, usage=usage))

    assert ok is True
    call = fake_http.calls[0]
    assert call["url"] == "http://api.test/api/voice-service/calls/u-1/ended"
    assert call["json"] == {"transcript": [], "summary": "Resumen", "duration": 30, "usage": usage}


def test_ended_without_usage_keeps_the_previous_body(fake_http):
    asyncio.run(RellachatApi("http://api.test/api", "s").ended("u-1", [], "Resumen", 30))

    assert fake_http.calls[0]["json"] == {"transcript": [], "summary": "Resumen", "duration": 30}
