"""Caller noise suppression (RNNoise) and the stricter VAD (no network)."""

import importlib.util

from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
from pipecat.audio.vad.vad_analyzer import VAD_CONFIDENCE, VAD_MIN_VOLUME, VAD_START_SECS

from rellachat import bot


def test_the_caller_gets_the_rnnoise_filter_when_installed():
    assert importlib.util.find_spec("pyrnnoise") is not None
    assert isinstance(bot.caller_audio_filter(), RNNoiseFilter)


def test_without_pyrnnoise_the_call_runs_unfiltered_and_warns_once(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        bot.importlib.util,
        "find_spec",
        lambda name, *args: None if name == "pyrnnoise" else real_find_spec(name, *args),
    )
    monkeypatch.setattr(bot, "_noise_filter_warned", False)
    warnings = []
    monkeypatch.setattr(bot.logger, "warning", lambda message: warnings.append(message))

    assert bot.caller_audio_filter() is None
    assert bot.caller_audio_filter() is None
    assert len(warnings) == 1


def test_the_vad_keeps_pipecat_defaults_so_short_answers_are_heard():
    # RNNoise already lowers the caller's level; a stricter VAD missed answers like "sí"
    params = bot.CALLER_VAD_PARAMS

    assert params.confidence == VAD_CONFIDENCE
    assert params.min_volume == VAD_MIN_VOLUME
    assert params.start_secs == VAD_START_SECS
