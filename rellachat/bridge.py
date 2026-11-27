"""Two-peer audio bridge between the WhatsApp caller and an advisor's browser.

Pipecat has no built-in bridge between two WebRTC peers, so we build one from
three small frame processors:

* ``CallerAudioGate`` sits right after the caller transport input. It decides
  where the caller's microphone audio goes: to the AI (STT), nowhere (on hold),
  or to the advisor (bridged).
* ``AudioInjector`` sits right before a transport output. Audio captured from
  the *other* peer is pushed into it from outside and leaves through that
  transport. On the caller side it also drops any late TTS audio once the
  call is bridged, so the AI can never talk over the advisor.
* ``AdvisorAudioTap`` sits after the advisor transport input and forwards the
  advisor's microphone audio to the caller-side injector.

``AdvisorLeg`` owns the advisor's SmallWebRTC connection and its tiny
pipeline (input -> tap -> injector -> output); it has no STT/LLM/TTS.

Caller mic -> CallerAudioGate --(BRIDGED)--> advisor AudioInjector -> advisor speaker
Advisor mic -> AdvisorAudioTap -----------> caller AudioInjector  -> caller speaker
"""

import asyncio
from collections.abc import Awaitable, Callable
from enum import StrEnum

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

AudioSink = Callable[[InputAudioRawFrame], Awaitable[None]]

BRIDGE_IN_SAMPLE_RATE = 16000
BRIDGE_OUT_SAMPLE_RATE = 16000


class CallerMode(StrEnum):
    """Where the caller's microphone audio is routed."""

    AI = "ai"  # to STT -> LLM -> TTS
    HOLD = "hold"  # dropped (transfer ringing, AI stays quiet)
    BRIDGED = "bridged"  # to the advisor


class CallerAudioGate(FrameProcessor):
    """Routes caller microphone audio according to ``mode``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = CallerMode.AI
        self.sink: AudioSink | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            if self.mode == CallerMode.AI:
                await self.push_frame(frame, direction)
            elif self.mode == CallerMode.BRIDGED and self.sink:
                try:
                    await self.sink(frame)
                except Exception as e:
                    logger.debug(f"Bridge sink dropped caller audio: {e}")
            # HOLD (or BRIDGED without a sink yet): drop.
            return

        await self.push_frame(frame, direction)


class AudioInjector(FrameProcessor):
    """Pushes audio received from the other peer out through this pipeline."""

    def __init__(self, *, drop_tts_when_bridged: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.bridged = False
        self._drop_tts = drop_tts_when_bridged
        self._running = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._running = True
        elif isinstance(frame, (EndFrame, CancelFrame)):
            self._running = False

        if self.bridged and self._drop_tts and isinstance(frame, TTSAudioRawFrame):
            return

        await self.push_frame(frame, direction)

    async def inject(self, frame: InputAudioRawFrame) -> None:
        """Queue a peer's microphone frame as output audio for this transport."""
        if not self._running:
            return
        await self.push_frame(
            OutputAudioRawFrame(
                audio=frame.audio,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )
        )


class AdvisorAudioTap(FrameProcessor):
    """Forwards the advisor's microphone audio to the caller side."""

    def __init__(self, sink: AudioSink, **kwargs):
        super().__init__(**kwargs)
        self._sink = sink

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            try:
                await self._sink(frame)
            except Exception as e:
                logger.debug(f"Bridge sink dropped advisor audio: {e}")
            return

        await self.push_frame(frame, direction)


class AdvisorLeg:
    """The advisor's WebRTC peer plus its audio-only bridge pipeline."""

    def __init__(
        self,
        *,
        call_label: str,
        ice_servers: list[IceServer],
        to_caller: AudioSink,
        on_closed: Callable[[], Awaitable[None]],
    ):
        self._label = call_label
        self._ice_servers = ice_servers
        self._to_caller = to_caller
        self._on_closed = on_closed
        self._connection: SmallWebRTCConnection | None = None
        self._runner: WorkerRunner | None = None
        self._task: asyncio.Task | None = None
        self._closed = False
        self.injector = AudioInjector(name=f"advisor-injector-{call_label}")

    async def start(self, sdp: str, sdp_type: str = "offer") -> dict:
        """Answer the advisor's offer and start bridging. Returns the SDP answer."""
        connection = SmallWebRTCConnection(self._ice_servers)
        self._connection = connection
        await connection.initialize(sdp=sdp, type=sdp_type)
        answer = connection.get_answer()
        if not answer or not answer.get("sdp"):
            raise RuntimeError("Advisor connection produced no SDP answer")

        @connection.event_handler("closed")
        async def _on_connection_closed(_connection):
            await self._handle_closed("connection closed")

        transport = SmallWebRTCTransport(
            webrtc_connection=connection,
            params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
        )

        @transport.event_handler("on_client_disconnected")
        async def _on_client_disconnected(_transport, _client):
            await self._handle_closed("advisor disconnected")

        pipeline = Pipeline(
            [
                transport.input(),
                AdvisorAudioTap(self._to_caller, name=f"advisor-tap-{self._label}"),
                self.injector,
                transport.output(),
            ]
        )
        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=BRIDGE_IN_SAMPLE_RATE,
                audio_out_sample_rate=BRIDGE_OUT_SAMPLE_RATE,
            ),
            idle_timeout_secs=None,
            enable_rtvi=False,
        )
        self._runner = WorkerRunner(handle_sigint=False)
        await self._runner.add_workers(worker)
        self._task = asyncio.create_task(self._run(), name=f"advisor-leg-{self._label}")
        return answer

    async def _run(self) -> None:
        try:
            assert self._runner is not None
            await self._runner.run()
        except Exception as e:
            logger.error(f"Advisor leg {self._label} crashed: {e}")
        finally:
            await self._handle_closed("pipeline finished")

    async def _handle_closed(self, reason: str) -> None:
        if self._closed:
            return
        self._closed = True
        logger.info(f"Advisor leg {self._label} closed: {reason}")
        try:
            await self._on_closed()
        except Exception as e:
            logger.error(f"Advisor leg {self._label} close callback failed: {e}")

    async def stop(self) -> None:
        """Hang up the advisor peer and stop its pipeline (idempotent)."""
        self._closed = True  # caller-initiated: do not call back on_closed
        if self._runner:
            try:
                await self._runner.cancel()
            except Exception as e:
                logger.debug(f"Advisor runner cancel failed: {e}")
        if self._connection:
            try:
                await self._connection.disconnect()
            except Exception as e:
                logger.debug(f"Advisor disconnect failed: {e}")
