"""One AI voice call: SmallWebRTC -> Cartesia STT -> Claude -> Cartesia TTS.

``VoiceCall`` owns the pipeline for a single call, its transcript, the
``transfer_to_advisor`` / ``end_call`` tools, the transfer state machine and
the callbacks to the Rellachat API.

Transfer states::

    AI --transfer_to_advisor--> TRANSFER_PENDING --advisor-offer--> BRIDGED
                                      |
                                      +--ring timeout / API failure--> AI
"""

import asyncio
import importlib.util
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from loguru import logger
from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndWorkerFrame,
    FunctionCallResultProperties,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.empty_user_turn import EmptyUserTurnConfig
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (
    MinWordsUserTurnStartStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from rellachat.api_client import RellachatApi
from rellachat.bridge import AdvisorLeg, AudioInjector, CallerAudioGate, CallerMode
from rellachat.business_tools import function_schemas
from rellachat.config import Settings
from rellachat.models import AdvisorUser, AgentConfig
from rellachat.outbound import ANSWER_TIMEOUT_SECS
from rellachat.prompts import (
    HOLD_MESSAGE,
    NO_ADVISOR_INSTRUCTION,
    UNANSWERED_SUMMARY,
    build_greeting,
    build_system_prompt,
    call_started_marker,
)
from rellachat.summarizer import summarize_call, summarize_for_transfer
from rellachat.usage import CallUsage, SttAudioMeter, UsageObserver

AUDIO_IN_SAMPLE_RATE = 16000
AUDIO_OUT_SAMPLE_RATE = 24000
# Thinking counts toward this cap: at 200 the model sometimes spent it all thinking and replied with no
# text (the agent went silent). Brevity comes from the prompt's style rules, not from this cap.
LLM_MAX_TOKENS = 400
# Words the caller must say before the agent stops talking: one, so a "sí" or "espera" cuts it off
# (interim transcripts count, so it stops as soon as the first word is recognized)
INTERRUPT_MIN_WORDS = 1
# Unrecognized turns in a row the agent still answers (asks to repeat) before staying quiet
EMPTY_TURN_RECOVERIES = 3


class CallState(StrEnum):
    AI = "ai"
    TRANSFER_PENDING = "transfer_pending"
    BRIDGED = "bridged"
    ENDED = "ended"


class TransferNotPendingError(Exception):
    """Raised when an advisor offer arrives but no transfer is ringing."""


class AnswerNotExpectedError(Exception):
    """Raised when an SDP answer arrives for an inbound call or one already answered."""


@dataclass
class WhatsAppLeg:
    """What we need to hang up the WhatsApp side from our end."""

    client: Any  # pipecat WhatsAppClient
    call_id: str


# Silero VAD at Pipecat's defaults: RNNoise already lowers the level of the caller's audio, so a stricter
# min_volume/confidence stopped detecting short answers ("sí") and the turn never ended.
CALLER_VAD_PARAMS = VADParams()

_noise_filter_warned = False


def caller_audio_filter() -> BaseAudioFilter | None:
    """RNNoise noise suppression for the caller's audio, or None when pyrnnoise is missing."""
    global _noise_filter_warned
    try:
        if importlib.util.find_spec("pyrnnoise") is None:
            raise ImportError("pyrnnoise is not installed")
        from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter

        return RNNoiseFilter()
    except Exception as e:
        if not _noise_filter_warned:
            _noise_filter_warned = True
            logger.warning(f"Noise suppression disabled (install the rnnoise extra): {e}")
        return None


def to_language(code: str | None) -> Language:
    """Map an agent language code (``es``, ``es-MX``) to a pipecat Language."""
    value = (code or "es").strip()
    for candidate in (value, value.split("-")[0].lower()):
        try:
            return Language(candidate)
        except ValueError:
            continue
    return Language.ES


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class VoiceCall:
    """A single AI-handled call (WhatsApp or browser test; phone calls subclass it).

    Subclasses change the media leg through ``_create_transport`` and ``_disconnect_media``.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        api: RellachatApi,
        agent: AgentConfig,
        connection: SmallWebRTCConnection | None,
        call_uuid: str | None = None,
        whatsapp: WhatsAppLeg | None = None,
        contact_name: str | None = None,
        initiated_by: str | None = None,
        answer_timeout_secs: float = ANSWER_TIMEOUT_SECS,
    ):
        self.settings = settings
        self.api = api
        self.agent = agent
        self.connection = connection
        self.call_uuid = call_uuid
        self.whatsapp = whatsapp
        self.contact_name = contact_name
        self.initiated_by = initiated_by
        self.test_mode = call_uuid is None
        # Outbound: we sent the SDP offer; the customer's answer and media come later
        self.outbound = agent.is_outbound
        self.answer_timeout_secs = answer_timeout_secs

        self.label = call_uuid or f"test-{connection.pc_id}"
        self.state = CallState.AI
        self.transcript: list[dict] = []
        self.answered_at: float | None = None
        # What the call consumed, reported on ``ended`` for billing
        self.usage = CallUsage(default_model=settings.anthropic_model)

        self._gate = CallerAudioGate(name=f"caller-gate-{self.label}")
        self._caller_injector = AudioInjector(
            drop_tts_when_bridged=True, name=f"caller-injector-{self.label}"
        )
        self._advisor: AdvisorLeg | None = None
        self._transfer_timer: asyncio.Task | None = None
        self._background: set[asyncio.Task] = set()
        self._runner: WorkerRunner | None = None
        self._worker: PipelineWorker | None = None
        self._remote_terminated = False
        self._finalized = False
        self._stop_requested = False
        self._answer_applied = False
        self._answer_timer: asyncio.Task | None = None
        self._unanswered = False
        self._cancelled = False

    # ------------------------------------------------------------------ utils

    def _spawn(self, coro, name: str) -> None:
        task = asyncio.create_task(coro, name=f"{name}-{self.label}")
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _add_transcript(self, role: str, text: str | None, at: str | None = None) -> None:
        text = (text or "").strip()
        if text:
            self.transcript.append({"role": role, "text": text, "at": at or now_iso()})

    # --------------------------------------------------------------- pipeline

    def _build_services(self):
        language = to_language(self.agent.language)
        stt = CartesiaSTTService(
            api_key=self.settings.cartesia_api_key,
            settings=CartesiaSTTService.Settings(
                language=language,
                **(
                    {"model": self.settings.cartesia_stt_model}
                    if self.settings.cartesia_stt_model
                    else {}
                ),
            ),
        )
        tts = CartesiaTTSService(
            api_key=self.settings.cartesia_api_key,
            settings=CartesiaTTSService.Settings(
                voice=self.agent.voice_id or self.settings.cartesia_voice_id,
                language=language,
            ),
        )
        llm = AnthropicLLMService(
            api_key=self.settings.anthropic_api_key,
            settings=AnthropicLLMService.Settings(
                model=self.settings.anthropic_model,
                system_instruction=build_system_prompt(
                    self.agent,
                    test_mode=self.test_mode,
                    contact_name=self.contact_name,
                    initiated_by=self.initiated_by,
                ),
                max_tokens=LLM_MAX_TOKENS,
            ),
        )
        return stt, tts, llm

    def _tools(self) -> ToolsSchema:
        tools = [
            FunctionSchema(
                name="end_call",
                description=(
                    "End the phone call. Use it when the conversation is over or the caller "
                    "says goodbye. Say a short farewell in the same turn before calling it."
                ),
                properties={},
                required=[],
            )
        ]
        if self.agent.transfer_enabled:
            tools.append(
                FunctionSchema(
                    name="transfer_to_advisor",
                    description=(
                        "Transfer the caller to a human advisor. Use it when the caller asks "
                        "for a person or you cannot solve the request. Do not say anything "
                        "before calling it; the system asks the caller to hold."
                    ),
                    properties={
                        "reason": {
                            "type": "string",
                            "description": "Short reason for the transfer, in Spanish.",
                        }
                    },
                    required=["reason"],
                )
            )
        tools.extend(self._business_tools())
        return ToolsSchema(standard_tools=tools)

    def _business_tools(self) -> list[FunctionSchema]:
        """The agent's Rellachat tools (calendar, products, orders, leads) the API sent."""
        return function_schemas(self.agent.tools)

    async def run(self) -> None:
        """Run the call until either side hangs up, then report it."""
        if self.outbound:
            self._answer_timer = asyncio.create_task(
                self._answer_timeout(self.answer_timeout_secs), name=f"answer-timer-{self.label}"
            )
        try:
            await self._run_pipeline()
        except Exception as e:
            logger.exception(f"Call {self.label} pipeline failed: {e}")
        finally:
            await self.finalize()

    def _create_transport(self):
        """The caller's media transport (WebRTC here; phone calls use Telnyx's WebSocket)."""
        return SmallWebRTCTransport(
            webrtc_connection=self.connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_filter=caller_audio_filter(),
            ),
        )

    async def _disconnect_media(self) -> None:
        """Close the caller's media leg once the call is over."""
        if self.connection is None:
            return
        try:
            await self.connection.disconnect()
        except Exception as e:
            logger.debug(f"Call {self.label}: disconnect failed: {e}")

    async def _run_pipeline(self) -> None:
        transport = self._create_transport()
        stt, tts, llm = self._build_services()

        llm.register_function("end_call", self._handle_end_call, cancel_on_interruption=False)
        if self.agent.transfer_enabled:
            llm.register_function(
                "transfer_to_advisor", self._handle_transfer, cancel_on_interruption=False
            )

        # Not cancelled on interruption: a booking the caller talks over must still finish
        for schema in self._business_tools():
            llm.register_function(
                schema.name, self._handle_business_tool, cancel_on_interruption=False
            )

        context = LLMContext(
            messages=[{"role": "user", "content": call_started_marker(self.agent)}],
            tools=self._tools(),
        )
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(params=CALLER_VAD_PARAMS),
                # Phone audio carries noise and echo: only transcribed words start a turn, and
                # cutting the agent off mid-sentence takes INTERRUPT_MIN_WORDS.
                user_turn_strategies=UserTurnStrategies(
                    start=[MinWordsUserTurnStartStrategy(min_words=INTERRUPT_MIN_WORDS)]
                ),
                # Never go silent after a few unrecognized turns: keep asking the caller to repeat
                empty_user_turn=EmptyUserTurnConfig(
                    max_consecutive_recoveries=EMPTY_TURN_RECOVERIES
                ),
            ),
        )

        pipeline = Pipeline(
            [
                transport.input(),
                self._gate,
                SttAudioMeter(self.usage.add_input_audio, name=f"stt-meter-{self.label}"),
                stt,
                user_aggregator,
                llm,
                tts,
                self._caller_injector,
                transport.output(),
                assistant_aggregator,
            ]
        )
        self._worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=AUDIO_IN_SAMPLE_RATE,
                audio_out_sample_rate=AUDIO_OUT_SAMPLE_RATE,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[UsageObserver(self.usage)],
            # Bridged calls carry no STT/TTS activity; hang-ups end the call instead.
            idle_timeout_secs=None,
        )
        self._runner = WorkerRunner(handle_sigint=False)
        await self._runner.add_workers(self._worker)

        @transport.event_handler("on_client_connected")
        async def on_client_connected(_transport, _client):
            await self._on_media_connected()

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(_transport, _client):
            logger.info(f"Call {self.label}: caller disconnected")
            await self._cancel_runner()

        @user_aggregator.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(_aggregator, _strategy, message):
            self._add_transcript("user", message.content, message.timestamp)

        @assistant_aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(_aggregator, message):
            self._add_transcript("assistant", message.content, message.timestamp)

        if not self.outbound:
            # Inbound: Meta's accept (or Telnyx's answer) already happened, so it is answered now
            self._mark_answered()

        if self._stop_requested:
            return
        await self._runner.run()

    def _mark_answered(self) -> None:
        """Start billing time and tell the API (once)."""
        if self.answered_at is not None:
            return
        self.answered_at = time.monotonic()
        if self._answer_timer:
            self._answer_timer.cancel()
            self._answer_timer = None
        if self.call_uuid:
            self._spawn(self.api.answered(self.call_uuid), "answered")

    async def _on_media_connected(self) -> None:
        """Caller media is flowing: inbound speaks the greeting, outbound lets the AI open."""
        logger.info(f"Call {self.label}: caller media connected")
        assert self._worker is not None
        if self.outbound:
            if self.answered_at is not None:
                return
            self._mark_answered()
            # The context starts with the outbound marker; the AI introduces itself and the reason
            await self._worker.queue_frames([LLMRunFrame()])
            return
        await self._worker.queue_frames(
            [TTSSpeakFrame(build_greeting(self.agent, self.contact_name))]
        )

    # --------------------------------------------------------------- outbound

    async def apply_answer(self, sdp: str, sdp_type: str, whatsapp: WhatsAppLeg) -> None:
        """Apply the customer's SDP answer and keep Meta's call to terminate it later."""
        if not self.outbound or self._answer_applied or self._finalized:
            raise AnswerNotExpectedError()
        self._answer_applied = True
        # Set first: a hang-up while the answer is applied must still terminate at Meta
        self.whatsapp = whatsapp
        await self.connection.apply_answer(sdp, sdp_type)

    async def _answer_timeout(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
        if self.answered_at is None and not self._finalized:
            logger.info(f"Call {self.label}: not answered in {seconds}s; ending it")
            self._unanswered = True
            await self._cancel_runner()

    async def cancel(self) -> None:
        """Meta never placed the call: tear it down without reporting ``ended``."""
        self._cancelled = True
        await self._cancel_runner()
        await self.finalize()

    async def _cancel_runner(self) -> None:
        self._stop_requested = True
        if self._runner:
            try:
                await self._runner.cancel()
            except Exception as e:
                logger.debug(f"Call {self.label}: runner cancel failed: {e}")

    # ------------------------------------------------------------------ tools

    async def _handle_end_call(self, params: FunctionCallParams) -> None:
        logger.info(f"Call {self.label}: end_call requested by the AI")
        await params.result_callback(
            {"status": "ending"}, properties=FunctionCallResultProperties(run_llm=False)
        )
        # Pushed after the farewell text, so the TTS audio is flushed before ending.
        await params.llm.push_frame(EndWorkerFrame(reason="end_call"))

    async def _handle_business_tool(self, params: FunctionCallParams) -> None:
        """Runs a business tool through the API; its result (or an error) goes back to the model."""
        arguments = params.arguments if isinstance(params.arguments, dict) else {}
        result = await self.api.run_tool(self.agent.tools_path, params.function_name, arguments)
        await params.result_callback(result)

    async def _handle_transfer(self, params: FunctionCallParams) -> None:
        reason = str((params.arguments or {}).get("reason") or "").strip()

        if self.test_mode:
            await params.result_callback(
                {
                    "status": "simulated",
                    "message": "Llamada de prueba: no hay transferencia real. Di que en una "
                    "llamada real transferirías ahora a un asesor y sigue ayudando.",
                }
            )
            return

        if self.state != CallState.AI:
            await params.result_callback(
                {"status": "already_in_progress"},
                properties=FunctionCallResultProperties(run_llm=False),
            )
            return

        logger.info(f"Call {self.label}: transfer requested ({reason})")
        self.state = CallState.TRANSFER_PENDING
        self._gate.mode = CallerMode.HOLD
        await params.result_callback(
            {"status": "transfer_started"},
            properties=FunctionCallResultProperties(run_llm=False),
        )
        await params.llm.push_frame(TTSSpeakFrame(HOLD_MESSAGE))
        self._spawn(self._request_transfer(reason), "transfer")

    # --------------------------------------------------------------- transfer

    async def _request_transfer(self, reason: str) -> None:
        history = list(self.transcript)
        if reason:
            history.append({"role": "assistant", "text": f"Motivo de la transferencia: {reason}"})
        summary = await summarize_for_transfer(
            self.settings.anthropic_api_key,
            self.settings.anthropic_model,
            history,
            usage=self.usage,
        )
        if self.state != CallState.TRANSFER_PENDING:
            return
        assert self.call_uuid is not None
        ok = await self.api.transfer(self.call_uuid, summary)
        if self.state != CallState.TRANSFER_PENDING:
            return
        if not ok:
            logger.warning(f"Call {self.label}: API rejected the transfer; AI resumes")
            await self._resume_ai(notify_api=False)
            return
        self._transfer_timer = asyncio.create_task(
            self._transfer_timeout(self.agent.ring_seconds), name=f"transfer-timer-{self.label}"
        )

    async def _transfer_timeout(self, seconds: int) -> None:
        await asyncio.sleep(seconds)
        if self.state == CallState.TRANSFER_PENDING:
            logger.info(f"Call {self.label}: no advisor answered in {seconds}s")
            await self._resume_ai(notify_api=True)

    async def _resume_ai(self, *, notify_api: bool) -> None:
        self.state = CallState.AI
        self._gate.mode = CallerMode.AI
        if notify_api and self.call_uuid:
            self._spawn(self.api.transfer_cancelled(self.call_uuid), "transfer-cancelled")
        if self._worker:
            await self._worker.queue_frames(
                [
                    LLMMessagesAppendFrame(
                        messages=[{"role": "developer", "content": NO_ADVISOR_INSTRUCTION}],
                        run_llm=True,
                    )
                ]
            )

    async def accept_advisor(self, sdp: str, sdp_type: str, user: AdvisorUser) -> dict:
        """Bridge an advisor's browser into the call. Returns the SDP answer."""
        if self.state != CallState.TRANSFER_PENDING:
            raise TransferNotPendingError()

        # Claim synchronously so the ring timer cannot fire in between.
        self.state = CallState.BRIDGED
        if self._transfer_timer:
            self._transfer_timer.cancel()
            self._transfer_timer = None
        self._gate.mode = CallerMode.BRIDGED

        advisor = AdvisorLeg(
            call_label=self.label,
            ice_servers=self.settings.ice_servers,
            to_caller=self._caller_injector.inject,
            on_closed=self._on_advisor_closed,
        )
        try:
            answer = await advisor.start(sdp, sdp_type)
        except Exception:
            logger.exception(f"Call {self.label}: advisor connection failed; AI resumes")
            await advisor.stop()
            await self._resume_ai(notify_api=True)
            raise

        self._advisor = advisor
        self._gate.sink = advisor.injector.inject
        self._caller_injector.bridged = True
        advisor_name = (user.name or "").strip() or "un asesor"
        self._add_transcript("advisor", f"[Llamada transferida a {advisor_name}]")
        logger.info(f"Call {self.label}: bridged with advisor {user.id}")
        assert self.call_uuid is not None
        # Background: the API is waiting on this request and may be single-threaded.
        self._spawn(self.api.transferred(self.call_uuid, user.id), "transferred")
        return answer

    async def _on_advisor_closed(self) -> None:
        if self.state != CallState.BRIDGED:
            return
        logger.info(f"Call {self.label}: advisor hung up; ending the call")
        self._add_transcript("advisor", "[El asesor colgó]")
        await self._cancel_runner()

    # ------------------------------------------------------------------- end

    def mark_remote_terminated(self) -> None:
        """Meta already ended the WhatsApp call; do not ask it to terminate again."""
        self._remote_terminated = True

    async def hangup(self) -> None:
        """Stop the call from our side (e.g. server shutdown)."""
        await self._cancel_runner()

    def _usage_payload(self) -> dict | None:
        """The call's usage for billing; None (field omitted) if it cannot be built."""
        try:
            return self.usage.to_payload()
        except Exception as e:
            logger.warning(f"Call {self.label}: usage report unavailable: {e}")
            return None

    async def finalize(self) -> None:
        """Tear down both legs and report the call to the API (exactly once)."""
        if self._finalized:
            return
        self._finalized = True
        self.state = CallState.ENDED

        if self._transfer_timer:
            self._transfer_timer.cancel()
        if self._answer_timer and self._answer_timer is not asyncio.current_task():
            self._answer_timer.cancel()
        if self._advisor:
            await self._advisor.stop()

        if self.whatsapp and not self._remote_terminated:
            try:
                await self.whatsapp.client._whatsapp_api.terminate_call_to_whatsapp(
                    self.whatsapp.call_id
                )
            except Exception as e:
                logger.warning(f"Call {self.label}: Meta terminate failed: {e}")
        await self._disconnect_media()

        if not self.call_uuid or self._cancelled:
            return
        duration = int(time.monotonic() - self.answered_at) if self.answered_at else 0
        if self.outbound and self.answered_at is None:
            # Never answered: no conversation to summarize; the API marks it failed
            summary = UNANSWERED_SUMMARY
        else:
            summary = await summarize_call(
                self.settings.anthropic_api_key,
                self.settings.anthropic_model,
                self.transcript,
                usage=self.usage,
            )
        await self.api.ended(
            self.call_uuid, self.transcript, summary, duration, usage=self._usage_payload()
        )
        logger.info(f"Call {self.label}: ended after {duration}s")
