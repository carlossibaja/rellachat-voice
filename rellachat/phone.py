"""Phone calls with real numbers (Telnyx): pending calls, the media stream and the call itself.

Flow: the Rellachat API prepares a call (``POST /phone/calls/prepare``) and gets a
one-time ``stream_url``; it hands that URL to Telnyx (``answer`` for inbound calls,
``dial`` for outbound ones). Telnyx then opens ``/phone/ws`` and sends ``connected``
and ``start`` events; ``start`` carries the stream id, the call control id and the
media encoding the ``TelnyxFrameSerializer`` needs. From there the call runs the same
pipeline as a WhatsApp call, only over Telnyx's WebSocket.
"""

import hmac
import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlencode

from loguru import logger
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from rellachat.bot import VoiceCall, caller_audio_filter
from rellachat.models import AgentConfig, PhonePrepareRequest
from rellachat.prompts import MESSAGING_TOOLS

# Seconds Telnyx has to open the stream of a prepared call before it is dropped
PHONE_STREAM_TTL_SECS = 120
# Seconds to wait for Telnyx's ``start`` event once the WebSocket is open
TELNYX_START_TIMEOUT_SECS = 10
# Messages read looking for ``start`` (Telnyx sends ``connected`` first)
TELNYX_START_MAX_MESSAGES = 5
SUPPORTED_ENCODINGS = frozenset({"PCMU", "PCMA"})
WS_PATH = "/phone/ws"


class DuplicateCallError(Exception):
    """Raised when a call uuid is prepared twice."""


class InvalidStreamError(Exception):
    """Raised when Telnyx's stream handshake is missing or unusable."""


@dataclass
class PendingPhoneCall:
    request: PhonePrepareRequest
    token: str
    expires_at: float


class PhoneCallRegistry:
    """Prepared phone calls waiting for Telnyx to open their media stream."""

    def __init__(
        self, ttl: float = PHONE_STREAM_TTL_SECS, clock: Callable[[], float] = time.monotonic
    ):
        self.ttl = ttl
        self.clock = clock
        self._pending: dict[str, PendingPhoneCall] = {}

    def __contains__(self, call_uuid: str) -> bool:
        self._purge()
        return call_uuid in self._pending

    def _purge(self) -> None:
        now = self.clock()
        for call_uuid in [u for u, p in self._pending.items() if p.expires_at <= now]:
            logger.info(f"Call {call_uuid}: Telnyx never opened the stream; dropped")
            self._pending.pop(call_uuid, None)

    def prepare(self, request: PhonePrepareRequest) -> str:
        """Register the call and return its one-time stream token."""
        if request.call_uuid in self:
            raise DuplicateCallError(request.call_uuid)
        token = secrets.token_urlsafe(32)
        self._pending[request.call_uuid] = PendingPhoneCall(request, token, self.clock() + self.ttl)
        return token

    def claim(self, call_uuid: str | None, token: str | None) -> PhonePrepareRequest | None:
        """The prepared call if the token matches (constant time); it can be claimed once."""
        if not call_uuid or not token or call_uuid not in self:
            return None
        pending = self._pending[call_uuid]
        if not hmac.compare_digest(pending.token.encode(), token.encode()):
            return None
        del self._pending[call_uuid]
        return pending.request

    def drop(self, call_uuid: str) -> PhonePrepareRequest | None:
        """Forget a pending call; returns it when it was still pending."""
        self._purge()
        pending = self._pending.pop(call_uuid, None)
        return pending.request if pending else None


def stream_url(base_url: str, call_uuid: str, token: str) -> str:
    """Public WebSocket URL Telnyx streams the call's media to."""
    return f"{base_url.rstrip('/')}{WS_PATH}?{urlencode({'call': call_uuid, 'token': token})}"


def phone_agent(agent: AgentConfig, direction: str) -> AgentConfig:
    """The agent for a phone call: no messaging tools, no transfer (Phase 1)."""
    tools = [
        definition
        for definition in agent.tools
        if not (
            isinstance(definition, dict)
            and isinstance(definition.get("function"), dict)
            and definition["function"].get("name") in MESSAGING_TOOLS
        )
    ]
    return agent.model_copy(
        update={
            "transport": "phone",
            "direction": direction,
            "transfer_enabled": False,
            "tools": tools,
        }
    )


# ---------------------------------------------------------------- handshake


@dataclass(frozen=True)
class TelnyxStart:
    stream_id: str
    call_control_id: str
    encoding: str


def parse_telnyx_start(message: dict) -> TelnyxStart | None:
    """The ``start`` event's ids and encoding; None for any other event."""
    if not isinstance(message, dict) or message.get("event") != "start":
        return None
    start = message.get("start") if isinstance(message.get("start"), dict) else {}
    stream_id = message.get("stream_id") or start.get("stream_id")
    call_control_id = start.get("call_control_id")
    media_format = start.get("media_format") if isinstance(start.get("media_format"), dict) else {}
    encoding = str(media_format.get("encoding") or "PCMU").upper()
    if not stream_id or not call_control_id:
        raise InvalidStreamError("start event without stream_id or call_control_id")
    if encoding not in SUPPORTED_ENCODINGS:
        raise InvalidStreamError(f"unsupported media encoding {encoding}")
    return TelnyxStart(str(stream_id), str(call_control_id), encoding)


async def read_telnyx_start(websocket) -> TelnyxStart:
    """Read Telnyx's first messages until ``start`` (``connected`` comes before it)."""
    for _ in range(TELNYX_START_MAX_MESSAGES):
        raw = await websocket.receive_text()
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            continue
        start = parse_telnyx_start(message)
        if start:
            return start
    raise InvalidStreamError("no start event from Telnyx")


def build_serializer(start: TelnyxStart, api_key: str) -> TelnyxFrameSerializer:
    """Telnyx serializer; with an API key it hangs up the call when the pipeline ends."""
    if not api_key:
        logger.warning("TELNYX_API_KEY is not set: calls the agent ends will not hang up at Telnyx")
    return TelnyxFrameSerializer(
        stream_id=start.stream_id,
        # Telnyx's encoding for the audio it sends us, and the same codec for ours (bidirectional)
        outbound_encoding=start.encoding,
        inbound_encoding=start.encoding,
        call_control_id=start.call_control_id,
        api_key=api_key or None,
        params=TelnyxFrameSerializer.InputParams(auto_hang_up=bool(api_key)),
    )


# --------------------------------------------------------------------- call


class PhoneVoiceCall(VoiceCall):
    """A ``VoiceCall`` whose caller is on Telnyx's media WebSocket instead of WebRTC."""

    def __init__(self, *, websocket, serializer: TelnyxFrameSerializer, **kwargs):
        super().__init__(connection=None, **kwargs)
        self.websocket = websocket
        self.serializer = serializer

    def _create_transport(self) -> FastAPIWebsocketTransport:
        return FastAPIWebsocketTransport(
            websocket=self.websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_filter=caller_audio_filter(),
                add_wav_header=False,
                serializer=self.serializer,
                # Telnyx sends no Origin; the one-time token already authorized this socket
                allowed_origins=[],
            ),
        )

    async def _disconnect_media(self) -> None:
        # The transport closes the WebSocket on end/cancel; the endpoint closes it otherwise
        return None
