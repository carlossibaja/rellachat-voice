"""Rellachat voice service: FastAPI server answering WhatsApp and phone (Telnyx) calls with AI.

Run with ``python rellachat/server.py`` from the repository root.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

if __package__ in (None, ""):
    # Allow ``python rellachat/server.py``: make the repo root importable.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp
from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from loguru import logger
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.whatsapp.api import WhatsAppConnectCall

from rellachat.api_client import RellachatApi
from rellachat.bot import AnswerNotExpectedError, TransferNotPendingError, VoiceCall, WhatsAppLeg
from rellachat.config import Settings, load_env
from rellachat.models import (
    AdvisorOfferRequest,
    ConnectRequest,
    OutboundAnswerRequest,
    OutboundOfferRequest,
    OutboundWhatsApp,
    PhonePrepareRequest,
    TerminateRequest,
    TestOfferRequest,
)
from rellachat.outbound import OutboundWebRTCConnection
from rellachat.phone import (
    TELNYX_START_TIMEOUT_SECS,
    DuplicateCallError,
    PhoneCallRegistry,
    PhoneVoiceCall,
    build_serializer,
    phone_agent,
    read_telnyx_start,
    stream_url,
)
from rellachat.prompts import PHONE_NOT_CONNECTED_SUMMARY, UNANSWERED_SUMMARY
from rellachat.security import secret_matches
from rellachat.whatsapp import (
    InvalidWebhookError,
    WhatsAppClientRegistry,
    parse_webhook,
    webhook_phone_number_id,
)

load_env()

SHUTDOWN_GRACE_SECS = 10
# WebSocket close code for a rejected Telnyx stream (policy violation)
WS_POLICY_VIOLATION = 1008


class ServiceState:
    """Process-wide state: settings, WhatsApp clients and live calls."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api = RellachatApi(settings.rellachat_api_url, settings.voice_service_secret)
        self.http: aiohttp.ClientSession | None = None
        self.clients: WhatsAppClientRegistry | None = None
        self.calls: dict[str, VoiceCall] = {}  # call_uuid -> call
        # Meta credentials of outbound calls, used to terminate them once answered
        self.outbound: dict[str, OutboundWhatsApp] = {}  # call_uuid -> credentials
        self.test_calls: set[VoiceCall] = set()
        # Phone (Telnyx) calls: prepared ones waiting for their stream, and the uuids of live ones
        self.phone = PhoneCallRegistry()
        self.phone_calls: set[str] = set()
        self.tasks: set[asyncio.Task] = set()

    def spawn(self, coro, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task


def create_app(
    settings: Settings | None = None,
    *,
    call_factory=VoiceCall,
    outbound_connection_factory=OutboundWebRTCConnection,
    phone_call_factory=PhoneVoiceCall,
) -> FastAPI:
    """Build the service; the factories are swappable so tests run without media or providers."""
    state = ServiceState(settings or Settings.from_env())

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async with aiohttp.ClientSession() as http:
            state.http = http
            state.clients = WhatsAppClientRegistry(http, state.settings.ice_servers)
            yield
            calls = list(state.calls.values()) + list(state.test_calls)
            for call in calls:
                await call.hangup()
            if state.tasks:
                await asyncio.wait(list(state.tasks), timeout=SHUTDOWN_GRACE_SECS)

    app = FastAPI(title="Rellachat voice service", lifespan=lifespan)
    app.state.service = state

    async def require_secret(x_voice_secret: str | None = Header(default=None)):
        if not secret_matches(x_voice_secret, state.settings.voice_service_secret):
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/health")
    async def health():
        return {"ok": True, "calls": len(state.calls), "test_calls": len(state.test_calls)}

    # ------------------------------------------------------------ WhatsApp

    async def run_call(call: VoiceCall) -> None:
        try:
            await call.run()
        finally:
            if call.call_uuid and state.calls.get(call.call_uuid) is call:
                state.calls.pop(call.call_uuid, None)
                state.outbound.pop(call.call_uuid, None)

    async def answer_whatsapp_call(body: ConnectRequest, request, client) -> None:
        started = False

        async def on_connection(connection: SmallWebRTCConnection, wa_call: WhatsAppConnectCall):
            nonlocal started
            started = True
            call = VoiceCall(
                settings=state.settings,
                api=state.api,
                agent=body.agent,
                connection=connection,
                call_uuid=body.call_uuid,
                whatsapp=WhatsAppLeg(client=client, call_id=wa_call.id),
                contact_name=body.contact.name,
            )
            state.calls[body.call_uuid] = call
            state.spawn(run_call(call), f"call-{body.call_uuid}")

        try:
            await client.handle_webhook_request(request, on_connection)
        except Exception as e:
            logger.error(f"Call {body.call_uuid}: could not answer WhatsApp call: {e}")
        if not started:
            await state.api.ended(body.call_uuid, [], "La IA no pudo contestar la llamada.", 0)

    @app.post("/calls/connect", status_code=202, dependencies=[Depends(require_secret)])
    async def calls_connect(body: ConnectRequest):
        try:
            request = parse_webhook(body.whatsapp.webhook, "connect")
        except InvalidWebhookError as e:
            raise HTTPException(status_code=422, detail=str(e))

        phone_number_id = body.whatsapp.phone_number_id or webhook_phone_number_id(request)
        if not phone_number_id or not body.whatsapp.token:
            raise HTTPException(
                status_code=422, detail="whatsapp.token and phone_number_id are required"
            )

        if body.call_uuid in state.calls:
            return {"ok": True, "duplicate": True}

        assert state.clients is not None
        client = state.clients.get_or_create(phone_number_id, body.whatsapp.token)
        state.spawn(answer_whatsapp_call(body, request, client), f"connect-{body.call_uuid}")
        return {"ok": True}

    @app.post("/calls/terminate", dependencies=[Depends(require_secret)])
    async def calls_terminate(body: TerminateRequest):
        try:
            request = parse_webhook(body.whatsapp.webhook, "terminate")
        except InvalidWebhookError as e:
            raise HTTPException(status_code=422, detail=str(e))

        call = state.calls.get(body.call_uuid)
        if call:
            call.mark_remote_terminated()

        phone_number_id = body.whatsapp.phone_number_id or webhook_phone_number_id(request)
        client = state.clients.get(phone_number_id) if state.clients and phone_number_id else None
        if client:
            try:
                await client.handle_webhook_request(request)
            except Exception as e:
                logger.warning(f"Call {body.call_uuid}: terminate webhook not handled: {e}")

        if call:
            await call.hangup()
        return {"ok": True, "found": call is not None}

    @app.post("/calls/{call_uuid}/advisor-offer", dependencies=[Depends(require_secret)])
    async def advisor_offer(call_uuid: str, body: AdvisorOfferRequest):
        call = state.calls.get(call_uuid)
        if not call:
            raise HTTPException(status_code=409, detail="Transfer is not pending")
        try:
            answer = await call.accept_advisor(body.sdp, body.type, body.user)
        except TransferNotPendingError:
            raise HTTPException(status_code=409, detail="Transfer is not pending")
        except Exception as e:
            logger.error(f"Call {call_uuid}: advisor offer failed: {e}")
            raise HTTPException(status_code=500, detail="Could not connect the advisor")
        return {"sdp": answer["sdp"], "type": answer.get("type", "answer")}

    # ------------------------------------------------------------ outbound

    @app.post("/calls/outbound/offer", dependencies=[Depends(require_secret)])
    async def outbound_offer(body: OutboundOfferRequest):
        """Create the business-side peer and its SDP offer; the API sends it to Meta's connect."""
        if body.call_uuid in state.calls:
            raise HTTPException(status_code=409, detail="Call already exists")

        connection = outbound_connection_factory(state.settings.ice_servers)
        try:
            offer = await connection.create_offer()
        except Exception as e:
            logger.error(f"Call {body.call_uuid}: could not create the SDP offer: {e}")
            await _disconnect_quietly(connection)
            raise HTTPException(status_code=500, detail="Could not create the SDP offer")

        if body.call_uuid in state.calls:  # raced with another offer for the same call
            await _disconnect_quietly(connection)
            raise HTTPException(status_code=409, detail="Call already exists")

        call = call_factory(
            settings=state.settings,
            api=state.api,
            agent=body.agent.model_copy(update={"direction": "outbound"}),
            connection=connection,
            call_uuid=body.call_uuid,
            whatsapp=None,  # set by /answer, once Meta gives the call id
            contact_name=body.contact.name,
            initiated_by=body.initiated_by.name,
        )
        state.calls[body.call_uuid] = call
        state.outbound[body.call_uuid] = body.whatsapp
        # The pipeline starts now; the AI waits for the customer's media before speaking
        state.spawn(run_call(call), f"call-{body.call_uuid}")
        return {"sdp": offer["sdp"], "type": "offer"}

    @app.post("/calls/{call_uuid}/answer", dependencies=[Depends(require_secret)])
    async def outbound_answer(call_uuid: str, body: OutboundAnswerRequest):
        """The customer's SDP answer (Meta connect webhook) for an outbound call."""
        call = state.calls.get(call_uuid)
        credentials = state.outbound.get(call_uuid)
        if not call or not credentials:
            raise HTTPException(status_code=404, detail="Unknown outbound call")

        assert state.clients is not None
        client = state.clients.get_or_create(credentials.phone_number_id, credentials.token)
        try:
            await call.apply_answer(body.sdp, body.type, WhatsAppLeg(client=client, call_id=body.wa_call_id))
        except AnswerNotExpectedError:
            raise HTTPException(status_code=409, detail="Call is not waiting for an answer")
        except Exception as e:
            logger.error(f"Call {call_uuid}: invalid SDP answer: {e}")
            raise HTTPException(status_code=422, detail="Invalid SDP answer")
        return {"ok": True}

    @app.post("/calls/{call_uuid}/cancel", dependencies=[Depends(require_secret)])
    async def outbound_cancel(call_uuid: str):
        """Meta never placed the call: tear it down without ``ended``. Unknown calls are ok (idempotent)."""
        if call_uuid in state.calls and call_uuid not in state.outbound:
            raise HTTPException(status_code=409, detail="Only outbound calls can be cancelled")
        call = state.calls.pop(call_uuid, None)
        state.outbound.pop(call_uuid, None)
        if call:
            await call.cancel()
        return {"ok": True, "found": call is not None}

    # ---------------------------------------------------------------- phone

    @app.post("/phone/calls/prepare", dependencies=[Depends(require_secret)])
    async def phone_prepare(body: PhonePrepareRequest):
        """Register a Telnyx call and return the one-time stream URL the API gives Telnyx."""
        base_url = state.settings.public_ws_base_url
        if not base_url:
            raise HTTPException(status_code=503, detail="PUBLIC_WS_BASE_URL is not set")
        if body.call_uuid in state.calls:
            raise HTTPException(status_code=409, detail="Call already exists")
        try:
            token = state.phone.prepare(body)
        except DuplicateCallError:
            raise HTTPException(status_code=409, detail="Call already exists")
        return {"stream_url": stream_url(base_url, body.call_uuid, token)}

    @app.post("/phone/calls/{call_uuid}/cancel", dependencies=[Depends(require_secret)])
    async def phone_cancel(call_uuid: str):
        """The call never went through at Telnyx: drop it without ``ended`` (idempotent)."""
        if call_uuid in state.calls and call_uuid not in state.phone_calls:
            raise HTTPException(status_code=409, detail="Only phone calls can be cancelled here")
        pending = state.phone.drop(call_uuid)
        call = state.calls.pop(call_uuid, None)
        state.phone_calls.discard(call_uuid)
        if call:
            await call.cancel()
        return {"ok": True, "found": pending is not None or call is not None}

    @app.post("/phone/calls/{call_uuid}/hangup", dependencies=[Depends(require_secret)])
    async def phone_hangup(call_uuid: str):
        """Telnyx hung up: finalize the call like a WhatsApp terminate (``ended`` comes from it)."""
        call = state.calls.get(call_uuid) if call_uuid in state.phone_calls else None
        if call:
            call.mark_remote_terminated()
            await call.hangup()
            return {"ok": True, "found": True}
        pending = state.phone.drop(call_uuid)
        if pending:
            # Never connected: nothing to summarize. Background: the API is waiting on this request.
            summary = (
                UNANSWERED_SUMMARY if pending.direction == "outbound" else PHONE_NOT_CONNECTED_SUMMARY
            )
            state.spawn(state.api.ended(call_uuid, [], summary, 0), f"phone-ended-{call_uuid}")
        return {"ok": True, "found": pending is not None}

    async def report_not_connected(call_uuid: str) -> None:
        state.spawn(
            state.api.ended(call_uuid, [], PHONE_NOT_CONNECTED_SUMMARY, 0),
            f"phone-ended-{call_uuid}",
        )

    @app.websocket("/phone/ws")
    async def phone_ws(websocket: WebSocket):
        """Telnyx's bidirectional media stream for a prepared call (one-time token)."""
        call_uuid = websocket.query_params.get("call")
        prepared = state.phone.claim(call_uuid, websocket.query_params.get("token"))
        if prepared is None:
            logger.warning(f"Phone stream rejected: unknown call or bad token ({call_uuid})")
            await websocket.close(code=WS_POLICY_VIOLATION)
            return

        await websocket.accept()
        try:
            start = await asyncio.wait_for(read_telnyx_start(websocket), TELNYX_START_TIMEOUT_SECS)
            expected = prepared.telnyx.call_control_id
            if expected and expected != start.call_control_id:
                raise ValueError(f"stream is for Telnyx call {start.call_control_id}")
        except Exception as e:
            logger.error(f"Call {prepared.call_uuid}: invalid Telnyx stream: {e}")
            await _close_quietly(websocket)
            await report_not_connected(prepared.call_uuid)
            return

        call = phone_call_factory(
            settings=state.settings,
            api=state.api,
            agent=phone_agent(prepared.agent, prepared.direction),
            websocket=websocket,
            serializer=build_serializer(start, state.settings.telnyx_api_key),
            call_uuid=prepared.call_uuid,
            contact_name=prepared.contact.name,
            initiated_by=prepared.initiated_by.name if prepared.initiated_by else None,
        )
        state.calls[prepared.call_uuid] = call
        state.phone_calls.add(prepared.call_uuid)
        logger.info(f"Call {prepared.call_uuid}: Telnyx stream {start.stream_id} connected")
        try:
            # The socket lives as long as this handler: run the call here, not in the background
            await run_call(call)
        finally:
            state.phone_calls.discard(prepared.call_uuid)
            await _close_quietly(websocket)

    # ---------------------------------------------------------- browser test

    @app.post("/test/offer", dependencies=[Depends(require_secret)])
    async def test_offer(body: TestOfferRequest):
        connection = SmallWebRTCConnection(state.settings.ice_servers)
        try:
            await connection.initialize(sdp=body.sdp, type=body.type)
        except Exception as e:
            logger.error(f"Test call: invalid offer: {e}")
            raise HTTPException(status_code=422, detail="Invalid SDP offer")
        answer = connection.get_answer()
        if not answer:
            raise HTTPException(status_code=500, detail="No SDP answer")

        call = VoiceCall(
            settings=state.settings, api=state.api, agent=body.agent, connection=connection
        )
        state.test_calls.add(call)

        async def run_test_call():
            try:
                await call.run()
            finally:
                state.test_calls.discard(call)

        state.spawn(run_test_call(), f"test-call-{connection.pc_id}")
        return {"sdp": answer["sdp"], "type": answer["type"], "pc_id": answer["pc_id"]}

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    return app


async def _close_quietly(websocket: WebSocket) -> None:
    try:
        await websocket.close()
    except Exception as e:
        logger.debug(f"WebSocket close failed: {e}")


async def _disconnect_quietly(connection) -> None:
    try:
        await connection.disconnect()
    except Exception as e:
        logger.debug(f"Peer disconnect failed: {e}")


def main() -> None:
    import uvicorn

    settings = Settings.from_env()
    if not settings.voice_service_secret:
        logger.warning("VOICE_SERVICE_SECRET is not set: every protected endpoint will return 401")
    uvicorn.run(
        create_app(settings),
        host=os.getenv("HOST", "0.0.0.0"),
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
