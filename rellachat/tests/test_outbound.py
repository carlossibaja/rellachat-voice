"""Outbound (business-initiated) AI calls: offer, answer, cancel, opener and teardown."""

import asyncio
import hashlib

import pytest
from fastapi.testclient import TestClient
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

from rellachat.bot import VoiceCall, WhatsAppLeg
from rellachat.config import Settings
from rellachat.models import AgentConfig, OutboundOfferRequest
from rellachat.outbound import (
    ANSWER_TIMEOUT_SECS,
    OutboundWebRTCConnection,
    filter_sdp_for_whatsapp,
)
from rellachat.prompts import (
    CALL_STARTED_MARKER,
    OUTBOUND_CALL_STARTED_MARKER,
    UNANSWERED_SUMMARY,
    build_system_prompt,
    call_started_marker,
)
from rellachat.server import create_app

SECRET = {"X-Voice-Secret": "s3cret"}


def agent(**overrides) -> AgentConfig:
    data = {
        "id": 7,
        "name": "Sofía",
        "identity_name": "Sofía",
        "company_name": "Rella",
        "purpose": "appointments",
        "instructions": "Confirma la cita de mañana.",
        "transfer_enabled": True,
    }
    data.update(overrides)
    return AgentConfig(**data)


def offer_body(call_uuid: str = "out-1") -> dict:
    return {
        "call_uuid": call_uuid,
        "whatsapp": {"token": "tok", "phone_number_id": "PNID-9"},
        "agent": {**agent().model_dump(), "direction": "outbound"},
        "contact": {"name": "Ana Pérez", "phone": "573001112233"},
        "initiated_by": {"id": 3, "name": "Carlos"},
    }


# ------------------------------------------------------------------ WebRTC peer


def test_the_offer_side_peer_creates_a_whatsapp_ready_offer_and_accepts_an_answer():
    async def scenario():
        caller = OutboundWebRTCConnection(ice_servers=[])
        callee = SmallWebRTCConnection(ice_servers=[])
        try:
            offer = await caller.create_offer()
            assert offer["type"] == "offer"
            sdp = offer["sdp"]
            assert "m=audio" in sdp
            assert "opus/48000/2" in sdp
            assert "a=sendrecv" in sdp
            fingerprints = [line for line in sdp.splitlines() if line.startswith("a=fingerprint:")]
            assert fingerprints and all(f.startswith("a=fingerprint:sha-256") for f in fingerprints)
            assert sdp.endswith("\r\n")

            await callee.initialize(sdp=sdp, type="offer")
            answer = callee.get_answer()
            await caller.apply_answer(answer["sdp"], "answer")
            assert caller.pc.remoteDescription.type == "answer"
            assert caller.pc.signalingState == "stable"
            # The pipeline's audio goes out on the first transceiver, as SmallWebRTCTransport expects
            assert caller.pc.getTransceivers()[0].kind == "audio"
        finally:
            await caller.disconnect()
            await callee.disconnect()

    asyncio.run(scenario())


def test_the_offer_sdp_keeps_only_sha256_fingerprints():
    sdp = "v=0\na=fingerprint:sha-256 AA\na=fingerprint:sha-384 BB\na=fingerprint:sha-512 CC\n"
    assert filter_sdp_for_whatsapp(sdp) == "v=0\r\na=fingerprint:sha-256 AA\r\n"


# ------------------------------------------------------------------- prompts


def test_outbound_prompt_tells_the_agent_it_is_calling_and_how_to_open():
    prompt = build_system_prompt(
        agent(direction="outbound"), contact_name="Ana Pérez", initiated_by="Carlos"
    )
    assert "Estás llamando" in prompt
    assert "Sofía" in prompt and "Rella" in prompt
    assert "motivo de la llamada" in prompt
    assert "Ana Pérez" in prompt
    assert "Carlos" in prompt
    assert "Confirma la cita de mañana." in prompt
    assert "Atiendes llamadas" not in prompt


# sha256 of the outbound prompt below before call instructions existed (T9b)
OUTBOUND_PROMPT_SHA256 = "8ebd05c9b158a9d0f2a714a87a00620e4b298d0054bd16189df165e05c9150f5"


def test_agent_config_accepts_optional_call_instructions():
    assert agent().call_instructions is None
    assert agent(call_instructions="Ofrécele la promo.").call_instructions == "Ofrécele la promo."
    body = offer_body()
    body["agent"]["call_instructions"] = "Ofrécele la promo."
    assert OutboundOfferRequest(**body).agent.call_instructions == "Ofrécele la promo."


def test_outbound_prompt_takes_the_advisors_instructions_as_the_call_goal():
    instructions = "Recuérdale que su pedido 1234 está listo para recoger hoy."
    prompt = build_system_prompt(
        agent(direction="outbound", call_instructions=f"  {instructions}  "),
        contact_name="Ana Pérez",
        initiated_by="Carlos",
    )
    assert "Objetivo de esta llamada" in prompt
    assert instructions in prompt
    assert "pidió Carlos" in prompt
    assert "no la leas" in prompt
    # Identity, company and the agent's own rules stay in place
    assert "Sofía" in prompt and "Rella" in prompt
    assert "Confirma la cita de mañana." in prompt
    assert "una sola pregunta a la vez" in prompt


def test_outbound_goal_without_advisor_name_still_reads_naturally():
    prompt = build_system_prompt(agent(direction="outbound", call_instructions="Ofrece la promo."))
    assert "Objetivo de esta llamada" in prompt
    assert "un asesor del equipo de Rella" in prompt


@pytest.mark.parametrize("value", [None, "", "   \n "])
def test_outbound_prompt_without_call_instructions_is_unchanged(value):
    prompt = build_system_prompt(
        agent(direction="outbound", call_instructions=value),
        contact_name="Ana Pérez",
        initiated_by="Carlos",
    )
    assert hashlib.sha256(prompt.encode()).hexdigest() == OUTBOUND_PROMPT_SHA256


def test_inbound_prompt_ignores_call_instructions():
    assert build_system_prompt(agent(call_instructions="Ofrece la promo.")) == build_system_prompt(
        agent()
    )


def test_inbound_prompt_is_unchanged_by_the_outbound_variant():
    inbound = build_system_prompt(agent())
    assert inbound == build_system_prompt(agent(direction="inbound"))
    assert inbound == build_system_prompt(agent(), contact_name="Ana", initiated_by="Carlos")
    assert "Atiendes llamadas telefónicas por WhatsApp." in inbound
    assert "Estás llamando" not in inbound


def test_the_first_context_message_depends_on_the_direction():
    assert call_started_marker(agent()) == CALL_STARTED_MARKER
    assert call_started_marker(agent(direction="outbound")) == OUTBOUND_CALL_STARTED_MARKER
    assert agent().direction == "inbound"
    assert agent(direction="outbound").is_outbound


# ----------------------------------------------------------------- VoiceCall


class FakeApi:
    def __init__(self):
        self.answered_calls: list[str] = []
        self.ended_calls: list[dict] = []

    async def answered(self, call_uuid):
        self.answered_calls.append(call_uuid)
        return True

    async def ended(self, call_uuid, transcript, summary, duration, usage=None):
        self.ended_calls.append(
            {"uuid": call_uuid, "summary": summary, "duration": duration, "transcript": transcript}
        )
        return True


class FakeConnection:
    pc_id = "fake-pc"

    def __init__(self, *_args, **_kwargs):
        self.answers: list[tuple[str, str]] = []
        self.disconnected = False

    async def create_offer(self):
        return {"sdp": "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n", "type": "offer"}

    async def apply_answer(self, sdp, type="answer"):
        self.answers.append((sdp, type))

    async def disconnect(self):
        self.disconnected = True


class FakeMetaApi:
    def __init__(self):
        self.terminated: list[str] = []

    async def terminate_call_to_whatsapp(self, call_id):
        self.terminated.append(call_id)
        return {"success": True}


class FakeWhatsAppClient:
    def __init__(self):
        self._whatsapp_api = FakeMetaApi()


class FakeWorker:
    def __init__(self):
        self.frames = []

    async def queue_frames(self, frames):
        self.frames.extend(frames)


def outbound_call(api=None, connection=None, **kwargs) -> VoiceCall:
    return VoiceCall(
        settings=Settings(),
        api=api or FakeApi(),
        agent=agent(direction="outbound"),
        connection=connection or FakeConnection(),
        call_uuid="out-1",
        contact_name="Ana",
        **kwargs,
    )


def test_answer_applies_the_customer_sdp_and_keeps_the_meta_call():
    async def scenario():
        connection = FakeConnection()
        call = outbound_call(connection=connection)
        leg = WhatsAppLeg(client=FakeWhatsAppClient(), call_id="wacid.OUT")
        await call.apply_answer("v=0\r\n", "answer", leg)
        assert connection.answers == [("v=0\r\n", "answer")]
        assert call.whatsapp is leg
        with pytest.raises(Exception):
            await call.apply_answer("v=0\r\n", "answer", leg)

    asyncio.run(scenario())


def test_an_inbound_call_rejects_an_sdp_answer():
    async def scenario():
        call = VoiceCall(
            settings=Settings(), api=FakeApi(), agent=agent(), connection=FakeConnection(),
            call_uuid="in-1",
        )
        with pytest.raises(Exception):
            await call.apply_answer("v=0\r\n", "answer", None)

    asyncio.run(scenario())


def test_outbound_media_connected_reports_answered_and_lets_the_ai_open():
    async def scenario():
        api = FakeApi()
        call = outbound_call(api=api)
        call._worker = FakeWorker()
        await call._on_media_connected()
        await call._on_media_connected()  # a second "connected" must not repeat it
        await asyncio.sleep(0)
        assert api.answered_calls == ["out-1"]
        assert call.answered_at is not None
        assert [type(f) for f in call._worker.frames] == [LLMRunFrame]

    asyncio.run(scenario())


def test_inbound_media_connected_still_speaks_the_greeting():
    async def scenario():
        call = VoiceCall(
            settings=Settings(), api=FakeApi(), agent=agent(greeting="Hola"),
            connection=FakeConnection(), call_uuid="in-1",
        )
        call._worker = FakeWorker()
        await call._on_media_connected()
        assert len(call._worker.frames) == 1
        assert isinstance(call._worker.frames[0], TTSSpeakFrame)
        assert call._worker.frames[0].text == "Hola"

    asyncio.run(scenario())


def test_ending_an_answered_outbound_call_terminates_it_at_meta_with_its_wa_call_id():
    async def scenario():
        api = FakeApi()
        call = outbound_call(api=api)
        client = FakeWhatsAppClient()
        await call.apply_answer("v=0\r\n", "answer", WhatsAppLeg(client=client, call_id="wacid.OUT"))
        call._worker = FakeWorker()
        await call._on_media_connected()
        await call.finalize()  # what end_call leads to once the farewell is spoken
        assert client._whatsapp_api.terminated == ["wacid.OUT"]
        assert [c["uuid"] for c in api.ended_calls] == ["out-1"]

    asyncio.run(scenario())


def test_a_cancelled_outbound_call_is_torn_down_without_reporting_ended():
    async def scenario():
        api = FakeApi()
        connection = FakeConnection()
        call = outbound_call(api=api, connection=connection)
        await call.cancel()
        await call.cancel()
        assert connection.disconnected
        assert api.ended_calls == []
        assert api.answered_calls == []

    asyncio.run(scenario())


def test_an_unanswered_outbound_call_ends_as_unanswered_after_the_timeout():
    async def scenario():
        api = FakeApi()
        connection = FakeConnection()
        call = outbound_call(api=api, connection=connection, answer_timeout_secs=0.05)

        async def ringing_forever():
            while not call._stop_requested:
                await asyncio.sleep(0.01)

        call._run_pipeline = ringing_forever
        await asyncio.wait_for(call.run(), timeout=2)
        assert api.answered_calls == []
        assert api.ended_calls == [
            {"uuid": "out-1", "summary": UNANSWERED_SUMMARY, "duration": 0, "transcript": []}
        ]
        assert connection.disconnected

    asyncio.run(scenario())


def test_the_default_answer_timeout_is_a_minute():
    assert ANSWER_TIMEOUT_SECS == 60


# -------------------------------------------------------------------- server


class FakeCall:
    instances: list["FakeCall"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.agent = kwargs["agent"]
        self.call_uuid = kwargs["call_uuid"]
        self.answers = []
        self.cancelled = False
        self.hung_up = False
        self.remote_terminated = False
        self._done = asyncio.Event()
        FakeCall.instances.append(self)

    async def run(self):
        await self._done.wait()

    async def apply_answer(self, sdp, sdp_type, leg):
        self.answers.append((sdp, sdp_type, leg))

    async def cancel(self):
        self.cancelled = True
        self._done.set()

    def mark_remote_terminated(self):
        self.remote_terminated = True

    async def hangup(self):
        self.hung_up = True
        self._done.set()


@pytest.fixture
def client():
    FakeCall.instances = []
    app = create_app(
        Settings(voice_service_secret="s3cret"),
        call_factory=FakeCall,
        outbound_connection_factory=FakeConnection,
    )
    with TestClient(app) as c:
        c.service = app.state.service
        yield c


@pytest.mark.parametrize(
    "path", ["/calls/outbound/offer", "/calls/abc/answer", "/calls/abc/cancel"]
)
def test_outbound_endpoints_require_the_secret(client, path):
    assert client.post(path, json={}).status_code == 401
    assert client.post(path, json={}, headers={"X-Voice-Secret": "nope"}).status_code == 401


def test_the_offer_endpoint_returns_the_sdp_offer_and_registers_the_call(client):
    response = client.post("/calls/outbound/offer", json=offer_body(), headers=SECRET)
    assert response.status_code == 200
    assert response.json() == {"sdp": "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n", "type": "offer"}
    call = client.service.calls["out-1"]
    assert call.agent.direction == "outbound"
    assert call.kwargs["contact_name"] == "Ana Pérez"
    assert call.kwargs["initiated_by"] == "Carlos"
    assert call.kwargs["whatsapp"] is None


def test_the_offer_forces_the_outbound_direction(client):
    body = offer_body()
    body["agent"].pop("direction")
    client.post("/calls/outbound/offer", json=body, headers=SECRET)
    assert client.service.calls["out-1"].agent.direction == "outbound"


def test_a_duplicate_offer_is_409(client):
    assert client.post("/calls/outbound/offer", json=offer_body(), headers=SECRET).status_code == 200
    assert client.post("/calls/outbound/offer", json=offer_body(), headers=SECRET).status_code == 409


def test_answer_applies_the_sdp_with_a_meta_leg_for_the_wa_call_id(client):
    client.post("/calls/outbound/offer", json=offer_body(), headers=SECRET)
    response = client.post(
        "/calls/out-1/answer", json={"sdp": "v=0\r\n", "wa_call_id": "wacid.OUT"}, headers=SECRET
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    sdp, sdp_type, leg = client.service.calls["out-1"].answers[0]
    assert (sdp, sdp_type, leg.call_id) == ("v=0\r\n", "answer", "wacid.OUT")
    meta = leg.client._whatsapp_api
    assert meta._phone_number_id == "PNID-9"
    assert meta._whatsapp_token == "tok"


def test_answer_for_an_unknown_call_is_404(client):
    response = client.post(
        "/calls/nope/answer", json={"sdp": "v=0\r\n", "wa_call_id": "wacid.X"}, headers=SECRET
    )
    assert response.status_code == 404


def test_cancel_tears_the_call_down_and_is_idempotent(client):
    client.post("/calls/outbound/offer", json=offer_body(), headers=SECRET)
    call = client.service.calls["out-1"]
    response = client.post("/calls/out-1/cancel", headers=SECRET)
    assert response.status_code == 200
    assert response.json() == {"ok": True, "found": True}
    assert call.cancelled
    assert "out-1" not in client.service.calls
    again = client.post("/calls/out-1/cancel", headers=SECRET)
    assert again.status_code == 200
    assert again.json() == {"ok": True, "found": False}


def test_meta_terminate_ends_an_outbound_call_through_the_normal_flow(client):
    from rellachat.tests.test_units import terminate_webhook

    client.post("/calls/outbound/offer", json=offer_body(), headers=SECRET)
    call = client.service.calls["out-1"]
    response = client.post(
        "/calls/terminate",
        json={
            "call_uuid": "out-1",
            "whatsapp": {"token": "tok", "phone_number_id": "PNID-9", "webhook": terminate_webhook()},
        },
        headers=SECRET,
    )
    assert response.status_code == 200
    assert response.json()["found"] is True
    assert call.remote_terminated and call.hung_up


def test_cancel_refuses_an_inbound_call(client):
    client.service.calls["in-1"] = FakeCall(agent=agent(), call_uuid="in-1")
    assert client.post("/calls/in-1/cancel", headers=SECRET).status_code == 409
    assert "in-1" in client.service.calls
