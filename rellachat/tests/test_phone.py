"""Phone calls over Telnyx: prepare/cancel/hangup, the media WebSocket and the phone prompt."""

import asyncio
import hashlib
import json
import time
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport
from starlette.websockets import WebSocketDisconnect

import rellachat.bot as bot_module
from rellachat.config import Settings
from rellachat.models import AgentConfig, PhonePrepareRequest
from rellachat.phone import (
    PHONE_STREAM_TTL_SECS,
    DuplicateCallError,
    InvalidStreamError,
    PhoneCallRegistry,
    PhoneVoiceCall,
    TelnyxStart,
    build_serializer,
    parse_telnyx_start,
    phone_agent,
)
from rellachat.prompts import UNANSWERED_SUMMARY, build_system_prompt
from rellachat.server import create_app

SECRET = {"X-Voice-Secret": "s3cret"}


def tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object", "properties": {}}},
    }


ALL_TOOLS = [
    tool(name)
    for name in [
        "buscar_productos",
        "enviar_producto",
        "enviar_precios",
        "crear_pedido",
        "agendar_cita",
        "consultar_disponibilidad",
    ]
]


def base_agent(**overrides) -> AgentConfig:
    data = {
        "id": 7,
        "name": "Sofía",
        "identity_name": "Sofía",
        "company_name": "Rella",
        "purpose": "sales",
        "instructions": "Sé amable.",
        "company_description": "Tienda.",
        "company_website": "rella.co",
        "tools": ALL_TOOLS,
        "tools_prompt": "Reglas API.",
    }
    data.update(overrides)
    return AgentConfig(**data)


def prepare_body(call_uuid: str = "ph-1", direction: str = "inbound", **overrides) -> dict:
    body = {
        "call_uuid": call_uuid,
        "direction": direction,
        "telnyx": {"call_control_id": "v2:ccid-1"} if direction == "inbound" else {},
        "agent": {**base_agent().model_dump(), "transport": "phone", "transfer_enabled": False},
        "contact": {"name": "Ana Pérez", "phone": "+15550001111"},
        "initiated_by": {"id": 3, "name": "Carlos"} if direction == "outbound" else None,
    }
    body.update(overrides)
    return body


def telnyx_messages(stream_id="st-1", call_control_id="v2:ccid-1", encoding="PCMU") -> list[str]:
    return [
        json.dumps({"event": "connected", "version": "1.0.0"}),
        json.dumps(
            {
                "event": "start",
                "sequence_number": "1",
                "stream_id": stream_id,
                "start": {
                    "call_control_id": call_control_id,
                    "media_format": {"encoding": encoding, "sample_rate": 8000, "channels": 1},
                    "from": "+15550001111",
                    "to": "+15550002222",
                },
            }
        ),
    ]


# ----------------------------------------------------------- WhatsApp prompts

# sha256 of the WhatsApp prompts before phone calls existed (P1b): they must not change
WHATSAPP_PROMPT_SHA256 = {
    "inbound": "1a3faafedcb9dc116e7c7cdf84d235000abcc5f9c58c7acb8782bdd4cba7ed98",
    "inbound_test": "d52d6455a2b07377ecb368ba7e0932031a5d7b2d906a3dc8379c3f4f7c7e3684",
    "inbound_no_transfer": "282ad1dd2de6780393720e98a5a99f144c48c900a55e32373fd0c6e7a576db1f",
    "outbound": "81825625c1e466eb3acdf1398121dfc421c0bd625d28771319118f9d215233bc",
}


def whatsapp_prompts() -> dict[str, str]:
    return {
        "inbound": build_system_prompt(base_agent()),
        "inbound_test": build_system_prompt(base_agent(), test_mode=True),
        "inbound_no_transfer": build_system_prompt(base_agent(transfer_enabled=False)),
        "outbound": build_system_prompt(
            base_agent(direction="outbound", call_instructions="Ofrece la promo."),
            contact_name="Ana",
            initiated_by="Carlos",
        ),
    }


def test_whatsapp_prompts_are_byte_identical_to_before_phone_calls():
    hashes = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in whatsapp_prompts().items()}
    assert hashes == WHATSAPP_PROMPT_SHA256


def test_the_agent_transport_defaults_to_whatsapp():
    assert base_agent().transport == "whatsapp"
    assert not base_agent().is_phone
    assert base_agent(transport="phone").is_phone


# -------------------------------------------------------------- phone prompts


def phone_prompt(direction: str = "inbound", **kwargs) -> str:
    agent = phone_agent(base_agent(transport="phone"), direction)
    return build_system_prompt(agent, **kwargs)


@pytest.mark.parametrize("direction", ["inbound", "outbound"])
def test_the_phone_prompt_never_offers_whatsapp_or_messages(direction):
    prompt = phone_prompt(direction, contact_name="Ana", initiated_by="Carlos")
    assert "WhatsApp" not in prompt
    assert "te lo envío" not in prompt
    assert "no puedes enviar" in prompt.lower()


@pytest.mark.parametrize("direction", ["inbound", "outbound"])
def test_the_phone_prompt_says_product_details_out_loud(direction):
    prompt = phone_prompt(direction)
    assert "buscar_productos" in prompt
    assert "precio" in prompt and "disponib" in prompt
    assert "en voz alta" in prompt


def test_the_phone_prompt_offers_no_transfer():
    prompt = phone_prompt()
    assert "transfer_to_advisor" not in prompt
    assert "No puedes transferir" in prompt


def test_the_phone_prompt_keeps_identity_purpose_and_call_goal():
    inbound = phone_prompt()
    assert "Sofía" in inbound and "Rella" in inbound
    assert "Atiendes llamadas telefónicas" in inbound
    outbound = phone_prompt("outbound", contact_name="Ana", initiated_by="Carlos")
    assert "Estás llamando" in outbound
    assert "Objetivo de esta llamada" not in outbound
    with_goal = build_system_prompt(
        phone_agent(
            base_agent(transport="phone", call_instructions="Ofrece la promo."), "outbound"
        ),
        initiated_by="Carlos",
    )
    assert "Ofrece la promo." in with_goal


def test_the_phone_agent_drops_messaging_tools_and_transfer():
    agent = phone_agent(base_agent(transfer_enabled=True), "outbound")
    names = [t["function"]["name"] for t in agent.tools]
    assert names == ["buscar_productos", "agendar_cita", "consultar_disponibilidad"]
    assert agent.transfer_enabled is False
    assert agent.transport == "phone"
    assert agent.direction == "outbound"


def test_without_product_search_the_phone_prompt_has_no_product_rule():
    agent = phone_agent(base_agent(tools=[tool("agendar_cita")]), "inbound")
    assert "buscar_productos" not in build_system_prompt(agent)


# ------------------------------------------------------------- pending calls


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def request(call_uuid="ph-1", direction="inbound") -> PhonePrepareRequest:
    return PhonePrepareRequest(**prepare_body(call_uuid, direction))


def test_a_prepared_call_is_claimed_once_with_its_token():
    registry = PhoneCallRegistry()
    token = registry.prepare(request())
    assert len(token) >= 32
    assert registry.claim("ph-1", "wrong") is None
    assert registry.claim("nope", token) is None
    claimed = registry.claim("ph-1", token)
    assert claimed is not None and claimed.call_uuid == "ph-1"
    assert registry.claim("ph-1", token) is None


def test_a_prepared_call_expires_unused():
    clock = Clock()
    registry = PhoneCallRegistry(clock=clock)
    token = registry.prepare(request())
    assert PHONE_STREAM_TTL_SECS == 120
    clock.now += PHONE_STREAM_TTL_SECS + 1
    assert registry.claim("ph-1", token) is None
    assert "ph-1" not in registry


def test_preparing_the_same_call_twice_is_a_duplicate():
    registry = PhoneCallRegistry()
    registry.prepare(request())
    with pytest.raises(DuplicateCallError):
        registry.prepare(request())


def test_the_registry_drops_a_pending_call():
    registry = PhoneCallRegistry()
    token = registry.prepare(request())
    assert registry.drop("ph-1") is not None
    assert registry.drop("ph-1") is None
    assert registry.claim("ph-1", token) is None


# --------------------------------------------------------------- Telnyx start


def test_the_start_message_gives_the_stream_call_and_encoding():
    start = parse_telnyx_start(json.loads(telnyx_messages()[1]))
    assert start == TelnyxStart(stream_id="st-1", call_control_id="v2:ccid-1", encoding="PCMU")
    assert parse_telnyx_start(json.loads(telnyx_messages()[0])) is None


def test_the_start_message_rejects_unsupported_encodings_and_missing_ids():
    with pytest.raises(InvalidStreamError):
        parse_telnyx_start(json.loads(telnyx_messages(encoding="L16")[1]))
    with pytest.raises(InvalidStreamError):
        parse_telnyx_start(json.loads(telnyx_messages(stream_id="")[1]))


def test_the_serializer_hangs_up_through_telnyx_with_the_call_ids():
    serializer = build_serializer(TelnyxStart("st-1", "v2:ccid-1", "PCMA"), "KEY123")
    assert isinstance(serializer, TelnyxFrameSerializer)
    assert serializer._stream_id == "st-1"
    assert serializer._call_control_id == "v2:ccid-1"
    assert serializer._api_key == "KEY123"
    assert serializer._params.auto_hang_up is True
    assert serializer._params.inbound_encoding == "PCMA"
    assert serializer._params.outbound_encoding == "PCMA"


def test_without_an_api_key_the_serializer_cannot_hang_up_but_still_works():
    serializer = build_serializer(TelnyxStart("st-1", "v2:ccid-1", "PCMU"), "")
    assert serializer._params.auto_hang_up is False


# ------------------------------------------------------------ PhoneVoiceCall


class FakeApi:
    def __init__(self):
        self.answered_calls: list[str] = []
        self.ended_calls: list[dict] = []

    async def answered(self, call_uuid):
        self.answered_calls.append(call_uuid)
        return True

    async def ended(self, call_uuid, transcript, summary, duration, usage=None):
        self.ended_calls.append(
            {"uuid": call_uuid, "summary": summary, "duration": duration, "usage": usage}
        )
        return True


class FakeWebSocket:
    headers: dict = {}


def phone_call(api, direction="inbound") -> PhoneVoiceCall:
    return PhoneVoiceCall(
        settings=Settings(),
        api=api,
        agent=phone_agent(base_agent(), direction),
        websocket=FakeWebSocket(),
        serializer=build_serializer(TelnyxStart("st-1", "v2:ccid-1", "PCMU"), "KEY"),
        call_uuid="ph-1",
        contact_name="Ana",
    )


def test_the_phone_call_uses_the_websocket_transport_with_the_telnyx_serializer():
    call = phone_call(FakeApi())
    transport = call._create_transport()
    assert isinstance(transport, FastAPIWebsocketTransport)
    assert transport._params.serializer is call.serializer
    assert transport._params.allowed_origins == []
    assert transport._params.audio_in_enabled and transport._params.audio_out_enabled


def test_a_telnyx_hangup_finalizes_the_phone_call_and_reports_ended(monkeypatch):
    async def fake_summary(*_args, **_kwargs):
        return "Resumen"

    monkeypatch.setattr(bot_module, "summarize_call", fake_summary)

    async def scenario():
        api = FakeApi()
        call = phone_call(api)

        async def live_until_stopped():
            call._mark_answered()
            while not call._stop_requested:
                await asyncio.sleep(0.01)

        call._run_pipeline = live_until_stopped
        task = asyncio.create_task(call.run())
        await asyncio.sleep(0.05)
        call.mark_remote_terminated()
        await call.hangup()
        await asyncio.wait_for(task, timeout=2)
        assert api.answered_calls == ["ph-1"]
        assert [(c["uuid"], c["summary"]) for c in api.ended_calls] == [("ph-1", "Resumen")]
        assert api.ended_calls[0]["usage"] is not None

    asyncio.run(scenario())


def test_a_cancelled_phone_call_never_reports_ended():
    async def scenario():
        api = FakeApi()
        call = phone_call(api, "outbound")
        await call.cancel()
        assert api.ended_calls == []

    asyncio.run(scenario())


# -------------------------------------------------------------------- server


class FakePhoneCall:
    instances: list["FakePhoneCall"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.agent = kwargs["agent"]
        self.call_uuid = kwargs["call_uuid"]
        self.serializer = kwargs["serializer"]
        self.ran = False
        self.cancelled = False
        self.hung_up = False
        self.remote_terminated = False
        FakePhoneCall.instances.append(self)

    async def run(self):
        self.ran = True

    async def cancel(self):
        self.cancelled = True

    def mark_remote_terminated(self):
        self.remote_terminated = True

    async def hangup(self):
        self.hung_up = True


@pytest.fixture
def client():
    FakePhoneCall.instances = []
    app = create_app(
        Settings(
            voice_service_secret="s3cret",
            telnyx_api_key="KEY123",
            public_ws_base_url="wss://voice.example.com/",
        ),
        phone_call_factory=FakePhoneCall,
    )
    with TestClient(app) as c:
        c.service = app.state.service
        c.service.api = FakeApi()
        yield c


def prepare(client, **kwargs):
    return client.post("/phone/calls/prepare", json=prepare_body(**kwargs), headers=SECRET)


def ws_path(stream_url: str) -> tuple[str, str]:
    url = urlparse(stream_url)
    query = parse_qs(url.query)
    return f"{url.path}?{url.query}", query["token"][0]


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.mark.parametrize(
    "path", ["/phone/calls/prepare", "/phone/calls/abc/cancel", "/phone/calls/abc/hangup"]
)
def test_phone_endpoints_require_the_secret(client, path):
    assert client.post(path, json={}).status_code == 401
    assert client.post(path, json={}, headers={"X-Voice-Secret": "nope"}).status_code == 401


def test_prepare_returns_a_public_stream_url_with_a_one_time_token(client):
    response = prepare(client)
    assert response.status_code == 200
    stream_url = response.json()["stream_url"]
    url = urlparse(stream_url)
    assert (url.scheme, url.netloc, url.path) == ("wss", "voice.example.com", "/phone/ws")
    query = parse_qs(url.query)
    assert query["call"] == ["ph-1"]
    assert len(query["token"][0]) >= 32
    assert "ph-1" in client.service.phone


def test_preparing_a_known_call_is_409(client):
    assert prepare(client).status_code == 200
    assert prepare(client).status_code == 409


def test_prepare_without_a_public_ws_url_is_503():
    app = create_app(Settings(voice_service_secret="s3cret"), phone_call_factory=FakePhoneCall)
    with TestClient(app) as c:
        assert prepare(c).status_code == 503


def test_the_websocket_rejects_a_bad_token(client):
    path, _token = ws_path(prepare(client).json()["stream_url"])
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/phone/ws?call=ph-1&token=bad") as ws:
            ws.receive_text()
    assert FakePhoneCall.instances == []
    assert "ph-1" in client.service.phone


def test_the_websocket_runs_the_call_with_the_telnyx_serializer_and_the_token_is_single_use(client):
    path, _token = ws_path(prepare(client).json()["stream_url"])
    with client.websocket_connect(path) as ws:
        for message in telnyx_messages(stream_id="st-9", call_control_id="v2:ccid-1"):
            ws.send_text(message)
        assert wait_for(lambda: FakePhoneCall.instances and FakePhoneCall.instances[0].ran)

    call = FakePhoneCall.instances[0]
    assert call.serializer._stream_id == "st-9"
    assert call.serializer._call_control_id == "v2:ccid-1"
    assert call.serializer._api_key == "KEY123"
    assert call.agent.transport == "phone"
    assert call.agent.direction == "inbound"
    assert call.agent.transfer_enabled is False
    assert call.kwargs["call_uuid"] == "ph-1"
    assert call.kwargs["contact_name"] == "Ana Pérez"
    assert "ph-1" not in client.service.phone

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(path) as ws:
            ws.receive_text()
    assert len(FakePhoneCall.instances) == 1


def test_an_outbound_phone_call_carries_the_advisor_and_direction(client):
    path, _ = ws_path(prepare(client, direction="outbound").json()["stream_url"])
    with client.websocket_connect(path) as ws:
        for message in telnyx_messages(call_control_id="v2:out-ccid"):
            ws.send_text(message)
        assert wait_for(lambda: FakePhoneCall.instances and FakePhoneCall.instances[0].ran)
    call = FakePhoneCall.instances[0]
    assert call.agent.direction == "outbound"
    assert call.kwargs["initiated_by"] == "Carlos"
    assert call.serializer._call_control_id == "v2:out-ccid"


def test_a_stream_for_another_telnyx_call_is_rejected_and_reported(client):
    path, _ = ws_path(prepare(client).json()["stream_url"])
    with client.websocket_connect(path) as ws:
        for message in telnyx_messages(call_control_id="v2:someone-else"):
            ws.send_text(message)
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()
    assert FakePhoneCall.instances == []
    assert wait_for(lambda: client.service.api.ended_calls)
    assert client.service.api.ended_calls[0]["uuid"] == "ph-1"


def test_cancel_drops_a_pending_call_without_ended_and_is_idempotent(client):
    prepare(client)
    response = client.post("/phone/calls/ph-1/cancel", headers=SECRET)
    assert response.json() == {"ok": True, "found": True}
    assert "ph-1" not in client.service.phone
    again = client.post("/phone/calls/ph-1/cancel", headers=SECRET)
    assert again.json() == {"ok": True, "found": False}
    time.sleep(0.05)
    assert client.service.api.ended_calls == []


def test_cancel_drops_a_connected_phone_call_without_ended(client):
    call = FakePhoneCall(agent=base_agent(), call_uuid="ph-2", serializer=None)
    client.service.calls["ph-2"] = call
    client.service.phone_calls.add("ph-2")
    response = client.post("/phone/calls/ph-2/cancel", headers=SECRET)
    assert response.json() == {"ok": True, "found": True}
    assert call.cancelled
    assert "ph-2" not in client.service.calls


def test_cancel_refuses_a_whatsapp_call(client):
    client.service.calls["wa-1"] = FakePhoneCall(
        agent=base_agent(), call_uuid="wa-1", serializer=None
    )
    assert client.post("/phone/calls/wa-1/cancel", headers=SECRET).status_code == 409


def test_hangup_finalizes_a_connected_phone_call(client):
    call = FakePhoneCall(agent=base_agent(), call_uuid="ph-2", serializer=None)
    client.service.calls["ph-2"] = call
    client.service.phone_calls.add("ph-2")
    response = client.post("/phone/calls/ph-2/hangup", headers=SECRET)
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert call.remote_terminated and call.hung_up


def test_hangup_of_an_unanswered_outbound_call_reports_it_unanswered(client):
    prepare(client, direction="outbound")
    response = client.post("/phone/calls/ph-1/hangup", headers=SECRET)
    assert response.json()["ok"] is True
    assert "ph-1" not in client.service.phone
    assert wait_for(lambda: client.service.api.ended_calls)
    ended = client.service.api.ended_calls[0]
    assert (ended["uuid"], ended["summary"], ended["duration"]) == ("ph-1", UNANSWERED_SUMMARY, 0)


def test_hangup_of_an_unknown_call_is_ok(client):
    response = client.post("/phone/calls/nope/hangup", headers=SECRET)
    assert response.status_code == 200
    assert response.json()["ok"] is True
