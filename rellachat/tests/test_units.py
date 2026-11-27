"""Deterministic unit tests (no network, no provider calls)."""

import pytest
from fastapi.testclient import TestClient
from pipecat.transports.whatsapp.api import WhatsAppConnectCallValue, WhatsAppWebhookRequest

from rellachat.bot import to_language
from rellachat.config import Settings, parse_ice_servers
from rellachat.models import AgentConfig
from rellachat.prompts import build_greeting, build_system_prompt, format_transcript
from rellachat.security import secret_matches
from rellachat.server import create_app
from rellachat.whatsapp import (
    InvalidWebhookError,
    parse_webhook,
    webhook_call_id,
    webhook_phone_number_id,
)

SDP_OFFER = (
    "v=0\r\no=- 7669997803033704573 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n"
    "a=group:BUNDLE 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\nc=IN IP4 0.0.0.0\r\n"
    "a=rtpmap:111 opus/48000/2\r\n"
)


def connect_webhook() -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "1234567890",
                "changes": [
                    {
                        "field": "calls",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "PNID-1",
                            },
                            "contacts": [{"profile": {"name": "Ana"}, "wa_id": "573001112233"}],
                            "calls": [
                                {
                                    "id": "wacid.ABC",
                                    "from": "573001112233",
                                    "to": "15551234567",
                                    "event": "connect",
                                    "timestamp": "1764100000",
                                    "direction": "USER_INITIATED",
                                    "session": {"sdp_type": "offer", "sdp": SDP_OFFER},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }


def terminate_webhook() -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "1234567890",
                "changes": [
                    {
                        "field": "calls",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "PNID-1",
                            },
                            "calls": [
                                {
                                    "id": "wacid.ABC",
                                    "from": "573001112233",
                                    "to": "15551234567",
                                    "event": "terminate",
                                    "timestamp": "1764100060",
                                    "direction": "USER_INITIATED",
                                    "status": "COMPLETED",
                                    "duration": 60,
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }


def agent(**overrides) -> AgentConfig:
    data = {
        "id": 7,
        "name": "Soporte",
        "language": "es",
        "identity_name": "Sofía",
        "purpose": "appointments",
        "company_name": "Rella Shop",
        "company_website": "https://rella.shop",
        "company_description": "Tienda de ropa en Bogotá.",
        "instructions": "Ofrece citas de lunes a viernes.",
        "greeting": None,
        "voice_id": None,
        "transfer_enabled": True,
        "transfer_ring_seconds": 25,
    }
    data.update(overrides)
    return AgentConfig(**data)


# ------------------------------------------------------------------ webhook


def test_parse_connect_webhook_into_pipecat_model():
    request = parse_webhook(connect_webhook(), "connect")
    assert isinstance(request, WhatsAppWebhookRequest)
    value = request.entry[0].changes[0].value
    assert isinstance(value, WhatsAppConnectCallValue)
    assert value.calls[0].from_ == "573001112233"
    assert value.calls[0].session.sdp.endswith("\r\n")
    assert webhook_phone_number_id(request) == "PNID-1"
    assert webhook_call_id(request) == "wacid.ABC"


def test_parse_terminate_webhook():
    request = parse_webhook(terminate_webhook(), "terminate")
    assert webhook_call_id(request) == "wacid.ABC"


def test_parse_webhook_rejects_wrong_event_and_garbage():
    with pytest.raises(InvalidWebhookError):
        parse_webhook(terminate_webhook(), "connect")
    with pytest.raises(InvalidWebhookError):
        parse_webhook({"object": "x"}, "connect")


# ------------------------------------------------------------------ prompts


def test_system_prompt_includes_agent_context_and_tools():
    prompt = build_system_prompt(agent())
    assert "Sofía" in prompt and "Rella Shop" in prompt
    assert "Tienda de ropa en Bogotá." in prompt
    assert "Ofrece citas de lunes a viernes." in prompt
    assert "agendamiento" in prompt
    assert "una sola pregunta a la vez" in prompt
    assert "transfer_to_advisor" in prompt
    assert "end_call" in prompt


def test_system_prompt_without_transfer_and_test_mode():
    prompt = build_system_prompt(agent(transfer_enabled=False), test_mode=True)
    assert "transfer_to_advisor" not in prompt
    assert "No puedes transferir" in prompt
    assert "llamada de prueba" in prompt


def test_greeting_uses_configured_or_default():
    assert build_greeting(agent(greeting="  Buenas, ¿qué tal?  ")) == "Buenas, ¿qué tal?"
    assert build_greeting(agent()) == "Hola, te habla Sofía de Rella Shop. ¿En qué te puedo ayudar?"
    assert "gracias por llamar" in build_greeting(agent(identity_name="", name="", company_name=""))


def test_greeting_fills_the_client_placeholder():
    greeting = "hola {{cliente}}, soy Sofía"
    assert build_greeting(agent(greeting=greeting), "Carlos Sibaja") == "hola Carlos, soy Sofía"
    assert build_greeting(agent(greeting="Hola {{ Cliente }}, ¿cómo estás?"), "Ana") == "Hola Ana, ¿cómo estás?"
    # Unknown caller: the placeholder goes away without leaving stray punctuation
    assert build_greeting(agent(greeting="hola {{cliente}}, soy Sofía"), None) == "hola, soy Sofía"
    assert build_greeting(agent(greeting="Hola {{nombre}}. ¿Qué tal?"), "  ") == "Hola. ¿Qué tal?"


def test_ring_seconds_defaults():
    assert agent(transfer_ring_seconds=None).ring_seconds == 25
    assert agent(transfer_ring_seconds=0).ring_seconds == 25
    assert agent(transfer_ring_seconds=40).ring_seconds == 40


def test_format_transcript_labels_roles():
    text = format_transcript(
        [
            {"role": "user", "text": "Hola", "at": "t"},
            {"role": "assistant", "text": "Buenas", "at": "t"},
            {"role": "advisor", "text": "", "at": "t"},
        ]
    )
    assert text == "Cliente: Hola\nAgente IA: Buenas"


def test_to_language():
    assert to_language("es").value == "es"
    assert to_language("es-MX").value == "es-MX"
    assert to_language("xx-YY").value == "es"


def test_parse_ice_servers():
    servers = parse_ice_servers(
        '[{"urls": "turn:t.example:3478", "username": "u", "credential": "p"}]'
    )
    assert servers[0].urls == "turn:t.example:3478" and servers[0].username == "u"
    assert parse_ice_servers("not json")[0].urls == "stun:stun.l.google.com:19302"
    assert parse_ice_servers(None)[0].urls == "stun:stun.l.google.com:19302"


# ------------------------------------------------------------------- secret


def test_secret_matches():
    assert secret_matches("s3cret", "s3cret")
    assert not secret_matches("wrong", "s3cret")
    assert not secret_matches(None, "s3cret")
    assert not secret_matches("", "")


@pytest.fixture
def client():
    app = create_app(Settings(voice_service_secret="s3cret"))
    with TestClient(app) as c:
        yield c


def test_health_is_public(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


@pytest.mark.parametrize(
    "path", ["/calls/connect", "/calls/terminate", "/calls/abc/advisor-offer", "/test/offer"]
)
def test_protected_endpoints_require_secret(client, path):
    assert client.post(path, json={}).status_code == 401
    assert client.post(path, json={}, headers={"X-Voice-Secret": "nope"}).status_code == 401


def test_advisor_offer_without_pending_transfer_is_409(client):
    response = client.post(
        "/calls/unknown/advisor-offer",
        json={"sdp": SDP_OFFER, "user": {"id": 1, "name": "Ana"}},
        headers={"X-Voice-Secret": "s3cret"},
    )
    assert response.status_code == 409


def test_connect_rejects_invalid_webhook(client):
    response = client.post(
        "/calls/connect",
        json={
            "call_uuid": "u-1",
            "whatsapp": {"token": "t", "phone_number_id": "PNID-1", "webhook": terminate_webhook()},
            "agent": agent().model_dump(),
            "contact": {"name": "Ana", "phone": "573001112233"},
        },
        headers={"X-Voice-Secret": "s3cret"},
    )
    assert response.status_code == 422


def test_the_stt_model_comes_from_the_environment(monkeypatch):
    from rellachat.config import Settings

    monkeypatch.setenv("CARTESIA_STT_MODEL", " ink-whisper ")
    assert Settings.from_env().cartesia_stt_model == "ink-whisper"
    monkeypatch.delenv("CARTESIA_STT_MODEL")
    assert Settings.from_env().cartesia_stt_model == "ink-2"
