"""Business tools: definitions -> LLM functions, the API call, and the prompt (no network)."""

import asyncio
import json

import httpx
import pytest

from rellachat import api_client
from rellachat.api_client import RellachatApi
from rellachat.business_tools import RESERVED_NAMES, function_schemas, valid_tools_path
from rellachat.models import AgentConfig
from rellachat.prompts import build_system_prompt

BOOK = {
    "type": "function",
    "function": {
        "name": "agendar_cita",
        "description": "Agenda una cita.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "inicio": {"type": "string"},
                "motivo": {"type": ["string", "null"]},
            },
            "required": ["inicio", "motivo", "ghost"],
            "additionalProperties": False,
        },
    },
}

AVAILABILITY = {
    "type": "function",
    "function": {"name": "consultar_disponibilidad", "description": "Horas libres.", "parameters": {"type": "object", "properties": {}}},
}


def agent(**extra) -> AgentConfig:
    return AgentConfig(identity_name="Sofía", company_name="Rella", **extra)


def test_definitions_become_function_schemas():
    schemas = function_schemas([BOOK, AVAILABILITY])

    assert [s.name for s in schemas] == ["agendar_cita", "consultar_disponibilidad"]
    assert schemas[0].description == "Agenda una cita."
    assert schemas[0].properties["motivo"] == {"type": ["string", "null"]}
    # A required key without a property is dropped
    assert schemas[0].required == ["inicio", "motivo"]


def test_invalid_duplicate_and_reserved_definitions_are_skipped():
    reserved = [{"type": "function", "function": {"name": name, "parameters": {}}} for name in RESERVED_NAMES]
    bad = [{}, {"function": "x"}, {"function": {"name": "Bad-Name"}}, "nope"]

    schemas = function_schemas([*reserved, *bad, AVAILABILITY, AVAILABILITY])

    assert [s.name for s in schemas] == ["consultar_disponibilidad"]
    assert function_schemas(None) == []


@pytest.mark.parametrize(
    "path,valid",
    [
        ("calls/abc-123/tools", True),
        ("agents/5/test/0b7f3c52-5d0e-4b7a-9a59-1f1c1c1c1c1c/tools", True),
        ("../secrets", False),
        ("calls/../../x", False),
        ("http://evil/tools", False),
        ("/calls/x/tools", False),
        ("", False),
        (None, False),
    ],
)
def test_tools_path_validation(path, valid):
    assert valid_tools_path(path) is valid


class FakeClient:
    """Stands in for httpx.AsyncClient, recording the request and replying with ``reply``."""

    calls: list[dict] = []
    reply: object = None

    def __init__(self, timeout):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json, headers):
        FakeClient.calls.append({"url": url, "json": json, "headers": headers, "timeout": self.timeout})
        if isinstance(FakeClient.reply, Exception):
            raise FakeClient.reply
        return FakeClient.reply


@pytest.fixture
def fake_http(monkeypatch):
    FakeClient.calls = []
    monkeypatch.setattr(api_client.httpx, "AsyncClient", FakeClient)
    return FakeClient


def test_run_tool_posts_the_arguments_and_returns_the_api_result(fake_http):
    fake_http.reply = httpx.Response(200, json={"ok": True, "cita_id": 7})
    api = RellachatApi("http://api.test/api/", "s3cret")

    result = asyncio.run(api.run_tool("calls/u-1/tools", "agendar_cita", {"inicio": "2026-10-15 09:30"}))

    assert result == {"ok": True, "cita_id": 7}
    call = fake_http.calls[0]
    assert call["url"] == "http://api.test/api/voice-service/calls/u-1/tools/agendar_cita"
    assert call["json"] == {"arguments": {"inicio": "2026-10-15 09:30"}}
    assert call["headers"]["X-Voice-Secret"] == "s3cret"
    assert call["timeout"] == api_client.TOOL_TIMEOUT


@pytest.mark.parametrize(
    "reply",
    [
        httpx.ConnectTimeout("timeout"),
        httpx.Response(500, text="boom"),
        httpx.Response(404, json={"message": "Not found"}),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json=["x"]),
    ],
)
def test_run_tool_failures_are_an_error_for_the_model(fake_http, reply):
    fake_http.reply = reply

    result = asyncio.run(RellachatApi("http://api.test/api", "s").run_tool("calls/u-1/tools", "agendar_cita", {}))

    assert result["ok"] is False
    assert result["error"]


def test_run_tool_without_api_url_or_a_bad_path_sends_nothing(fake_http):
    assert asyncio.run(RellachatApi("", "s").run_tool("calls/u/tools", "x", {}))["ok"] is False
    assert asyncio.run(RellachatApi("http://api.test", "s").run_tool("../x", "x", {}))["ok"] is False
    assert asyncio.run(RellachatApi("http://api.test", "s").run_tool(None, "x", {}))["ok"] is False
    assert fake_http.calls == []


def test_the_prompt_explains_the_business_tools_and_carries_the_api_rules():
    prompt = build_system_prompt(
        agent(tools=[BOOK, AVAILABILITY], tools_prompt="AGENDA:\n- Fecha y hora actual del negocio: miércoles 2026-10-14 09:10", tools_path="calls/u/tools")
    )

    assert "Herramientas del negocio" in prompt
    assert "déjame revisar" in prompt
    assert "sirvan a tu propósito" in prompt
    assert "agendar, reprogramar o cancelar" in prompt
    assert "espera que confirme" in prompt
    assert "AGENDA:" in prompt
    assert "WhatsApp" in prompt  # the call itself is on WhatsApp; messaging rule below is separate
    assert "te lo envío por WhatsApp" not in prompt


def test_messaging_tools_tell_the_caller_it_goes_to_whatsapp_and_tests_are_simulated():
    send = {"type": "function", "function": {"name": "enviar_producto", "parameters": {"type": "object", "properties": {}}}}

    prompt = build_system_prompt(agent(tools=[send]), test_mode=True)

    assert "te lo envío por WhatsApp" in prompt
    assert "se simulan" in prompt


def test_without_tools_the_prompt_has_no_business_section():
    assert "Herramientas del negocio" not in build_system_prompt(agent())


def test_the_agent_config_accepts_the_api_tool_fields():
    config = AgentConfig.model_validate(
        {"id": 1, "tools": [BOOK], "tools_prompt": "x", "tools_path": "calls/u/tools", "appointment_duration_minutes": 30}
    )

    assert config.tools == [BOOK]
    assert config.tools_path == "calls/u/tools"
    assert json.loads(json.dumps(config.tools))[0]["function"]["name"] == "agendar_cita"
