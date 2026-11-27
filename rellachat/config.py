"""Runtime configuration for the Rellachat voice service.

Values come from environment variables, optionally loaded from
``rellachat/.env`` (never committed). Settings are read lazily so tests can
override the environment.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pipecat.transports.smallwebrtc.connection import IceServer

ENV_FILE = Path(__file__).resolve().parent / ".env"
# Also accepted: the repo root .env (git-ignored), for setups that keep it next to pyproject.toml
ROOT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"
DEFAULT_PORT = 7860
DEFAULT_STT_MODEL = "ink-2"
DEFAULT_ICE_SERVERS = [IceServer(urls="stun:stun.l.google.com:19302")]


def load_env() -> None:
    """Load ``rellachat/.env``, then the repo root ``.env``, without overriding real env vars."""
    for env_file in (ENV_FILE, ROOT_ENV_FILE):
        if env_file.exists():
            load_dotenv(env_file, override=False)


def parse_ice_servers(raw: str | None) -> list[IceServer]:
    """Parse the ``ICE_SERVERS`` JSON env var.

    Accepts a JSON list whose items are either URL strings or objects with
    ``urls`` and optional ``username``/``credential``. Invalid input falls back
    to the public Google STUN server.
    """
    if not raw or not raw.strip():
        return list(DEFAULT_ICE_SERVERS)
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("ICE_SERVERS must be a JSON list")
        servers: list[IceServer] = []
        for item in data:
            if isinstance(item, str):
                servers.append(IceServer(urls=item))
            elif isinstance(item, dict) and item.get("urls"):
                servers.append(
                    IceServer(
                        urls=item["urls"],
                        username=item.get("username"),
                        credential=item.get("credential"),
                    )
                )
            else:
                raise ValueError(f"Invalid ICE server entry: {item!r}")
        return servers or list(DEFAULT_ICE_SERVERS)
    except Exception as e:
        logger.warning(f"Ignoring invalid ICE_SERVERS ({e}); using default STUN")
        return list(DEFAULT_ICE_SERVERS)


@dataclass
class Settings:
    """Service settings resolved from the environment."""

    anthropic_api_key: str = ""
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    cartesia_api_key: str = ""
    cartesia_voice_id: str = ""
    # Cartesia STT model: ink-2 ends turns faster and hears short answers ("sí") better than ink-whisper
    cartesia_stt_model: str = DEFAULT_STT_MODEL
    rellachat_api_url: str = ""
    voice_service_secret: str = ""
    port: int = DEFAULT_PORT
    ice_servers: list[IceServer] = field(default_factory=lambda: list(DEFAULT_ICE_SERVERS))
    # Telnyx phone calls: the API key lets the serializer hang up calls the agent ends, and the
    # public ``wss://`` base is where Telnyx streams media (the service must be reachable there)
    telnyx_api_key: str = ""
    public_ws_base_url: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from the current process environment."""
        port_raw = os.getenv("PORT", "")
        try:
            port = int(port_raw) if port_raw else DEFAULT_PORT
        except ValueError:
            port = DEFAULT_PORT
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            anthropic_model=os.getenv("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL,
            cartesia_api_key=os.getenv("CARTESIA_API_KEY", ""),
            cartesia_voice_id=os.getenv("CARTESIA_VOICE_ID", ""),
            cartesia_stt_model=(os.getenv("CARTESIA_STT_MODEL") or "").strip() or DEFAULT_STT_MODEL,
            rellachat_api_url=(os.getenv("RELLACHAT_API_URL") or "").rstrip("/"),
            voice_service_secret=os.getenv("VOICE_SERVICE_SECRET", ""),
            port=port,
            ice_servers=parse_ice_servers(os.getenv("ICE_SERVERS")),
            telnyx_api_key=os.getenv("TELNYX_API_KEY", ""),
            public_ws_base_url=(os.getenv("PUBLIC_WS_BASE_URL") or "").strip().rstrip("/"),
        )
