"""Callbacks from the voice service to the Rellachat API.

Every call is best-effort: failures are logged and never raised, so an API
outage cannot crash a live phone call.
"""

from typing import Any

import httpx
from loguru import logger

from rellachat.business_tools import error_result, valid_tools_path
from rellachat.security import SECRET_HEADER

# Seconds a business tool may take: the caller is waiting on the line
TOOL_TIMEOUT = 10.0


class RellachatApi:
    """Thin client for ``{RELLACHAT_API_URL}/voice-service/calls/{uuid}/...``."""

    def __init__(self, base_url: str, secret: str, timeout: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._secret = secret
        self._timeout = timeout

    async def _post(
        self, call_uuid: str, action: str, payload: dict[str, Any] | None = None
    ) -> bool:
        if not self._base_url:
            logger.warning(f"RELLACHAT_API_URL not set; skipping '{action}' for call {call_uuid}")
            return False
        url = f"{self._base_url}/voice-service/calls/{call_uuid}/{action}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    url,
                    json=payload or {},
                    headers={SECRET_HEADER: self._secret, "Accept": "application/json"},
                )
            if response.status_code >= 400:
                logger.error(
                    f"API '{action}' for call {call_uuid} failed: {response.status_code} {response.text[:300]}"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"API '{action}' for call {call_uuid} errored: {e}")
            return False

    async def answered(self, call_uuid: str) -> bool:
        return await self._post(call_uuid, "answered")

    async def transfer(self, call_uuid: str, summary: str) -> bool:
        return await self._post(call_uuid, "transfer", {"summary": summary})

    async def transfer_cancelled(self, call_uuid: str) -> bool:
        return await self._post(call_uuid, "transfer-cancelled")

    async def transferred(self, call_uuid: str, user_id: int | str) -> bool:
        return await self._post(call_uuid, "transferred", {"user_id": user_id})

    async def ended(
        self, call_uuid: str, transcript: list[dict], summary: str, duration: int
    ) -> bool:
        return await self._post(
            call_uuid,
            "ended",
            {"transcript": transcript, "summary": summary, "duration": duration},
        )

    async def run_tool(
        self, tools_path: str | None, name: str, arguments: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Run one business tool through the API; any failure is an error result for the model."""
        if not self._base_url or not valid_tools_path(tools_path):
            return error_result()
        url = f"{self._base_url}/voice-service/{tools_path}/{name}"
        try:
            async with httpx.AsyncClient(timeout=TOOL_TIMEOUT) as client:
                response = await client.post(
                    url,
                    json={"arguments": arguments or {}},
                    headers={SECRET_HEADER: self._secret, "Accept": "application/json"},
                )
            result = response.json() if response.status_code < 400 else None
        except Exception:
            return error_result()
        return result if isinstance(result, dict) and "ok" in result else error_result()
