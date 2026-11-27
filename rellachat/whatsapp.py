"""WhatsApp helpers: webhook parsing and one WhatsAppClient per phone number."""

from typing import Any

import aiohttp
from pipecat.transports.smallwebrtc.connection import IceServer
from pipecat.transports.whatsapp.api import (
    WhatsAppConnectCallValue,
    WhatsAppTerminateCallValue,
    WhatsAppWebhookRequest,
)
from pipecat.transports.whatsapp.client import WhatsAppClient


class InvalidWebhookError(ValueError):
    """The forwarded Meta webhook body is not a usable calls event."""


def parse_webhook(body: dict[str, Any], expected_event: str) -> WhatsAppWebhookRequest:
    """Validate the forwarded Meta webhook and check it carries ``expected_event``."""
    try:
        request = WhatsAppWebhookRequest.model_validate(body)
    except Exception as e:
        raise InvalidWebhookError(f"Invalid WhatsApp webhook: {e}") from e

    value_type = (
        WhatsAppConnectCallValue if expected_event == "connect" else WhatsAppTerminateCallValue
    )
    for entry in request.entry:
        for change in entry.changes:
            if isinstance(change.value, value_type) and any(
                call.event == expected_event for call in change.value.calls
            ):
                return request
    raise InvalidWebhookError(f"Webhook has no '{expected_event}' call event")


def webhook_phone_number_id(request: WhatsAppWebhookRequest) -> str | None:
    """Business phone_number_id from the webhook metadata."""
    for entry in request.entry:
        for change in entry.changes:
            return change.value.metadata.phone_number_id
    return None


def webhook_call_id(request: WhatsAppWebhookRequest) -> str | None:
    """Meta call id (wacid) of the first call in the webhook."""
    for entry in request.entry:
        for change in entry.changes:
            for call in change.value.calls:
                return call.id
    return None


class WhatsAppClientRegistry:
    """Keeps one WhatsAppClient per phone_number_id.

    ``WhatsAppClient.update_whatsapp_phone_number_id`` does not rebuild the
    Graph URL, so clients are never re-pointed; only the token is refreshed.
    No app secret is passed: the Rellachat API already verified the webhook.
    """

    def __init__(self, session: aiohttp.ClientSession, ice_servers: list[IceServer]):
        self._session = session
        self._ice_servers = ice_servers
        self._clients: dict[str, WhatsAppClient] = {}
        self._tokens: dict[str, str] = {}

    def get(self, phone_number_id: str) -> WhatsAppClient | None:
        return self._clients.get(phone_number_id)

    def get_or_create(self, phone_number_id: str, token: str) -> WhatsAppClient:
        client = self._clients.get(phone_number_id)
        if client is None:
            client = WhatsAppClient(
                whatsapp_token=token,
                phone_number_id=phone_number_id,
                session=self._session,
                ice_servers=self._ice_servers,
            )
            self._clients[phone_number_id] = client
            self._tokens[phone_number_id] = token
        elif token and self._tokens.get(phone_number_id) != token:
            client.update_whatsapp_token(token)
            self._tokens[phone_number_id] = token
        return client
