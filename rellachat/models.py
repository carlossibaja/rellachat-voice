"""Request/response models for the voice service HTTP contract."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_TRANSFER_RING_SECONDS = 25


class AgentConfig(BaseModel):
    """Voice agent configuration sent by the Rellachat API."""

    model_config = ConfigDict(extra="ignore")

    id: int | str | None = None
    name: str = ""
    language: str = "es"
    identity_name: str = ""
    purpose: str = "customer_support"
    company_name: str = ""
    company_website: str | None = None
    company_description: str | None = None
    instructions: str | None = None
    greeting: str | None = None
    voice_id: str | None = None
    transfer_enabled: bool = True
    transfer_ring_seconds: int | None = DEFAULT_TRANSFER_RING_SECONDS

    @property
    def ring_seconds(self) -> int:
        """Transfer ring window, falling back to the default when unset/invalid."""
        value = self.transfer_ring_seconds
        if not value or value <= 0:
            return DEFAULT_TRANSFER_RING_SECONDS
        return value


class WhatsAppPayload(BaseModel):
    """WhatsApp credentials plus the original Meta webhook body."""

    model_config = ConfigDict(extra="ignore")

    token: str | None = None
    phone_number_id: str | None = None
    webhook: dict[str, Any]


class Contact(BaseModel):
    """Caller contact info."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    phone: str | None = None


class ConnectRequest(BaseModel):
    """Body of ``POST /calls/connect``."""

    call_uuid: str
    whatsapp: WhatsAppPayload
    agent: AgentConfig
    contact: Contact = Field(default_factory=Contact)


class TerminateRequest(BaseModel):
    """Body of ``POST /calls/terminate``."""

    call_uuid: str
    whatsapp: WhatsAppPayload


class AdvisorUser(BaseModel):
    """Advisor who accepted the transfer."""

    model_config = ConfigDict(extra="ignore")

    id: int | str
    name: str | None = None


class AdvisorOfferRequest(BaseModel):
    """Body of ``POST /calls/{call_uuid}/advisor-offer``."""

    sdp: str
    type: str = "offer"
    user: AdvisorUser


class TestOfferRequest(BaseModel):
    """Body of ``POST /test/offer``."""

    agent: AgentConfig
    sdp: str
    type: str = "offer"
