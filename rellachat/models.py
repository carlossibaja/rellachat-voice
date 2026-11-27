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
    appointment_duration_minutes: int | None = None
    # Business tools (OpenAI function definitions), what the model must know to use them, and
    # the API path they run through (``calls/{uuid}/tools`` or a test session's path)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tools_prompt: str | None = None
    tools_path: str | None = None
    # ``outbound`` when the business places the call (T9); inbound keeps the original behavior
    direction: str = "inbound"

    @property
    def is_outbound(self) -> bool:
        return self.direction == "outbound"

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


class OutboundWhatsApp(BaseModel):
    """Meta credentials of the business number placing an outbound call."""

    model_config = ConfigDict(extra="ignore")

    token: str = Field(min_length=1)
    phone_number_id: str = Field(min_length=1)


class Initiator(BaseModel):
    """Advisor who placed the outbound AI call."""

    model_config = ConfigDict(extra="ignore")

    id: int | str | None = None
    name: str | None = None


class OutboundOfferRequest(BaseModel):
    """Body of ``POST /calls/outbound/offer``."""

    call_uuid: str = Field(min_length=1)
    whatsapp: OutboundWhatsApp
    agent: AgentConfig
    contact: Contact = Field(default_factory=Contact)
    initiated_by: Initiator = Field(default_factory=Initiator)


class OutboundAnswerRequest(BaseModel):
    """Body of ``POST /calls/{call_uuid}/answer``: the customer's SDP answer from Meta."""

    sdp: str = Field(min_length=1)
    type: str = "answer"
    wa_call_id: str = Field(min_length=1)


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
