"""Business-initiated (outbound) WhatsApp calls: the caller-side WebRTC peer.

Pipecat's ``WhatsAppClient`` only answers user-initiated calls, where Meta sends
the SDP offer. For an outbound call the business sends the offer: the Rellachat
API passes the one created here to Meta's ``connect`` action and forwards the
customer's SDP answer back to ``apply_answer``. Meta signaling stays in the API.
"""

from aiortc import RTCSessionDescription
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

# Seconds the customer has to answer before the call is torn down as unanswered
ANSWER_TIMEOUT_SECS = 60


def filter_sdp_for_whatsapp(sdp: str) -> str:
    """Keep only SHA-256 DTLS fingerprints (all WhatsApp supports), with CRLF line endings.

    Same rule as Pipecat's ``WhatsAppClient._filter_sdp_for_whatsapp`` for answers.
    """
    lines = [
        line
        for line in sdp.splitlines()
        if not (line.startswith("a=fingerprint:") and not line.startswith("a=fingerprint:sha-256"))
    ]
    return "\r\n".join(lines) + "\r\n"


class OutboundWebRTCConnection(SmallWebRTCConnection):
    """A ``SmallWebRTCConnection`` that makes the SDP offer instead of answering one.

    ``SmallWebRTCConnection`` (Pipecat 1.17) only exposes ``initialize(offer)`` ->
    answer. This adds the offerer path on the same aiortc ``RTCPeerConnection``, so
    ``SmallWebRTCTransport`` uses it unchanged: the audio transceiver is the first
    one (``replace_audio_track`` / ``audio_input_track`` use index 0) and the
    ``connected`` event fires once ICE/DTLS complete after the answer is applied.
    """

    async def create_offer(self) -> dict[str, str]:
        """Create the audio-only SDP offer (ICE candidates gathered, WhatsApp-filtered)."""
        self.pc.addTransceiver("audio", direction="sendrecv")
        offer = await self.pc.createOffer()
        # aiortc gathers every ICE candidate before this returns (no trickle ICE)
        await self.pc.setLocalDescription(offer)
        return {"sdp": filter_sdp_for_whatsapp(self.pc.localDescription.sdp), "type": "offer"}

    async def apply_answer(self, sdp: str, type: str = "answer") -> None:
        """Apply the customer's SDP answer (forwarded from Meta's connect webhook)."""
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type))
