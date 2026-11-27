"""Short Spanish summaries of a call transcript via the Anthropic Messages API."""

from anthropic import AsyncAnthropic
from loguru import logger

from rellachat.prompts import (
    SUMMARY_SYSTEM_PROMPT,
    TRANSFER_SUMMARY_SYSTEM_PROMPT,
    format_transcript,
)


async def _summarize(api_key: str, model: str, system: str, transcript: list[dict]) -> str:
    text = format_transcript(transcript)
    if not text:
        return "La llamada terminó sin conversación registrada."
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set; skipping summary")
        return ""
    try:
        client = AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model=model,
            max_tokens=400,
            system=system,
            messages=[{"role": "user", "content": f"Transcripción:\n{text}"}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return ""


async def summarize_call(api_key: str, model: str, transcript: list[dict]) -> str:
    """Summary stored on the call when it ends."""
    return await _summarize(api_key, model, SUMMARY_SYSTEM_PROMPT, transcript)


async def summarize_for_transfer(api_key: str, model: str, transcript: list[dict]) -> str:
    """Summary shown to advisors while the transfer rings."""
    summary = await _summarize(api_key, model, TRANSFER_SUMMARY_SYSTEM_PROMPT, transcript)
    return summary or "El cliente pidió hablar con un asesor."
